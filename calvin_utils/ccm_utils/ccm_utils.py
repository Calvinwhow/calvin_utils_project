from nilearn import plotting
import nibabel as nib
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from tqdm import tqdm

class ConvergentMapGenerator:
    def __init__(self, corr_map_dict, data_loader, mask_path=None, out_dir=None, weight=False):
        self.corr_map_dict = corr_map_dict
        self.data_loader = data_loader
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.weight = weight
        self._handle_nans()
        
    def _handle_nans(self):
        drop_list = []
        for key in self.corr_map_dict.keys():
            if np.isnan(self.corr_map_dict[key]).all():
                print(f"Warning: The correlation map for {key} contains only NaNs and will be excluded from the analysis.")
                drop_list.append(key)
            elif np.isnan(self.corr_map_dict[key]).any():
                self.corr_map_dict[key] = np.nan_to_num(self.corr_map_dict[key], nan=0, posinf=1, neginf=-1)
            else:
                continue
        
        for key in drop_list:
            del self.corr_map_dict[key]
            
    def generate_weighted_average_r_map(self):
        r_maps = np.array(list(self.corr_map_dict.values()))
        if self.weight:
            weights = []
            for dataset_name in self.corr_map_dict.keys():
                data = self.data_loader.load_dataset(dataset_name)
                weights.append(data['niftis'].shape[0])
            weights = np.array(weights)
            return np.average(r_maps, axis=0, weights=weights)
        else:
            return np.mean(r_maps, axis=0)

    def generate_agreement_map(self):
        r_maps = np.array(list(self.corr_map_dict.values()))
        signs = np.sign(r_maps)
        agreement = np.all(signs == signs[0], axis=0)
        return agreement.astype(int)
    
    def _load_nifti(self, path):
        img = nib.load(path)
        return img.get_fdata()
    
    def _mask_array(self, data_array, threshold=0):
        if self.mask_path is None:
            from nimlab import datasets as nimds
            mask = nimds.get_img("mni_icbm152")
        else:
            mask = nib.load(self.mask_path)

        mask_data = mask.get_fdata()
        mask_indices = mask_data.flatten() > threshold
        
        masked_array = data_array.flatten()[mask_indices]
        return masked_array
    
    def _unmask_array(self, data_array, threshold=0):
        if self.mask_path is None:
            from nimlab import datasets as nimds
            mask = nimds.get_img("mni_icbm152")
        else:
            mask = nib.load(self.mask_path)

        mask_data = mask.get_fdata()
        mask_indices = mask_data.flatten() > threshold
        
        unmasked_array = np.zeros(mask_indices.shape)
        unmasked_array[mask_indices] = data_array.flatten()
        return unmasked_array.reshape(mask_data.shape), mask.affine

    def _save_map(self, map_data, file_name):
        unmasked_map, mask_affine = self._unmask_array(map_data)
        img = nib.Nifti1Image(unmasked_map, affine=mask_affine)
        if self.out_dir is not None:
            file_path = os.path.join(self.out_dir, 'convergence_map', file_name)
            nib.save(img, file_path)
        return img

    def _visualize_map(self, img, title):
        plotting.view_img(img, title=title).open_in_browser()
        
    def generate_and_save_maps(self):
        # Generate weighted average r map
        weighted_avg_map = self.generate_weighted_average_r_map()
        try:
            weighted_avg_img = self._save_map(weighted_avg_map, 'weighted_average_r_map.nii.gz')
            self._visualize_map(weighted_avg_img, 'Weighted Average R Map')
        except:
            pass

        # Generate agreement map
        agreement_map = self.generate_agreement_map()
        try:
            agreement_img = self._save_map(agreement_map, 'agreement_map.nii.gz')
            self._visualize_map(agreement_img, 'Agreement Map')
        except:
            pass
    
    def save_individual_r_maps(self):
        for dataset_name, r_map in self.corr_map_dict.items():
            r_img = self._save_map(r_map, f'{dataset_name}_correlation_map.nii.gz')
            self._visualize_map(r_img, f'{dataset_name} Correlation Map')

class LOOCVAnalyzer(ConvergentMapGenerator):
    def __init__(self, corr_map_dict, data_loader, mask_path=None, out_dir=None, weight=False, method='spearman', convergence_type='agreement', similarity='cos', n_bootstrap=1000, roi_path=None):
        """
        Initialize the LOOCVAnalyzer.

        Parameters:
        -----------
        corr_map_dict : dict
            Dictionary containing correlation maps for each dataset.
        data_loader : DataLoader
            Instance of DataLoader to load datasets.
        mask_path : str, optional
            Path to the mask file.
        out_dir : str, optional
            Output directory to save maps.
        weight : bool, optional
            Whether to weight the datasets.
        method : str, optional
            Correlation method to use ('spearman' or 'pearson').
        n_bootstrap : int, optional
            Number of bootstrap samples to generate.
        convergence_type : str, optional
            Type of convergence to use ('agreement' or other types). Default is 'agreement'.
        similarity : str, optional
            Similarity measure to use ('cos' for cosine similarity or other measures). Default is 'cos'.
            Number of bootstrap samples to generate. Default is 1000.
        roi_path : str, optional
            Path to ROI file to use in place of convergent map. 
            Cosine similarity is best to use with this choice.
        """
        super().__init__(corr_map_dict, data_loader, mask_path, out_dir, weight)
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.similarity = similarity
        self.convergence_type = convergence_type
        self.roi_path = roi_path
        self.correlation_calculator = CorrelationCalculator(method=method)
        self.results = self.perform_loocv()
        self.results_df = self.results_to_dataframe()
    
    def results_to_dataframe(self):
        """
        Convert the LOOCV results to a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the R-value, lower confidence interval, upper confidence interval, and mean R-value for each dataset.
        """
        columns = ['Dataset', 'CI Lower', 'CI Upper', 'Mean R']
        data = []
        for i, (ci_lower, ci_upper, mean_r) in enumerate(self.results):
            try:
                dataset_name = list(self.corr_map_dict.keys())[i]
                data.append([dataset_name, ci_lower, ci_upper, mean_r])
            except:
                continue
        return pd.DataFrame(data, columns=columns)

    def generate_convergent_roi(self):
        """
        Generate the convergent map using the region of interest (ROI) file.

        Returns:
        --------
        np.array
            Convergent map.
        """
        roi_data = self._load_nifti(self.roi_path)
        roi_data = self._mask_array(roi_data)
        return roi_data

    def perform_loocv(self):
        """
        Perform Leave-One-Out Cross-Validation (LOOCV) analysis.

        Returns:
        --------
        list of tuple
            List of tuples containing the R-value and confidence intervals for each dataset.
        """
        results = []
        dataset_names = list(self.corr_map_dict.keys())
        for i, test_dataset_name in enumerate(dataset_names):
            print("Evaluating dataset:", test_dataset_name)
            # Load the test dataset
            test_data = self.data_loader.load_dataset(test_dataset_name)
            test_niftis = test_data['niftis']
            test_indep_var = test_data['indep_var']

            # TRAIN - Generate the convergent map using the training datasets (or an ROI)
            if self.roi_path is not None:
                convergent_map = self.generate_convergent_roi()
            elif self.convergence_type == 'average':
                train_dataset_names = dataset_names[:i] + dataset_names[i+1:]
                self.corr_map_dict = self.generate_correlation_maps(train_dataset_names)
                convergent_map = self.generate_weighted_average_r_map()
            elif self.convergence_type == 'agreement':
                train_dataset_names = dataset_names[:i] + dataset_names[i+1:]
                self.corr_map_dict = self.generate_correlation_maps(train_dataset_names)
                convergent_map = self.generate_agreement_map()
            else:
                raise ValueError("Invalid convergence type (self.convergence_type). Please choose 'average', 'agreement', or set path to a region of interest to test (self.roi_path).")

            # TEST - use the convergent map on the test dataset
            similarities = self.calculate_similarity(test_niftis, convergent_map)
            ci_lower, ci_upper, mean_r = self.correlate_similarity_with_outcomes(similarities, test_indep_var)
            results.append((ci_lower, ci_upper, mean_r))
        return results

    def generate_correlation_maps(self, dataset_names):
        """
        Generate correlation maps for the given dataset names.

        Parameters:
        -----------
        dataset_names : list of str
            List of dataset names.

        Returns:
        --------
        dict
            Dictionary containing correlation maps for each dataset.
        """
        correlation_maps = {}
        for dataset_name in dataset_names:
            data = self.data_loader.load_dataset(dataset_name)
            self.correlation_calculator._process_data(data)
            correlation_maps[dataset_name] = self.correlation_calculator.correlation_map
        return correlation_maps

    def calculate_similarity(self, patient_maps, convergent_map):
        """
        Calculate cosine similarity between patient maps and the convergent map.

        Parameters:
        -----------
        patient_maps : np.array
            Array of patient maps.
        convergent_map : np.array
            Convergent map.

        Returns:
        --------
        list of float
            List of cosine similarity values.
        """
        if self.similarity == 'cos':
            similarities = [self.cosine_similarity(patient_map, convergent_map) for patient_map in patient_maps]
        elif self.similarity == 'spcorr':
            similarities = [pearsonr(patient_map, convergent_map)[0] for patient_map in patient_maps]
        else:
            raise ValueError("Invalid similarity measure (self.similarity). Please choose 'cos' or 'spcorr'.")
        return similarities
    
    def cosine_similarity(self, a, b):
        """
        Calculate the cosine similarity between two vectors.

        Parameters:
        -----------
        a : np.array
            First vector.
        b : np.array
            Second vector.

        Returns:
        --------
        float
            Cosine similarity value.
        """
        a = a.flatten()
        b = b.flatten()
        numerator = np.dot(a, b)
        denominator = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
        similarity = numerator / denominator
        return similarity
    
    def correlate_similarity_with_outcomes(self, similarities, indep_var):
        """
        Correlate similarity values with independent variables and calculate confidence intervals.

        Parameters:
        -----------
        similarities : list of float
            List of cosine similarity values.
        indep_var : np.array
            Array of independent variable values.

        Returns:
        --------
        tuple
            R-value, lower confidence interval, and upper confidence interval.
        """
        resampled_r = []
        for _ in tqdm(range(self.n_bootstrap), 'Running bootstraps'):
            resampled_indices = np.random.choice(len(similarities), len(similarities), replace=True)
            resampled_similarities = np.array(similarities)[resampled_indices]
            resampled_indep_var = np.array(indep_var)[resampled_indices]
            
            if self.method == 'spearman':
                resampled_r.append(spearmanr(resampled_similarities.flatten(), resampled_indep_var.flatten())[0])
            else:
                resampled_r.append(pearsonr(resampled_similarities.flatten(), resampled_indep_var.flatten())[0])
                
        ci_lower = np.percentile(resampled_r, 2.5)
        ci_upper = np.percentile(resampled_r, 97.5)
        mean_r = np.mean(resampled_r)
        return ci_lower, ci_upper, mean_r
