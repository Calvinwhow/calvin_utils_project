from nilearn import plotting
import nibabel as nib
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from tqdm import tqdm
from calvin_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import time

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

class CorrelationAnalysis:
    """
    A class to perform correlation analysis on neuroimaging data.
    
    Attributes:
    -----------
    data_dict_path : str
        Path to the dictionary containing dataset paths.
    method : str, optional
        Method to calculate correlation ('pearson' or 'spearman'). Default is 'pearson'.
    n_permutations : int, optional
        Number of permutations for the permutation test. Default is 1000.
    out_dir : str, optional
        Directory to save the results. Default is None.
    corr_map_dict : dict
        Dictionary to store correlation maps.
    data_loader : DataLoader
        Instance of DataLoader to load datasets.
    original_similarity_matrix : np.ndarray
        Similarity matrix calculated from original data.
    permuted_similarity_tensor : np.ndarray
        Tensor of similarity matrices calculated from permuted data.
        
    Methods:
    --------
    generate_correlation_maps():
        Generates correlation maps for all datasets.
    calculate_similarity_matrix(corr_map_dict):
        Calculates the similarity matrix from correlation maps.
    permute_and_recompose():
        Permutes the data and recomposes the correlation maps.
    calculate_permuted_similarity_matrix(permuted_corr_map_dict):
        Calculates the similarity matrix from permuted correlation maps.
    repeat_permutation_process():
        Repeats the permutation process to generate permuted similarity matrices.
    save_results():
        Saves the original and permuted similarity matrices to the output directory.
    calculate_p_value(method='two_tail'):
        Calculates the p-value based on the permutation test.
    calculate_max_stat_p_values(method):
        Calculates p-values using the maximum statistic method.
    calculate_pairwise_p_values_uncorrected(method):
        Calculates uncorrected pairwise p-values.
    calculate_pairwise_p_values(method='two_tail', max_stat=False):
        Calculates pairwise p-values, with an option for maximum statistic correction.
    run():
        Runs the entire correlation analysis process and returns the p-value and pairwise p-values.
    """
    
    def __init__(self, data_dict_path, method='pearson', n_permutations=1000, out_dir=None):
        self.data_dict_path = data_dict_path
        self.method = method
        self.n_permutations = n_permutations
        self.out_dir = out_dir
        self.corr_map_dict = None
        self.data_loader = None
        self.original_similarity_matrix = None
        self.permuted_similarity_tensor = None

    def generate_correlation_maps(self):
        self.data_loader = DataLoader(self.data_dict_path)
        correlation_calculator = CorrelationCalculator(method=self.method, verbose=False)
        
        self.corr_map_dict = correlation_calculator.process_all_datasets(self.data_loader.dataset_paths_dict)

    def calculate_similarity_matrix(self, corr_map_dict):
        dataset_names = list(corr_map_dict.keys())
        n = len(dataset_names)
        similarity_matrix = np.full((n, n), np.nan)
        
        for i in range(n):
            for j in range(i, n):
                t0 = time.time()
                mx_1 = corr_map_dict[dataset_names[i]].flatten()
                mx_2 = corr_map_dict[dataset_names[j]].flatten()

                t1 = time.time()
                if self.method == 'pearson':
                    similarity, _ = pearsonr(mx_1, mx_2)
                elif self.method == 'spearman':
                    similarity, _ = pearsonr(mx_1, mx_2)
                t2 = time.time()
                # CorrelationCalculator._check_for_nans(similarity, nanpolicy='stop')
                similarity_matrix[i,j] = similarity
                similarity_matrix[j,i] = similarity
                t3 = time.time()
                print(f"Time for matrix assignment : {t1-t0}")
                print(f"Time for spearmanr: {t2-t1}")
                print(f"Time for ij, ji: {t3-t2}")
        return similarity_matrix

    def permute_and_recompose(self):
        permuted_corr_map_dict = {}
        for dataset_name in self.data_loader.dataset_paths_dict.keys():
            data = self.data_loader.load_dataset(dataset_name)
            np.random.shuffle(data['indep_var'])  # permute
            correlation_calculator = CorrelationCalculator(method=self.method, verbose=False)
            correlation_calculator._process_data(data)
            permuted_corr_map_dict[dataset_name] = correlation_calculator.correlation_map
        return permuted_corr_map_dict

    def calculate_permuted_similarity_matrix(self, permuted_corr_map_dict):
        return self.calculate_similarity_matrix(permuted_corr_map_dict)

    def repeat_permutation_process(self):
        self.generate_correlation_maps()
        self.original_similarity_matrix = self.calculate_similarity_matrix(self.corr_map_dict)
        
        n = len(self.corr_map_dict)
        self.permuted_similarity_tensor = np.zeros((self.n_permutations, n, n))
        
        for i in tqdm(range(self.n_permutations), desc='Running permutations'):
            permuted_corr_map_dict = self.permute_and_recompose()
            permuted_similarity_matrix = self.calculate_permuted_similarity_matrix(permuted_corr_map_dict)
            self.permuted_similarity_tensor[i] = permuted_similarity_matrix

    def save_results(self):
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            np.save(f'{self.out_dir}/original_similarity_matrix.npy', self.original_similarity_matrix)
            np.save(f'{self.out_dir}/permuted_similarity_tensor.npy', self.permuted_similarity_tensor)

    def calculate_p_value(self, method='two_tail'):
        observed_avg = np.nanmean(self.original_similarity_matrix)
        permuted_avg = np.nanmean(self.permuted_similarity_tensor, axis=(1, 2))
        if method == 'two_tail':
            p_value = np.mean(np.abs(permuted_avg) > np.abs(observed_avg))
        elif method == 'one_tail':
            p_value = np.mean(permuted_avg > observed_avg)
        else:
            raise ValueError('Invalid method. Please choose either "one_tail" or "two_tail".')
        print(f'The {method} p-value is: {p_value}. This is the proportion of permuted averages that are greater than the observed average.')
        return p_value

    def calculate_max_stat_p_values(self, method):
        n = self.original_similarity_matrix.shape[0]
        p_value_matrix = np.full((n, n), np.nan)
        max_permuted_values = np.max(np.abs(self.permuted_similarity_tensor), axis=(1, 2)) if method == 'two_tail' else np.max(self.permuted_similarity_tensor, axis=(1, 2))

        for i in range(n):
            for j in range(i, n):
                observed_value = np.abs(self.original_similarity_matrix[i, j]) if method == 'two_tail' else self.original_similarity_matrix[i, j]
                p_value = np.mean(max_permuted_values > observed_value)
                p_value_matrix[i, j] = p_value
                p_value_matrix[j, i] = p_value
        return p_value_matrix

    def calculate_pairwise_p_values_uncorrected(self, method):
        n = self.original_similarity_matrix.shape[0]
        p_value_matrix = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(i, n):
                observed_value = self.original_similarity_matrix[i, j]
                permuted_values = self.permuted_similarity_tensor[:, i, j]
                if method == 'two_tail':
                    p_value = np.mean(np.abs(permuted_values) > np.abs(observed_value))
                elif method == 'one_tail':
                    p_value = np.mean(permuted_values > observed_value)
                p_value_matrix[i,j] = p_value
                p_value_matrix[j,i] = p_value
        return p_value_matrix

    def calculate_pairwise_p_values(self, method='two_tail', max_stat=False, verbose=False):
        if max_stat:
            p_val_mx =  self.calculate_max_stat_p_values(method)
        else:
            p_val_mx =  self.calculate_pairwise_p_values_uncorrected(method)
        if verbose:
            print("Pairwise P-values")
            print(p_val_mx)
        return p_val_mx

    def run(self, tails='two_tail', fwe=False, verbose=False):
        self.repeat_permutation_process()
        self.save_results()
        
        if tails == 'two_tail':
            print('Calculating two-tailed p-values')
        elif tails == 'one_tail':
            print('Calculating one-tailed p-values')
        else:
            raise ValueError('Invalid method. Please choose either "one_tail" or "two_tail".')
        
        p_value = self.calculate_p_value(tails)
        pairwise_p_values = self.calculate_pairwise_p_values(tails, fwe)
        if verbose:
            print('Similarity Matrix: ')
            print(self.original_similarity_matrix)
        print(f"Overall p-value: {p_value}")
        return p_value, pairwise_p_values
    
    def matrix_heatmap(self, similarity_matrix, type='similarity', mask_half=False, output_path=None):
        # Remove the diagonal of the similarity matrix
        if mask_half: 
            similarity_matrix = np.tril(similarity_matrix)
            np.fill_diagonal(similarity_matrix, np.nan)

        if type == 'similarity':
            cmap = LinearSegmentedColormap.from_list('RedBlackGreen', ['red', 'black', 'green'])
            norm = TwoSlopeNorm(vmin=np.nanmin(similarity_matrix), vcenter=0, vmax=np.nanmax(similarity_matrix))
            output_file = 'similarity_heatmap.svg'
        elif type == 'pvals':
            bounds = [0, 0.0001, 0.001, 0.01, 0.05, 1]
            cmap = cm.get_cmap('viridis', len(bounds) - 1)
            norm = cm.colors.BoundaryNorm(bounds, cmap.N)
            output_file = 'pvals_heatmap.svg'
        else:
            raise ValueError("Invalid input. Please choose either 'similarity' or 'pvals'.")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(similarity_matrix, square=True, linewidths=1.0, cmap=cmap, norm=norm, ax=ax, cbar=True)
        ax.set_xticks(np.arange(len(self.corr_map_dict)) + 0.5)
        ax.set_yticks(np.arange(len(self.corr_map_dict)) + 0.5)
        ax.set_xticklabels(list(self.corr_map_dict.keys()), rotation=90)
        ax.set_yticklabels(list(self.corr_map_dict.keys()), rotation=0)

        if output_path:
            plt.savefig(os.path.join(output_path, output_file))
        plt.show()