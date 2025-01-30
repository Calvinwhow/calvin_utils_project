from nilearn import plotting
import nibabel as nib
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata
import pandas as pd
from tqdm import tqdm
from calvin_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from calvin_utils.ccm_utils.stat_utils_jax import _calculate_pearson_r_map_jax, _rankdata_jax
import time

class ConvergentMapGenerator:
    def __init__(self, corr_map_dict, data_loader, mask_path=None, out_dir=None, weight=False):
        self.corr_map_dict = corr_map_dict
        self.data_loader = data_loader
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.weight = weight
        if out_dir is not None: os.makedirs(out_dir, exist_ok=True)
        self._handle_nans()
        
    def _handle_nans(self):
        print("Removing NaNS from self.coor_map_dict.")
        for dataset, matrix in self.corr_map_dict.items():
            self.corr_map_dict[dataset] = CorrelationCalculator._check_for_nans(matrix, nanpolicy='remove', verbose=False)
            
    def generate_weighted_average_r_map(self, override_corr_map_dict=None):
        if override_corr_map_dict is None:
            local_corr_map_dict = self.corr_map_dict
        else:
            local_corr_map_dict = override_corr_map_dict
            
        r_maps = np.array(list(local_corr_map_dict.values()))
        
        if self.weight:
            weights = []
            for dataset_name in local_corr_map_dict.keys():
                data = self.data_loader.load_dataset(dataset_name)
                weights.append(data['niftis'].shape[0])
            weights = np.array(weights)
            return np.mean(r_maps, axis=0, weights=weights)
        else:
            return np.mean(r_maps, axis=0)

    def generate_agreement_map(self, override_corr_map_dict=None, return_signed=True):
        '''
        Generates mask of regions that share signs among all maps. 
        If return signed, map will range from [-1,0,1]
        zeroes do not factor into cosine simlarity against the agreement map given the nature of the dot product.
        '''
        if override_corr_map_dict is None:
            local_corr_map_dict = self.corr_map_dict
        else:
            local_corr_map_dict = override_corr_map_dict
        r_maps = np.array(list(local_corr_map_dict.values()))
        signs = np.sign(r_maps)
        agreement = np.all(signs == signs[0], axis=0)
        if return_signed:                            # Returns -1, 0, or 1 based on agreement
            return agreement.astype(int) * signs[0] 
        else:                                        # Returns only 0 or 1
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
        
        file_path = os.path.join(self.out_dir, file_name)
        if self.out_dir is not None:
            if bool(os.path.splitext(file_path)[1]):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            else: 
                os.makedirs(self.out_dir, exist_ok=True)
            nib.save(img, file_path)
            
        return img

    def _visualize_map(self, img, title):
        plotting.view_img(img, title=title).open_in_browser()
    
    def _align_signs(self, images_dict):
        """
        Align the signs of all brain images in `images_dict` so that the average
        pairwise Pearson's R is (roughly) maximized by flipping signs where needed.
        
        images_dict: dict[str, np.ndarray]
            Keys are dataset names, values are the corresponding brain images.
        
        Returns
        -------
        aligned_dict : dict[str, np.ndarray]
            A copy of `images_dict` with some values possibly flipped (multiplied by -1).
        """
        keys = list(images_dict.keys())
        # Flatten each image to 1D for correlation
        flattened = [images_dict[k].ravel() for k in keys]
        
        # 1) Build a reference by summing all flattened images
        reference = np.sum(flattened, axis=0)
        
        # 2) Flip sign of each image if it negatively correlates with the reference
        aligned_dict = {}
        for idx, k in enumerate(keys):
            r_val = np.corrcoef(flattened[idx], reference)[0, 1]
            if r_val < 0:
                aligned_dict[k] = -images_dict[k]
            else:
                aligned_dict[k] = images_dict[k]

        return aligned_dict


    def generate_and_save_maps(self, verbose=False, group_dict={}, dir='', align_all_maps=True):
        if group_dict == {}:
            group_dict = {ds: 'all_datasets' for ds in self.corr_map_dict.keys()}
            
        for group in set(group_dict.values()):
            print(group)
            # get the datasets that belong to the group you are working on
            datasets_in_group = [key for key, val in group_dict.items() if val == group]
            local_corr_map_dict = {dataset: self.corr_map_dict[dataset] for dataset in datasets_in_group}
            
            # if align maps, set them all so they have the same sign spatial correlation
            if align_all_maps: local_corr_map_dict = self._align_signs(local_corr_map_dict)
            
            # Generate weighted average r map
            weighted_avg_map = self.generate_weighted_average_r_map(local_corr_map_dict)
            weighted_avg_img = self._save_map(weighted_avg_map, f'{dir}{group}_weighted_average_r_map.nii.gz')
            try:
                if verbose: self._visualize_map(weighted_avg_img, f'{group} Weighted Average R Map')
            except: pass

            # Generate agreement map
            agreement_map = self.generate_agreement_map(local_corr_map_dict)
            agreement_img = self._save_map(agreement_map, f'{dir}{group}_agreement_map.nii.gz')
            try:
                if verbose: self._visualize_map(agreement_img, f'{group} Agreement Map')
            except: pass
    
    def save_individual_r_maps(self, verbose=False, dir=''):
        for dataset_name, r_map in self.corr_map_dict.items():
            r_img = self._save_map(r_map, f'{dir}{dataset_name}_correlation_map.nii.gz')
            if verbose: self._visualize_map(r_img, f'{dataset_name} Correlation Map')


from math import log, exp, sqrt
class LOOCVAnalyzer(ConvergentMapGenerator):
    def __init__(self, corr_map_dict, data_loader, mask_path=None, out_dir=None, weight=False, method='spearman', convergence_type='agreement', similarity='cos', n_bootstrap=1000, roi_path=None, group_dict={}):
        """
        Initialize the LOOCVAnalyzer.

        Parameters:
        -----------
        corr_map_dict : dict
            corr_map_dict
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
        group_dict : dict, optional
            a dictionary with dataset names as keys and their group as values.
            If entered, matching groups will be used to create their group-level convergent map
            Then one group-level convergent map will predict the other group-level convergent map. 
        """
        super().__init__(corr_map_dict, data_loader, mask_path, out_dir, weight)
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.similarity = similarity
        self.convergence_type = convergence_type
        self.roi_path = roi_path
        self.group_dict = group_dict
        self.out_dir=out_dir
        self.correlation_calculator = CorrelationCalculator(method=method, verbose=False, use_jax=False)
         
    def run(self):
        self.results = self.perform_loocv()
        self.results_df = self.results_to_dataframe()
        self.results_df.to_csv(f'{self.out_dir}/loocv_results.csv')
    
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
        dataset_names = list(self.corr_map_dict.keys())
        for i, dataset_name in enumerate(dataset_names):
            dataset_name
            ci_lower, ci_upper, mean_r = self.results[i]
            data.append([dataset_name, ci_lower, ci_upper, mean_r])
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

    def _prepare_train_dataset(self, i, dataset_names_list, test_dataset_name):
        if self.group_dict == {}:                                       # If empty, have all groups predict test data
            return dataset_names_list[:i] + dataset_names_list[i+1:]    # Return groups that aren't the test group
        
        taining_dataset_list = []
        test_group = self.group_dict.get(test_dataset_name)             # If full, have opposite group predict test data
        for k,v in self.group_dict.items():
            if v != test_group: 
                taining_dataset_list.append(k)                          # Aggregates all opposite group dataset names
            
            if len(set(self.group_dict.values)) > 2:                         # Unclear how to handle the case where there are multiple groups
                raise ValueError("Over 2 groups have been assigned in the group_dict argument. This is not yet supported.")
            if len(set(self.group_dict.values)) ==1:                         # Not possible
                raise ValueError("Only 1 group has been assigned in the group_dict argument. This is not supported.")
        return taining_dataset_list

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
            # Load the training dataset
            print("Evaluating dataset:", test_dataset_name)
            train_dataset_names = self._prepare_train_dataset(i, dataset_names, test_dataset_name)            
            
            # Load the test dataset
            test_data = self.data_loader.load_dataset(test_dataset_name)
            test_niftis = CorrelationCalculator._check_for_nans(test_data['niftis'], nanpolicy='remove', verbose=False)
            test_indep_var = CorrelationCalculator._check_for_nans(test_data['indep_var'], nanpolicy='remove', verbose=False)

            # TRAIN - Generate the convergent map using the training datasets (or an ROI)
            if self.roi_path is not None:
                convergent_map = self.generate_convergent_roi()
            elif self.convergence_type == 'average':
                local_corr_map_dict = self.generate_correlation_maps(train_dataset_names)
                convergent_map = self.generate_weighted_average_r_map(local_corr_map_dict)
            elif self.convergence_type == 'agreement':
                local_corr_map_dict = self.generate_correlation_maps(train_dataset_names)
                convergent_map = self.generate_agreement_map(local_corr_map_dict)
            else:
                raise ValueError("Invalid convergence type (self.convergence_type). Please choose 'average', 'agreement', or set path to a region of interest to test (self.roi_path).")

            # TEST - use the convergent map on the test dataset
            test_niftis = CorrelationCalculator._check_for_nans(test_niftis, nanpolicy='remove', verbose=False)
            convergent_map = CorrelationCalculator._check_for_nans(convergent_map, nanpolicy='remove', verbose=False)
            test_indep_var = CorrelationCalculator._check_for_nans(test_indep_var, nanpolicy='remove', verbose=False)
            
            similarities = self.calculate_similarity(test_niftis, convergent_map)
            ci_lower, ci_upper, mean_r = self.correlate_similarity_with_outcomes(similarities, test_indep_var, test_dataset_name)
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
        patient_maps = CorrelationCalculator._check_for_nans(patient_maps, nanpolicy='remove', verbose=False)
        
        if self.similarity == 'cos':
            similarities = [self.cosine_similarity(patient_map, convergent_map) for patient_map in patient_maps]
        elif self.similarity == 'spcorr':
            similarities = [pearsonr(patient_map.flatten(), convergent_map.flatten())[0] for patient_map in patient_maps]
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

    def _calculate_confidence_interval(self, data, ci_method='percentile', alpha=0.05, return_mean=False):
        """
        Calculate confidence intervals (CI) for a given distribution.
        
        Parameters:
            data (array-like): Input distribution of data.
            ci_method (str): Method to calculate CI. Options are 'analytic' or 'percentile'.
            alpha (float): Significance level for the CI (default is 0.05 for a 95% CI).
        
        Returns:
            tuple: (CI lower bound, CI upper bound, Mean)
        """        
        # Calculate the mean
        mean = np.mean(data)
        if ci_method == 'analytic':
            # Use the analytic method (normal approximation)
            z = 1.96  # For 95% CI
            std_err = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard error of the mean
            ci_lower = mean - z * std_err
            ci_upper = mean + z * std_err
        
        elif ci_method == 'percentile':
            # Use the percentile method
            ci_lower = np.percentile(data, 100 * (alpha / 2))
            ci_upper = np.percentile(data, 100 * (1 - alpha / 2))
        elif ci_method == 'hybrid':
            ci_lower_1 = np.percentile(data, 100 * (alpha / 2))
            ci_upper_1 = np.percentile(data, 100 * (1 - alpha / 2))
            
            # Use the analytic method (normal approximation)
            z = 1.96  # For 95% CI
            std_err = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard error of the mean
            ci_lower_2 = mean - z * std_err
            ci_upper_2 = mean + z * std_err
            
            ci_lower = ( ci_lower_1 + ci_lower_2 ) / 2
            ci_upper = ( ci_upper_1 + ci_upper_2 ) / 2
            
        else:
            raise ValueError("ci_method must be 'analytic' or 'percentile'")
        if return_mean:
            return ci_lower, ci_upper, mean
        else:
            return ci_lower, ci_upper, data[0]

    def _generate_scatterplot(self, similarities, indep_var, dataset_name):
        """
        Generate and save a scatterplot of similarity vs. outcome with Spearman correlation.

        Parameters:
        -----------
        similarities : list of float
            List of cosine similarity values (X-axis).
        indep_var : np.array
            Array of independent variable values (Y-axis).
        dataset_name : str
            Name of the dataset (used for title and filename).
        """
        rho, p = spearmanr(similarities, indep_var)
        r, pr = pearsonr(similarities, indep_var)

        # Create scatterplot
        plt.figure(figsize=(6, 4))
        df = pd.DataFrame({"Similarity": similarities, "Outcome": indep_var})
        if self.similarity=='cos':
            xlab = 'Cosine Similarity'
        else:
            xlab = 'Spatial Correlation'
        sns.scatterplot(data=df, x="Similarity", y="Outcome")
        
        # Title with rho, p-value, and dataset name
        plt.title(f"{dataset_name}: Spearman ρ = {rho:.2f}, p = {p:.2e} \n Pearson r = {r:.2f}, p = {pr:.2e}")
        plt.xlabel("Similarity")
        plt.ylabel("Outcome")

        # Save plot
        os.makedirs(self.out_dir+'/scatterplots', exist_ok=True)
        plt.savefig(os.path.join(self.out_dir, f"scatterplots/{dataset_name}_scatterplot.svg"), bbox_inches="tight")
        plt.close()
        
    def correlate_similarity_with_outcomes(self, similarities, indep_var, dataset_name='test_data', gen_scatterplot=True):
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
        if gen_scatterplot:
            self._generate_scatterplot(similarities, indep_var.flatten(), dataset_name)
            
        for _ in tqdm(range(self.n_bootstrap), 'Running bootstraps'):
            resampled_indices = np.random.choice(len(similarities), len(similarities), replace=True)
            if _ == 0:      # on the first iteration, store the actual data
                resampled_similarities = np.array(similarities)
                resampled_indep_var = np.array(indep_var)
            else:
                resampled_similarities = np.array(similarities)[resampled_indices]
                resampled_indep_var = np.array(indep_var)[resampled_indices]
            
            if self.method == 'spearman':
                resampled_r.append(spearmanr(resampled_similarities.flatten(), resampled_indep_var.flatten())[0])
            else:
                resampled_r.append(pearsonr(resampled_similarities.flatten(), resampled_indep_var.flatten())[0])
        
        ci_lower, ci_upper, mean_r = self._calculate_confidence_interval(resampled_r)
        return ci_lower, ci_upper, mean_r
    
    def compute_fixed_effects_by_group(self, group_dict):
        """
        Perform an inverse-variance fixed-effect meta-analysis of mean R-values by group.

        Parameters
        ----------
        group_dict : dict
            Dictionary mapping each dataset_name -> group_name, e.g.:
            {
            'dataset_1': 'group1',
            'dataset_2': 'group2',
            'dataset_3': 'group1'
            }

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['Group', 'R_Fixed', 'CI_Lower', 'CI_Upper'],
            where each row represents the meta-analytic fixed-effect of R
            among all datasets in that group.
        """

        def fisher_z(r):
            return 0.5 * log((1 + r) / (1 - r))

        def inv_fisher_z(z):
            return (exp(2*z) - 1) / (exp(2*z) + 1)

        def prep_group_dict(group_dict):
            if group_dict == {}:
                for dataset in self.corr_map_dict.keys():
                    group_dict[dataset] = 'Overall Fixed Effects'
            return group_dict
        
        # 0. Prepare the dict mapping datasets to fixed effects groups
        group_dict = prep_group_dict(group_dict)

        # 1. Gather mean R and sample sizes by group
        #    (We assume self.results lines up with self.corr_map_dict.keys())
        grouped_z_and_w = {}
        for (dataset_name, (_, _, mean_r)) in zip(self.corr_map_dict.keys(), self.results):
            group = group_dict.get(dataset_name)
            if group is None:
                continue  # Skip datasets not in the dictionary

            # Convert mean R to Fisher's Z
            z = fisher_z(mean_r)

            # Sample size (for variance of Fisher's Z)
            n = len(self.data_loader.load_dataset(dataset_name)['indep_var'])
            w = n - 3  # weight = inverse variance = (n-3)

            grouped_z_and_w.setdefault(group, []).append((z, w))

        # 2. Perform meta-analysis for each group
        rows = []
        for group, z_w_pairs in grouped_z_and_w.items():
            # Weighted average in Z-space
            sum_w = sum(w for _, w in z_w_pairs)
            z_fixed = sum(z * w for z, w in z_w_pairs) / sum_w

            # Variance and standard error
            var_fixed = 1.0 / sum_w
            se_fixed = sqrt(var_fixed)

            # 95% CI in Z-space
            z_lower, z_upper = z_fixed - 1.96 * se_fixed, z_fixed + 1.96 * se_fixed

            # Back-transform to correlation
            r_fixed = inv_fisher_z(z_fixed)
            r_lower = inv_fisher_z(z_lower)
            r_upper = inv_fisher_z(z_upper)

            rows.append([group, r_lower, r_upper, r_fixed])

        # 3. Return a one-row-per-group DataFrame
        return pd.DataFrame(rows, columns=['Dataset', 'CI Lower', 'CI Upper', 'Mean R'])


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
    datasets_to_flip: 
        list of dataset names. if detected, will flip multiply correlation map by -1.
    run():
        Runs the entire correlation analysis process and returns the p-value and pairwise p-values.
    """
    
    def __init__(self, data_dict_path, method='pearson', n_permutations=1000, out_dir=None, use_jax=False, datasets_to_flip=[]):
        self.data_dict_path = data_dict_path
        self.method = method
        self.n_permutations = n_permutations
        self.out_dir = out_dir
        self.corr_map_dict = None
        self.data_loader = None
        self.original_similarity_matrix = None
        self.permuted_similarity_tensor = None
        self.use_jax = use_jax
        self.datasets_to_flip = datasets_to_flip
        self.correlation_calculator = CorrelationCalculator(method=self.method, verbose=False, use_jax=self.use_jax)
        self.data_loader = DataLoader(self.data_dict_path)

    def generate_correlation_maps(self, permute=False):
        corr_map_dict = {}
        for dataset_name in self.data_loader.dataset_paths_dict.keys():
            if self.method == 'pearson':
                data = self.data_loader.load_dataset(dataset_name)
            elif self.method =='spearman':
                data = self.data_loader.load_dataset(dataset_name, nifti_type='niftis_ranked')
            
            if permute: np.random.shuffle(data['indep_var'])  # permute the independent variable
            
            self.correlation_calculator._process_data(data)
            if dataset_name in self.datasets_to_flip:
                self.correlation_calculator.correlation_map *= -1
            corr_map_dict[dataset_name] = self.correlation_calculator.correlation_map
        return corr_map_dict

    def calculate_similarity_matrix(self, corr_map_dict):
        dataset_names = list(corr_map_dict.keys())
        n = len(dataset_names)
        similarity_matrix = np.full((n, n), np.nan)
        
        for i in range(n):
            for j in range(i, n):
                mx_1 = corr_map_dict[dataset_names[i]].flatten()
                mx_2 = corr_map_dict[dataset_names[j]].flatten()

                mx_1 = self.correlation_calculator._check_for_nans(mx_1, nanpolicy='remove', verbose=False)
                mx_2 = self.correlation_calculator._check_for_nans(mx_2, nanpolicy='remove', verbose=False)

                # if self.method == 'pearson' and self.spcorr_method == 'pearson':
                    # if self.use_jax:
                    #     similarity = _calculate_pearson_r_map_jax(mx_1, mx_2)[0][0]
                    # else:
                        # similarity, _ = pearsonr(mx_1, mx_2) 
                # elif self.method == 'spearman':
                #     if self.use_jax:
                #         mx_1 = _rankdata_jax(mx_1[:, np.newaxis])
                #         mx_2 = _rankdata_jax(mx_2[:, np.newaxis])
                #         similarity = _calculate_pearson_r_map_jax(mx_1, mx_2)[0][0]
                #     else:
                #         similarity, _ = spearmanr(mx_1, mx_2)
                similarity, _ = pearsonr(mx_1, mx_2) 
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix

    def repeat_permutation_process(self):
        self.corr_map_dict = self.generate_correlation_maps(permute=False)
        self.original_similarity_matrix = self.calculate_similarity_matrix(self.corr_map_dict)
        
        n = len(self.corr_map_dict)
        self.permuted_similarity_tensor = np.zeros((self.n_permutations, n, n))
        
        for i in tqdm(range(self.n_permutations), desc='Running permutations'):
            permuted_corr_map_dict = self.generate_correlation_maps(permute=True)
            permuted_similarity_matrix = self.calculate_similarity_matrix(permuted_corr_map_dict)
            self.permuted_similarity_tensor[i] = permuted_similarity_matrix

    def save_results(self):
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            np.save(f'{self.out_dir}/original_similarity_matrix.npy', self.original_similarity_matrix)
            np.save(f'{self.out_dir}/permuted_similarity_tensor.npy', self.permuted_similarity_tensor)

    def calculate_p_value(self, method='two_tail', absolute_similarity=True):
        """
        method : str
            if true, will take the absolute value of the average similarity before computing. 
        absolute_similarity : bool
            if true, will take the absolute value of the similarity matrices before computing.
            This tests if the topology was similar.
            If false, will test if there was a significant similarity, considering signs. 
        """
        # Remove the diagonal from the comparison
        diag_mask = ~np.eye(self.original_similarity_matrix.shape[0], dtype=bool)  # False on diagonal, True elsewhere
        local_obsv = self.original_similarity_matrix * diag_mask 
        local_perm = self.permuted_similarity_tensor * diag_mask  # Apply across all permutations
        
        # check the absolute correlation.
        if absolute_similarity:
            local_obsv = np.abs(local_obsv)
            local_perm = np.abs(local_perm)
        
        # get averages
        observed_avg = np.nanmean(local_obsv)
        permuted_avg = np.nanmean(local_perm, axis=(1, 2))
        print(observed_avg, permuted_avg)

        # Get p values
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
    
    def matrix_heatmap(self, similarity_matrix, type='similarity', mask_half=False, output_path=None, limit=None):
        # Possibly mask the upper half
        if mask_half:
            similarity_matrix = np.tril(similarity_matrix)

        # Optionally replace diagonal
        np.fill_diagonal(similarity_matrix, np.nan)
        
        if type == 'similarity':
            # 1) Define a colormap with positions in [0..1]
            #    (0.0 → red, 0.4 → black, 0.6 → black, 1.0 → green)
            cmap = LinearSegmentedColormap.from_list(
                'RedBlackGreen',
                [
                    (0, 'red'),   
                    (0.5, 'black'),
                    (0.5, 'black'),
                    (1.0, 'green')
                ]
            )

            # 2) Dynamically set vmin & vmax from data, keep center at 0
            if limit is None:
                minimum = np.nanmin(np.abs(similarity_matrix))
                maximum = np.nanmax(np.abs(similarity_matrix))
                limit = np.max(np.array([minimum, maximum]))
                
            norm = TwoSlopeNorm(
                vmin= -limit,
                vcenter=0,
                vmax= limit
            )
            if bool(os.path.splitext(output_path)[1]):
                output_file = os.path.basename(output_path)
                output_path = os.path.dirname(output_path)
            else:
                output_file = 'heatmap_similarity.svg'

        elif type == 'pvals':
            # p-value colormap example
            import matplotlib.cm as cm
            bounds = [0, 0.0001, 0.001, 0.01, 0.05, 1]
            cmap = cm.get_cmap('viridis', len(bounds) - 1)
            norm = cm.colors.BoundaryNorm(bounds, cmap.N)
            output_file = 'heatmap_pvals.svg'

        else:
            raise ValueError("Invalid input. Please choose either 'similarity' or 'pvals'.")

        # If you really want the diagonal = 1 afterward:
        np.fill_diagonal(similarity_matrix, 1)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            similarity_matrix,
            square=True,
            linewidths=1.0,
            cmap=cmap,
            norm=norm,
            ax=ax,
            cbar=True
        )

        ax.set_xticks(np.arange(len(self.corr_map_dict)) + 0.5)
        ax.set_yticks(np.arange(len(self.corr_map_dict)) + 0.5)
        ax.set_xticklabels(list(self.corr_map_dict.keys()), rotation=90)
        ax.set_yticklabels(list(self.corr_map_dict.keys()), rotation=0)

        if output_path:
            plt.savefig(os.path.join(output_path, output_file))
        plt.show()
        return limit