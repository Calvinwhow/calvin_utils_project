from nilearn import plotting
import nibabel as nib
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata
import pandas as pd
from tqdm import tqdm
from calvin_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.ccm_utils.optimization import NiftiOptimizer
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from calvin_utils.ccm_utils.stat_utils_jax import _calculate_pearson_r_map_jax, _rankdata_jax
import time

class ConvergentMapGenerator:
    def __init__(self, corr_map_dict, data_loader, mask_path=None, out_dir=None, weight=False, optimizer=False, align_all_maps=False):
        self.corr_map_dict = corr_map_dict
        self.data_loader = data_loader
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.weight = weight
        if self.weight:
            self.prefix = 'weighted_avg'
        else:
            self.prefix = 'unweighted_avg'
        self.optimizer = optimizer
        self.align_all_maps=align_all_maps
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
            return np.average(r_maps, axis=0, weights=weights)
        else:
            return np.average(r_maps, axis=0)

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
    
    def _align_signs(self, images_dict, reference=None):
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
        if reference is None: 
            reference = flattened[0]
        elif reference=='sum':
            reference = np.sum(flattened, axis=0)
        else:
            reference = [self.corr_map_dict[reference].ravel()]
            
        # 2) Flip sign of each image if it negatively correlates with the reference
        aligned_dict = {}
        for idx, k in enumerate(keys):
            r_val = np.corrcoef(flattened[idx], reference)[0, 1]
            if r_val < 0:
                aligned_dict[k] = -images_dict[k]
            else:
                aligned_dict[k] = images_dict[k]

        return aligned_dict
    
    def generate_optimal_average_r_map(self, override_corr_map_dict):
        if override_corr_map_dict is None:
            local_corr_map_dict = self.corr_map_dict
        else:
            local_corr_map_dict = override_corr_map_dict
            
        optimizer = NiftiOptimizer(local_corr_map_dict, self.data_loader, load_in_time=True)
        optimal_average_r_map = optimizer.optimize()
        return optimal_average_r_map

    def generate_and_save_maps(self, verbose=False, group_dict={}, dir=''):
        if group_dict == {}:
            group_dict = {ds: 'all_datasets' for ds in self.corr_map_dict.keys()}
            
        for group in set(group_dict.values()):
            # get the datasets that belong to the group you are working on
            datasets_in_group = [key for key, val in group_dict.items() if val == group]
            local_corr_map_dict = {dataset: self.corr_map_dict[dataset] for dataset in datasets_in_group}
            
            # if align maps, set them all so they have the same sign spatial correlation
            if self.align_all_maps: 
                local_corr_map_dict = self._align_signs(local_corr_map_dict)
            
            # Generate weighted average r map
            weighted_avg_map = self.generate_weighted_average_r_map(local_corr_map_dict)
            weighted_avg_img = self._save_map(weighted_avg_map, f'{dir}{group}_{self.prefix}_r_map.nii.gz')
            try:
                if verbose: self._visualize_map(weighted_avg_img, f'{group} {self.prefix} R Map')
            except: pass

            # Generate agreement map
            agreement_map = self.generate_agreement_map(local_corr_map_dict)
            agreement_img = self._save_map(agreement_map, f'{dir}{group}_agreement_map.nii.gz')
            try:
                if verbose: self._visualize_map(agreement_img, f'{group} Agreement Map')
            except: pass
            
            # Generate optimized r map
            if self.optimizer:
                optimized_avg_map = self.generate_optimal_average_r_map(local_corr_map_dict)
                optimized_avg_img = self._save_map(optimized_avg_map, f'{dir}{group}_optimized_avg_map.nii.gz')
                try:
                    if verbose: self._visualize_map(optimized_avg_img, f'{group} Optimal Avg Map')
                except: pass
                
    def _mask_array(self, data_array, threshold=0):
        if self.mask_path is None:
            masked_array = data_array.flatten()
        else:
            mask = nib.load(self.mask_path)
            mask_data = mask.get_fdata()
            mask_indices = mask_data.flatten() > threshold
            masked_array = data_array.flatten()[mask_indices]
        return masked_array
    
    def _unmask_array(self, data_array, threshold=0):
        if self.mask_path is None:
            unmasked_array = data_array.flatten()
        else:
            mask = nib.load(self.mask_path)
            mask_data = mask.get_fdata()
            mask_indices = mask_data.flatten() > threshold
            unmasked_array = np.zeros(mask_indices.shape)
            
            print(data_array.shape, mask_indices.shape, unmasked_array.shape)
            
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
    
    def save_individual_r_maps(self, verbose=False, dir=''):
        for dataset_name, r_map in self.corr_map_dict.items():
            r_img = self._save_map(r_map, f'{dir}{dataset_name}_correlation_map.nii.gz')
            if verbose: self._visualize_map(r_img, f'{dataset_name} Correlation Map')


from math import log, exp, sqrt
class LOOCVAnalyzer(ConvergentMapGenerator):
    def __init__(self, corr_map_dict, data_loader, mask_path=None, out_dir=None, weight=False, method='spearman', similarity='spatial_correl', optimizer=False, n_bootstrap=1000, roi_path=None, group_dict={}, datasets_to_flip = [], align_all_maps=False):
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
        similarity : str, optional
            Similarity measure to use ('cos' for cosine similarity or other measures). Options: 'cosine' and 'spatial_correl'
            If 'spatial_correl', will use the weighted average map with spatial correlation (pearson r).
            If 'cosine', will use the agreement map with cosine similarity.
            Default is 'spatial_correl'.
        optimizer : bool, optional
            If set to True, will call optimization.py and us gradient ascent to optimize the combination of maps into the composite map.
        roi_path : str, optional
            Path to ROI file to use in place of convergent map. 
            Cosine similarity is best to use with this choice.
        group_dict : dict, optional
            a dictionary with dataset names as keys and their group as values.
            If entered, matching groups will be used to create their group-level convergent map
            Then one group-level convergent map will predict the other group-level convergent map. 
        datasets_to_flip : list, optional
            A list of keys corresponding to datasets which should have their correlation maps multiplied by -1. 
            This is typically performed to align maps to enable testing of topology, controlling for sign. 
        """
        super().__init__(corr_map_dict, data_loader, mask_path, out_dir, weight, optimizer=optimizer, align_all_maps=align_all_maps)
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.similarity = similarity
        self.roi_path = roi_path
        self.group_dict = group_dict
        self.out_dir=out_dir
        self.datasets_to_flip = datasets_to_flip
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
        """Perform Leave-One-Out Cross-Validation (LOOCV) analysis."""
        results = []
        dataset_names = list(self.corr_map_dict.keys())
        for i, test_dataset_name in enumerate(dataset_names):
           
            # Load the training dataset
            train_dataset_names = self._prepare_train_dataset(i, dataset_names, test_dataset_name)    
            
            # Load the test dataset
            test_data = self.data_loader.load_dataset(test_dataset_name)
            test_niftis = CorrelationCalculator._check_for_nans(test_data['niftis'], nanpolicy='remove', verbose=False)
            test_indep_var = CorrelationCalculator._check_for_nans(test_data['indep_var'], nanpolicy='remove', verbose=False)
            
            # Orient Dataset to Account for Sign Flips
            if test_dataset_name in self.datasets_to_flip:
                print(f'flipping: {test_dataset_name}')
                test_indep_var = test_indep_var * -1

            # TRAIN - Generate the convergent map using the training datasets (or an ROI)
            if self.roi_path is not None:
                convergent_map = self.generate_convergent_roi()
            elif self.similarity == 'spatial_correl':
                print("Training on: ", train_dataset_names)
                print("Testing on held-out datset: ", test_dataset_name)
                local_corr_map_dict = self.generate_correlation_maps(train_dataset_names)
                convergent_map = self.generate_weighted_average_r_map(local_corr_map_dict)
            elif self.similarity == 'cosine':
                print("Training on: ", train_dataset_names)
                print("Testing on held-out datset: ", test_dataset_name)
                local_corr_map_dict = self.generate_correlation_maps(train_dataset_names)
                convergent_map = self.generate_agreement_map(local_corr_map_dict)
            else:
                raise ValueError("Invalid similarity type (self.similarity). Please choose 'spatial_correl', 'cosine', or set path to a region of interest to test (self.roi_path).")

            # TEST - use the convergent map on the test dataset
            test_niftis = CorrelationCalculator._check_for_nans(test_niftis, nanpolicy='remove', verbose=False)
            convergent_map = CorrelationCalculator._check_for_nans(convergent_map, nanpolicy='remove', verbose=False)
            test_indep_var = CorrelationCalculator._check_for_nans(test_indep_var, nanpolicy='remove', verbose=False)
            
            similarities = self.calculate_similarity(test_niftis, convergent_map)
            ci_lower, ci_upper, mean_r = self.correlate_similarity_with_outcomes(similarities, test_indep_var, test_dataset_name)
            ci_lower, ci_upper, mean_r = self._align_correlations(ci_lower, ci_upper, mean_r)
            results.append((ci_lower, ci_upper, mean_r))
        return results
    
    def _align_correlations(self, ci_lower, ci_upper, mean_r):
        '''Reorients the correlations to be positive if forcibly aligning maps'''
        if self.align_all_maps:
            if mean_r < 0:
                stored_ci = ci_upper
                ci_upper = ci_lower * -1
                ci_lower = stored_ci * -1
                mean_r = mean_r * -1
        return ci_lower, ci_upper, mean_r

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
        
        if self.similarity == 'cosine':
            similarities = [self.cosine_similarity(patient_map, convergent_map) for patient_map in patient_maps]
        elif self.similarity == 'spatial_correl':
            similarities = [pearsonr(patient_map.flatten(), convergent_map.flatten())[0] for patient_map in patient_maps]
        else:
            raise ValueError("Invalid similarity measure (self.similarity). Please choose 'cos' or 'spatial_correl'.")
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

    def _generate_scatterplot(self, similarities, indep_var, dataset_name, flip_axes=False):
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
        flip_axes : bool
            Plot X on Y and Y on X. 
        """
        
        rho, p = spearmanr(similarities, indep_var)
        r, pr = pearsonr(similarities, indep_var)

        # Create DataFrame for Seaborn
        df = pd.DataFrame({"Similarity": similarities, "Outcome": indep_var})
        xlab = 'Cosine Similarity' if self.similarity == 'cosine' else 'Spatial Correlation'
        # Create LM plot
        plt.figure(figsize=(6, 6))
        if flip_axes:
            sns.lmplot(data=df, x="Outcome", y="Similarity", height=6, aspect=1, 
               scatter_kws={'alpha': 0.98, 'color': '#8E8E8E', 's': 150, 'edgecolors': 'white', 'linewidth': 2, 'zorder': 3}, 
               line_kws={'color': '#8E8E8E', 'zorder': 2})
        else:
            sns.lmplot(data=df, x="Similarity", y="Outcome", height=6, aspect=1, 
               scatter_kws={'alpha': 0.98, 'color': '#8E8E8E', 's': 150, 'edgecolors': 'white', 'linewidth': 2, 'zorder': 3}, 
               line_kws={'color': '#8E8E8E', 'zorder': 2})

        # Title with dataset name
        plt.title(f"{dataset_name}", fontsize=20)
        plt.xlabel(xlab, fontsize=20)
        plt.ylabel("Outcome", fontsize=20)

        # Dynamically place stats inside the plot
        x_pos = 0.05 if rho > 0 else 0.05
        y_pos = 0.95 if rho > 0 else 0.15
        plt.text(
            x_pos, y_pos, 
            f"Rho = {rho:.2f}, p = {p:.2e}\nR = {r:.2f}, p = {pr:.2e}",
            fontsize=16, 
            transform=plt.gca().transAxes, 
            verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0, edgecolor='none')
        )
        
        # Set font sizes and line thickness for the scatterplot
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=16)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Save plot
        os.makedirs(self.out_dir+'/scatterplots', exist_ok=True)
        plt.savefig(os.path.join(self.out_dir, f"scatterplots/{dataset_name}_scatterplot.svg"), bbox_inches="tight")
        plt.show()
        
                
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

    def _shuffle_for_roi_comparison(self, niftis, indep_var, method):
        if method == "bootstrap":
            idx = np.random.choice(len(niftis), size=len(niftis), replace=True)
        elif method == "permutation":
            idx = np.arange(len(niftis))
            np.random.shuffle(idx)
        elif method =="observed":
            idx = np.arange(len(niftis))
        else:
            raise ValueError("Invalid method. Please choose 'bootstrap' or 'permutation'.")
        sub_niftis = niftis[idx]
        sub_indep_var = indep_var[idx].flatten()
        return sub_niftis, sub_indep_var
    
    def _compute_r_differences(self, roi1_map, roi2_map, dataset_names, method, n_iter, delta_r2, spearman=True):
        all_r_diffs = {}
        for dataset_name in dataset_names:
            r_diffs = []
            iter_count = 0
            with tqdm(total=n_iter) as pbar:
                while iter_count < n_iter:
                    data = self.data_loader.load_dataset(dataset_name)
                    test_niftis = CorrelationCalculator._check_for_nans(data['niftis'], nanpolicy='remove', verbose=False)
                    test_indep_var = CorrelationCalculator._check_for_nans(data['indep_var'], nanpolicy='remove', verbose=False)
                    sub_niftis, sub_indep_var = self._shuffle_for_roi_comparison(test_niftis, test_indep_var, method)
                    if self.similarity == 'cosine':
                        sim1 = [self.cosine_similarity(n, roi1_map) for n in sub_niftis]
                        sim2 = [self.cosine_similarity(n, roi2_map) for n in sub_niftis]
                    else:
                        sim1 = [pearsonr(n.flatten(), roi1_map.flatten())[0] for n in sub_niftis]
                        sim2 = [pearsonr(n.flatten(), roi2_map.flatten())[0] for n in sub_niftis]
                        
                    if spearman:
                        stat1 = spearmanr(sim1, sub_indep_var, nan_policy='omit')[0]
                        stat2 = spearmanr(sim2, sub_indep_var, nan_policy='omit')[0]
                    else:
                        mask1 = ~np.isnan(sim1) & ~np.isnan(sub_indep_var)
                        mask2 = ~np.isnan(sim2) & ~np.isnan(sub_indep_var)
                        stat1 = pearsonr(np.array(sim1)[mask1], np.array(sub_indep_var)[mask1])[0]
                        stat2 = pearsonr(np.array(sim2)[mask2], np.array(sub_indep_var)[mask2])[0]
                    if np.isnan(stat1) or np.isnan(stat2):
                        continue  # skip this iteration and do not increment iter_count
                    if delta_r2:
                        stat1 = stat1 ** 2
                        stat2 = stat2 ** 2
                    
                    delta = stat1 - stat2
                    r_diffs.append(delta)
                    for roi, stat in zip(['roi1', 'roi2'], [stat1, stat2]):
                        self.r_values[roi].append(stat)
                    iter_count += 1
                    pbar.update(1)
            all_r_diffs[dataset_name] = r_diffs
        return all_r_diffs
        
    def _calculate_probability(self, all_r_diffs, roi1_map, roi2_map, dataset_names, method, delta_r2):
        """Calculate the probability based on the method (bootstrap or permutation)"""
        p_vals = {}
        print(f"Below results used delta explained variance (r-squared): {delta_r2}")
        if method == "bootstrap":
            print("Checking the probability that ROI 1 outperforms ROI 2 by comparing # of times it outperformed it in")
            for dataset_name, diffs in all_r_diffs.items():
                pval = np.mean(np.array(diffs) > 0)
                print(f'{dataset_name} \n    -Avg delta value: {np.mean(np.array(diffs))}| Probability Roi 1 is generally superior: {pval}.')
                p_vals[dataset_name] = pval
        elif method == "permutation":
            print("Checking the probability that ROI 1 outperforms ROI 2 using permutation testing.")
            observed_diffs = self._compute_r_differences(roi1_map, roi2_map, dataset_names, method='observed', n_iter=1, delta_r2=delta_r2)
            for dataset_name, diffs in all_r_diffs.items():
                obsv_diff = observed_diffs[dataset_name]
                pval = np.mean(np.array(diffs) > obsv_diff)
                print(f'{dataset_name} \n    -delta value: {obsv_diff}| permutation p-value: {pval}.')
                p_vals[dataset_name] = pval
            overall_pval = np.mean(np.mean(list(all_r_diffs.values()), axis=1) > np.mean(list(observed_diffs.values())))
            print(f'Overall permutation p-value: {overall_pval}')
        else:
            raise ValueError("method must be 'bootstrap' or 'permutation'")
        
        
        # --- Overall bootstrap p-value and average delta ---
        if method == "bootstrap":
            diff_matrix = np.array(list(all_r_diffs.values()))  # shape: (n_datasets, n_iter)
            mean_diffs = np.mean(diff_matrix, axis=0)           # shape: (n_iter,)
            overall_pval = np.mean(mean_diffs > 0)
            print(f"Overall bootstrap:\n    avg delta R = {np.mean(mean_diffs):.4f}, Probability ROI 1 is generally superior = {overall_pval:.4f}")

        # --- Overall permutation p-value and average delta ---
        elif method == "permutation":
            observed_mean_diff = np.mean(list(observed_diffs.values()))
            diff_matrix = np.array(list(all_r_diffs.values()))  # shape: (n_datasets, n_iter)
            mean_diffs = np.mean(diff_matrix, axis=0)           # shape: (n_iter,)
            overall_pval = np.mean(mean_diffs > observed_mean_diff)
            print(f"Overall permutation:\n     observed avg delta R = {observed_mean_diff:.4f}, p = {overall_pval:.4f}")
        return p_vals
    
    def compare_roi_correlations(self, roi1, roi2, method="bootstrap", n_iter=10000, delta_r2=True, seed=None):
        """Compare the correlation of two ROI maps with outcome variables using bootstrap or permutation test"""
        self.roi_path = roi1
        roi1_map = self.generate_convergent_roi()
        self.roi_path = roi2
        roi2_map = self.generate_convergent_roi()
        
        dataset_names = list(self.corr_map_dict.keys())
        self.r_values = {'roi1': [], 'roi2': []}
        
        all_r_diffs = self._compute_r_differences(roi1_map, roi2_map, dataset_names, method, n_iter, delta_r2)
        self.prob = self._calculate_probability(all_r_diffs, roi1_map, roi2_map, dataset_names, method, delta_r2)


class CorrelationAnalysis:
    """
    A class to perform convergent correlation analysis on neuroimaging data.
    
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
        list of dataset names. if detected, will flip multiply correlation map by -1. Use this to manually align maps to a similar topology. 
    topology_command:
        string. set topology_command to one of: None | 'aligned' | 'r2' | 'absval'. This method is applied to maps before measuring similarity.
    run():
        Runs the entire correlation analysis process and returns the p-value and pairwise p-values.
    """
    
    def __init__(self, data_dict_path, method='pearson', n_permutations=1000, out_dir=None, use_jax=False, datasets_to_flip=[], topology_command=None):
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
        self.topology_command = topology_command
        self.correlation_calculator = CorrelationCalculator(method=method, verbose=False, use_jax=self.use_jax)
        self.data_loader = DataLoader(self.data_dict_path)

    def generate_correlation_maps(self, permute=False, datasets_list=[]):
        """
        Generates correlation maps for the specified datasets.
        Args:
            permute (bool, optional): If True, permutes the independent variable before 
                calculating correlations. Defaults to False.
            dataset_list (list, optional): List of dataset names to process. If empty, 
                all datasets in `data_loader.dataset_paths_dict` are used. Defaults to [].
        Returns:
            dict: A dictionary where keys are dataset names and values are the 
                corresponding correlation maps.
        """
        
        if datasets_list == []:
            datasets_list = self.data_loader.dataset_paths_dict.keys()
    
        corr_map_dict = {}
        for dataset_name in datasets_list:
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
    
    def align_spatial_correlation(self, matrix_1, matrix_2):
        """
        Align the spatial correlation values of each r,c pair in the given array
        to the value at index (0,0). If the correlation is negative relative to (0,0),
        multiply it by -1.

        Parameters:
        -----------
        matrix_1 : np.ndarray
            The similarity matrix to align.
        matrix_2 : np.ndarray
            The similarity matrix to align.

        Returns:
        --------
        np.ndarray
            The aligned similarity matrix.
        """
        r = pearsonr(x=matrix_1, y=matrix_2)[0]
        if r < 0:
            matrix_2 = matrix_2 * -1
        return matrix_1, matrix_2

    
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

                if self.topology_command is None:
                    similarity, _ = pearsonr(mx_1, mx_2) 
                elif self.topology_command == 'r2':
                    similarity, _ = pearsonr(np.square(mx_1), np.square(mx_2)) 
                elif self.topology_command == 'absval':
                    similarity, _ = pearsonr(np.abs(mx_1), np.abs(mx_2)) 
                elif self.topology_command == 'aligned':
                    mx_1, mx_2 = self.align_spatial_correlation(mx_1, mx_2)
                    similarity, _ = pearsonr(mx_1, mx_2)
                else:
                    raise ValueError(f"Error: {self.topology_command} not recognized. Please set topology_command to one of: None | 'r2' | 'absval'.")
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix

    def save_results(self, prefix=''):
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            np.save(f'{self.out_dir}/{prefix}original_similarity_matrix.npy', self.original_similarity_matrix)
            np.save(f'{self.out_dir}/{prefix}permuted_similarity_tensor.npy', self.permuted_similarity_tensor)

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

        # Get p values
        if method == 'two_tail':
            p_value = np.mean(np.abs(permuted_avg) > np.abs(observed_avg))
        elif method == 'one_tail':
            p_value = np.mean(permuted_avg > observed_avg)
        else:
            raise ValueError('Invalid method. Please choose either "one_tail" or "two_tail".')
        print(f"The observed average similarity is: {observed_avg}")
        print(f'The {method} p-value is: {p_value}. This is the proportion of permuted averages that are greater than the observed average.')
        return p_value, observed_avg

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
            plt.savefig(os.path.join(output_path, f'similarity-{self.observed_average}_p-{self.p_value}_{output_file}'))
        plt.show()
        return limit
    
    def perform_permutation_testing(self, datasets_list=[]):
        if datasets_list == []:
            datasets_list = self.data_loader.dataset_paths_dict.keys()
            
        self.corr_map_dict = self.generate_correlation_maps(permute=False, datasets_list=datasets_list)
        self.original_similarity_matrix = self.calculate_similarity_matrix(self.corr_map_dict)
        
        n = len(self.corr_map_dict)
        self.permuted_similarity_tensor = np.zeros((self.n_permutations, n, n))
        
        for i in tqdm(range(self.n_permutations), desc='Running permutations'):
            permuted_corr_map_dict = self.generate_correlation_maps(permute=True, datasets_list=datasets_list)
            permuted_similarity_matrix = self.calculate_similarity_matrix(permuted_corr_map_dict)
            self.permuted_similarity_tensor[i] = permuted_similarity_matrix

    def run(self, tails='two_tail', fwe=False, verbose=False):
        self.perform_permutation_testing()
        self.save_results()
        
        if tails == 'two_tail':
            print('Calculating two-tailed p-values')
        elif tails == 'one_tail':
            print('Calculating one-tailed p-values')
        else:
            raise ValueError('Invalid method. Please choose either "one_tail" or "two_tail".')
        
        p_value, observed_avg = self.calculate_p_value(tails)
        pairwise_p_values = self.calculate_pairwise_p_values(tails, fwe)
        if verbose:
            print('Similarity Matrix: ')
            print(self.original_similarity_matrix)
        print(f"Overall p-value: {p_value}")
        self.p_value = p_value; self.observed_average = observed_avg
        return p_value, pairwise_p_values
    
    def run_lofo(self, folds, tails='two_tail', verbose=False):
        """
        Perform Leave-One-Out Cross-Validation (LOOCV) using the provided array of lists.
    
        Parameters:
        -----------
        folds : list of lists
            An array of lists, where each list contains dataset names to fold out.
            Example: [['dataset1'], ['dataset2'], ['dataset3']] for LOOCV.
        tails : {'one_tail', 'two_tail'}, optional
            Method to calculate p-values. Default is 'two_tail'.
        verbose : bool, optional
            Whether to print detailed logs. Default is False.
    
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the LOOCV results, including R-values and confidence intervals.
        """    
        for fold_idx, test_datasets in enumerate(folds):
            if verbose:
                print(f"Running LOOCV fold {fold_idx + 1}/{len(folds)} with test datasets: {test_datasets}")
    
            # Step 1: Split datasets into training and testing sets
            train_datasets = [ds for ds in self.data_loader.dataset_paths_dict.keys() if ds not in test_datasets]
    
            # Step 2: Run Permutation Process on Training Datasets
            self.perform_permutation_testing(datasets_list=train_datasets)
            
            if tails == 'two_tail':
                print('Calculating two-tailed p-values')
            elif tails == 'one_tail':
                print('Calculating one-tailed p-values')
            else:
                raise ValueError('Invalid method. Please choose either "one_tail" or "two_tail".')
            
            p_value, observed_avg = self.calculate_p_value(method=tails)
            pairwise_p_values = self.calculate_pairwise_p_values(method=tails, max_stat=False)

            print(f"  Left Out Dataset: {test_datasets}")
            print(f"    P-value: {p_value:.4f}")
            print(f"    Observed Average Similarity: {observed_avg:.4f}")
            print(f"    Pairwise P-values:")
            for j, train_dataset in enumerate(train_datasets):
                print(f"      {train_dataset}: {pairwise_p_values[j, j]:.4f}")
            prefix = '_'.join(train_datasets) + '_'
            self.save_results(prefix=prefix)
    
    @staticmethod
    def calculate_slice_p_value(
        observed_matrix,
        permuted_tensor,
        row_indices,
        col_indices,
        method='two_tail',
        absolute_similarity=True,
        remove_diagonal=True
    ):
        """
        Compute the p-value for a submatrix (slice) of the similarity matrix.

        Parameters
        ----------
        observed_matrix : np.ndarray
            The original NxN similarity matrix (2D).
        permuted_tensor : np.ndarray
            The permuted similarity tensor of shape (num_permutations, N, N).
        row_indices : array-like
            Indices of rows to include in the slice.
        col_indices : array-like
            Indices of columns to include in the slice.
        method : {'two_tail','one_tail'}, optional
            - 'two_tail': checks if the absolute permuted average is greater than the absolute observed average.
            - 'one_tail': checks if the permuted average is greater than the observed average.
        absolute_similarity : bool, optional
            If True, takes the absolute value of the similarities before computing.
        remove_diagonal : bool, optional
            If True and row_indices == col_indices, sets the diagonal in that sub-slice to NaN before computing.

        Returns
        -------
        p_value : float
            The proportion of permutations whose average in the specified slice
            exceeds the observed average (two-tail uses absolute values).
        observed_avg : float
            The mean similarity value in this slice of the observed matrix.
        """

        # 1) Extract submatrices
        sub_obsv = observed_matrix[np.ix_(row_indices, col_indices)]
        sub_perm = permuted_tensor[:, row_indices][:, :, col_indices]

        # 2) Optionally remove diagonal if slice is square
        if remove_diagonal and (set(row_indices) == set(col_indices)):
            diag_mask_sub = ~np.eye(len(row_indices), dtype=bool)
            sub_obsv = sub_obsv * diag_mask_sub
            sub_perm = sub_perm * diag_mask_sub[None, :, :]

        # 3) Apply absolute value if needed
        if absolute_similarity:
            sub_obsv = np.abs(sub_obsv)
            sub_perm = np.abs(sub_perm)

        # 4) Compute averages
        observed_avg = np.nanmean(sub_obsv)
        permuted_avg = np.nanmean(sub_perm, axis=(1, 2))

        # 5) Calculate p-value
        if method == 'two_tail':
            p_value = np.mean(np.abs(permuted_avg) > np.abs(observed_avg))
        elif method == 'one_tail':
            p_value = np.mean(permuted_avg > observed_avg)
        else:
            raise ValueError('method must be either "one_tail" or "two_tail"')

        return p_value, observed_avg

    
### GENERAL NIFTI FUNCTIONS ###
                    
def mask_array(mask_path, data_array, threshold=0):
    if mask_path is None:
        from nimlab import datasets as nimds
        mask = nimds.get_img("mni_icbm152")
    else:
        mask = nib.load(mask_path)
    mask_data = mask.get_fdata()
    mask_indices = mask_data.flatten() > threshold
    masked_array = data_array.flatten()[mask_indices]
    return masked_array

def unmask_array(mask_path, data_array, threshold=0):
    if mask_path is None:
        from nimlab import datasets as nimds
        mask = nimds.get_img("mni_icbm152")
    else:
        mask = nib.load(mask_path)
    mask_data = mask.get_fdata()
    mask_indices = mask_data.flatten() > threshold
    unmasked_array = np.zeros(mask_indices.shape)
    unmasked_array[mask_indices] = data_array.flatten()
    return unmasked_array.reshape(mask_data.shape), mask.affine

def save_map(map_data, out_dir, file_name, mask_path=None):
    unmasked_map, mask_affine = unmask_array(data_array=map_data, mask_path=mask_path)
    img = nib.Nifti1Image(unmasked_map, affine=mask_affine)
    file_path = os.path.join(out_dir, file_name)
    if out_dir is not None:
        if bool(os.path.splitext(file_path)[1]):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        else: 
            os.makedirs(out_dir, exist_ok=True)
        nib.save(img, file_path)
        
    return img

def visualize_map(img, title):
    plotting.view_img(img, title=title).open_in_browser()

def save_individual_r_maps(data, verbose=False, out_dir='', filename='generated_map.nii.gz', mask_path=None):
    r_img = save_map(map_data=data, out_dir=out_dir, file_name=filename, mask_path=mask_path)
    if verbose: visualize_map(img=r_img, title=filename)