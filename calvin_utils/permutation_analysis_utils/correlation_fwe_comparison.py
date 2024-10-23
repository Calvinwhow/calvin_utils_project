import pandas as pd
from tqdm import tqdm
import numpy as np
from calvin_utils.permutation_analysis_utils.correlation_fwe import CalvinFWEMap
import nibabel as nib
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean

class CalvinFWEWrapper:
    def __init__(self, neuroimaging_dataframe1, variable_dataframe1, 
                        mask_threshold, mask_path, out_dir, method='spearman', 
                        max_stat_method=None, vectorize=True,
                        roi_path=None, roi_threshold=0,
                        neuroimaging_dataframe2=None, variable_dataframe2=None, 
                        map_path=None, use_spearman=False, two_tail=False):
        '''
        map_path (str): path to a single nifti you want to perform a spatial correlation upon
        roi_path (str): path to the ROI you want to restrict your evaluation within. 
        roi_threshold (float): the value to threshold your ROI by for binarization to create the ROI mask.
        use_spearman (bool): whether to perform spatial correlation with Spearman Correlation. 
            if false, defaults to Pearson
        two_tail (bool): whether to perform two-tailed p-value testing
        '''
        self.roi_path = roi_path
        self.roi_threshold = roi_threshold
        self.map_path = map_path
        self.use_spearmanr=use_spearman
        self.two_tail=two_tail
        self.get_roi_mask() # generates self.roi_indides
        self.calvin_fwe1 = CalvinFWEMap(neuroimaging_dataframe=neuroimaging_dataframe1, 
                                        variable_dataframe=variable_dataframe1, 
                                        mask_threshold=mask_threshold, 
                                        mask_path=mask_path, 
                                        out_dir=out_dir, 
                                        method=method, 
                                        max_stat_method=max_stat_method, 
                                        vectorize=vectorize)
        if neuroimaging_dataframe2 is not None: self.calvin_fwe2 = CalvinFWEMap(neuroimaging_dataframe=neuroimaging_dataframe2, 
                                        variable_dataframe=variable_dataframe2, 
                                        mask_threshold=mask_threshold, 
                                        mask_path=mask_path, 
                                        out_dir=out_dir, 
                                        method=method, 
                                        max_stat_method=max_stat_method, 
                                        vectorize=vectorize)  
         
    #----Basic Preparatory Functions----#
    def check_array(self, data):
        if type(data) != np.array:
            data = np.array(data)
        return data
            
    def get_roi_mask(self):
        if self.roi_path is not None:
            self.roi_data_3d = nib.load(self.roi_path).get_fdata()
            self.roi_data_flat = self.roi_data_3d.flatten()
            self.roi_indices_flat = np.where(self.roi_data_flat > self.roi_threshold)[0]

    def apply_roi_mask(self, dataframe):
        return dataframe.loc[self.roi_indices_flat]
    
    def remove_roi_mask(self, dataframe):
        full_data_flat = np.zeros_like(self.roi_data_flat)
        full_data_flat[self.roi_indices_flat] = dataframe.values
        return full_data_flat.reshape(self.roi_data_3d.shape)
    
    def remove_nans_and_infs(self, array1, array2):
        mask = ~np.isnan(array1) & ~np.isnan(array2) & ~np.isinf(array1) & ~np.isinf(array2)
        return array1[mask], array2[mask]

    #----Permutation Wrapping Functions----#
    def get_observed_maps(self):
        observed_1 = self.calvin_fwe1.unmask_dataframe(self.calvin_fwe1.get_correlation_map())
        
        if self.map_path is None:
            observed_2 = self.calvin_fwe2.unmask_dataframe(self.calvin_fwe2.get_correlation_map())
        else:
            observed_2 = pd.DataFrame(nib.load(self.map_path).get_fdata().flatten(), columns=observed_1.columns)
            self.observed_2 = observed_2
        if self.roi_path is not None:
            observed_1 = self.apply_roi_mask(observed_1)
            observed_2 = self.apply_roi_mask(observed_2)
        observed_1, observed_2 = self.remove_nans_and_infs(observed_1, observed_2)
        return observed_1, observed_2

    def get_permuted_maps(self, n_permutations=1000):
        for _ in tqdm(range(n_permutations), desc='Running permutation'):
            perm1 = self.calvin_fwe1.unmask_dataframe(self.calvin_fwe1.get_correlation_map(permute=True))
            if self.map_path is None:
                perm2 = self.calvin_fwe2.unmask_dataframe(self.calvin_fwe2.get_correlation_map(permute=True))
            else:
                perm2 = self.observed_2
            if self.roi_path is not None:
                perm1 = self.apply_roi_mask(perm1)
                perm2 = self.apply_roi_mask(perm2)
            perm1, perm2 = self.remove_nans_and_infs(perm1, perm2)
            yield perm1, perm2
            
    def get_bootstrapped_maps(self, n_bootstraps=1000):
        for _ in tqdm(range(n_bootstraps), desc='Running bootstraps'):
            boot1 = self.calvin_fwe1.unmask_dataframe(self.calvin_fwe1.get_correlation_map(permute=False, bootstrap=True))
            if self.map_path is None:
                boot2 = self.calvin_fwe2.unmask_dataframe(self.calvin_fwe2.get_correlation_map(permute=False, bootstrap=True))
            else:
                boot2 = self.observed_2
            if self.roi_path is not None:
                boot1 = self.apply_roi_mask(boot1)
                boot2 = self.apply_roi_mask(boot2)
            boot1, boot2 = self.remove_nans_and_infs(boot1, boot2)
            yield boot1, boot2
            
    def run_analysis(self, comparison_function, n_permutations=1000, bootstrap=False):
        observed_1, observed_2 = self.get_observed_maps()
        observed_result = comparison_function(observed_1, observed_2)
        
        results = []
        if bootstrap:                   # Run bootstrap
            for boot1, boot2 in self.get_bootstrapped_maps(n_permutations):
                boot_result = comparison_function(boot1, boot2)
                results.append(boot_result)
        else:                           # Run permutation
            for perm1, perm2 in self.get_permuted_maps(n_permutations):
                permuted_result = comparison_function(perm1, perm2)
                results.append(permuted_result)
        
        self.p_value_calculation(observed_result, results)
        return observed_result, results

    def p_value_calculation(self, observed_results, permuted_results):
        """
        Returns:
            np.ndarray: Array of p-values corresponding to the uncorrected statistic values.
        """
        # Calculate P-Values
        observed_results = (np.abs(observed_results) if self.two_tail else observed_results)
        permuted_results = (np.abs(permuted_results) if self.two_tail else permuted_results)
        
        observed_results = self.check_array(observed_results)
        permuted_results = self.check_array(permuted_results)
        if observed_results.shape == (1,):
            observed_results = observed_results.reshape(-1,1)
        p_values = np.mean(observed_results >= permuted_results, axis=0)
        print(f"Observed: {observed_results}, p-value {p_values}, using 2-tail: {self.two_tail}.")

    #----Functions for Statistical Analysis----#
    def calculate_pearson_correlation(self, map1, map2):
        flat_map1 = map1.values.flatten()
        flat_map2 = map2.values.flatten()
        if self.use_spearmanr: correlation, _ = spearmanr(flat_map1, flat_map2) 
        else: correlation, _ = pearsonr(flat_map1, flat_map2)
        return correlation

    def calculate_peak_voxel_distance(self, map1, map2, max=True, abs=True):
        '''
        Params:
        max (bool) : This will take the maximum in the 2 maps by default. Set to False to take the minima.
        abs (bool) : This will take the absolute correlation in the 2 maps by default. 

        However, if you want 
        '''
        if max: 
            peak_index1 = np.argmax(np.abs(map1) if abs else map1)
            peak_index2 = np.argmax(np.abs(map2)  if abs else map1)
        else:
            peak_index1 = np.argmax(np.min(map1) if abs else map1)
            peak_index2 = np.argmax(np.min(map2)  if abs else map1)
        
        peak_voxel1_3d = np.array(np.unravel_index(peak_index1, self.roi_data_3d.shape))
        peak_voxel2_3d = np.array(np.unravel_index(peak_index2, self.roi_data_3d.shape))

        distance = euclidean(peak_voxel1_3d, peak_voxel2_3d)
        return distance
    
    def calculate_peak_correlation(self, map1, map2, abs=False):
        map_1_max = np.max(np.abs(map1) if abs else map1)
        map_2_max = np.max(np.abs(map2) if abs else map2)
        return map_1_max - map_2_max

    #----Functions to Call Statistical Analysis----#
    def run_pearson_analysis(self, n_permutations=1000):
        return self.run_analysis(self.calculate_pearson_correlation, n_permutations)

    def run_peak_voxel_analysis(self, n_permutations=1000):
        return self.run_analysis(self.calculate_peak_voxel_distance, n_permutations)
    
    def run_peak_corr_analysis(self, n_permutations=1000):
        return self.run_analysis(self.calculate_peak_correlation, n_permutations)
    
    def bootstrap_peak_corr(self, n_permutations=1000):
        return self.run_analysis(self.calculate_peak_correlation, n_permutations, bootstrap=True)

