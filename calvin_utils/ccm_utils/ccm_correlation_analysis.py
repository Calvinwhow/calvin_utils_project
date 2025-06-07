import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from nilearn import plotting
from scipy.stats import spearmanr, pearsonr, rankdata
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.cm as cm

from calvin_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.ccm_utils.optimization.convergent_map_optimizer import NiftiOptimizer

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