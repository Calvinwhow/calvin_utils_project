import os
import numpy as np
import pandas as pd
import forestplot as fp
from jax import jit
import jax.numpy as jnp
from scipy.stats import rankdata
from calvin_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.ccm_utils.stat_utils_jax import _rankdata_jax, calculate_spearman_r_map_jax, _calculate_pearson_r_map_jax

class CorrelationCalculator:
    def __init__(self, method='pearson', verbose=False, use_jax=False):
        self.method = method
        self.verbose = verbose
        self.use_jax = use_jax

    @staticmethod
    def _check_for_nans(array, nanpolicy='remove', verbose=False):
        '''Check array for nan and raise error if present'''
        if np.isnan(array).any():
            if verbose:
                print("NaN detected in the array.")
                print(f"Array shape: {array.shape}")
                print(f"Array with NaNs: {array}")
                nan_indices = np.argwhere(np.isnan(array))
                print(f"Indices of NaNs: {nan_indices}")
                for index in nan_indices:
                    print(f"NaN at index: {index}, value: {array[tuple(index)]}")
                inf_indices = np.argwhere(np.isinf(array))
                if inf_indices.size > 0:
                    print(f"Indices of Infs: {inf_indices}")
                    for index in inf_indices:
                        print(f"Inf at index: {index}, value: {array[tuple(index)]}")
            if nanpolicy=='remove':
                max_val = np.nanmax(array)
                min_val = np.nanmin(array)
                array = np.nan_to_num(array, nan=0, posinf=max_val, neginf=min_val)
            elif nanpolicy=='permit':
                array = array
            elif nanpolicy=='stop':
                raise ValueError("The array contains NaNs.")
            else:
                raise ValueError("Selected nanpolicy does not exist. choose 'stop' | 'permit' | 'remove' ")
        return array
            
    def _rankdata(self, array, vectorize=True):
        """Vectorized ranking function using NumPy. Handles ties sloppily."""
        if self.use_jax:
            return _rankdata_jax(array, vectorize)
        
        if vectorize:       #very fast, but no tie handling.
            ranks = np.empty_like(array, dtype=float)       #shape (voxels, n) initialized array
            rows = np.arange(array.shape[0])[:, None]       #shape (voxels, 1) Row indices, ascending in order. 
            cols = np.arange(array.shape[1])                #shape (1,      n) Col indices, ascending in order. 
            
            idx_sorted = np.argsort(array, axis=0)          #shape (voxels, n) Row indices, ascending in rank. For each column. 
            ranks[idx_sorted, cols] = rows                  #shape (voxels, n) Assigns row indices by ascending rank. For each column. 
        else:               #very slow, but handles ties better.
            ranks = np.apply_along_axis(rankdata, 0, array) 
        return ranks                                        #returns the rank matrix, not the sorted data. 

    def _calculate_spearman_r_map(self, niftis, indep_var):
        """Calculate the Spearman rank-order correlation coefficient for each voxel in a fully vectorized manner."""
        # Rank the data using scipy.stats.rankdata to handle ties
        if self.use_jax:
            return calculate_spearman_r_map_jax(niftis, indep_var)
        
        # self.ranked_niftis = self._rankdata(niftis, vectorize=True) # TODO--ONLY RANK THE NIFTIS THE FIRST TIME. THEN, STORE THE RANK AND USE IT IN FUTURE! ?OVERWRITE THE ORIGINAL NIFTI DATA W/ RANKS
        ranked_indep_var = rankdata(indep_var)[:, np.newaxis]
        rho = self._calculate_pearson_r_map(niftis, ranked_indep_var)
        return rho

    def _calculate_pearson_r_map(self, niftis, indep_var):
        if self.use_jax:
            return _calculate_pearson_r_map_jax(niftis, indep_var)
        
        X = indep_var
        Y = niftis
        X_BAR = X.mean(axis=0)[:, np.newaxis]
        Y_BAR = Y.mean(axis=0)[np.newaxis, :]
        X_C = X - X_BAR
        Y_C = Y - Y_BAR
        NUMERATOR = np.dot(X_C.T, Y_C)
        SST_X = np.sum((X - X_BAR)**2, axis=0)
        SST_Y = np.sum((Y - Y_BAR)**2, axis=0)
        DENOMINATOR = np.sqrt(SST_X * SST_Y)
        r = NUMERATOR / DENOMINATOR
        
        if self.verbose:
            print(f"Shape of X: {X.shape}")
            print(f"Shape of Y: {Y.shape}")
            print(f"Shape of X_BAR: {X_BAR.shape}")
            print(f"Shape of Y_BAR: {Y_BAR.shape}")
            print(f"Shape of X_C: {X_C.shape}")
            print(f"Shape of Y_C: {Y_C.shape}")
            print(f"Shape of NUMERATOR: {NUMERATOR.shape}")
            print(f"Shape of DENOMINATOR: {DENOMINATOR.shape}")
            print("Shape of r: ", r.shape)
        return r

    def _process_data(self, data):
        self._check_for_nans(data['niftis'])
        self._check_for_nans(data['indep_var'])
        if self.method == 'pearson':
            self.correlation_map = self._calculate_pearson_r_map(data['niftis'], data['indep_var'])
        elif self.method == 'spearman':
            self.correlation_map = self._calculate_spearman_r_map(data['niftis'], data['indep_var'])
        return self.correlation_map
    
    def generate_correlation_maps(self, data_loader):
        corr_map_dict = {}
        for dataset_name in data_loader.dataset_paths_dict.keys():
            if self.method == 'pearson':
                data = data_loader.load_dataset(dataset_name)
            elif self.method =='spearman':
                data = data_loader.load_dataset(dataset_name, nifti_type='niftis_ranked')            
            corr_map_dict[dataset_name] = self._process_data(data)
        return corr_map_dict

class MetaConvergenceForestPlot:
    """
    This class takes a DataFrame containing meta-convergence data and generates a forest plot
    to visualize the confidence intervals and mean R values for each dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the meta-convergence data with columns 'Dataset', 'CI Lower', 'CI Upper', and 'Mean R'.
    sig_digits : int, optional
        Number of significant digits to display in the forest plot (default: 2).
    out_dir : str, optional
        Directory path where the plot should be saved (default: None).
    table : bool, optional
        Flag to indicate whether the plot should be displayed as a table (default: False).
    """

    def __init__(self, data, sig_digits=2, out_dir=None, table=False):
        self.data = data
        self.sig_digits = sig_digits
        self.out_dir = out_dir
        self.table = table

    def table_prep(self):
        """
        If we want to use a table with less than 6 regressors, the table output will be malformed.
        To address this, we fill the bottom of the self.data_for_plot DataFrame with np.NaN
        to expand over 6 rows.
        """
        if self.table and len(self.data) < 7:
            additional_rows_needed = 7 - len(self.data)
            
            # Create a DataFrame with the additional rows filled with np.NaN
            additional_rows = pd.DataFrame({
                'Mean R': [np.NaN] * additional_rows_needed,
                'CI Lower': [np.NaN] * additional_rows_needed,
                'CI Upper': [np.NaN] * additional_rows_needed,
                'Dataset': [''] * additional_rows_needed  # Assuming label can remain as an empty string
            })

            # Append the additional rows to the data DataFrame
            self.data = pd.concat([self.data, additional_rows], ignore_index=True)

    def create_and_display_forest_plot(self, x_label="Mean R", estimate="Mean R", varlabel="Dataset", ll="CI Lower", hl="CI Upper"):
        """
        Generate and display a forest plot from the meta-convergence data using forestplot.py.
        """
        # Generate the forest plot
        ax = fp.forestplot(dataframe=self.data,
                # Necessary inputs
                estimate=estimate,  # Column containing estimated effect size 
                varlabel=varlabel,  # Column containing variable label
                
                # Additional Plotting Inputs
                ll=ll, hl=hl,  # Columns containing conf. int. lower and higher limits
                
                # Axis Labels
                xlabel=x_label,
                ylabel='Est.(95% Conf. Int.)',
                
                # Forest Plot Configuration
                decimal_precision=self.sig_digits,
                capitalize='capitalize',
                color_alt_rows=False,  # Gray alternate rows
                table=self.table,
                flush=False,
                
                # Image Configuration
                **{"marker": "D",  # Set marker symbol as diamond
                    "markersize": 150,  # Adjust marker size
                    "xlinestyle": (0, (10, 5)),  # Long dash for x-reference line 
                    "xlinecolor": "#808080",  # Gray color for x-reference line
                    "xtick_size": 12,  # Adjust x-ticker fontsize
                    'fontfamily': 'helvetica'
                    }  
            )
        self.fig = ax.figure
        self.fig.show()
        
    def figure_saver(self, title="meta_convergence_forest_plot"):
        """
        Method to save the forest plot.
        """
        # Save the plot as PNG and SVG
        os.makedirs(self.out_dir+'/forest_plots', exist_ok=True)
            
        if self.out_dir:
            self.fig.savefig(os.path.join(self.out_dir, f"forest_plots/{title}.png"), bbox_inches='tight')
            self.fig.savefig(os.path.join(self.out_dir, f"forest_plots/{title}.svg"), bbox_inches='tight')
            print(f'Saved to {self.out_dir} as {title}.svg and .png')
            
    def run(self):
        """
        Orchestrator method.
        """
        self.table_prep()
        self.create_and_display_forest_plot()
        self.figure_saver()