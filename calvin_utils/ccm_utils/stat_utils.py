from calvin_utils.ccm_utils.npy_utils import DataLoader
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import os
import forestplot as fp

class CorrelationCalculator:
    def __init__(self, method='pearson', verbose=False):
        self.method = method
        self.verbose = verbose

    def _calculate_pearson_r_map(self, niftis, indep_var):
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
        return r

    def _calculate_spearman_r_map(self, niftis, indep_var):
        '''Not easily broadcast, sorry!'''
        n_voxels = niftis.shape[1]
        rho = np.zeros(n_voxels)
        for i in tqdm(range(n_voxels), desc='Running Spearman Rho'):
            rho[i], _ = spearmanr(indep_var, niftis[:, i])
            
        if self.verbose:
            print(f"Shape of niftis: {niftis.shape}")
            print(f"Shape of rho: {rho.shape}")
        return rho
    
    def _process_data(self, data):
        if self.method == 'pearson':
            self.correlation_map = self._calculate_pearson_r_map(data['niftis'], data['indep_var'])
        elif self.method == 'spearman':
            self.correlation_map = self._calculate_spearman_r_map(data['niftis'], data['indep_var'])
    
    def process_all_datasets(self, data_dict):
        correlation_maps = {}
        for dataset_name in data_dict.keys():
            data = DataLoader.load_dataset_static(data_dict, dataset_name)
            self._process_data(data)
            correlation_maps[dataset_name] = self.correlation_map
        return correlation_maps

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

    def create_and_display_forest_plot(self, x_label="Mean R"):
        """
        Generate and display a forest plot from the meta-convergence data using forestplot.py.
        """
        # Generate the forest plot
        ax = fp.forestplot(dataframe=self.data,
                # Necessary inputs
                estimate="Mean R",  # Column containing estimated effect size 
                varlabel="Dataset",  # Column containing variable label
                
                # Additional Plotting Inputs
                ll="CI Lower", hl="CI Upper",  # Columns containing conf. int. lower and higher limits
                
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
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            self.fig.savefig(os.path.join(self.out_dir, f"{title}.png"), bbox_inches='tight')
            self.fig.savefig(os.path.join(self.out_dir, f"{title}.svg"), bbox_inches='tight')
            print(f'Saved to {self.out_dir} as {title}.svg and .png')
            
    def run(self):
        """
        Orchestrator method.
        """
        self.table_prep()
        self.create_and_display_forest_plot()
        self.figure_saver()