import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import norm, pearsonr, spearmanr, ttest_ind
from calvin_utils.ccm_utils.resampling_plot import ResampleVisualizer

class CompareCorrelations:
    def __init__(self, dv_path: str,  iv_paths: list[str] = None, n_bootstraps: int = 1000, seed: int = 42, method: str = 'spearman'):
        """
        Initializes the CompareCorrelations object.
        Args:
            dv_path (str): File path to a CSV with observed outcomes in a column, observations in rows
                Should share same order as CSVs chosen in DV_paths
            iv_paths (list[str], optional): List of file paths containing dependent variables in columns. Defaults to None.
                Expects a single column, with observations in rows. Should have same order across CSVs. 
                Names for the plot are derived from the basename of these CSVs.
            n_bootstraps (int, optional): Number of bootstrap samples to use for statistical analysis. Defaults to 1000.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            method (str, optional): Type of correlation to use. Options: 'spearman' or 'pearson'
        Attributes:
            _dv_df (pd.DataFrame or None): DataFrame to store observed labels, initialized as None.
            _pred_dfs (dict): Dictionary to store DataFrames of predicted labels for each classifier.
            boot_idx_cache (any): Cache for bootstrap indices.
            n_bootstraps (int): Number of bootstrap samples.
            rng (np.random.RandomState): Random number generator initialized with the given seed.
            auc_dist (dict): Dictionary to store AUC distributions for each classifier.
            labels_path (str or None): Path to the file containing the true labels.
            dv_path (list[str] or None): List of file paths containing predicted labels.
        """
        self.method      = method
        self._dv_df      = None
        self._iv_dfs     = {}
        self.corr_dist    = {}
        self.n_bootstraps = n_bootstraps
        self.rng = np.random.RandomState(seed)
        self.dv_path = dv_path
        self.iv_paths = iv_paths
        self.boot_idx_cache = self._paired_boot_indices()
        self._get_corr_dist()
        
    ### Setter/Getter Logic ###
    @property
    def dv_path(self) -> str:
        return self._dv_path

    @dv_path.setter
    def dv_path(self, path: str):
        if not isinstance(path, str):
            raise ValueError("labels_path must be a string CSV filepath")
        self._dv_path = path
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        self._dv_df = df.astype(float)

    @property
    def iv_paths(self) -> list[str]:
        return list(self._iv_dfs.keys())

    @iv_paths.setter
    def iv_paths(self, paths: list[str]):
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("iv_paths must be a list of CSV filepaths")
        if len(paths) != 2:
            raise ValueError(f"iv_paths should only contain two paths to compare. Detected {len(paths)}.")
        self._iv_dfs = {}
        for idx, p in enumerate(paths):
            df = pd.read_csv(p)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            if self.dv_path is not None:
                self._validate(p, df)
            name = os.path.splitext(os.path.basename(p))[0]
            self._iv_dfs[name] = df.astype(float)

    @property
    def observations_df(self) -> pd.DataFrame:
        return self._dv_df

    @property
    def predictions_dfs(self) -> dict[str, pd.DataFrame]:
        """Returns a dict mapping filepath → its predictions DataFrame."""
        return self._iv_dfs

    def _validate(self, path: str, df: pd.DataFrame):
        """Ensure one predictions-DF lines up exactly with obs_df cols & shape."""
        if df.shape != self._dv_df.shape:
            raise ValueError(f"[{path}] shape mismatch: obs {self._dv_df.shape} vs pred {df.shape}")
        
    def _paired_boot_indices(self):
        """Returns an (n_bootstraps x n) integer array of row-indices. Generated once, and reused for each model"""
        n = self._dv_df.shape[0]
        return self.rng.randint(0, n, size=(self.n_bootstraps, n))
    
    ### Statistical Methods ###
    def _get_correlation(self, iv, dv):
        if self.method=='spearman':
            return spearmanr(iv, dv)[0]
        else:
            return pearsonr(iv, dv)[0]
        
    def _get_ttest(self, arr1, arr2):
        print("t-test results of bootstrap: \n\t", ttest_ind(arr1, arr2))

    ### Bootstrapping Logic ###
    def _bootstrap_correlations(self, iv_df: pd.DataFrame):
        dv = self._dv_df.to_numpy().ravel()
        iv_df = iv_df.to_numpy().ravel()
        correlations = [self._get_correlation(iv_df[idx], dv[idx]) for idx in self.boot_idx_cache]
        return np.asarray(correlations)
    
    def _get_corr_dist(self):
        for k in self._iv_dfs.keys():
            self.corr_dist[k] = self._bootstrap_correlations(self._iv_dfs[k])
    
    ### Plotting Methods ###
    def superiority_plot(self, out_dir=None):
        for model_a, model_b in combinations(self.corr_dist.keys(), 2):
            corr_a = self.corr_dist[model_a]
            corr_b = self.corr_dist[model_b]
            self._get_ttest(corr_a,corr_b)
            resample_viz = ResampleVisualizer(stat_array_1=corr_a,stat_array_2=corr_b, model1_name=model_a, model2_name=model_b, stat='Correlation', out_dir=out_dir)
            resample_viz.draw()
            
    def run(self, out_dir=None):
        """
        Run the comparison of classifiers and generate plots.
        Args:
            out_dir (str, optional): Directory to save the output plots. Defaults to None.
        """
        self.superiority_plot(out_dir)
        print("Plots saved to:", out_dir)