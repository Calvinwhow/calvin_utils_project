import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.genmod.generalized_linear_model import GLMResults
from calvin_utils.statistical_utils.response_surface_utils.regression_response_surface import GLMPredictionComparison

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _parse_formula(formula):
    """
    Parse the formula (either user-provided or from the fitted model) to extract the outcome and independent variables.

    Returns
    -------
    tuple
        A tuple containing:
        - outcome (str): The name of the dependent variable (outcome).
        - independent_vars (list): List of unique independent variables (excluding interactions).
    """
    # Split by '~' to separate outcome from predictors
    outcome, predictors = formula.split('~')
    outcome = outcome.strip()

    # Split the predictors by '*', ':', and '+', and strip each
    predictors_list = predictors.replace('*', '+').replace(':', '+').split('+')
    predictors_list_stripped = [var.strip() for var in predictors_list]
    independent_vars = list(dict.fromkeys(predictors_list_stripped))  # Preserve order and get unique variables

    return outcome, independent_vars

class GLMPredictionPermutationTest:
    def __init__(self, data_df: pd.DataFrame, formula: str, cohort_col: str, two_tailed : bool = False,
                 method: str = 'pearsonr', n_permutations: int = 1000, random_state: Optional[int] = None):
        """
        Initialize the permutation test class.
        
        Parameters
        ----------
        data_df : pd.DataFrame
            The primary dataframe containing all necessary data, including the cohort indicator.
        formula : str
            The formula string for the GLM model (e.g., 'outcome ~ var1 + var2 + var1:var2').
        cohort_col : str
            The name of the column in data_df that indicates the cohort.
        two_tailed : bool
            Whether to run a 2-tailed test. 
            - Two-tails checks if the observed spcorrel is more similar or anti-similar than by chance. 
            - One-tail checks if the obsvered spcorrl is more similar than by chance. 
        method : str, optional
            The scipy.stats correlation method to use. Options: 'pearsonr', 'spearmanr', 'kendalltau'. Default is 'pearsonr'.
        n_permutations : int, optional
            Number of permutations to perform. Default is 1000.
        random_state : int, optional
            Seed for reproducibility. Default is None.
        """
        self.data_df = data_df.copy()
        self.formula = formula
        self.cohort_col = cohort_col
        self.method = method
        self.two_tailed = two_tailed
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_state)
        
        self.observed_corrs: Dict[str, float] = {}
        self.p_values: Dict[str, float] = {}
        
        self._validate_parameters()
        self.outcome_col, _ = _parse_formula(self.formula)
        
    def _validate_parameters(self):
        """Validate input parameters."""
        if not isinstance(self.n_permutations, int) or self.n_permutations < 1:
            raise ValueError("n_permutations must be a positive integer.")
        if self.cohort_col not in self.data_df.columns:
            raise ValueError(f"Cohort column '{self.cohort_col}' not found in data.")
    
    def _permute_data(self, df, method='permute_outcomes') -> pd.DataFrame:
        """
        Shuffle cohort labels in the dataframe IF METHOD == LABELS. If method !=labels, will shuffle the outcomes within each cohort. 

        Returns
        -------
        pd.DataFrame
            A new dataframe with shuffled cohort labels.
        """
        df = df.copy()
        if method == 'labels':
            df[self.cohort_col] = np.random.permutation(df[self.cohort_col].values)
        elif method == 'permute_outcomes':
            outcome_permuted = []
            for cohort, group in df.groupby(self.cohort_col):
                permuted = np.random.permutation(group[self.outcome_col].values)
                outcome_permuted.append(pd.Series(permuted, index=group.index))
            outcome_permuted = pd.concat(outcome_permuted)
            df.loc[outcome_permuted.index, self.outcome_col] = outcome_permuted
        else:
            raise ValueError(f"{method} is not supported. Choose 'labels' or 'permute_outcomes'.")
        return df
    
    def _get_spcorrs(self, df):
        glm_comparison = GLMPredictionComparison(data_df=df, formula=self.formula,cohort_col=self.cohort_col,method=self.method)
        return glm_comparison.run()
        
    def _compute_observed_correlations(self):
        """Compute observed spatial correlations using GLMPredictionComparison."""
        logger.info("Computing observed spatial correlations.")
        self.observed_corrs = self._get_spcorrs(self.data_df)
        self.null_corrs = {key: [] for key in self.observed_corrs.keys()}
        logger.info("Observed spatial correlations computed.")
    
    def _compute_null_distribution(self):
        """Generate null distribution through permutation testing."""
        logger.info(f"Starting permutation testing with {self.n_permutations} permutations.")
        milestone = max(1, self.n_permutations // 10)
        for i in range(0, self.n_permutations):
            df_permuted = self._permute_data(self.data_df)              # Permute Data
            permuted_corrs = self._get_spcorrs(df_permuted)
            for key, corr in permuted_corrs.items():                    # Append permuted correlations to null_corrs
                self.null_corrs[key].append(corr)
                
            if (i + 1) % milestone == 0:                                # Logging progress
                logger.info(f"Completed {i + 1}/{self.n_permutations} permutations.")
        logger.info("Permutation testing completed.")
    
    def _organize_correlations(self) -> tuple:
        """
        Organize observed and permuted spatial correlations into a NumPy array.

        Each column corresponds to a cohort pair, and each row corresponds to a permutation.

        Returns
        -------
        tuple of 2 np.ndarrays (null, observed)
            A 2D NumPy array with shape (n_permutations, n_cohort_pairs),
            where each column represents a cohort pair and each row represents a permutation iteration.
        """
        permutation_lengths = [len(corrs) for corrs in self.null_corrs.values()]       # Ensure all cohort pairs have the same number of permutations
        if len(set(permutation_lengths)) != 1:
            raise ValueError("All null_corrs lists must have the same number of permutations.")
        cohort_pairs = sorted(self.null_corrs.keys())                                   # Sort cohort pairs for consistent column ordering
        null_corrs_lists = [self.null_corrs[pair] for pair in cohort_pairs]             # Extract null correlations for each cohort pair in sorted order
        null_corrs_array = np.column_stack(null_corrs_lists)                            # Convert the list of lists into a 2D NumPy array
        
        obs_corrs_lists = [self.observed_corrs[pair] for pair in cohort_pairs]
        obs_corrs_array = np.column_stack(np.array(obs_corrs_lists))
        return (null_corrs_array, obs_corrs_array)
        
    def _compute_p_value(self, observed: float, null_dist: List[float]) -> float:
        """It does what you think it does."""
        if self.two_tailed: 
            return np.mean(np.abs(null_dist) >= np.abs(observed)) # More similar or anti-similar than by chance
        else:
            return np.mean(null_dist >= observed)                 # More similar than by change
        
    def _compute_p_value_average_similarity(self):
        null_arr, obs_arr = self._organize_correlations()
        null_avg = np.mean(null_arr, axis=1)                    # Average similarity of the permutations
        obs_avg = np.mean(obs_arr)                              # Average similarity of the groups
        p = self._compute_p_value(observed=obs_avg, null_dist=null_avg)
        logger.info(f"P-value for average similarity ({obs_avg}) = {p}")
        
        
    def _calculate_p_values(self):
        """Calculate p-values based on the null distribution."""
        logger.info("Calculating p-values based on null distributions.")
        for key, observed in self.observed_corrs.items():
            null_dist = self.null_corrs.get(key)
            p_value = self._compute_p_value(observed, null_dist)
            self.p_values[key] = p_value
            logger.info(f"Results for {key}: R = {observed} | p = {p_value}")
    
    def run(self):
        """
        Execute the permutation test process.
        
        Steps:
        1. Compute observed spatial correlations.
        2. Perform permutation testing to generate null distributions.
        3. Calculate p-values for observed correlations.
        """
        self._compute_observed_correlations()
        self._compute_null_distribution()
        self._compute_p_value_average_similarity()
        self._calculate_p_values()
        logger.info("Permutation test completed.")
    
    def get_results(self) -> pd.DataFrame:
        """
        Retrieve the observed correlations and p-values.
        
        Returns
        -------
        pd.DataFrame
            A dataframe containing observed correlations and p-values for each cohort pair.
        """
        results = pd.DataFrame({
            'Cohort_Pair': list(self.observed_corrs.keys()),
            'Observed_Correlation': list(self.observed_corrs.values()),
            'P_Value': [self.p_values.get(k, np.nan) for k in self.observed_corrs.keys()]
        })
        return results
    
    def save_results(self, out_dir: str):
        """
        Save the observed correlations and p-values to a CSV file.
        
        Parameters
        ----------
        out_dir : str
            Directory to save the results CSV file.
        """
        os.makedirs(out_dir, exist_ok=True)
        results = self.get_results()
        file_path = os.path.join(out_dir, 'spatial_correlation_permutation_results.csv')
        results.to_csv(file_path, index=False)
        logger.info(f"Saved permutation test results to {file_path}")
    
    def save_null_distributions(self, out_dir: str):
        """
        Save the null distributions for each cohort pair to CSV files.
        
        Parameters
        ----------
        out_dir : str
            Directory to save the null distribution CSV files.
        """
        os.makedirs(out_dir, exist_ok=True)
        for key, null_dist in self.null_corrs.items():
            file_path = os.path.join(out_dir, f'null_distribution_{key}.csv')
            pd.DataFrame({'Correlation': null_dist}).to_csv(file_path, index=False)
            logger.info(f"Saved null distribution for {key} to {file_path}")
