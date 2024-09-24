import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from calvin_utils.ml_utils.graph_feature_selector.loocv import run_loocv_for_formula

class LOOCVManager:
    """
    A manager class to perform Leave-One-Out Cross-Validation (LOOCV) for multiple formulae in parallel.
    
    Attributes:
    -----------
    data_df : pd.DataFrame
        The DataFrame containing all necessary variables.
    formulae : list
        A list of formula strings to be evaluated.
    model_type : str
        The type of model to be fitted. Options are 'ols', 'krr', 'binomial_logit', 'multinomial_logit'.
    kwargs : dict
        Additional parameters for the StatisticalModeler.
    
    Methods:
    --------
    run_loocv_for_formula(formula):
        Performs LOOCV for a single formula and returns the performance metrics.
    run_all():
        Executes LOOCV for all formulae in parallel and aggregates the results.
    """
    
    def __init__(self, data_df, formulae, model_type='ols', disable_tqdm=False, **kwargs):
        """
        Initializes the LOOCVManager with the dataset, list of formulae, and model specifications.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            The DataFrame containing all necessary variables.
        formulae : list
            A list of formula strings to be evaluated.
        model_type : str, optional, default='ols'
            The type of model to be fitted. Options are 'ols', 'krr', 'binomial_logit', 'multinomial_logit'.
        kwargs : dict
            Additional parameters for the StatisticalModeler.
        disable_tqdm : bool
            Whther to silence the TQDM loading bar running a LOOCV
        """
        self.data_df = data_df
        self.formulae = formulae
        self.model_type = model_type
        self.kwargs = kwargs
        self.disable = disable_tqdm

    def multiprocess_organizer(self, max_workers):
        """
        Distributed an LOOCV for a formula to a core
        
        Parameters:
        -----------
        max_workers : int
            The maximum number of worker processes to use. Defaults to the number of processors on the machine.
        
        Returns:
        --------
        results
            A list containing all dicts of formulae and their performance metrics.
        """
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_formula = {executor.submit(run_loocv_for_formula, formula, self.data_df, self.model_type, **self.kwargs): formula for formula in self.formulae}
            
            # As LOOCVs finish, extract their data
            for future in as_completed(future_to_formula):
                formula = future_to_formula[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    # In case of unexpected exceptions
                    results.append({'Formula': formula, 'Error': str(exc)})
        return results
    
    def series_organizer(self):
        """Runs serial LOOCV and returns list containing dict of {formula: metrics} where metrics is a dict"""
        results = []
        for formula in tqdm(self.formulae, desc='Running Formulae', disable=self.disable):
            data = run_loocv_for_formula(formula, self.data_df, self.model_type, **self.kwargs)
            results.append(data)
        return results
    
    def run_all(self, multiprocess=False, max_workers=None, debug=False):
        """
        Executes LOOCV for all formulae in parallel and aggregates the results.
        
        Parameters:
        -----------
        multiprocess : bool, optional
            Whether to multiprocess or run in series.
        max_workers : int, optional
            The maximum number of worker processes to use. Defaults to the number of processors on the machine.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing all formulas and their corresponding performance metrics.
        """
        if debug: t1 = time()
        if multiprocess:
            results = self.multiprocess_organizer(max_workers)
        else:
            results = self.series_organizer()
        if debug: print(f"Compute time: {time() - t1} seconds")
        return pd.DataFrame(results)