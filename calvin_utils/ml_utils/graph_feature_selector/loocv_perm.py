import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from calvin_utils.ml_utils.graph_feature_selector.loocv_manager import LOOCVManager


class LOOCVPermutationTester:
    def __init__(self, data_df, results_df, model_type='ols', num_permutations=10, 
                 multiprocess=True, max_workers=None, **kwargs):
        self.results_df = results_df.dropna(subset='R2')
        self.original_data_df = data_df.copy()
        self.model_type = model_type
        self.kwargs = kwargs
        self.num_permutations = num_permutations
        self.multiprocess = multiprocess
        self.max_workers = max_workers
        self.formulae = self.get_formulae()
        self.response_variable = self._extract_response_variable()
        
    def get_formulae(self):
        return self.results_df['Formula'].to_list()

    def _extract_response_variable(self):
        first_formula = self.formulae[0]
        return first_formula.split('~')[0].strip()

    def _shuffle_response(self, data_df):
        shuffled_df = data_df.copy()
        shuffled_df[self.response_variable] = np.random.permutation(shuffled_df[self.response_variable].values)
        return shuffled_df

    def _run_loocv(self, data_df):
        manager = LOOCVManager(data_df=data_df, formulae=self.formulae, model_type=self.model_type, disable_tqdm=True, **self.kwargs)
        return manager.run_all(multiprocess=False, max_workers=self.max_workers)

    def _run_single_permutation(self, _):
        shuffled_df = self._shuffle_response(self.original_data_df)
        return self._run_loocv(shuffled_df)

    def _run_permutations(self):
        permutation_distributions = {formula: [] for formula in self.formulae}
        permutations = range(self.num_permutations)

        if self.multiprocess:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(tqdm(executor.map(self._run_single_permutation, permutations), 
                                    total=self.num_permutations, desc='Permutations'))
        else:
            results = [self._run_single_permutation(i) for i in tqdm(permutations, desc='Permutations')]

        for permuted_df in results:
            for _, row in permuted_df.iterrows():
                formula = row['Formula']
                metric = row['Entropy'] if pd.notna(row.get('Entropy')) else row['RMSE']
                permutation_distributions[formula].append(metric)

        return permutation_distributions

    def _calculate_pvalues(self, permutation_distributions):
        p_values = []
        for _, row in self.results_df.iterrows():
            formula = row['Formula']
            observed_metric = row['Entropy'] if pd.notna(row.get('Entropy')) else row['RMSE']
            permuted_metrics = permutation_distributions[formula]
            p_val = np.mean(np.array(permuted_metrics) <= observed_metric)
            p_values.append(p_val)
        self.results_df['p-value'] = p_values
        return self.results_df

    def run(self):        
        print(f"Running {self.num_permutations} permutations...")
        permutation_distributions = self._run_permutations()
        print("Calculating p-values...")
        results_df = self._calculate_pvalues(permutation_distributions)
        return results_df