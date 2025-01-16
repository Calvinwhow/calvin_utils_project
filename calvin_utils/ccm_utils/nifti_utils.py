from calvin_utils.file_utils.import_functions import GiiNiiFileImport
import os
import pandas as pd
import numpy as np
import shutil
import json
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import CP
from tensorly.regression.cp_regression import CPRegressor
import statsmodels.api as sm

class DatasetNiftiImporter(GiiNiiFileImport):
    def __init__(self, df, dataset_col, nifti_col, indep_var_col, covariate_cols, out_dir, mask_path=None, regression_method='tensor'):
        self.df = df
        self.dataset_col = dataset_col
        self.nifti_col = nifti_col
        self.indep_var_col = indep_var_col
        self.covariate_cols = covariate_cols
        self.out_dir = out_dir
        self.mask_path = mask_path
        self.data_dict = {}
        self.regression_method = regression_method
        self._prepare_data_dict()
        self._create_dataset_dict()
        
    def _prep_dataset_dir(self, dataset):
        dataset_dir = os.path.join(self.out_dir, 'tmp', dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    def _prepare_data_dict(self):
        for dataset in self.df[self.dataset_col].unique():
            print("Importing dataset: ", dataset)
            dataset_df = self.df[self.df[self.dataset_col] == dataset]
            dataset_df.columns = self.df.columns
            dataset_dir = self._prep_dataset_dir(dataset)

            self.data_dict[dataset] = {
                'niftis': pd.DataFrame(),
                'indep_var': pd.DataFrame(),
                'covariates': pd.DataFrame()
            }

            nifti_paths = dataset_df[self.nifti_col].tolist()
            nifti_importer = GiiNiiFileImport(import_path=None, file_column=None, file_pattern=None)
            self.data_dict[dataset]['niftis'] = nifti_importer.import_matrices(nifti_paths)
            self.data_dict[dataset]['indep_var'] = self.prep_dmatrix(dataset_df.loc[:, [self.indep_var_col]])
            self.data_dict[dataset]['covariates'] = self.prep_dmatrix(dataset_df.loc[:, self.covariate_cols])

            self._process_dataset(dataset, dataset_dir)
            
    def prep_dmatrix(self, df):
        """
        Converts columns in the DataFrame to numeric. If conversion fails, converts them to categorical.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame whose columns need to be converted.

        Returns:
        --------
        pandas.DataFrame
            The DataFrame with columns converted to numeric or categorical.
        """
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                df[col] = df[col].astype('category').cat.codes
        return df
    
    def _add_intercept(self, covariates_arr):
        intercept = np.ones((covariates_arr.shape[0], 1)) # shape (samples, 1)
        return np.hstack((intercept, covariates_arr)) # shape (samples, N_covariates + 1)
    
    def _tensor_regression(self, endog, exog):
        # Ensure endog and exog are tensors
        endog = tl.tensor(endog)
        exog = tl.tensor(exog)

        # Perform tensor regression using CP decomposition
        model = CPRegressor(weight_rank=1)
        model.fit(exog, endog)
        residuals = endog - model.predict(exog)
        return residuals

    def _ols_regression(self, endog, exog):
        residuals = np.zeros_like(endog)
        for i in tqdm(range(endog.shape[1]), desc='OLS Regression'):
            model = sm.OLS(endog[:, i], exog).fit()
            residuals[:, i] = model.resid
        return residuals
    
    def _run_regression(self, masked_niftis_array, indep_var_array, covariates_array):
        print(masked_niftis_array.shape, indep_var_array.shape, covariates_array.shape)
        if self.regression_method == 'tensor':
            # Perform tensor regression
            nifti_residuals = self._tensor_regression(masked_niftis_array, covariates_array)
            indep_residuals = self._tensor_regression(indep_var_array, covariates_array)
        elif self.regression_method == 'ols':
            # Perform OLS regression
            nifti_residuals = self._ols_regression(masked_niftis_array, covariates_array)
            indep_residuals = self._ols_regression(indep_var_array, covariates_array)
        else:
            raise ValueError("Invalid regression method. Choose 'tensor' or 'ols'.")
        return nifti_residuals, indep_residuals
    
    def _save_residuals(self, dataset_dir, nifti_residuals_arr, indep_residuals_arr, covariates_arr):
        np.save(os.path.join(dataset_dir, 'niftis.npy'), nifti_residuals_arr)
        np.save(os.path.join(dataset_dir, 'indep_var.npy'), indep_residuals_arr)
        np.save(os.path.join(dataset_dir, 'covariates.npy'), covariates_arr)
        
    def _process_dataset(self, dataset, dataset_dir):
        niftis = self.data_dict[dataset]['niftis'] # Get the nifti
        _, _, masked_niftis = self.mask_dataframe(niftis, mask_path=self.mask_path) # mask the nifti 
        masked_niftis_arr = masked_niftis.values.T                          # shape (samples, voxels)
        indep_var_arr = self.data_dict[dataset]['indep_var'].values         # shape (samples, 1)
        covariates_arr = self.data_dict[dataset]['covariates'].values       # shape (samples, N_covariates)
        covariates_arr = self._add_intercept(covariates_arr)                # shape (samples, N_covariates+1)
        
        nifti_residuals, indep_residuals = self._run_regression(masked_niftis_arr, indep_var_arr, covariates_arr)

        self._save_residuals(dataset_dir, nifti_residuals, indep_residuals, covariates_arr)

    def cleanup(self):
        tmp_dir = os.path.join(self.out_dir, 'tmp')
        shutil.rmtree(tmp_dir)
        
    def _create_dataset_dict(self):
        dataset_dict = {}
        for dataset in self.data_dict.keys():
            dataset_dir = self._prep_dataset_dir(dataset)
            dataset_dict[dataset] = {
                'niftis': os.path.join(dataset_dir, 'niftis.npy'),
                'indep_var': os.path.join(dataset_dir, 'indep_var.npy'),
                'covariates': os.path.join(dataset_dir, 'covariates.npy')
            }
        dict_path = os.path.join(self.out_dir, 'tmp', 'dataset_dict.json')
        with open(dict_path, 'w') as f:
            json.dump(dataset_dict, f)
        print(f"Dataset dictionary saved to: {dict_path}")
        return dataset_dict