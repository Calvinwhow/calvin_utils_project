import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Tuple

def import_covariates(control_covariates_csv_path: str, patient_covariates_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Import the covariates given paths. 
    Remove NaNs
    """
    control_covariates_df = pd.read_csv(control_covariates_csv_path, index_col=0).dropna(axis=1)
    patient_covariates_df = pd.read_csv(patient_covariates_csv_path, index_col=0).dropna(axis=1)
    
    return control_covariates_df, patient_covariates_df

class CalvinWMap():
    """
    This is a class to orchestrate W-mapping process. It is not optimal, but it is easy to code and easy to follow.
    Will initialize with the requisite dictionaries containing dataframes of information, as well as covariate dataframes. 
    
    Improvements:
    Hat-matrix based vectorization to extract betas. 
    Apply betas on vectorized basis to extract predictions. 
    vectorized extraction of prediction error standard deviation. 
    betas can be used on the patient data rather than calling the regression. 
    Vectorized standaridization of prediction error. 
    """
    def __init__(self, dataframes_dict: dict, control_dataframes_dict: dict, control_covariates_df: pd.DataFrame, patient_covariates_df: pd.DataFrame, use_intercept: bool=False, mask: bool=True):
        """
        Need to provide the dataframe dictionaries and dataframes of importance. 
        
        Args:
        - dataframes_dict (dict): Dictionary containing patient dataframes.
        - control_dataframes_dict (dict): Dictionary containing control dataframes.
        - control_covariates_df (pd.DataFrame): DataFrame where each column represents represents a control subject,
                                        and each row represents the covariate. 
        - patient_covariates_df (pd.DataFrame): Same as above, but for patients. 
        - use_intercept (bool): If true, model will use an intercept for the GLM, which is atypical. Defaults to False. 
        - mask (bool): If true, will mask the data to conserve memory
        """
        self.dataframes_dict =  dataframes_dict
        self.control_dataframes_dict = control_dataframes_dict
        self.control_covariates_df = control_covariates_df
        self.patient_covariates_df = patient_covariates_df
        self.use_intercept = use_intercept
        self.mask = mask
    
    def threshold_probabilities(self, df: pd.DataFrame, threshold: float=0.2) -> pd.DataFrame:
        """
        This will mask the raw probabilities. 
        Generally, VBM probabilities under 0.2 are masked out.
        
        Will then find all voxels which are nonzero across all dataframes and create a mask from them. 
        Will then return the masked dataframe and the mask for computational speed.
        """
        df = df.where(df > threshold, 0)
        nonzero_mask = df.sum(axis=1) > 0
        return df, nonzero_mask

    def sort_dataframes(self, voxel_df: pd.DataFrame, covariate_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Will sort the rows of the voxelwise DF and the covariate DF to make sure they are identically organized.
        Then will check that the columns are equivalent. 
        """
        # Force Columns to Match
        voxel_cols = set(voxel_df.columns.astype(str).sort_values().values)
        covariate_cols = set(covariate_df.columns.astype(str).sort_values().values)
        shared_columns = list(voxel_cols.intersection(covariate_cols))
        # This will occur when columns have strange naming, such as subject 1 being 0001 verus 1. 
        if len(shared_columns) == 0:
            voxel_cols = voxel_df.columns.astype(int).astype(str).sort_values().values
            covariate_cols = covariate_df.columns.astype(int).astype(str).sort_values().values
            
            voxel_df.columns = voxel_cols
            covariate_df.columns = covariate_cols
            
            shared_columns = list(set(voxel_cols).intersection(set(covariate_cols)))
            
        return voxel_df.loc[:, shared_columns], covariate_df.loc[:, shared_columns]
    
    def mask_dataframe(self, control_df: pd.DataFrame, patient_df: pd.DataFrame, threshold: float=0.2):
        """
        Simple masking function.
        """
        # Now you can use the function to apply a threshold to patient_df and control_df
        patient_df, _ = self.threshold_probabilities(patient_df, threshold)
        control_df, nonzero_mask = self.threshold_probabilities(control_df, threshold)
        
        whole_mask = control_df.index
        masked_patient_df = patient_df.loc[nonzero_mask, :]
        masked_control_df = control_df.loc[nonzero_mask, :]
        return whole_mask, nonzero_mask, masked_patient_df, masked_control_df
    
    def unmask_dataframe(self, whole_mask: pd.Index, nonzero_mask:pd.Index, patient_df: pd.DataFrame, patient_w_scores:pd.DataFrame):
        """
        Simple unmasking function.
        """
        unmasked_w_score = pd.DataFrame(index=whole_mask, columns=patient_df.columns, data=0)
        unmasked_w_score.loc[nonzero_mask, :] = patient_w_scores.loc[nonzero_mask, :]
        return unmasked_w_score
    
    def calculate_w_scores_vectorized(self, control_df: pd.DataFrame, patient_df: pd.DataFrame, debug: bool=True) -> pd.DataFrame:
        """
        Calculate voxelwise W-scores in a vectorized manner using linear regression.
        This applies a single linear regression across the entire dataset, resulting in inherent smoothing. 
        It is STRONGLY advised to set mask=True when running this.

        This function performs a linear regression using sklearn's LinearRegression, fitted on control data
        and applied to both control and patient data. The regression is done once across all voxels simultaneously,
        treating each voxel's values across subjects as independent responses. This vectorized approach
        efficiently handles the calculations by leveraging matrix operations, which are computationally
        optimized in libraries like numpy and sklearn.

        Args:
            control_df (pd.DataFrame): DataFrame where each column represents a control subject,
                                    and each row represents flattened image data for a voxel.
            patient_df (pd.DataFrame): DataFrame where each column represents a patient,
                                    and each row represents flattened image data for a voxel.
            debu (bool): if true, prints out summary metrics

        Returns:
            pd.DataFrame: A DataFrame of the same shape as patient_df, containing the W-scores for each voxel.

        Explanation of Process:
            1. **Setup Response Variables:** Both control and patient data are transposed to shape subjects as rows
            and voxels as columns, facilitating simultaneous regression across all voxels.
            2. **Design Matrix:** A common design matrix is created from sorted_control_covariate_df, which is used
            for fitting the model. This matrix contains the predictors (covariates) for each subject.
            3. **Fit the Model:** The LinearRegression model is fitted using the control data. The model is designed
            to handle multiple response variables (voxels) simultaneously, which are treated as independent.
            This method ensures that the relationship modeled in the voxelwise approach is maintained across
            all voxels simultaneously, mirroring the structure where each voxel is analyzed independently but more efficiently.
            4. **Prediction and Error Calculation:** The model predicts both control and patient data. Residuals
            are computed for control predictions to determine the variability unexplained by the model across voxels.
            5. **Compute W-scores:** W-scores are calculated by dividing the prediction errors (patient data minus
            their predictions) by the voxel-specific residual standard deviation, providing a normalized measure
            of deviation for each patient voxel relative to the control model.
        """
        # Optional masking for memory consveration
        if self.mask:
            whole_mask, nonzero_mask, patient_df, control_df = self.mask_dataframe(control_df, patient_df)
        
        # Design matrix X for control group, outcomes Y for control group
        X_control = self.sorted_control_covariate_df.T 
        Y_control = control_df.T.values
        # Fit model on control data across all voxels
        control_model = LinearRegression(fit_intercept=self.use_intercept)
        control_model.fit(X_control, Y_control)

        # Design matrix X for experimental group, outcomes Y for experimental group
        X_patient = self.sorted_patient_covariate_df.T
        Y_patient = patient_df.T.values
        
        # Predict on experimental group and calculate errors
        PREDICTION = control_model.predict(X_patient)
        RESIDUALS = Y_patient - PREDICTION
        RSS = np.sum(RESIDUALS**2, axis=0)
        DF = Y_patient.shape[0] - X_patient.shape[1] - int(self.use_intercept)
        RSD = np.sqrt(RSS / DF)

        # Compute W-scores for patient data
        w_scores = RESIDUALS / RSD
        if debug:
            print(X_patient.shape, Y_patient.shape, PREDICTION.shape, RESIDUALS.shape, RSS.shape, RSD.shape, w_scores.shape)

        # Reshape W-scores to DataFrame format
        w_scores_df = pd.DataFrame(w_scores.T, index=control_df.index, columns=patient_df.columns)
        if self.mask:
            w_scores_df = self.unmask_dataframe(whole_mask, nonzero_mask, patient_df, w_scores_df)
        
        return w_scores_df


    def calculate_w_scores(self, control_df: pd.DataFrame, patient_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to calculate voxelwise W-map.
        1) This will first perform a regression on the control voxel to identify the standard deviation of the error. 
        2) Then this will use the first model to predict the value of the experimental voxels.
        3) Then this will divide the prediction error by the residual standard deviation, giving the W-score.
        4) This will then iterate over all voxels until a W-map is complete. 
        
        This is a slow function, but it is easy to code. 
        
        Args: 
        control_df (pd.DataFrame): DataFrame where each column represents a control subject, 
                                and each row represents flattened image data for a voxel.
        patient_df (pd.DataFrame): DataFrame where each column represents a patient, 
                                and each row represents flattened image data for a voxel.
                                
        Note:
        The covariates_df MUST have the same subject ID in for column names as the dataframe with the voxels
        """
        if self.mask:
            whole_mask, nonzero_mask, patient_df, control_df = self.mask_dataframe(control_df, patient_df)
        patient_w_scores = pd.DataFrame(index=patient_df.index, columns=patient_df.columns)

        for voxel in tqdm(control_df.index, desc='Fitting voxelwise model'):
            ## CONTROL FIT
            # Set predictors to shape (samples, regressors)
            X_control = self.sorted_control_covariate_df.T  

            # Set observations to shape (samples, 1)
            y_control = control_df.loc[voxel, :].values.reshape(-1, 1)

            # Fit linear regression to control data
            control_model = LinearRegression(fit_intercept=self.use_intercept)
            control_model.fit(X=X_control, y=y_control)
            
            
            ## EXPERIMENTAL FIT
            # Predict on patient data
            X_patient = self.sorted_patient_covariate_df.T  # Transpose to match orientation as above.
            Y_patient = patient_df.loc[voxel, :].values.reshape(-1, 1)
            Yi_patient = control_model.predict(X_patient)
            
            # Derive Mean Squared Error 
            RESIDUALS = Y_patient - Yi_patient
            SSE = np.sum(RESIDUALS**2)
            
            # Derive Adjusted Degrees of Freedom (DF = n - p)
            DF = Y_patient.shape[0] - X_patient.shape[1] - int(self.use_intercept)
            
            # Derive Residual Standard Deviation (Root(Sum Squared Errors / Adjusted Degrees of Freedom)) AKA Root(MSE)
            RSD = np.sqrt(SSE/DF)
            
            # Calculate W-scores
            patient_w_scores.loc[voxel, :] = RESIDUALS.flatten() / RSD
            
        # Unmask W-scores
        if self.mask:
            patient_w_scores = self.unmask_dataframe(whole_mask, nonzero_mask, patient_df, patient_w_scores)
        return patient_w_scores
            
    def process_atrophy_dataframes(self, dataframes_dict, control_dataframes_dict, vectorize=True):
        """
        Processes the provided dataframes to calculate z-scores and determine significant atrophy.

        Parameters:
        - dataframes_dict (dict): Dictionary containing patient dataframes.
        - control_dataframes_dict (dict): Dictionary containing control dataframes.
        - vector (bool): If set to false, will consider the statistical distribution of each voxel independently.
            If set to true, will 

        Returns:
        - tuple: A tuple containing two dictionaries - atrophy_dataframes_dict and significant_atrophy_dataframes_dict.
        """
        
        atrophy_dataframes_dict = {}
        significant_atrophy_dataframes_dict = {}

        for k in dataframes_dict.keys():
            # Make sure the covariates line up with the voxels
            self.sorted_control_voxel_df, self.sorted_control_covariate_df = self.sort_dataframes(control_dataframes_dict[k], self.control_covariates_df)
            self.sorted_patient_voxel_df, self.sorted_patient_covariate_df = self.sort_dataframes(dataframes_dict[k], self.patient_covariates_df)
            
            # Submit
            if vectorize:
                atrophy_dataframes_dict[k] = self.calculate_w_scores_vectorized(control_df=self.sorted_control_voxel_df, patient_df=self.sorted_patient_voxel_df)
            else:
                atrophy_dataframes_dict[k] = self.calculate_w_scores(control_df=self.sorted_control_voxel_df, patient_df=self.sorted_patient_voxel_df)
            
            # Threshold
            if k == 'cerebrospinal_fluid':
                significant_atrophy_dataframes_dict[k] = atrophy_dataframes_dict[k].where(atrophy_dataframes_dict[k] > 2, 0)
            else:
                significant_atrophy_dataframes_dict[k] = atrophy_dataframes_dict[k].where(atrophy_dataframes_dict[k] < -2, 0)
            print('Dataframe: ', k)
            display(dataframes_dict[k])
            print('------------- \n')
        
        return atrophy_dataframes_dict, significant_atrophy_dataframes_dict
    
    def run(self):
        """
        Orchestration method. 
        """
        atrophy_dataframes_dict, significant_atrophy_dataframes_dict = self.process_atrophy_dataframes(self.dataframes_dict, self.control_dataframes_dict)
        return atrophy_dataframes_dict, significant_atrophy_dataframes_dict