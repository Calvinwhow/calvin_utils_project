import json
import numpy as np
from scipy.stats import t
from calvin_utils.ccm_utils.npy_utils import DataLoader

class VoxelwiseRegression:
    def __init__(self, json_path, mask_path=None, out_dir=None):
        self.json_path = json_path
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.data_loader = DataLoader(self.json_path)
        self.design_tensor, self.outcome_tensor, self.contrast_matrix, self.exchangeability_blocks, self.weight_vector = self.load_data()
        self.n_obs, self.n_preds, self.dim3_X, self.dim3_Y, self.n_contrasts, self.n_voxels, self.n_outputs = self.set_variables()
        
    #### Setter/Getter methods ####
    def load_data(self):
        with open(self.json_path, 'r') as f:
            paths = json.load(f)['voxelwise_regression']
        design_tensor = np.load(paths['design_matrix'])                        # shape: (observations, predictors,  voxels)
        outcome_data = np.load(paths['outcome_data'])                          # shape: (observations, regressions, voxels)
        contrast_matrix = np.load(paths['contrast_matrix'])                    # shape: (contrasts, predictors)
        weight_vector = np.load(paths["weights_vector"])                       # shape: (observations, )
        exchangeability_blocks = np.load(paths["exchangeability_block"]) if "exchangeability_block" in paths else None
        return design_tensor, outcome_data, contrast_matrix, exchangeability_blocks, weight_vector
        
    def set_variables(self):
        n_obs,   n_preds,  n_voxels_X = self.design_tensor.shape
        n_obs_y, n_cols_Y, n_voxels_Y = self.outcome_tensor.shape
        n_contrasts, n_preds_cmx      = self.contrast_matrix.shape
        return n_obs, n_preds, n_voxels_X, n_voxels_Y, n_contrasts, max(n_voxels_X, n_voxels_Y), n_cols_Y

    ### REGRESSION HELPERS ###
    def _get_targets(self, permutation):
        """
        Returns the regressor (design tensor) and regressand (outcome data), 
        optionally permuting the outcome data if permutation is True.
        """
        regressor = self.design_tensor  # never permute regressor
        regressand = self.outcome_tensor
        weights = self.weight_vector
        if permutation:
            if self.exchangeability_blocks is None:
                resample_idx = np.random.permutation(regressand.shape[0])
            else:
                block_labels = self.exchangeability_blocks.ravel()
                unique_blocks = np.unique(block_labels)
                resample_idx = []
                for block in unique_blocks:
                    block_indices = np.where(block_labels == block)[0]
                    resample_idx.extend(block_indices)
                resample_idx = np.array(resample_idx)
            regressand = regressand[resample_idx, :, :]
            weights = weights[resample_idx]
        return regressor, regressand, weights  
    
    def _prep_targets(self, regressor, regressand, weights, voxel_idx, regression_idx=0):
        """Ensure shape of X, Y, and W matrices"""
        if regressor.shape[2] == self.n_voxels:          # for voxel-wise design
            X = regressor[:, :, voxel_idx]
        else:                                            # for broadcast design
            X = regressor[:, :, 0]                       

        if regressand.shape[2] == self.n_voxels:         # for voxel-wise outcome
            Y = regressand[:, regression_idx, voxel_idx]
        else:                                            # for multi-output voxel-wise outcome
            Y = regressand[:, regression_idx, 0]
        return X, Y, weights
    
    #### Regression Methods ####
    def get_r2(self, Y, Y_HAT, W, e=1e-6):
        """
        Calculate R-squared value from 1 - SS_Residuals/SS_TotalObservations
        Y is the observed data
        Y_HAT is the predicted data
        """
        Y_mean = np.sum(W * Y)
        SS_residual = np.sum((Y - Y_HAT)**2)
        SS_total = np.sum(W * (Y - Y_mean)**2)
        return 1 - (SS_residual / (SS_total+e))
    
    def apply_contrasts(self, XtX_inv, BETA, MSE, e=1e-6, get_p=False):
        """
        t = (C @ BETA) / sqrt(diag(C @ XtX_inv @ C.T) * MSE)

        'C' is (c, p)
        'BETA' is (p,) or (p, 1)
        'XtX_inv' is (p, p)
        'MSE' is scalar (residual variance)
        """
        C = self.contrast_matrix
        NUM = C @ BETA
        var_diag = np.diag(C @ XtX_inv @ C.T)  
        DEN = np.sqrt(var_diag * MSE)      
        T = NUM / (DEN+e)
        if get_p:
            df = self.n_obs - self.n_preds
            P = 2 * t.sf(np.abs(T), df=df)
        return T

    def _run_regression(self, X, Y, W):
        """
        Runs a standard linear regression. Gets Betas and T values.
            
        Parameters
        ----------
        X : (n_obs, n_preds) design matrix
        Y : (n_obs,) data (single voxel)
        W : (n_obs,) weights  (use w_i = sample_size_i  or any precision proxy)
        """
        wsqrt = np.sqrt(W)                                              # (n_obs,) <-- rooting w array enables its multiple multiplications to remain equivalet to X.T @ W @ X
        Xw = X * wsqrt[:, None]                                         # scale each row
        Yw = Y * wsqrt                                                   # scale each row
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)                             # (preds × preds)
        BETA = XtX_inv @ Xw.T @ Yw
        Y_HAT  = X @ BETA                                               # (obs,) <- (obs, preds) @ (preds,)
        residuals = Y - Y_HAT                                           # (n_sub, n_voxels) <- (n_sub, n_voxels) - (n_sub, n_voxels)
        dof = self.n_obs - self.n_preds                                 # scalar  <- (n_sub, ) - (n_cov, )
        mse = np.sum(W * residuals**2, axis=0) / dof                    # (n_voxels,) <- summed (n_sub, n_voxels) along n_sub
        T = self.apply_contrasts(XtX_inv, BETA, mse)                    # (n_cov, n_voxels) <- (n_cov, n_voxels) / (n_cov, n_voxels)
        R2 = self.get_r2(Y, Y_HAT, W)                                    # (n_voxels,) <- (n_sub, n_voxels) - (n_sub, n_voxels)
        return BETA, T, R2                                              # beta is (predictors,), while T and P are (contrasts,)
    
    def voxelwise_regression(self, permutation=False):
        '''Relies on hat matrix (X'@(X'X)^-1@X')@Y to calculate beta, t-values, and p-values'''
        BETA = np.zeros((self.n_preds, self.n_voxels))
        T = np.zeros((self.n_contrasts, self.n_voxels))
        R2 = np.zeros((1, self.n_voxels))        
        regressor, regressand, weights = self._get_targets(permutation)
        for idx in (range(self.n_voxels) if permutation else tqdm(range(self.n_voxels), desc='Running voxelwise regressions')):
            X, Y, W = self._prep_targets(regressor, regressand, weights, idx)
            BETA[:,idx], T[:,idx], R2[:,idx] = self._run_regression(X, Y, W)
        return BETA, T, R2
    
    ### P-VALUE METHODS ###
    def _get_max_stat(self, arr, pseudo_var_smooth=True, t=75):
        """Return the 99.9th percentile of the absolute values in arr. Or just the raw maximum if pseudo_var_smooth is false (this is subject to chaotic noise)."""
        if pseudo_var_smooth:        
            return np.nanpercentile(np.abs(arr), t, axis=1)  # Calculate along rows, ignoring NaNs
        else: 
            return np.nanmax(np.abs(arr), axis=1)  # Calculate along rows

    def run_permutation(self, n_permutations):
        if n_permutations < 1:
            print("No permutations requested.")
            return
        Tp = np.zeros_like(self.T)        
        R2p = np.zeros_like(self.R2) 
        for i in tqdm(range(n_permutations), desc='running permutations'):
            _, permT, permR2 = self.voxelwise_regression(permutation=True)
            max_statsT = self._get_max_stat(permT)
            max_statsR2 = self._get_max_stat(permR2)
            Tp += (max_statsT[:, None] > np.abs(self.T)).astype(int)  #max t is already absval. self.T must be set to absval for a 2-sample t test. 
            R2p += (max_statsR2 > self.R2).astype(int)                #R2 does not need to be absval. It is inherently 1-sided t test.
        self.Tp = Tp / n_permutations 
        self.R2p = R2p / n_permutations
        
    ### Nifti Saving Methods ####
    def _unmask_array(self, data_array):
        """
        Unmasks a vectorized image to full-brain shape using self.mask_path.
        Returns:
            unmasked_array: full-brain NIfTI-like array
            mask_affine: affine transformation from mask
        """
        if self.mask_path is None:
            raise ValueError("Mask path is not provided. Provide the mask used to create the data_array.")
        else:
            mask = nib.load(self.mask_path)
            mask_data = mask.get_fdata()
            mask_indices = mask_data.flatten() > 0  # Assuming mask is binary
            unmasked_array = np.zeros(mask_indices.shape)
            unmasked_array[mask_indices] = data_array.flatten()
        return unmasked_array.reshape(mask_data.shape), mask.affine

    def _save_map(self, map_data, file_name):
        """
        Saves unmasked NIfTI image to disk.
        """
        if self.out_dir is None:
            return
        
        unmasked_map, mask_affine = self._unmask_array(map_data)
        img = nib.Nifti1Image(unmasked_map, affine=mask_affine)
        file_path = os.path.join(self.out_dir, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        nib.save(img, file_path)
        return img
    
    def _save_nifti_maps(self):
        """
        Example method that unmasks and saves NIfTI maps for BETA & T.
        Assumes you have self._unmask_array() and self._save_map() in place.
        """
        if not self.out_dir or not self.mask_path:
            return
        
        # Save betas: shape (n_preds, n_voxels)
        if hasattr(self, 'BETA'):
            for i in range(self.n_preds):
                beta_name = f"beta_predictor_{i}.nii.gz"
                self._save_map(self.BETA[i, :], beta_name)
        # Save T-values for Betas: shape (n_preds, n_voxels)
        if hasattr(self, 'T'):
            for c in range(self.n_contrasts):
                self._save_map(self.T[c, :], f"contrast_{c}_tval.nii.gz")
        # Save overall R2 (measure of model overall model fit)
        if hasattr(self, 'R2'):
            self._save_map(self.R2, f"R2_vals.nii.gz")

        # Save FWE-corrected significance masks if we have permutation results
        if hasattr(self, 'Tp'):
            for c in range(self.n_contrasts):
                sig_mask = (self.Tp[c, :] < 0.05)
                sig_tvals = np.where(sig_mask, self.T[c, :], np.nan)
                self._save_map(sig_tvals, f"contrast_{c}_tval_FWE.nii.gz")
                self._save_map(self.Tp[c, :], f"contrast_{c}_pval_FWE.nii.gz")
                
        # Save FWE-corrected significance masks if we have permutation results
        if hasattr(self, 'R2p'):
            sig_mask = (self.R2p < 0.05)
            sig_r2vals = np.where(sig_mask, self.R2, np.nan)
            self._save_map(sig_r2vals, f"R2_FWE.nii.gz")
            self._save_map(self.R2p, f"R2_pval_FWE.nii.gz")

    #### Public Code ####
    def full_multiout_regression(self):
        n_out = self.outcome_tensor.shape[1]
        results = []
        for j in range(n_out):
            BETA, T, R2 = self.voxelwise_regression(regression_idx=j)
            results.append((BETA, T, R2))
        return results

    def run_all_outputs(self, n_permutations=0):
        """
        Orchestrates full multi-output regression.
        For each output channel in outcome_tensor:
        - Runs regression
        - Runs permutation testing
        - Saves results into a separate subdirectory
        """
        base_out_dir = self.out_dir

        for j in range(self.n_outputs):
            print(f"\nRunning regression for output {j}")
            self.out_dir = os.path.join(base_out_dir, f"regression_{j}")
            os.makedirs(self.out_dir, exist_ok=True)

            # Run one regression for this output
            self.BETA, self.T, self.R2 = self.voxelwise_regression(regression_idx=j)
            self.run_permutation(n_permutations)
            self._save_nifti_maps()

    def run(self, n_permutations=0):
        self.BETA, self.T, self.R2 = self.voxelwise_regression()
        self.run_permutation(n_permutations)
        self._save_nifti_maps()