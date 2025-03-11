import numpy as np
from scipy.stats import spearmanr
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from tqdm import tqdm

from time import time

class NiftiOptimizer:
    def __init__(self, corr_map_dict, data_loader, load_in_time=True):
        self.data_loader = data_loader
        self.corr_map_names = [k for k in corr_map_dict.keys()]
        self.corr_map_dict = self._handle_nans(corr_map_dict)
        self.W = self._initialize_weights(weighted=False)
        self.MAPS = self._initialize_maps(self.corr_map_dict)
        
        self.datasets = self._init_data(load_in_time)
        self._readout()
        self.converged = False
        self.adam = AdamOptimizer(self.W, lr=0.1)
        self.convergence_monitor = ConvergenceMonitor(max_i=500)

    def _handle_nans(self, corr_map_dict):
        print("Removing NaNS from self.coor_map_dict.")
        for dataset, matrix in corr_map_dict.items():
            corr_map_dict[dataset] = CorrelationCalculator._check_for_nans(matrix, nanpolicy='remove', verbose=False)
        return corr_map_dict
    
    def _initialize_maps(self, corr_map_dict):
        '''Returns the niftis as a flattened array. Shape (N, Voxels)'''
        n_obs = self.W.shape[1]
        n_vox = len(corr_map_dict[self.corr_map_names[0]].flatten())
        M = np.zeros((n_obs, n_vox))
        for i, k in enumerate(corr_map_dict.keys()):
            M[i,:] = corr_map_dict[k]
        return M

    def _initialize_weights(self, weighted=False):
        '''Initializes weights in proportion to sample size. Shape (1, N)'''
        weights = []
        for k in self.corr_map_names:
            if weighted:
                data = self.data_loader.load_dataset(k)
                weights.append(data['niftis'].shape[0])
            else:
                weights.append(np.random.normal(1,.2)) #provide near-equal weighting using RNG
        weights = np.array(weights)
        weights = np.reshape(weights, (1, len(weights)))
        return weights / np.sum(weights)
    
    def _init_data(self, load_in_time=False):
        if load_in_time:
            return None
        
        datasets = {}
        for k in self.data_loader.dataset_names_list:
            datasets[k] = self.data_loader.load_dataset(k)
            datasets[k]['niftis'] = CorrelationCalculator._check_for_nans(datasets[k]['niftis'], nanpolicy='remove', verbose=False)
            datasets[k]['indep_var'] = CorrelationCalculator._check_for_nans(datasets[k]['indep_var'], nanpolicy='remove', verbose=False)
        return datasets
        
    def _readout(self):
        print("===Imaging Optimizer Initialized===")
        print(f"Initializing: \n\t Weights: {self.W.shape}  \n\t Training Maps: {self.MAPS.shape}")
        print("===        Training Data        ===")
        for k in self.corr_map_dict.keys():
            print(f"\t Training Dataset (For weighted average): {k}")
        print("===         Testing Data        ===")
        for k in self.data_loader.dataset_names_list:
            print(f"\t Testing Dataset (To predict outcomes using weighted average): {k}")
    
    def _converge_maps(self, W=None, verbose=False):
        '''
        Calculate the convergent map. 
        Allow weights to be passed as an argument (for perturbation).
        If no weight argument, will use the weights object (for optimization)
        '''
        if W is None:           
            W = self.W
        convergent_map = W @ self.MAPS
        convergent_map = CorrelationCalculator._check_for_nans(convergent_map, nanpolicy='remove', verbose=False)
        return convergent_map
    
    def _cosine_similarity(self, A, B):
        """Calculate the cosine similarity between two vectors."""
        A = A.flatten()
        B = B.flatten()
        NUMERATOR = np.dot(A, B)
        DENOMINATOR = np.sqrt(np.sum(A**2)) * np.sqrt(np.sum(B**2))
        return NUMERATOR/DENOMINATOR
    
    def _broadcast_cosine_similarity(self, patient_maps, convergent_map):
        '''broadcast matmul of patient maps against convergent map'''
        NUM = np.matmul(patient_maps,convergent_map.T)    #shape (n_obs,  ) <-  (n_obs, n_vox) @ (n_vox, 1)
        A = np.sqrt(np.sum(convergent_map**2, axis=1))    #shape (n_obs,  )   <-  (1, n_vox) summed over n_vox
        B = np.sqrt(np.sum(patient_maps**2,   axis=1))     #shape (n_obs,  )   <-  (n_obs, n_vox) summed over n_vox
        DEN = A*B
        CS =  NUM.flatten() / DEN
        return CS
    
    def _calculate_similarity(self, patient_maps, convergent_map):
        """Orchestrate similarity calculation"""
        return self._broadcast_cosine_similarity(patient_maps, convergent_map)
    
    def _rho_function(self, AVG_MAP, verbose=False):
        '''
        Step 1 - Get similarity of each patient map to convergent map
        Step 2 - Correlate those similarities to outcomes.
        Step 3 - Repeat for each dataset
        '''
        RHO_LIST = []
        for k in self.data_loader.dataset_names_list:
            # Step 1 - Load data in 
            if self.datasets is None:
                dataset = self.data_loader.load_dataset(k)
                niftis = CorrelationCalculator._check_for_nans(dataset['niftis'], nanpolicy='remove', verbose=False)
                Y = CorrelationCalculator._check_for_nans(dataset['indep_var'], nanpolicy='remove', verbose=False)
            else:
                niftis = self.datasets[k]['niftis']
                Y = self.datasets[k]['indep_var']   
                
            # Step 2 - Correlate similarity to Y variable
            X = self._calculate_similarity(niftis, AVG_MAP)            
            RHO, _ = spearmanr(X, Y)
            RHO_LIST.append(RHO)
        if verbose: print(f"RHO_LIST): {RHO_LIST}")
        return np.array(RHO_LIST)
    
    def _target_function(self, RHO_ARR):
        """Measure the Root Mean Rho. Like RMAE"""
        mean_abs_rho = np.mean(np.abs(RHO_ARR))
        return np.sqrt(mean_abs_rho)

    def _penalty_all_weights(self,threshold=0.33, downscale=1000):
        '''Block all weights from going over threshold'''
        p = np.sum( 1e-6 / (threshold - np.abs(self.W)) )
        return p / downscale

    def _pentalty_per_weight(self, threshold=0.00, penalty_factor=0.005):
        '''Gently penalize any nonzero weight'''
        mask = np.abs(self.W) > threshold
        contacts_over = np.abs(self.W) * mask
        return np.sum(contacts_over * penalty_factor)
    
    def _loss_function(self, RHO_ARR, verbose=True):
        '''L = T'''
        T = self._target_function(RHO_ARR)
        P1 = self._penalty_all_weights()
        P2 = self._pentalty_per_weight()
        return T - P1 #- P2

    def _orchestrate_loss(self, W=None, verbose=False):
        # step 1 - generate convergent map  
        AVG_MAP = self._converge_maps(W, verbose)      
        AVG_MAP = CorrelationCalculator._check_for_nans(AVG_MAP, nanpolicy='stop', verbose=False)        
        # step 2 - measure how well similarity relates to independent variable
        RHO_ARR = self._rho_function(AVG_MAP, verbose)
        # step 3 - get Loss
        LOSS = self._loss_function(RHO_ARR, verbose)
        return LOSS
        
    def _partial_difference_quotient(self, LOSS, h, W_H):
        '''Runs forward loss Partial Difference Quotient'''
        LOSS_FWD = self._orchestrate_loss(W=W_H)
        return (LOSS_FWD - LOSS) / h
    
    def _clip_gradient(self, GRADIENT):
        return np.clip(GRADIENT, a_min=-0.05, a_max=0.05)
        
    def _orchestrate_gradient(self, LOSS, h):
        '''For each weight, perturb it and get the loss.'''
        GRADIENT = np.zeros((self.W.shape[0], self.W.shape[1]))           # shape (1, n_vars)
        H_MATRIX = np.eye(self.W.shape[1], self.W.shape[1])               # shape (n_vars, n_vars)
        for i in range(self.W.shape[1]):
            W_H = self.W + H_MATRIX[i, :]                                 # shape (1, n_vars)
            GRADIENT[0, i] = self._partial_difference_quotient(LOSS, h, W_H)
        
        GRADIENT = self._clip_gradient(GRADIENT)
        return GRADIENT
    
    def optimize(self, h=0.001, verbose=False):
        '''
        Gets loss
        Gets gradient
        Updates params
        Checks convergence
        Does it all over again
        Returns the final map
        '''
        LOSS = 0
        while tqdm(self.converged==False, desc=f'Loss: {LOSS}, {self.convergence_monitor.iterations}/{self.convergence_monitor.max_iterations}'):
            LOSS = self._orchestrate_loss(verbose=verbose)
            GRADIENT = self._orchestrate_gradient(LOSS=LOSS, h=h)
            self.W = self.adam.step(gradient=GRADIENT)
            self.W = self.W / np.sum(self.W)                # normalize weights back to -1 1
            self.converged = self.convergence_monitor.check_convergence(weights=self.W, gradient=GRADIENT, loss=LOSS)
        return self._converge_maps()

class AdamOptimizer:
    def __init__(self, weights, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        '''
        weights - for optimization
        lr - learning rate
        beta1 - moment bias scaling factor
        beta2 - variance bias scaling factor
        epsilon - negligible small number to avoid div by zero
        m - first moment vector (mean of gradient)
        v - second moment vector (variance of gradient)
        t - time step
        '''
        self.weights = weights
        self.lr = lr
        self.beta1 = beta1
        self.beta2  = beta2
        self.epsilon = epsilon
        self.m = np.zeros(weights.shape)
        self.v = np.zeros(weights.shape)
        self.t = 0
        
    def _update(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1-self.beta1) * gradient            # momentum
        self.v = self.beta2 * self.v + (1-self.beta2) * (gradient**2)       # variance
        m_hat = self.m / (1 - self.beta1 ** self.t)                         # bias corrected momentum (for time step)
        v_hat = self.v / (1 - self.beta2 ** self.t)                         # bias corrected variance (for time step)
        self.weights += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)   # update that weights 
        
    def step(self, gradient):
        self._update(gradient)
        return self.weights
        
class ConvergenceMonitor:
    def __init__(self, tol=1e-5, max_i=500, grad_tol=1e-6, param_tol=1e-6, loss_tol=1e-6, plateau_window=20):
        '''
        tol - general convergence tolerance
        max_i - max number iterations allowed
        grad_tol - threshold for gradient norm
        param_tol - threshold for change in params
        loss_tol - threshold for change in loss
        plateau_window - window to check for change in loss
        '''
        self.tol = tol
        self.max_iterations = max_i 
        self.grad_tol = grad_tol
        self.param_tol=param_tol
        self.loss_tol = loss_tol
        self.plateau_window = plateau_window
        self.iterations = 0
        self.prev_weights = None
        self.prev_loss = None
        self.l1_history = []
        self.loss_history = []
        
    def check_convergence(self, weights, gradient, loss=None):
        self.iterations += 1
        self.prev_weights = np.copy(weights)
        self.prev_loss = loss
        self.loss_history.append(loss)
        
        if self.iterations < 50:
            return False
        
        # Check iterations
        if self.iterations > self.max_iterations:
            print(f"Max iterations reached: {self.max_iterations}")
            return True
        
        # Check L1 Norm
        l1_norm = np.linalg.norm(gradient)
        if l1_norm < self.grad_tol:
            print(f"Converged. L1 norm below thresold: {l1_norm} < {self.grad_tol}")
            return True
        
        # Check Gradient Plateau
        self.l1_history.append(l1_norm)
        if len(self.l1_history) > self.plateau_window:
            recent_l1 = np.array(self.l1_history[-self.plateau_window:])
            delta = np.max(recent_l1) - np.min(recent_l1)
            if delta < self.grad_tol:
                print(f"Converged. L1 norm has plateaud: {delta} < {self.grad_tol}")
                return True
            
        # if self.prev_weights is not None:    
        #     param_change = np.max(np.abs(weights - self.prev_weights))
        #     if param_change < self.param_tol:
        #         print(f"Converged. Weight change thresold: {param_change} < {self.param_tol}")
        
        # # Check loss change
        # if self.prev_loss is not None: 
        #     loss_change = np.abs(loss - self.prev_loss)
        #     if loss_change < self.loss_tol:
        #         print(f"Converged. Loss change thresold: {loss_change} < {self.loss_tol}")
        #         return True 
        
        # check plateau in loss 
        if len(self.loss_history) > self.plateau_window:
            recent_losses = np.array(self.loss_history[-self.plateau_window:])
            delta = np.max(recent_losses) - np.min(recent_losses)
            if delta < self.loss_tol:
                print("Converged. Loss stagnated.")
                return True 
        
        return False