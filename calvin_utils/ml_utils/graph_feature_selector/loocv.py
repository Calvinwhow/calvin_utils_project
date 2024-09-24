import itertools
import patsy
import pandas as pd
import numpy as np
import sys
import os 
import contextlib 
from tqdm import tqdm
import random

import warnings
warnings.filterwarnings('ignore', message="A NumPy version >=")

@contextlib.contextmanager
def suppress_stdout():
    """
    A context manager to suppress stdout temporarily.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
            
def run_loocv_for_formula(formula, data_df, model_type, **kwargs):
    """
    Performs LOOCV for a single formula and returns the performance metrics.
    This is called by the class LOOCVManager for submitting multiprocessed LOOCVs.
    
    Parameters:
    -----------
    formula : str
        The formula string to be evaluated.
    data_df : pd.DataFrame
        The dataframe with all data in it.
    model_type : str
        The type of model submitted
    kwargs : dict
        Additional kwargs, generally expecting {alpha: __, gamma:___} for a KRR.
    
    Returns:
    --------
    dict
        A dictionary containing the formula and its performance metrics.
    """
    # Generate Design Matrices
    try:
        design_matrix_handler = DesignMatrixHandler()
        Y, X = design_matrix_handler.define_design_matrix_df(formula, data_df)
        Y = design_matrix_handler.Y_preprocess(Y)
        
        # Calculate LOOCV Metrics
        loocv = LOOCVOrchestrator(Y=Y, X=X, model_type=model_type, **kwargs)
        metrics = loocv.calculate_performance()
        metrics['Formula'] = formula
    except Exception as e:
        print(f"Error caught: {e}")
        metrics = {
            'R2': np.nan,
            'RMSE': np.nan,
            'Entropy': np.nan,
            'Average_AIC': np.nan,
            'Formula': formula
        }
    return metrics

class FormulaGenerator:
    """
    A class to generate all possible regression formulas based on given predictor variables and optional interactions.

    Methods:
    --------
    generate_all_formulas(outcome_var, predictors, max_interaction_level=1):
        Generates all possible formulas with optional interactions up to a specified level.
    """

    def __init__(self, outcome_var, predictors, max_interaction_level=3):
        """
        Initializes the FormulaGenerator with the outcome variable, predictors, and interaction level.

        Parameters:
        -----------
        outcome_var : str
            The name of the dependent (outcome) variable.
        predictors : list of str
            List of predictor variables to be used in formula generation.
        max_interaction_level : int, optional, default=1
            Maximum level of interaction terms (e.g., 2 for two-way, 3 for three-way interactions).
        """
        self.outcome_var = outcome_var
        self.predictors = predictors
        self.max_interaction_level = max_interaction_level
        
    @staticmethod
    def generate_all_formulas(predictors, dep_var, max_predictors_in_model=3, max_interaction_level=2):
        """
        Limits number of involved predictors, then limits their combination into interactions
        Returns formulas, a list of strings which are composed of strings: Y ~ v1 + v2 + v1*v2
        """
        formulas = []

        # Generate all combinations of predictors of size max_predictors_in_model
        for predictor_group in tqdm(itertools.combinations(predictors, max_predictors_in_model), desc='Formula generation'):            
            # Initialize the formula with simple effects
            formula = dep_var + ' ~ '
            formula += ' + '.join(predictor_group)
            
            # Add interaction terms based on max_interaction_level
            for interaction_group in itertools.combinations(predictor_group, max_interaction_level):
                interaction_term = '*'.join(interaction_group)
                formula += ' + ' + interaction_term
                formulas.append(formula)  # Store each formula
        return formulas



class DesignMatrixHandler:
    """
    A class to build design matrices using patsy based on a given formula.

    Methods:
    --------
    define_design_matrix(formula, data_df, voxelwise_variable=None):
        Defines the design matrix based on the patsy formula and returns it as a DataFrame.
    """

    def __init__(self):
        pass

    @staticmethod
    def define_design_matrix_df(formula, data_df, voxelwise_variable=None):
        """
        Defines the design matrix based on the patsy formula and returns it as a DataFrame.

        Parameters:
        -----------
        formula : str
            The patsy formula to construct the design matrix.
        data_df : pd.DataFrame
            The DataFrame from which to construct the design matrix.
        voxelwise_variable : str, optional
            The column in data_df with paths to the voxelwise regressor files (e.g., Nifti files).

        Returns:
        --------
        tuple of pd.DataFrame
            Tuple containing the design matrix for the dependent variable and the design matrix for the independent variables.
        """
        if voxelwise_variable is not None:
            vars = patsy.ModelDesc.from_formula(formula)
            # Remove voxelwise_variable from the formula
            vars.rhs_termlist = [term for term in vars.rhs_termlist 
                                 if voxelwise_variable not in str(term)]
            y, X = patsy.build_design_matrices([vars.lhs_termlist, vars.rhs_termlist],
                                               data_df, return_type='dataframe')
            # Add voxelwise_variable as a separate column
            X[voxelwise_variable] = data_df[voxelwise_variable]
        else:
            y, X = patsy.dmatrices(formula, data_df, return_type='dataframe')
        return y.to_numpy(), X.to_numpy()
    
    @staticmethod
    def Y_preprocess(Y):
        if Y.shape[1]==2:
            return Y[:,0]
        else:
            return Y
  

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, log_loss
import statsmodels.api as sm
from statsmodels.base.model import LikelihoodModelResults
from scipy.special import softmax
class StatisticalModeler:
    """
    A class to fit various statistical models and evaluate them using appropriate metrics.
    
    Attributes:
    -----------
    y : pd.DataFrame or np.ndarray
        The dependent variable matrix (response variable).
    X : pd.DataFrame or np.ndarray
        The independent variable matrix (predictors).
    model_type : str
        The type of model to be fitted. Options are 'ols', 'krr', 'binomial_logit', 'multinomial_logit'.
    
    Methods:
    --------
    fit_model():
        Fits the specified model type.
    predict(X):
        Predicts using the fitted model.
    calculate_r_squared():
        Calculates R-squared for OLS or Kernel Ridge Regression.
    calculate_pseudo_r_squared():
        Calculates pseudo R-squared for logit models.
    calculate_rmse():
        Calculates the Root Mean Squared Error (RMSE).
    calculate_cross_entropy():
        Calculates cross-entropy loss for logit models.
    calculate_aic():
        Calculates the Akaike Information Criterion (AIC).
    """

    def __init__(self, y, X, model_type='ols', **kwargs):
        """
        Initializes the StatisticalModeler with the given design matrices and model type.

        Parameters:
        -----------
        y : pd.DataFrame or np.ndarray
            The dependent variable matrix (response variable).
        X : pd.DataFrame or np.ndarray
            The independent variable matrix (predictors).
        model_type : str, optional, default='ols'
            The type of model to be fitted. Options are 'ols', 'krr', 'binomial_logit', 'multinomial_logit'.
        kwargs : dict
            Additional parameters for the specific model type.
            This is expected for kernel ridge regression, in the form: kwargs = {'gamma': 0.1, 'alpha':0.1}
        """
        self.y = y
        self.X = X
        self.model_type = model_type
        self.model = None
        self.kwargs = kwargs

    def fit_model(self, Y, X):
        """
        Fits the specified model type based on y and X.

        Raises:
        -------
        ValueError if the model type is not supported.
        """
        if self.model_type == 'ols':
            self.model = sm.OLS(Y, X).fit()
        elif self.model_type == 'krr':
            self.model = KernelRidge(**self.kwargs).fit(Y, X)
        elif self.model_type == 'binomial_logit':
            self.model = sm.Logit(Y, X).fit()
        elif self.model_type == 'multinomial_logit':
            self.model = sm.MNLogit(Y, X).fit()
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

    def predict(self, X):
        """
        Predicts using the fitted model.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            The independent variables for which predictions are to be made.

        Returns:
        --------
        np.ndarray
            The predicted values or probabilities.
        """
        if self.model_type in ['ols', 'multinomial_logit']:
            return self.model.predict(X)
        elif self.model_type == 'binomial_logit':
            return self.model.predict(X)
        elif self.model_type == 'krr':
            return self.model.predict(X)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")

    def ols_r2(self, Y, Y_HAT, debug=False):
        SR = np.square( Y - Y_HAT )
        SSR = np.sum(SR, axis = 0)
        
        STO = np.square( Y- np.mean(Y_HAT) )
        SSTO = np.sum(STO, axis = 0)

        R2 = SSR/SSTO
        return R2 
    
    def mcfaddens_r2(self, Y, Y_HAT, debug=False):
        """
            Classification model detected. Employing McFadden's R2.
            This skews towards smaller numbers, with values between 0.2-0.4 being excellent
            See McFadden 1974 - Conditional Logit Analysis of Qualitative Choice Behavior
        """
        if self.model_type == 'binomial_logit':
            L_model = np.sum(( Y * np.log(np.clip(Y_HAT,  1e-15, 1)) ) + ( (1 - Y) * np.log(np.clip(1 - Y_HAT, 1e-15, 1)) ))
            p_null = np.mean(Y)  # Proportion of positives (k/n)
            L_null = np.sum(Y * np.log(p_null) + (1 - Y) * np.log(1 - p_null))
        elif self.model_type == 'multinomial_logit':
            L_model = np.sum(np.log(np.clip(Y_HAT[np.arange(len(Y)), Y], 1e-15, 1)))
            p_null = np.mean(Y, axis=0)  # Proportion of each class, assuming n,k vector of 0s and 1s
            L_null = np.sum(np.log(np.clip(p_null[np.arange(len(Y)), Y], 1e-15, 1)))
        if debug: print("L_model: ", L_model, " | p_null: ", p_null," | L_null: ", L_null)
        # McFadden's R-squared
        return 1 - (L_model / L_null)
        
    def calculate_r_squared(self, Y, Y_HAT, debug=False):
        """
        Calculates R-squared for OLS or Kernel Ridge Regression models.
        R2 = SSR/SSTO = 1 - (SSE/SSTO)
        
        for McFadden's:
        R2 = 1 - (Likelihood model)/(Likelihood null)
        L_model = ∑( (Y*ln(Y_HAT)) + (1 - Y)*ln(1-Y_hat) ), where Y_HAT = probability(class) from model
        L_null = ∑( (Y*ln(Y_HAT)) + (1 - Y)*ln(1-Y_hat) ), where Y_HAT = p(class) = N(classes)/N(total)

        Returns:
        --------
        float
            R-squared value, AKA the coefficient of determination.
        """
        if self.model_type in ['ols', 'krr']:
            R2 = self.ols_r2(Y=Y, Y_HAT=Y_HAT)
        elif self.model_type in ['binomial_logit', 'multinomial_logit']:
            R2 = self.mcfaddens_r2(Y=Y, Y_HAT=Y_HAT)
        else: 
            print(f"R2 not implemented for {self.model_type}")
        return R2

    def calculate_rmse(self, Y, Y_HAT):
       return np.sqrt(mean_squared_error(Y, Y_HAT))

    def binomial_crossentrophy(self, Y, Y_HAT):
        BCE = -(1/Y.shape[0]) * np.sum( ( Y * np.log(np.clip(Y_HAT,  1e-15, 1)) ) + ( (1 - Y) * np.log(np.clip(1 - Y_HAT, 1e-15, 1)) ))
        return BCE
    
    def categorical_crossentropy(self, Y, Y_HAT):
        CCE =  -(1/Y.shape[0]) * np.sum(np.log(np.clip(Y_HAT[np.arange(len(Y)), Y], 1e-15, 1)))
        return CCE
    
    def rmse(self, Y, Y_HAT):
        if self.model_type in ['ols', 'krr']: 
            return self.calculate_rmse(Y, Y_HAT)
        else:
            return None

    def entropy(self, Y, Y_HAT):
        if self.model_type == 'binomial_logit':
            return self.binomial_crossentrophy(Y=Y, Y_HAT=Y_HAT)
        elif self.model_type == 'multinomial_logit':
            return self.categorical_crossentropy(Y=Y, Y_HAT=Y_HAT)
        else:
            return None

    def calculate_aic(self):
        """
        Calculates the Akaike Information Criterion (AIC).
        AIC = 2k - 2ln(L)

        Returns:
        --------
        float
            AIC value.
        """
        if self.model_type in ['ols', 'binomial_logit', 'multinomial_logit']:
            return self.model.aic
        else:
            raise ValueError("AIC is not available for this model type.")

class LOOCVOrchestrator:
    """
    A class to perform Leave-One-Out Cross-Validation (LOOCV) using the StatisticalModeler.
    
    Attributes:
    -----------
    Y : np.ndarray
        The dependent variable vector.
    X : np.ndarray
        The independent variable matrix.
    model_type : str
        The type of model to be fitted. Options are 'ols', 'krr', 'binomial_logit', 'multinomial_logit'.
    kwargs : dict
        Additional parameters for the StatisticalModeler.
    
    Methods:
    --------
    perform_loocv():
        Executes the LOOCV process and returns performance metrics.
    """

    def __init__(self, Y, X, model_type='ols', **kwargs):
        """
        Initializes the LOOCVOrchestrator with the given data and model specifications.

        Parameters:
        -----------
        Y : np.ndarray
            The dependent variable vector.
        X : np.ndarray
            The independent variable matrix.
        model_type : str, optional, default='ols'
            The type of model to be fitted. Options are 'ols', 'krr', 'binomial_logit', 'multinomial_logit'.
        kwargs : dict
            Additional parameters for the StatisticalModeler.
        """
        self.Y = Y
        self.X = X
        self.model_type = model_type
        self.kwargs = kwargs
        
        self.n_samples = Y.shape[0]
        self.Y_HAT_storage = []
        self.AIC_storage = []
        
    def split_data(self, i):
        """Split the data: leave out the i-th sample"""
        Y_train = np.delete(self.Y, i)
        X_train = np.delete(self.X, i, axis=0)
        Y_test = self.Y[i]
        X_test = self.X[i].reshape(1, -1)
        return X_train, Y_train, X_test, Y_test

    def perform_loocv(self):
        """
        Performs Leave-One-Out Cross-Validation.

        Returns:
        --------
        dict
            A dictionary containing R², RMSE, Entropy, and Average AIC.
        """
        for i in range(self.n_samples):
            X_train, Y_train, X_test, Y_test = self.split_data(i)

            # Initialize and fit the model
            with suppress_stdout():
                modeler = StatisticalModeler(y=Y_train, X=X_train, model_type=self.model_type, **self.kwargs)
                modeler.fit_model(Y=Y_train, X=X_train)
                Y_pred = modeler.predict(X_test)
            self.Y_HAT_storage.append(Y_pred[0])  # Assuming Y_pred is a 1D array
            self.AIC_storage.append( modeler.calculate_aic() )
            
    def calculate_performance(self):
        self.perform_loocv()
        # Convert lists to NumPy arrays
        Y_HAT = np.array(self.Y_HAT_storage)
        AIC = np.array(self.AIC_storage)

        # Calculate Metrics using StatisticalModeler
        metric_modeler = StatisticalModeler(y=self.Y, X=self.X, model_type=self.model_type, **self.kwargs)

        # Calculate metrics. This uses calculations from Y_HAT of each LOOCV vs Y
        R2 = metric_modeler.calculate_r_squared(Y=self.Y, Y_HAT=Y_HAT)
        RMSE = metric_modeler.rmse(Y=self.Y, Y_HAT=Y_HAT)
        E = metric_modeler.entropy(Y=self.Y, Y_HAT=Y_HAT)
        AIC = np.mean(AIC)

        # Compile performance metrics
        performance_metrics = {
            'R2': R2,
            'RMSE': RMSE,
            'Entropy': E,
            'Average_AIC': AIC
        }

        return performance_metrics
