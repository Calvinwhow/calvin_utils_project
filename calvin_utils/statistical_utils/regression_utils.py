import statsmodels.formula.api as smf
import statsmodels.api as sm

class RegressOutCovariates():
    """
    Will regress on values and return residuals. Will add the residuals to a dataframe as <name>_residual and return the DF
    """
    @staticmethod
    def generate_formula(dependent_variable_list, covariates_list, intercept, verbose=True):
        formula_dict = {}
        for dep_var in dependent_variable_list:
            formula = dep_var
            for covariate in covariates_list:
                if covariate == covariates_list[0]:
                    formula += f" ~ {covariate}"
                else:
                    formula += f" + {covariate}"
            if intercept:
                pass
            else:
                 formula += f" - 1"
            formula_dict[dep_var] = formula
            print(f"\n Formula for {dep_var}: \n", formula) if verbose else None
        return formula_dict
    
    @staticmethod
    def regress_out_covariates(df, formula_dict):
        adjusted_indep_vars_list = []
        for indep_var, formula in formula_dict.items():
            fitted_model = smf.ols(data=df, formula=formula).fit()
            residuals = fitted_model.resid
            df[f"{indep_var}_residual"] = residuals
            adjusted_indep_vars_list.append(f"{indep_var}_residual")
        return df, adjusted_indep_vars_list
    
    @staticmethod
    def regress_out_covariates_using_endog_exog(endog, exog, add_intercept=True):
        if add_intercept: 
            exog = sm.add_constant(exog)
        fitted_model = sm.OLS(endog=endog, exog=exog).fit()
        return fitted_model.resid
    
    @staticmethod
    def run(df, dependent_variable_list, covariates_list, intercept=True):
        """
        Params:
        
        df: pandas DF containing your covariates and independent variables
        dependent_variable_list: a list of dependent variables as found in the dataframe columns. 
        covariates_list: a list of covariates as found in the dataframe columns. 
        """
        formula_dict = RegressOutCovariates.generate_formula(dependent_variable_list, covariates_list, intercept)
        df, adjusted_indep_vars_list = RegressOutCovariates.regress_out_covariates(df, formula_dict)
        return df, adjusted_indep_vars_list
    
def convert_to_ordinal(data_df, columns):
    """
    Convert unique values in specified columns of a DataFrame to ordinal values and print the mapping.

    Parameters:
    - data_df (pd.DataFrame): DataFrame containing the data to be converted.
    - columns (list): List of column names to be converted to ordinal values.

    Returns:
    - ordinal_df (pd.DataFrame): DataFrame with specified columns converted to ordinal values.
    - mapping_dict (dict): Dictionary showing the mapping of original values to ordinal values for each column.
    """
    ordinal_df = data_df.copy()
    mapping_dict = {}

    for column in columns:
        if column in ordinal_df.columns:
            unique_values = ordinal_df[column].unique()
            unique_values.sort()  # Ensure the values are sorted before assigning ordinals
            mapping_dict[column] = {value: idx for idx, value in enumerate(unique_values)}
            ordinal_df[column] = ordinal_df[column].map(mapping_dict[column])

    print("Mapping of unique values to ordinal values:")
    for column, mapping in mapping_dict.items():
        print(f"{column}: {mapping}")

    return ordinal_df