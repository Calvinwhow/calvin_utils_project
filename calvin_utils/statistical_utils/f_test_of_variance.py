import pandas as pd
import numpy as np
from scipy.stats import f

def calculate_variance(data, reference):
    if reference is None:
        reference = np.mean(data)
    deviance = data - reference
    variance = np.sum(deviance**2) / (len(data)-1)
    return variance

def calculate_pooled_variance(data, reference):
    pooled_variance = 0
    pooled_weight = 0
    for cohort in data:
        if reference is None: reference = np.mean(cohort)
        variance = calculate_variance(cohort, reference)
        weight = len(cohort) #since len iterates from 0, this is technically == n-1
        
        pooled_variance = pooled_variance + variance*weight
        pooled_weight = pooled_weight + weight
    return pooled_variance/pooled_weight

def prepare_data(categories, iv, dv, df, categorical_col):
    if len(categories) != 2:
        raise ValueError("The IV column must have exactly 2 unique groups.")
    
    group1, group2 = categories
    print("Checking ", categories)
    if categorical_col is None:
        return df[df[iv] == group1][dv], df[df[iv] == group2][dv]
    else:
        # Initialize containers for d1 and d2
        d1_vals = []
        d2_vals = []
        
        # Iterate over each category and extract data from each group
        for category in df[categorical_col].unique():
            group1_category_data = df[(df[iv] == group1) & (df[categorical_col] == category)][dv]
            group2_category_data = df[(df[iv] == group2) & (df[categorical_col] == category)][dv]
            
            # Append data to the respective lists
            d1_vals.append(group1_category_data)
            d2_vals.append(group2_category_data)
        
        return d1_vals, d2_vals

def f_test_of_variance(v1, v2):
    return v1 / v2

def p_value_of_f_stat(d1, d2, f_stat):
    df1 = len(d1) - 1
    df2 = len(d2) - 1
    return f.sf(f_stat, df1, df2)
    
def orchestrate_f_test(df: pd.DataFrame, dv_col: str, iv_col: str, categorical_col: str, comparison_value=None):
    """
    Calculate an F-test comparing the variances of two groups split by the IV column, 
    where deviance from a comparison value (or group mean) is measured.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    dv_col : str
        The dependent variable (DV) column.
    iv_col : str
        The independent variable (IV) column used for grouping.
    comparison_value : float, optional
        The value to compare deviations from. If None, the group mean will be used.
    categorical_col: str
        cohort columns

    Returns
    -------
    None
        Prints the F-statistic and the corresponding p-value.
    """
    # Split the dataframe into two groups based on unique values of the IV column
    unique_groups = df[iv_col].unique()
    d1, d2 = prepare_data(categories=unique_groups, iv=iv_col, dv=dv_col, df=df, categorical_col=categorical_col)
    if categorical_col is None:
        v1 = calculate_variance(data=d1, reference=comparison_value)
        v2 = calculate_variance(data=d2, reference=comparison_value)
    else:
        v1 = calculate_pooled_variance(data=d1, reference=comparison_value)
        v2 = calculate_pooled_variance(data=d2, reference=comparison_value)
    
    f = f_test_of_variance(v1,v2)
    p = p_value_of_f_stat(d1,d2,f)

    # Print results
    print(f"F-statistic: {f}")
    print(f"p-value: {p}")