import pandas as pd
from typing import Tuple
from itertools import combinations

def generate_composite(dataframes_dict: dict, csfgm_only: bool=True, thresholded: bool=True) -> dict:
    """
    Generates composite DataFrames by combining data from multiple DataFrames in the input dictionary.
    For each combination of two or more DataFrames, creates a new DataFrame by summing their values.
    For the 'cerebrospinal_fluid' DataFrame, flips the sign of its values before combining.

    Parameters:
    - dataframes_dict (dict): A dictionary where keys are strings representing the names of DataFrames, 
                              and values are pandas DataFrames.

    Returns:
    - dict: The input dictionary updated with additional keys for each combination of DataFrames. 
            Each new key contains the sum of the combined DataFrames' values.
    """
    for k in dataframes_dict.keys():
        if k == 'cerebrospinal_fluid':
            dataframes_dict[k] = dataframes_dict[k] * -1
    
    if csfgm_only and not thresholded:
        dataframes_dict['csf-plus-gm'] = dataframes_dict['cerebrospinal_fluid'] + dataframes_dict['grey_matter']
    else:
        keys = list(dataframes_dict.keys())
        for r in range(2, len(keys)+1):  # Create combinations for r-tuples where r ranges from 2 to len(keys)
            for key_tuple in combinations(keys, r):
                combined_key = '+'.join(key_tuple)
                combined_df = pd.DataFrame()
                
                # Iterate over each dataframe within the combination
                for k in key_tuple:
                    if combined_df.empty: 
                        combined_df = dataframes_dict[k].copy()
                    else:
                        combined_df += dataframes_dict[k]  # Add the values to the combined DataFrame

                dataframes_dict[combined_key] = combined_df  # Store the combined DataFrame in the dictionary under the new key
    return dataframes_dict   

def generate_composite_maps(dataframes_dict: dict, thresholded: bool=True) -> dict:
    """
    Combines DataFrames in the input dictionary to create composite maps based on absolute values. 
    Generates combinations of DataFrames for all possible combinations of keys in the dictionary, 
    starting from pairs up to the full set of keys. Adds each resulting DataFrame back to the dictionary 
    with a key formatted as 'K1+K2+...+Kn', where K1, K2, ..., Kn are the original keys of the combined DataFrames.

    Parameters:
    - dataframes_dict (dict): A dictionary where each key is a string representing the DataFrame's name, 
                              and each value is a pandas DataFrame.
    - thresholded (bool): A boolean indicating whether to use thresholded values (True) or unthresholded values (False).

    Returns:
    - dict: The original dictionary updated with additional keys for each combination of DataFrames. 
            Each new key contains the sum of absolute values of the combined DataFrames.
    """
    dataframes_dict = generate_composite(dataframes_dict, thresholded=thresholded)
    return dataframes_dict
