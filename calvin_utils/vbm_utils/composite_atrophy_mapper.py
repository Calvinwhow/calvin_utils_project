import pandas as pd
from typing import Tuple
from itertools import combinations

def generate_composite(dataframes_dict: dict, csfgm_only: bool=True, sign_flip_csf: bool=False) -> dict:
    """
    Generates composite DataFrames by combining data from multiple DataFrames in the input dictionary.
    For each combination of two or more DataFrames, creates a new DataFrame by summing their values.
    For the 'cerebrospinal_fluid' DataFrame, flips the sign of its values before combining.

    Parameters:
    - dataframes_dict (dict): A dictionary where keys are strings representing the names of DataFrames, 
                              and values are pandas DataFrames.
    - csfgm_only (bool): Whether to only make compound maps of CSF and GM. 
    - sign_flip_csf (bool): whether to flip sign of CSF or other segments. If true, multiplies CSF
        by negative 1, setting more atrophy to be more negative. If false, multiplies all others by -1,
        setting more atrophy to be more positive. 

    Returns:
    - dict: The input dictionary updated with additional keys for each combination of DataFrames. 
            Each new key contains the sum of the combined DataFrames' values.
    """
    if sign_flip_csf: 
        print("MULTIPLYING CSF BY -1. NEGATIVE VALUES IN ALL OUTPUTS REPRESENT MORE ATROPHY")
        dataframes_dict['cerebrospinal_fluid'] = dataframes_dict['cerebrospinal_fluid']*-1
    else:
        print("MULTIPLYING GM AND WM BY -1. POSITIVE VALUES IN ALL OUTPUTS REPRESENT MORE ATROPHY")
        for key in dataframes_dict.keys():
            if key != 'cerebrospinal_fluid': dataframes_dict[key] = dataframes_dict[key]*-1
        
    if csfgm_only:
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

def generate_composite_maps(dataframes_dict: dict, csfgm_only=True) -> dict:
    """
    Combines DataFrames in the input dictionary to create composite maps based on absolute values. 
    Generates combinations of DataFrames for all possible combinations of keys in the dictionary, 
    starting from pairs up to the full set of keys. Adds each resulting DataFrame back to the dictionary 
    with a key formatted as 'K1+K2+...+Kn', where K1, K2, ..., Kn are the original keys of the combined DataFrames.

    Parameters:
    - dataframes_dict (dict): A dictionary where each key is a string representing the DataFrame's name, 
                              and each value is a pandas DataFrame.
    - csfgm_only (bool): Whether to only make compound maps of CSF and GM. 

    Returns:
    - dict: The original dictionary updated with additional keys for each combination of DataFrames. 
            Each new key contains the sum of absolute values of the combined DataFrames.
    """
    dataframes_dict = generate_composite(dataframes_dict, csfgm_only=csfgm_only, sign_flip_csf=False)
    return dataframes_dict
