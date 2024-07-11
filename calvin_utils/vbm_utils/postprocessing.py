import pandas as pd
import numpy as np
from nimlab import datasets as nimds
import nibabel as nib
from tqdm import tqdm
import os

def save_csv_to_bids(dataframes_dict, bids_base_dir, analysis='atrophy_results', ses=None, dry_run=True):
    """
    Saves csv to a BIDS directory structure.
    
    Parameters:
    - dataframes_dict (dict): Dictionary containing dataframes with NIFTI data.
    - bids_base_dir (str): The base directory where the BIDS structure starts.
    - ses (str, optional): Session identifier. If None, defaults to '01'.
    
    Note:
    This function assumes a predefined BIDS directory structure and saves the CSV 
    accordingly.
    """
    
    for key, value in tqdm(dataframes_dict.items()):            
        # Define BIDS Directory Architecture
        sub_no = 'all'
        if ses is None:
            ses_no = '01'
        else:
            ses_no = ses
        
        # Define and Initialize the Save Directory
        out_dir = os.path.join(bids_base_dir, 'neuroimaging_analyses', f'ses-{ses_no}', f'sub-{sub_no}', analysis)
        os.makedirs(out_dir, exist_ok=True)
        
        # Save Image to BIDS Directory
        if dry_run:
            print(out_dir+f'/{key}.csv')
        else:
            value.to_csv(out_dir+f'/{key}.csv')
            print('Saved to: ', out_dir+f'/{key}.csv')

def sort_dataframes_by_index(damage_df_dict):
    """
    Attempts to convert the index of each dataframe in damage_df_dict to integers 
    and then sorts the dataframe by its index in ascending order.
    
    Parameters:
    - damage_df_dict (dict): Dictionary containing dataframes to be sorted.
    
    Returns:
    - dict: The sorted damage_df_dict.
    """
    
    sorted_df_dict = {}
    
    for k, df in damage_df_dict.items():
        try:
            # Convert index to integers
            df.index = df.index.astype(int)
            
            # Sort dataframe by index
            sorted_df = df.sort_index(ascending=True)
            sorted_df_dict[k] = sorted_df
            
            # Display the results
            print('Dataframe: ', k)
            display(sorted_df)
            print('------------- \n')
            
        except ValueError:
            # If conversion to integer fails, just use the original dataframe
            sorted_df_dict[k] = df

    return sorted_df_dict


def calculate_damage_scores(thresholded_atrophy_df_dict, region_of_interest_df, count_voxels=True):
    """
    Calculates damage scores for each region of interest based on thresholded atrophy data.
    
    Parameters:
    - thresholded_atrophy_df_dict (dict): Dictionary containing dataframes with thresholded atrophy data.
    - region_of_interest_df (DataFrame): DataFrame containing regions of interest.
    
    Returns:
    - dict: A dictionary containing damage scores for each region of interest for each patient.
    """
    
    damage_df_dict = {}
    
    for k, v in thresholded_atrophy_df_dict.items():
        
        # Initialize the Dataframe
        damage_df_dict[k] = pd.DataFrame(index=v.columns, columns=region_of_interest_df.columns)
        
        # Iterate Over Each Region of Interest
        for matrix in damage_df_dict[k].columns:
            # Extract Damage Score for Each Patient
            for subject in damage_df_dict[k].index:
                # Mask the subject dataframe to the matrix at hand
                intersection = v[subject].where(region_of_interest_df[matrix] > 0, 0)
                
                # Use multiplication to zero values outside ROI
                weighted_overlap = intersection * region_of_interest_df[matrix]
                
                # Assess overall damage score & atrophied voxels within ROI
                damage_score = weighted_overlap.sum()
                num_voxels = np.count_nonzero(weighted_overlap)
                
                # Assign values to DF 
                damage_df_dict[k].loc[subject, matrix] = damage_score
                damage_df_dict[k].loc[subject, 'num_atrophied_voxels_'+matrix] = num_voxels
                
        print('Dataframe: ', k)
        display(damage_df_dict[k])
        print('------------- \n')
    
    return damage_df_dict

def calculate_total_atrophy_voxels(thresholded_atrophy_df_dict, damage_df_dict, mask_path=None):
    """
    Calculates the total number of atrophy voxels within a mask and updates the damage_df_dict with this information.
    
    Parameters:
    - thresholded_atrophy_df_dict (dict): Dictionary containing dataframes with thresholded atrophy data.
    - damage_df_dict (dict): Dictionary containing dataframes to which the 'Total Atrophy Voxels' column will be added.
    - mask_path (str): path to mask. Defaults to None. 
    
    Returns:
    - dict: The updated damage_df_dict with a new column 'Total Atrophy Voxels'.
    """
    
    # Get the mask and brain indices
    if mask_path is None:
        try:
            mni_mask = nimds.get_img("mni_icbm152")
            mask_data = mni_mask.get_fdata().flatten()
        except:
            raise FileNotFoundError("Nimlab package is not installed. Please provide the absolute path to your brain mask.")
    else:
        mask_data = nib.load(mask_path).get_fdata().flatten()
        
    brain_indices = np.where(mask_data > 0)[0]
    
    for k, v in thresholded_atrophy_df_dict.items():
        # initialize the column
        damage_df_dict[k]['Total Atrophy Voxels'] = np.nan
        for patient_id in v.columns:
            # Filter the dataframe using brain indices
            filtered_df = v[patient_id].iloc[brain_indices]
            damage_df_dict[k].loc[patient_id, 'Total Atrophy Voxels'] = np.count_nonzero(filtered_df)
        
        print('Dataframe: ', k)
        display(damage_df_dict[k])
        print('------------- \n')
    
    return damage_df_dict
