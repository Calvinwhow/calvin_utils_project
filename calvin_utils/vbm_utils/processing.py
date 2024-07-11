import os
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from itertools import combinations
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti


def threshold_probabilities(patient_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    patient_df = patient_df.where(patient_df > threshold, 0)
    return patient_df

def calculate_z_scores(control_df: pd.DataFrame, patient_df: pd.DataFrame, matter_type=None) -> pd.DataFrame:
    """
    Function to calculate voxel-wise mean, standard deviation for control group and z-scores for patient group.

    Args:
    control_df (pd.DataFrame): DataFrame where each column represents a control subject, 
                               and each row represents flattened image data for a voxel.
    patient_df (pd.DataFrame): DataFrame where each column represents a patient, 
                               and each row represents flattened image data for a voxel.

    Returns:
    patient_z_scores (pd.DataFrame): DataFrame of voxel-wise z-scores calculated for each patient using control mean and std.
    """

    # # Mask the dataframes to only consider tissues over acceptable probability thresholds
    # # Using p>0.2, as typical masking to MNI152 segments uses P > 0.2 for a given segment
    
    # # Now you can use the function to apply a threshold to patient_df and control_df
    threshold = 0.2
    patient_df = threshold_probabilities(patient_df, threshold)
    control_df = threshold_probabilities(control_df, threshold)

    # Calculate mean and standard deviation for each voxel in control group
    control_mean = control_df.mean(axis=1)
    control_std = control_df.std(axis=1)

    # Initialize DataFrame to store patient z-scores
    patient_z_scores = pd.DataFrame()

    # Calculate z-scores for each patient using control mean and std
    for patient in patient_df.columns:
        patient_z_scores[patient] = (patient_df[patient] - control_mean) / control_std

    return patient_z_scores

def process_atrophy_dataframes(dataframes_dict, control_dataframes_dict):
    """
    Processes the provided dataframes to calculate z-scores and determine significant atrophy.

    Parameters:
    - dataframes_dict (dict): Dictionary containing patient dataframes.
    - control_dataframes_dict (dict): Dictionary containing control dataframes.

    Returns:
    - tuple: A tuple containing two dictionaries - atrophy_dataframes_dict and significant_atrophy_dataframes_dict.
    """
    
    atrophy_dataframes_dict = {}
    significant_atrophy_dataframes_dict = {}

    for k in dataframes_dict.keys():
        atrophy_dataframes_dict[k] = calculate_z_scores(control_df=control_dataframes_dict[k], patient_df=dataframes_dict[k])
        if k == 'cerebrospinal_fluid':
            significant_atrophy_dataframes_dict[k] = atrophy_dataframes_dict[k].where(atrophy_dataframes_dict[k] > 2, 0)
        else:
            significant_atrophy_dataframes_dict[k] = atrophy_dataframes_dict[k].where(atrophy_dataframes_dict[k] < -2, 0)
        print('Dataframe: ', k)
        display(dataframes_dict[k])
        print('------------- \n')

    return atrophy_dataframes_dict, significant_atrophy_dataframes_dict

def save_nifti_to_bids(dataframes_dict, bids_base_dir, analysis='tissue_segment_z_scores', ses=None, dry_run=True):
    """
    Saves NIFTI images to a BIDS directory structure.
    
    Parameters:
    - dataframes_dict (dict): Dictionary containing dataframes with NIFTI data.
    - bids_base_dir (str): The base directory where the BIDS structure starts.
    - ses (str, optional): Session identifier. If None, defaults to '01'.
    
    Note:
    This function assumes a predefined BIDS directory structure and saves the NIFTI 
    images accordingly. The function currently has the view_and_save_nifti call commented out 
    for safety. Uncomment this call if you wish to actually save the NIFTI images.
    
    Example:
    >>> dfs = { ... }  # some dictionary with dataframes
    >>> save_nifti_to_bids(dfs, '/path/to/base/dir')
    """
    
    for k in tqdm(dataframes_dict.keys()):
        for col in dataframes_dict[k].columns:
            
            # Define BIDS Directory Architecture
            sub_no = col
            if ses is None:
                ses_no = '01'
            else:
                ses_no = ses
            
            # Define and Initialize the Save Directory
            out_dir = os.path.join(bids_base_dir, f'sub-{sub_no}', f'ses-{ses_no}', analysis)
            os.makedirs(out_dir, exist_ok=True)
            
            # Save Image to BIDS Directory
            if dry_run:
                print(out_dir+f'/sub-{sub_no}_{k}')
            else:
                view_and_save_nifti(matrix=dataframes_dict[k][col],
                                    out_dir=out_dir,
                                    output_name=(f'sub-{sub_no}_{k}'))