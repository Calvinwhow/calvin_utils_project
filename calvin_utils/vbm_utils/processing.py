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

def save_nifti_to_bids(dataframes_dict, bids_base_dir, mask_path, analysis='tissue_segment_z_scores', ses=None, dry_run=True):
    """
    Saves NIFTI images to a BIDS-compliant directory structure.

    Parameters:
    - dataframes_dict (dict): A dictionary where keys are tissue types or categories, 
                                and values are DataFrames containing NIFTI data for each subject.
    - bids_base_dir (str): The base directory where the BIDS structure starts.
    - mask_path (str): Path to the mask file used for reference when saving NIFTI images.
    - analysis (str, optional): The name of the analysis folder to save the NIFTI images under. 
                                    Defaults to 'tissue_segment_z_scores'.
    - ses (str, optional): Session identifier. If None, defaults to '01'.
    - dry_run (bool, optional): If True, prints the output directory paths without saving files. 
                                    Defaults to True.

    Note:
    This function assumes a predefined BIDS directory structure and saves the NIFTI images 
    accordingly. If `dry_run` is set to True, the function will only print the intended save 
    paths without creating any files. To actually save the NIFTI images, set `dry_run` to False.
    """
    def construct_bids_path(bids_base_dir, sub_no, ses_no, analysis):
        """
        Constructs the BIDS-compliant output directory path.

        Parameters:
        - bids_base_dir (str): Base directory for BIDS structure.
        - sub_no (str): Subject identifier.
        - ses_no (str): Session identifier.
        - analysis (str): Folder name.

        Returns:
        - str: Full path to the output directory.
        """
        return os.path.join(bids_base_dir, f'sub-{sub_no}', f'ses-{ses_no}', analysis)

    def save_or_print_nifti(dataframe, col, out_dir, tissue_type, dry_run, mask_path):
        """
        Saves or prints the NIFTI file path for a given subject.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing NIFTI data.
        - col (str): Column name representing the subject.
        - out_dir (str): Output directory path.
        - tissue_type (str): Tissue type or category.
        - dry_run (bool): If True, only prints the path; otherwise, saves the file.
        - mask_path (str): File to use as mask
        """
        output_name = f'sub-{col}_{tissue_type}'
        if dry_run:
            print(os.path.join(out_dir, output_name))
        else:
            view_and_save_nifti(matrix=dataframe[col],
                                out_dir=out_dir,
                                output_name=output_name,
                                ref_file=mask_path)

    ses_no = ses if ses else '01'

    for tissue_type, dataframe in tqdm(dataframes_dict.items()):
        for col in dataframe.columns:
            sub_no = col
            out_dir = construct_bids_path(bids_base_dir, sub_no, ses_no, analysis)
            os.makedirs(out_dir, exist_ok=True)
            save_or_print_nifti(dataframe, col, out_dir, tissue_type, dry_run, mask_path)