import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from itertools import combinations
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti

def get_tiv(tissue_dict: dict[str, pd.DataFrame], voxel_res: float = 2.0) -> np.ndarray:
    """
    Return total intracranial volume (one scalar per patient).
    Assumes each DataFrame has shape (n_voxels, n_subjects) and *modulated* intensities in mm³/voxel.
    """
    keys = ("grey_matter", "white_matter", "cerebrospinal_fluid")
    missing = [k for k in keys if k not in tissue_dict]
    if missing:
        raise KeyError(f"Need GM, WM, and CSF. Missing {missing}")

    voxel_sum = sum(tissue_dict[k].values for k in keys)      # (vox × subj)
    tiv = voxel_sum.sum(axis=0) * voxel_res**3                 # (n_subjects,)
    return tiv


def threshold_probabilities(patient_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    '''This step avoids generation of ridiculously high z-scores due to very low probabilities.'''
    patient_df = patient_df.where(patient_df > threshold, 0)
    return patient_df

def calculate_z_scores(ctrl: pd.DataFrame, pat: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate voxel-wise z-scores for patients using control mean and std.

    Args:
        ctrl (pd.DataFrame): Controls, columns are subjects, rows are voxels.
        pat (pd.DataFrame): Patients, columns are subjects, rows are voxels.

    Returns:
        pd.DataFrame: Z-scores for each patient.
    """
    ctrl_arr = ctrl.values                    # (N_vox × N_ctrl)
    pat_arr  = pat.values                     # (N_vox × N_pat)
    mean = ctrl_arr.mean(axis=1, keepdims=True)  # (N_vox × 1)
    std  = ctrl_arr.std(axis=1, ddof=0, keepdims=True) # (N_vox × 1)
    z_arr = (pat_arr - mean) / std               # broadcasting (N_vox × N_pat)
    return pd.DataFrame(z_arr, index=pat.index, columns=pat.columns), mean, std

def process_tissue(df, tiv):
    '''This will threshold the probabilities and normalize by TIV.'''
    df = threshold_probabilities(df, threshold=0.2)  # Thresholding control probabilities
    return df / tiv[np.newaxis, :]  # Normalize by TIV

def process_atrophy(data_dict, ctrl_dict):
    """Calculates z-scores and significant atrophy masks for each tissue type."""
    zscore_dict = {}
    zscore_mask_dict = {}
    stats_dict = {}
    
    pt_tiv = get_tiv(data_dict)    # Total Intracranial Volume for patients
    ctrl_tiv = get_tiv(ctrl_dict)  # Total Intracranial Volume for controls

    for tissue in data_dict:
        pat_df = process_tissue(data_dict[tissue], pt_tiv)  # Process patient data
        ctrl_df = process_tissue(ctrl_dict[tissue], ctrl_tiv)
        zscores, mean, std = calculate_z_scores(pat=pat_df, ctrl=ctrl_df)
        if tissue == 'cerebrospinal_fluid':
            sig_mask = zscores.where(zscores > 2, 0)
        else:
            sig_mask = zscores.where(zscores < -2, 0)
            
        zscore_dict[tissue] = zscores
        zscore_mask_dict[tissue] = sig_mask
        stats_dict[tissue] = (mean, std)
        
    return zscore_dict, zscore_mask_dict, stats_dict

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
    """
    def process_name(name):
        """Process the subject name to ensure it is in BIDS format."""
        for pattern in ['_smwp1', '_smwp2', '_smwp3', '_mwp1', '_mwp2', '_mwp3' 'smwp1', 'smwp2', 'smwp3', 'mwp1', 'mwp2', 'mwp3', '_T1', 'T1', '_resampled', 'resampled', '.nii', '.nii.gz']:
            name = name.replace(pattern, '')
        return name
    
    def construct_bids_path(bids_base_dir, sub_no, ses_no, analysis):
        return os.path.join(bids_base_dir, f'sub-{sub_no}', f'ses-{ses_no}', analysis)

    def save_or_print_nifti(dataframe, col, ses, tissue_type, out_dir, dry_run, mask_path):
        output_name = f'sub-{col}_{ses}_{tissue_type}'
        if dry_run:
            print(os.path.join(out_dir, output_name))
        else:
            view_and_save_nifti(matrix=dataframe[col],
                                out_dir=out_dir,
                                output_name=output_name,
                                ref_file=mask_path)

    ses_no = ses if ses else '01'

    for tissue_type, dataframe in tqdm(dataframes_dict.items()):
        dataframe = dataframe.rename(columns=process_name)
        for col in dataframe.columns:
            out_dir = construct_bids_path(bids_base_dir, col, ses_no, analysis)
            os.makedirs(out_dir, exist_ok=True)
            save_or_print_nifti(dataframe, col, ses, tissue_type, out_dir, dry_run, mask_path)