import os
from calvin_utils.file_utils.import_matrices import import_matrices_from_folder 

def import_dataframes_by_tissue(base_directory, shared_glob_path, tissue_to_import):
    """
    Imports dataframes based on tissue types from specified directories and glob name patterns.
    
    Parameters:
    - base_directory (str): The base directory where the data resides.
    - shared_glob_path (str): The shared directory path for all tissues.
    - tissue_to_import (list): List of tissues to be imported.
    
    Returns:
    - dict: A dictionary containing dataframes for each specified tissue.
    """

    segments_dict = {}
    for tissue in tissue_to_import:
        glob_path = os.path.join(shared_glob_path, ('*'+tissue+'*'))
        segments_dict[tissue] = glob_path

    dataframes_dict = {}
    nan_handler = {'nan': 0, 'posinf':20, 'neginf':-20}
    for k, v in segments_dict.items():
        dataframes_dict[k] = import_matrices_from_folder(connectivity_path=base_directory, file_pattern=v, convert_nan_to_num=nan_handler)
        print(f'Imported data {k} data with {dataframes_dict[k].shape[0]} voxels and {dataframes_dict[k].shape[1]} patients')
        print(f'Example filename per subject: {dataframes_dict[k].columns[0]}')
        print('\n--------------------------------\n')

    return dataframes_dict

def import_segments(base_dir, gm_pattern, wm_pattern, csf_pattern) -> dict:
    """
    Import control dataframes for grey matter, white matter, and CSF using specified glob patterns.

    Args:
        base_dir (str): Base directory containing the data.
        gm_pattern (str): Glob pattern for grey matter files.
        wm_pattern (str): Glob pattern for white matter files.
        csf_pattern (str): Glob pattern for CSF files.

    Returns:
        dict: Dictionary with keys 'grey_matter', 'white_matter', 'cerebrospinal_fluid' and corresponding dataframes as values.
    """
    patterns = {'grey_matter': gm_pattern, 'white_matter': wm_pattern, 'cerebrospinal_fluid': csf_pattern}
    dfs = {}
    for tissue, pattern in patterns.items():
        dfs[tissue] = import_matrices_from_folder(connectivity_path=base_dir, file_pattern=pattern)
        print(f'Imported {tissue}: {dfs[tissue].shape[0]} voxels, {dfs[tissue].shape[1]} patients')
        print(f'Example subject filename: {dfs[tissue].columns[-1]}')
        print('--------------------------------')
    return dfs