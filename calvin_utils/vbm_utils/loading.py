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

def import_control_dataframes(base_directory, control_grey_matter_glob_name_pattern, control_white_matter_glob_name_pattern, control_csf_glob_name_pattern):
    """
    Imports control dataframes from specified directories and glob name patterns.

    Parameters:
    - base_directory (str): The base directory where the data resides.
    - control_grey_matter_glob_name_pattern (str): Glob pattern for grey matter data.
    - control_white_matter_glob_name_pattern (str): Glob pattern for white matter data.
    - control_csf_glob_name_pattern (str): Glob pattern for cerebrospinal fluid data.

    Returns:
    - dict: A dictionary containing control dataframes for grey matter, white matter, and cerebrospinal fluid.
    """
    
    segments_dict = {
        'grey_matter': {'path': base_directory, 'glob_name_pattern': control_grey_matter_glob_name_pattern},
        'white_matter': {'path': base_directory, 'glob_name_pattern': control_white_matter_glob_name_pattern},
        'cerebrospinal_fluid': {'path': base_directory, 'glob_name_pattern': control_csf_glob_name_pattern}
    }

    control_dataframes_dict = {}
    for k, v in segments_dict.items():
        control_dataframes_dict[k] = import_matrices_from_folder(connectivity_path=v['path'], file_pattern=v['glob_name_pattern']);
        print(f'Imported data {k} data with {control_dataframes_dict[k].shape[0]} voxels and {control_dataframes_dict[k].shape[1]} patients')
        print(f'Example subject filename: {control_dataframes_dict[k].columns[-1]}')
        print('--------------------------------')

    return control_dataframes_dict

def import_dataframes_from_folders(base_directory, grey_matter_glob_name_pattern, white_matter_glob_name_pattern, csf_glob_name_pattern):
    """
    Imports dataframes from specified directories and glob name patterns.
    
    Parameters:
    - base_directory (str): The base directory where the data resides.
    - grey_matter_glob_name_pattern (str): Glob pattern for grey matter data.
    - white_matter_glob_name_pattern (str): Glob pattern for white matter data.
    - csf_glob_name_pattern (str): Glob pattern for cerebrospinal fluid data.
    
    Returns:
    - dict: A dictionary containing dataframes for grey matter, white matter, and cerebrospinal fluid.
    """
    

    segments_dict = {
        'grey_matter': {'path': base_directory, 'glob_name_pattern': grey_matter_glob_name_pattern},
        'white_matter': {'path': base_directory, 'glob_name_pattern': white_matter_glob_name_pattern},
        'cerebrospinal_fluid': {'path': base_directory, 'glob_name_pattern': csf_glob_name_pattern}
    }

    dataframes_dict = {}

    for k, v in segments_dict.items():
        dataframes_dict[k] = import_matrices_from_folder(connectivity_path=v['path'], file_pattern=v['glob_name_pattern'])
        print(f'Imported data {k} data with {dataframes_dict[k].shape[0]} voxels and {dataframes_dict[k].shape[1]} patients')
        print(f'These are the filenames per subject {dataframes_dict[k].columns}')
        print('--------------------------------')

    return dataframes_dict
