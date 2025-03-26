import os
import re
import glob
import nibabel as nib
from tqdm import tqdm
# from nimlab import datasets as nimds
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template

def remove_specific_mwp_integer_pattern(text):
    # Define the pattern to search for: 'mwp' followed by [1], [2], or [3]
    pattern = r'mwp[123]'
    # Replace the first occurrence of the pattern with an empty string
    return re.sub(pattern, '', text, count=1)

def extract_and_rename_subject_id(dataframe, split_command_dict):
    """
    Renames the columns of a dataframe based on specified split commands.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe whose columns need to be renamed.
    - split_command_dict (dict): A dictionary where the key is the split string 
                                 and the value is the order to take after splitting 
                                 (0 for before the split, 1 for after the split, etc.).

    Returns:
    - pd.DataFrame: Dataframe with renamed columns.
    """

    raw_names = dataframe.columns
    name_mapping = {}

    # For each column name in the dataframe
    for name in raw_names:
        new_name = name  # Default to the original name in case it doesn't match any split command

        # Check each split command to see if it applies to this column name
        for k, v in split_command_dict.items():
            if k in new_name:
                new_name = remove_specific_mwp_integer_pattern(new_name)
                if k !='':
                    new_name = new_name.split(k)[v]
        # Add the original and new name to the mapping
        name_mapping[name] = new_name

    # Rename columns in the dataframe based on the mapping
    return dataframe.rename(columns=name_mapping)

def rename_dataframe_subjects(dataframes_dict, preceding_id, proceeding_id):
    """
    Renames the subjects in the provided dataframes based on the split commands.

    Parameters:
    - dataframes_dict (dict): A dictionary containing dataframes with subjects to be renamed.
    - preceding_id (str): The delimiter for taking the part after the split.
    - proceeding_id (str): The delimiter for taking the part before the split.

    Returns:
    - dict: A dictionary containing dataframes with subjects renamed.
    """
    
    split_command_dict = {preceding_id: 1, proceeding_id: 0}
    
    for k, v in dataframes_dict.items():
        dataframes_dict[k] = extract_and_rename_subject_id(dataframe=dataframes_dict[k], split_command_dict=split_command_dict)
        print('Dataframe: ', k)
        print(dataframes_dict[k])
        print('------------- \n')

    return dataframes_dict

def binarize_image(img):
    data = img.get_fdata()
    data[data != 0] = 1
    return nib.Nifti1Image(data, img.affine, img.header)
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nilearn.image import resample_to_img

def quick_downsample_with_dilation(input_path, output_path, mask_path, dilation=1):
    # Load images
    mni_mask = nib.load(mask_path)
    img = nib.load(input_path)

    # Morphological dilation: expand lesion so it’s not lost
    data = img.get_fdata()
    data[data != 0] = 1
    # Use binary dilation
    dilated_data = ndi.binary_dilation(data, iterations=dilation).astype(data.dtype)
    dilated_img = nib.Nifti1Image(dilated_data, img.affine, img.header)

    # Downsample with nearest neighbor
    resampled_img = resample_to_img(
        dilated_img, 
        mni_mask, 
        interpolation='nearest'
    )

    # Optional: erode to revert the dilation
    # Erosion with the same number of iterations
    resampled_data = resampled_img.get_fdata()
    eroded_data = ndi.binary_erosion(resampled_data, iterations=dilation).astype(resampled_data.dtype)

    # Save
    final_img = nib.Nifti1Image(eroded_data, resampled_img.affine, resampled_img.header)
    nib.save(final_img, output_path)
    
def downsample_image(input_path, output_path, res=2, mask_path=None, binarize=False):
    """
    Function to downsample a 3D image to a new voxel size using a target affine.
    
    Args:
    input_path (str): Filepath to the input image.
    output_path (str): Filepath to save the output image.
    target_voxel_size (list): Target voxels to resample to.
    """
    # Load the image
    if mask_path is not None: 
        mni_mask = nib.load(mask_path)
    else:
        mni_mask = load_mni152_template(res)

    img = nib.load(input_path)
    resampled_img = resample_to_img(img, mni_mask, interpolation='nearest')
    nib.save(resampled_img, output_path)
    
def resample_images_in_folder(input_folder_pattern, mask_path=None, binarize=False):
    """
    Function to resample all 3D images in a folder to a new voxel size.
    
    Args:
    input_folder_pattern (str): Glob pattern to find the input images.
    target_voxel_size (list): Target voxels to resample to.
    binarize (bool): Whether to binarize
    """
    # Find all input image filepaths
    input_filepaths = glob.glob(input_folder_pattern)
    print('Will search:, ', input_folder_pattern)

    # Loop over each input image
    output_path_list = []
    for input_path in tqdm(input_filepaths):
        # Define the output path
        base, ext = os.path.splitext(input_path)
        if ext == '.gz':
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext
        output_path = base + '_resampled' + ext

        quick_downsample_with_dilation(input_path, output_path, mask_path=mask_path)
        output_path_list.append(output_path)
    
    return output_path_list

def downsample_to_mni152_images_in_folder(input_folder_pattern, mask_path=None, res=2, dry_run=True):
    """
    Function to downsample all 3D images in a folder to a new voxel size.
    
    Args:
    input_folder_pattern (str): Glob pattern to find the input images.
    res (int): Resolution to downsample to (default is 2mm).
    target_voxel_size (list): Target voxels to resample to.
    """
    # Find all input image filepaths
    input_filepaths = glob.glob(input_folder_pattern)
    print('Will search:, ', input_folder_pattern)

    # Loop over each input image
    output_path_list = []
    for input_path in tqdm(input_filepaths):
        # Define the output path
        base, ext = os.path.splitext(input_path)
        if ext == '.gz':
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext
        output_path = base + '_resampled' + ext

        # Downsample the image
        if dry_run:
            pass
        else:
            downsample_image(input_path, output_path, mask_path=mask_path, res=res)
        output_path_list.append(output_path)
    
    return output_path_list

def downsample_orchestrator(base_directory, grey_matter_glob_name_pattern, white_matter_glob_name_pattern, csf_glob_name_pattern):
    """
    Downsamples images to MNI152 2x2x2mm standard space and saves them in a specified directory.
    
    Parameters:
    - base_directory (str): The base directory where the images are located.
    - grey_matter_glob_name_pattern (str): Glob pattern for grey matter data.
    - white_matter_glob_name_pattern (str): Glob pattern for white matter data.
    - csf_glob_name_pattern (str): Glob pattern for cerebrospinal fluid data.
    
    Returns:
    - dict: A dictionary containing paths of the downsampled images for each segment.
    """
    
    segments_dict = {
        'grey_matter': {'path': base_directory, 'glob_name_pattern': grey_matter_glob_name_pattern},
        'white_matter': {'path': base_directory, 'glob_name_pattern': white_matter_glob_name_pattern},
        'cerebrospinal_fluid': {'path': base_directory, 'glob_name_pattern': csf_glob_name_pattern}
    }

    output_paths_dict = {}

    for k, v in segments_dict.items():
        output_paths_dict[k] = downsample_to_mni152_images_in_folder(os.path.join(v['path'], v['glob_name_pattern']), dry_run=False)
        print(f'Downsampled {k} segment data \n ')
        print(f'Saved files to {output_paths_dict[k]}')
        print('-------------------------------- \n')

    return output_paths_dict
