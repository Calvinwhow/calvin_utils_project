import numpy as np
import pandas as pd

def convert_ijk_to_mni(ijk, affine):
    return affine @ np.append(ijk, 1)

def convert_mni_to_ijk(mni, affine):
    inv_affine = np.linalg.pinv(affine)
    return inv_affine @ np.append(mni, 1)

def convert_talairach_to_mni(talairach_coords):
    """
    Convert Talairach coordinates to MNI coordinates using a predefined transformation matrix.

    Parameters:
    -----------
    talairach_coords : np.array
        1D array representing the Talairach coordinates (x, y, z).

    Returns:
    --------
    mni_coords : np.array
        1D array representing the MNI coordinates (x, y, z).
    """
    # Predefined transformation matrix from Talairach to MNI
    talairach_to_mni_matrix = np.array([
        [0.9900, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.9688, 0.0460, 0.0000],
        [0.0000, -0.0485, 0.9189, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])

    # Convert Talairach coordinates to homogeneous coordinates
    if len(talairach_coords) == 3:
        talairach_coords_homogeneous = np.append(talairach_coords, 1)
    mni_coords_homogeneous = talairach_to_mni_matrix @ talairach_coords_homogeneous
    return mni_coords_homogeneous[:3]

def convert_mni_to_talairach_mni(mni_coords):
    """
    Convert MNI coordinates to Talairach coordinates using a predefined transformation matrix.

    Parameters:
    -----------
    mni_coords : np.array
        1D array representing the Talairach coordinates (x, y, z).

    Returns:
    --------
    tal_coords : np.array
        1D array representing the MNI coordinates (x, y, z).
    """
    # Predefined transformation matrix from Talairach to MNI
    talairach_to_mni_matrix = np.array([
        [0.9900, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.9688, 0.0460, 0.0000],
        [0.0000, -0.0485, 0.9189, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])
    invmx = np.linalg.pinv(talairach_to_mni_matrix)

    # Convert Talairach coordinates to homogeneous coordinates
    if len(mni_coords) == 3:
        mni_coords = np.append(mni_coords, 1)
    tal_coords = invmx @ mni_coords
    return tal_coords[:3]

def convert_coordinates_in_df(df, x_col, y_col, z_col, conversion_type):
    """
    Convert coordinates from one space to another and store the converted coordinates in new columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the coordinates to be converted.
    x_col : str
        Column name for the x coordinates.
    y_col : str
        Column name for the y coordinates.
    z_col : str
        Column name for the z coordinates.
    conversion_type : str
        Type of conversion to perform. Options are:
        - 'ijk_to_mni'
        - 'mni_to_ijk'
        - 'talairach_to_mni'
        - 'mni_to_talairach'

    Returns:
    --------
    pd.DataFrame
        DataFrame with the converted coordinates in new columns.
    """
    conversion_functions = {
        'ijk_to_mni': convert_ijk_to_mni,
        'mni_to_ijk': convert_mni_to_ijk,
        'talairach_to_mni': convert_talairach_to_mni,
        'mni_to_talairach': convert_mni_to_talairach_mni
    }

    if conversion_type not in conversion_functions:
        raise ValueError(f"Invalid conversion type: {conversion_type}")

    convert_func = conversion_functions[conversion_type]

    # Prepare new column names
    new_x_col = f"x_{conversion_type.split('_')[-1]}"
    new_y_col = f"y_{conversion_type.split('_')[-1]}"
    new_z_col = f"z_{conversion_type.split('_')[-1]}"

    # Perform the conversion
    converted_coords = df.apply(lambda row: convert_func(np.array([row[x_col], row[y_col], row[z_col]])), axis=1)
    df[[new_x_col, new_y_col, new_z_col]] = pd.DataFrame(converted_coords.tolist(), index=df.index)

    return df