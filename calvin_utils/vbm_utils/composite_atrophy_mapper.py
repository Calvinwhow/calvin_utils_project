import pandas as pd
from typing import Tuple
from itertools import combinations
import numpy as np

def generate_tensor(dict):
    arrays = [np.asarray(dict[k]) for k in dict.keys()]
    voxels, patients = arrays[0].shape
    V = len(arrays)
    tensor = np.empty((voxels, patients, V), dtype=arrays[0].dtype)
    for i, arr in enumerate(arrays):
        tensor[:, :, i] = arr  # shape (voxels, patients, V)
    return tensor

def generate_norm(arr):
    """Generates the L2 norm of a tensor"""
    return np.sqrt(np.sum(arr**2, axis=2))

def generate_norm_map(pt_dict, ctrl_dict, flip_csf=True) -> pd.DataFrame:
    """Generates a composite atrophy map with L2 norm of Z scores"""
    if flip_csf:
        print("Flipping CSF sign.")
        if 'cerebrospinal_fluid' in pt_dict:
            pt_dict['cerebrospinal_fluid'] = -pt_dict['cerebrospinal_fluid']
        if 'cerebrospinal_fluid' in ctrl_dict:
            ctrl_dict['cerebrospinal_fluid'] = -ctrl_dict['cerebrospinal_fluid']
            
    ctrl_tensor = generate_tensor(ctrl_dict)
    pt_tensor = generate_tensor(pt_dict)
    ctrl_norm = generate_norm(ctrl_tensor)
    pt_norm = generate_norm(pt_tensor)
    
    mean = ctrl_norm.mean(axis=1)
    std  = ctrl_norm.std(axis=1)
    z = (pt_norm - mean[:, np.newaxis]) / std[:, np.newaxis]
    
    first_key = next(iter(pt_dict))
    z = pd.DataFrame(z, columns=pt_dict[first_key].columns, index=pt_dict[first_key].index)
    return z, mean, std

def generate_composite(dataframes: dict, csfgm_only: bool = True, sign_flip_csf: bool = False) -> dict:
    """
    Combine DataFrames to create composite maps.
    Optionally flip CSF or other segment signs.
    If csfgm_only is True, only combine CSF and GM; otherwise, combine all possible subsets.
    """
    dfs = dataframes.copy()
    if sign_flip_csf:
        print("Flipping CSF sign.")
        if 'cerebrospinal_fluid' in dfs:
            dfs['cerebrospinal_fluid'] = -dfs['cerebrospinal_fluid']
    else:
        print("Flipping GM and WM sign.")
        for k in dfs:
            if k != 'cerebrospinal_fluid':
                dfs[k] = -dfs[k]

    if csfgm_only:
        if 'cerebrospinal_fluid' in dfs and 'grey_matter' in dfs:
            dfs['csf+gm'] = dfs['cerebrospinal_fluid'] + dfs['grey_matter']
    else:
        keys = list(dfs.keys())
        for r in range(2, len(keys) + 1):
            for combo in combinations(keys, r):
                name = '+'.join(combo)
                dfs[name] = sum(dfs[k] for k in combo)
    return dfs