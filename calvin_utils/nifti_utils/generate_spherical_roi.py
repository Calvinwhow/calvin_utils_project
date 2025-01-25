import os
import numpy as np
import nibabel as nib
from nilearn import plotting

def calculate_voxelwise_mni_coords(brain_img):
    brain_mask = brain_img.get_fdata()
    ijk_mask = np.array(np.meshgrid(
        np.arange(brain_mask.shape[0]),
        np.arange(brain_mask.shape[1]),
        np.arange(brain_mask.shape[2]),
        indexing='ij'
    ))                                                                                  #shape (3, i, j, k). An array stacking the ijk coords into 3 tensors
    ijk_arr = ijk_mask.reshape(3,-1).T                                                  #shape (n, 3). An array with the ijk coords in sub-arrays. 
    ijk1 = np.zeros((ijk_arr.shape[0],4))                                               #shape (n,4)
    ijk1[:,:3] = ijk_arr                                                                #shape (n,4). An array with the ijk coords in sub-arrays
    ijk1[:, 3] = 1                                                                      #shape (n,4). Prepared for homologous transform
    affine = brain_img.affine                                                           #shape (4,4). The affine matrix of the brain mask
    mni_coords = np.dot(affine, ijk1.T).T[:, :3]                                        #shape (n,3). The MNI coordinates of each voxel
    return mni_coords

def calculate_pairwise_distance(mni_coords_point, mni_coords_brain):
    dist_from_point = np.linalg.norm(mni_coords_brain - mni_coords_point, axis=1)       #shape (n,). The euclidean distance from the point to each voxel
    return dist_from_point

def points_within_sphere(distances, radius):
    return (distances <= radius).astype(int)

def generate_sphere_mask(points, brain_img):
    return np.reshape(points, brain_img.get_fdata().shape)

def mask_within_brain(sphere_mask, brain_img):
    brain_mask = brain_img.get_fdata()
    brain_indices = brain_mask == 1
    sphere_mask[~brain_indices] = 0
    return sphere_mask

def save_mask(mask, brain_img, mni_arr, out_dir, filename='sphere_roi'):
    affine = brain_img.affine
    mask_img = nib.Nifti1Image(mask, affine, header=brain_img.header)
    mask_img.to_filename(os.path.join(out_dir, filename))
    return plotting.view_img(mask_img, cut_coords=(mni_arr[0],mni_arr[1],mni_arr[2]), black_bg=False, opacity=.75, cmap='ocean_hot')

def prep_bids(subid, out_dir):
    bids_dir = os.path.join(out_dir, f'sub-{subid}', 'roi')
    os.makedirs(bids_dir, exist_ok=True)
    filename = f'sub-{subid}-MNI152_T1_2mm-tms_sphere_roi.nii.gz'
    return bids_dir, filename

def get_closest_brain_edge(center_mni, brain_img):
    mni_coords_brain = calculate_voxelwise_mni_coords(brain_img)                    #shape (n,3)    
   
    brain_mask = brain_img.get_fdata()                                          
    brain_indices = brain_mask.flatten() == 1                                       #shape (k,3)
    mni_coords_in_brain = mni_coords_brain[brain_indices]                           #shape (k,3)
                  
    edge_distances = calculate_pairwise_distance(center_mni, mni_coords_in_brain)   #shape (k,3)
    min_distance_index = np.argmin(edge_distances)                                  #shape (1,)
    nearest_edge_coord = mni_coords_in_brain[min_distance_index]                    #shape (1,)   
    return nearest_edge_coord

def generate_spherical_rois_from_df(df, x_col, y_col, z_col, subcol, radius, mask_path, out_dir, project_on_to_brain=False):
    # Load the brain mask image
    brain_img = nib.load(mask_path)

    for index, row in df.iterrows():
        x = df.loc[index, x_col]
        y = df.loc[index, y_col]
        z = df.loc[index, z_col]
        center_mni = np.array([x, y, z])

        # Calculate voxelwise MNI Mask
        mni_coords_brain = calculate_voxelwise_mni_coords(brain_img)
        distances = calculate_pairwise_distance(center_mni, mni_coords_brain)
        points_in_sphere = points_within_sphere(distances, radius)
        sphere_mask = generate_sphere_mask(points_in_sphere, brain_img)
        
        if project_on_to_brain:
            closest_coord = get_closest_brain_edge(center_mni, brain_img)               # shape (3,) representing MNI coordinates of closest brain voxel
            distances = calculate_pairwise_distance(closest_coord, mni_coords_brain)    # 
            points_in_sphere = points_within_sphere(distances, radius)
            sphere_mask = generate_sphere_mask(points_in_sphere, brain_img)
            
        sphere_mask = mask_within_brain(sphere_mask, brain_img)

        # Save the mask
        subject_id = row[subcol]  # Assuming there's a 'subject_id' column
        bids_dir, filename = prep_bids(subid=subject_id, out_dir=out_dir)
        save_mask(sphere_mask, brain_img, center_mni, bids_dir, filename)