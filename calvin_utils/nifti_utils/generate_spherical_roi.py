import os
import ast
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import plotting
from typing import Sequence, Union, List

class SphereROIBuilder:
    """
    Build spherical ROIs in MNI space and save them as NIfTI masks.

    Parameters
    ----------
    mask_path : str | Path
        Path to an MNI-registered binary brain mask (value = 1 inside brain).
    out_dir   : str | Path
        Top-level directory where BIDS-style ROI files will be written.
    radius    : float
        Sphere radius in millimetres.
    """

    def __init__(self, mask_path: Union[str, Path], out_dir: Union[str, Path], radius: float):
        self.brain_img: nib.Nifti1Image = nib.load(str(mask_path))
        self.radius: float = float(radius)
        self.out_dir: Path = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._mni_coords_brain: np.ndarray = self._voxelwise_mni_coords() # pre-compute MNI coordinates for every voxel (n_vox × 3)

    ### validation methods ###
    @staticmethod
    def _normalize_coord_entry(entry) -> np.ndarray:
        """
        Accepts:
            • list / tuple / np.ndarray, shape (3,) or (n,3)
            • string representation of either of the above, e.g.
              "[34, -18, 52]"  or  "[[12,34,56], [-7,8,9]]"
        Returns
        -------
        arr : np.ndarray, shape (n, 3)
        """
        if isinstance(entry, str):
            entry = entry.strip()
            entry = ast.literal_eval(entry)  # safe eval of list/tuple syntax
        arr = np.asarray(entry, dtype=float)
        if arr.ndim == 1:                    # single triplet → make it 2-D
            if arr.size != 3:
                raise ValueError("Each coordinate must have exactly 3 values.")
            arr = arr[None, :]               # (1,3)
        elif arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Coordinate entry must be (n,3).")
        return arr
    
    ### orchestration methods ###
    def generate_from_single_coord(
        self,
        centre_xyz: Sequence[float] | str,
        *,
        subject: str,
        session: str | None = None,
        project_on_brain: bool = True,
        suffix: str = ""
    ) -> str:
        """
        Build **one** spherical ROI from a single MNI coordinate triplet
        and write it to disk.

        Parameters
        ----------
        centre_xyz : [x, y, z] list/tuple/ndarray **or** its string repr.
        subject    : BIDS “sub-” label.
        session    : BIDS “ses-” label (defaults to “ses-01”).
        project_on_brain : Snap centre to nearest in-brain voxel.
        suffix     : Extra text to append to the filename.

        Returns
        -------
        str – absolute path to the saved NIfTI mask.
        """
        centre = self._normalize_coord_entry(centre_xyz)[0]  # (1,3) → (3,)
        return self._build_and_save(
            centre_xyz=centre,
            subject=subject,
            session=session,
            project_on_brain=project_on_brain,
            suffix=suffix,
        )

    def generate_from_separate_coord_cols(self, df, x_col: str, y_col: str, z_col: str, sub_col: str, session_col: str | None = None, project_on_brain: bool = True) -> pd.DataFrame:
        """
        Generate spherical regions of interest (ROIs) from a dataframe.

        This method iterates over the rows of the provided dataframe and creates 
        a spherical ROI for each row based on the specified x, y, and z coordinates. 
        The ROIs are saved with metadata including subject, session (if provided), 
        and an optional projection onto a brain surface.

        Args:
            df (pd.DataFrame): The dataframe containing ROI information.
            x_col (str): The column name for the x-coordinate.
            y_col (str): The column name for the y-coordinate.
            z_col (str): The column name for the z-coordinate.
            sub_col (str): The column name for the subject identifier.
            session_col (str | None, optional): The column name for the session identifier. 
                Defaults to None.
            project_on_brain (bool, optional): Whether to project the ROI onto a brain 
                surface (geodesic projection moves ROI inside the brain). Defaults to True.
        """
        for row_idx, row in df.iterrows():
            centre = np.array([row[x_col], row[y_col], row[z_col]], dtype=float)
            df.loc[row_idx, "roi_path"] = self._build_and_save(centre, subject=row[sub_col], session=row[session_col] if session_col else None, project_on_brain=project_on_brain)
        return df 

    def generate_from_coordlist_column(self, df, coord_col: str, sub_col: str, session_col: str | None = None, project_on_brain: bool = True, save_individual_roi: bool = False) -> pd.DataFrame:
        """
        Generates spherical regions of interest (ROIs) from a DataFrame column containing coordinates. If multiple lists of coordinates are provided,
        each list is treated as a separate set of coordinates for which a spherical ROI is generated. All ROIs are saved inside one nifti. 

        This method processes a DataFrame where a specified column contains either single [x, y, z] 
        coordinate lists or lists of multiple coordinate lists. For each coordinate or list of coordinates, 
        a spherical ROI is generated and saved. Optionally, the coordinates can be projected onto a brain 
        surface.

        Args:
            df (pd.DataFrame): The input DataFrame containing coordinate data.
            coord_col (str): The name of the column in the DataFrame containing the coordinates.
            sub_col (str): The name of the column specifying the subject identifier.
            session_col (str | None, optional): The name of the column specifying the session identifier. 
                Defaults to None.
            project_on_brain (bool, optional): Whether to project the coordinates onto a brain surface. 
                Defaults to True (geodesic projection moves ROI inside the brain).

        """
        for row_idx, row in df.iterrows():
            coord_entry = row[coord_col]
            coords = self._normalize_coord_entry(coord_entry)
            sphere = None
            for idx, centre_xyz in enumerate(coords, start=1):
                if save_individual_roi:
                    self._build_and_save(np.asarray(centre_xyz, dtype=float),subject=row[sub_col],session=row[session_col] if session_col else None,project_on_brain=project_on_brain,suffix=f"_{idx:02d}" if len(coords) > 1 else "")
                current_sphere = self._build_sphere(np.asarray(centre_xyz, dtype=float), project_on_brain)
                sphere = current_sphere if sphere is None else sphere + current_sphere
            df.loc[row_idx, "roi_path"] = self._save_built_sphere(sphere, row[sub_col], row[session_col] if session_col else None, project_on_brain, suffix = "_composite")
        return df 

    ### internal utils ###
    
    def _voxelwise_mni_coords(self) -> np.ndarray:
        data = self.brain_img.get_fdata()
        ijk = np.stack(np.meshgrid(
            np.arange(data.shape[0]),
            np.arange(data.shape[1]),
            np.arange(data.shape[2]),
            indexing="ij"
        ), axis=0)               # 3 × X × Y × Z
        ijk = ijk.reshape(3, -1).T                       # n_vox × 3
        ijk_h = np.c_[ijk, np.ones(len(ijk))]            # homogeneous
        return (self.brain_img.affine @ ijk_h.T).T[:, :3]

    @staticmethod
    def _pairwise_distance(point_xyz: np.ndarray, coords_xyz: np.ndarray) -> np.ndarray:
        return np.linalg.norm(coords_xyz - point_xyz, axis=1)

    @staticmethod
    def _points_within_radius(distances: np.ndarray, radius: float) -> np.ndarray:
        return (distances <= radius).astype(int)

    def _build_sphere_mask(self, centre_xyz: np.ndarray) -> np.ndarray:
        d = self._pairwise_distance(centre_xyz, self._mni_coords_brain)
        mask_flat = self._points_within_radius(d, self.radius)
        mask = mask_flat.reshape(self.brain_img.shape)
        mask[self.brain_img.get_fdata() != 1] = 0
        return mask

    def _nearest_brain_voxel(self, point_xyz: np.ndarray) -> np.ndarray:
        inside = self._mni_coords_brain[self.brain_img.get_fdata().ravel() == 1]
        d = self._pairwise_distance(point_xyz, inside)
        return inside[np.argmin(d)]
    
    def _build_sphere(self,centre_xyz: np.ndarray, project_on_brain: bool) -> None:
        if project_on_brain:
            centre_xyz = self._nearest_brain_voxel(centre_xyz)
        return self._build_sphere_mask(centre_xyz)
    
    def _save_built_sphere(self, sphere: np.ndarray, subject: str, session: str | None, project_on_brain: bool, suffix: str = "") -> None:
        bids_dir, fname = self._prep_bids(subject, session, suffix)
        self._save_mask(sphere, (0,0,0), bids_dir / fname)
        return os.path.join(bids_dir, fname)

    def _build_and_save(self,centre_xyz: np.ndarray, subject: str, session: str | None, project_on_brain: bool, suffix: str = "") -> None:
        if project_on_brain:
            centre_xyz = self._nearest_brain_voxel(centre_xyz)
        mask = self._build_sphere_mask(centre_xyz)
        bids_dir, fname = self._prep_bids(subject, session, suffix)
        self._save_mask(mask, centre_xyz, bids_dir / fname)
        return os.path.join(bids_dir, fname)

    ### file I/O ###
    def _prep_bids(self, sub: str, ses: str | None, suffix: str) -> tuple[Path, str]:
        ses_part = f"ses-{ses}" if ses else "ses-01"
        roi_dir = self.out_dir / f"sub-{sub.replace(' ', '_')}" / ses_part.replace(" ", "_") / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        fname = f"sub-{sub}_MNI152_T1_{self.radius}mm_sphere_roi{suffix}.nii.gz".replace(" ", "_")
        return roi_dir, fname

    def _save_mask(self, mask: np.ndarray, centre_xyz: np.ndarray, path: Path, verbose: bool = False) -> None:
        nib.save(nib.Nifti1Image(mask, self.brain_img.affine, header=self.brain_img.header), path)
        if verbose: plotting.view_img(path.as_posix(),cut_coords=tuple(centre_xyz),black_bg=False,opacity=0.75,cmap="ocean_hot")