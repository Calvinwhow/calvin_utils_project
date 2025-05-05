from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans


class NiftiCentroidStats:
    """
    Centroid statistics for two groups of NIfTI files.

    Dependencies
    ------------
    numpy
    scipy
    nibabel
    scikit‑learn   # only if `multi_centroids=True`

    Usage
    -----
    stats = NiftiCentroidStats(
        grp1_glob='/path/to/group1/**/*.nii*',
        grp2_glob='/path/to/group2/**/*.nii*',
        multi_centroids=False          # True ↔ k‑means on each file
    )
    summary, ttest = stats.run()
    """
    def __init__(self, grp1_glob: str, grp2_glob: str, *, multi_centroids: bool = False):
        self.grp1_files = self._expand(glob(grp1_glob, recursive=True))
        self.grp2_files = self._expand(glob(grp2_glob, recursive=True))
        self.multi = multi_centroids

    @staticmethod
    def _expand(paths):
        '''keep .nii and .nii.gz only'''
        return [p for p in paths if Path(p).suffix in {'.nii', '.gz'}]

    @staticmethod
    def _voxel_to_world(vox_xyz, affine):
        """Map voxel XYZ (float) to world coordinates using affine."""
        xyz1 = np.append(vox_xyz, 1.0)
        return (affine @ xyz1)[:3]

    def _centroid_of_file(self, f):
        img = nib.load(f)
        data = img.get_fdata()
        idx = np.transpose(np.nonzero(data))  # (N, 3) voxel indices
        if self.multi:
            labels = KMeans(n_clusters=2, n_init='auto').fit_predict(idx)
            centroids = [idx[labels == k].mean(0) for k in (0, 1)]
        else:
            centroids = [idx.mean(0)]
        return np.vstack([self._voxel_to_world(c, img.affine) for c in centroids])

    def _centroids_for_group(self, file_list):
        """Return (n,3) array of centroids for a group of files."""
        all_xyz = [self._centroid_of_file(f) for f in file_list]
        return np.vstack(all_xyz)

    def run(self):
        xyz1 = self._centroids_for_group(self.grp1_files)
        xyz2 = self._centroids_for_group(self.grp2_files)

        summary = {
            'group1': {'n': len(xyz1), 'mean': xyz1.mean(0), 'std': xyz1.std(0, ddof=1)},
            'group2': {'n': len(xyz2), 'mean': xyz2.mean(0), 'std': xyz2.std(0, ddof=1)},
        }

        ttest = {
            axis: ttest_ind(xyz1[:, i], xyz2[:, i], equal_var=False)
            for i, axis in enumerate(('x', 'y', 'z'))
        }
        return summary, ttest


# quick CLI‑style example (remove or adapt for real use)
if __name__ == '__main__':
    stats = NiftiCentroidStats(
        grp1_glob='/data/groupA/**/*.nii*',
        grp2_glob='/data/groupB/**/*.nii*',
        multi_centroids=False,
    )
    summary, ttest = stats.run()
    print(summary)
    print({k: {'t': v.statistic, 'p': v.pvalue} for k, v in ttest.items()})
