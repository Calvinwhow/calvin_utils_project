from glob import glob
from pathlib import Path
import pprint

import nibabel as nib
import numpy as np
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans


class NiftiCentroidStats:
    """
    Compare centroids of two sets of NIfTI files.

    Parameters
    ----------
    grp1_glob, grp2_glob : str
        Recursive glob patterns for the two groups (e.g. '/data/A/**/*.nii*').
    n_centroids : int, default 1
        How many centroids to extract **per file** (k‑means if > 1).
    mirror : bool, default True
        If n_centroids > 1 and *mirror* is True, x‑coordinates of every
        centroid‑bucket are forced to share the same sign (midline mirroring)
        and **all buckets are then concatenated**, so the test is run on a
        *single* merged centroid array per group.

    Public methods
    --------------
    run()                 → summary_xyz, ttests_xyz
    compare_norms()       → summary_norm, ttests_norm
    """

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        grp1_glob: str,
        grp2_glob: str,
        *,
        n_centroids: int = 1,
        mirror: bool = True,
    ):
        if n_centroids < 1:
            raise ValueError("n_centroids must be ≥ 1")
        self.n_centroids = n_centroids
        self.mirror = mirror and n_centroids > 1
        self.grp1_files = self._keep_nii(glob(grp1_glob, recursive=True))
        self.grp2_files = self._keep_nii(glob(grp2_glob, recursive=True))

    # ------------------------------------------------------------------ #
    # public orchestrators
    # ------------------------------------------------------------------ #
    def run(self, name="Each coordinate (XYZ)"):
        """Welch t‑tests on X, Y, Z."""
        if self.mirror:
            xyz1, xyz2 = self._get_merged_mirrored_both()
            summary = self._xyz_summary(xyz1, xyz2)
            ttests  = self._xyz_ttests(xyz1, xyz2)
            self._show_results(summary, ttests, name + "  (mirrored & merged)")
            return summary, ttests

        # no mirroring → keep buckets separate
        grp1, grp2 = self._get_centroid_sets_both()
        summary = {f"centroid_{i}": self._xyz_summary(g1, g2)
                   for i, (g1, g2) in enumerate(zip(grp1, grp2))}
        ttests  = {f"centroid_{i}": self._xyz_ttests(g1, g2)
                   for i, (g1, g2) in enumerate(zip(grp1, grp2))}
        self._show_results(summary, ttests, name)
        return summary, ttests

    def compare_norms(self, name="Euclidean norm of XYZ"):
        """Welch t‑test on ||XYZ||."""
        if self.mirror:
            xyz1, xyz2 = self._get_merged_mirrored_both()
            n1, n2 = np.linalg.norm(xyz1, axis=1), np.linalg.norm(xyz2, axis=1)
            summary, ttests = self._norm_stats_and_test(n1, n2)
            self._show_results(summary, ttests, name + "  (mirrored & merged)")
            return summary, ttests

        grp1, grp2 = self._get_centroid_sets_both()
        summary, ttests = {}, {}
        for i, (g1, g2) in enumerate(zip(grp1, grp2)):
            n1, n2 = np.linalg.norm(g1, axis=1), np.linalg.norm(g2, axis=1)
            s, t = self._norm_stats_and_test(n1, n2)
            summary[f"centroid_{i}"], ttests[f"centroid_{i}"] = s['merged'], t['norm']
        self._show_results(summary, ttests, name)
        return summary, ttests

    # ------------------------------------------------------------------ #
    # helpers – file handling / centroid extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _keep_nii(paths):
        return [p for p in paths if Path(p).suffix in {'.nii', '.gz'}]

    @staticmethod
    def _voxel_to_world(vox_xyz, affine):
        return (affine @ np.append(vox_xyz, 1.0))[:3]

    def _centroids_of_file(self, f):
        """Return (n_centroids,3) world‑space centroids for one file."""
        img   = nib.load(f)
        idx   = np.transpose(np.nonzero(img.get_fdata()))
        if self.n_centroids == 1:
            ctrs = [idx.mean(0)]
        else:
            kmeans = KMeans(n_clusters=self.n_centroids, n_init='auto').fit(idx)
            ctrs = [idx[kmeans.labels_ == k].mean(0) for k in range(self.n_centroids)]
            ctrs.sort(key=lambda c: c[0])        # left → right order
        return np.vstack([self._voxel_to_world(c, img.affine) for c in ctrs])

    # bucket‑by‑centroid arrays
    def _centroid_sets_for_group(self, files):
        buckets = [[] for _ in range(self.n_centroids)]
        for f in files:
            ctrs = self._centroids_of_file(f)
            for i, c in enumerate(ctrs):
                buckets[i].append(c)
        return [np.vstack(b) for b in buckets]

    def _get_centroid_sets_both(self):
        return (self._centroid_sets_for_group(self.grp1_files),
                self._centroid_sets_for_group(self.grp2_files))

    # ------------------------------------------------------------------ #
    # mirroring / merging
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mirror_bucket(arr):
        """Flip x so the bucket's mean x becomes non‑negative."""
        if np.mean(arr[:, 0]) < 0:
            arr = arr.copy()
            arr[:, 0] *= -1
        return arr

    def _merged_mirrored(self, bucket_list):
        mirrored = [self._mirror_bucket(b) for b in bucket_list]
        return np.vstack(mirrored)

    def _get_merged_mirrored_both(self):
        g1_buckets, g2_buckets = self._get_centroid_sets_both()
        return (self._merged_mirrored(g1_buckets),
                self._merged_mirrored(g2_buckets))

    # ------------------------------------------------------------------ #
    # stats helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _xyz_summary(xyz1, xyz2):
        return {
            'group1': {'n': len(xyz1), 'mean': xyz1.mean(0), 'std': xyz1.std(0, ddof=1)},
            'group2': {'n': len(xyz2), 'mean': xyz2.mean(0), 'std': xyz2.std(0, ddof=1)},
        }

    @staticmethod
    def _xyz_ttests(xyz1, xyz2):
        if min(len(xyz1), len(xyz2)) < 2:
            return {ax: {'t': np.nan, 'p': np.nan} for ax in 'xyz'}
        return {ax: {'t': res.statistic, 'p': res.pvalue}
                for ax, res in zip('xyz',
                                   (ttest_ind(xyz1[:, i], xyz2[:, i], equal_var=False)
                                    for i in range(3)))}

    @staticmethod
    def _norm_stats_and_test(n1, n2):
        summary = {
            'merged': {
                'group1': {'n': len(n1), 'mean': n1.mean(), 'std': n1.std(ddof=1)},
                'group2': {'n': len(n2), 'mean': n2.mean(), 'std': n2.std(ddof=1)},
            }
        }
        t, p = (np.nan, np.nan) if min(len(n1), len(n2)) < 2 else \
               ttest_ind(n1, n2, equal_var=False)
        ttests = {'norm': {'t': t, 'p': p}}
        return summary, ttests

    # pretty printing
    @staticmethod
    def _show_results(summary, tests, name):
        print(f"\n----- {name} -----")
        print("Summary:")
        pprint.pprint(summary, compact=True)
        print("T‑tests:")
        pprint.pprint(tests, compact=True)


# ------------------------------------------------------------------ #
# demo
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    stats = NiftiCentroidStats(
        grp1_glob='/data/groupA/**/*.nii*',
        grp2_glob='/data/groupB/**/*.nii*',
        n_centroids=2,
        mirror=True,
    )
    stats.run()
    stats.compare_norms()
