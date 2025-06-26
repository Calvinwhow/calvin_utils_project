import re
import pathlib
import numpy as np
import nibabel as nib

def split_atlas(atlas_nii: str, labels_txt: str, out_dir: str | pathlib.Path) -> None:
    """
    Parameters
    ----------
    atlas_nii   : str | Path  to integer valued NIfTI, e.g. AAL.nii.gz
    labels_txt  : str | Path  to tabular text/LUT file: <index> <region_name> <other>
    out_dir     : str | Path  to directory where masks will be written
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load once
    atlas_img = nib.load(atlas_nii)
    atlas = atlas_img.get_fdata()
    affine, header = atlas_img.affine, atlas_img.header

    # iterate over label file
    with open(labels_txt) as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # split on any whitespace; ignore extra columns
            idx_str, name = re.split(r"\s+", line, maxsplit=1)
            idx = int(idx_str)
            if idx == 0:
                continue  # skip background

            mask = (atlas == idx).astype(np.uint8)
            if not mask.any():
                continue  # atlas lacks this label

            # safe filename
            fname = re.sub(r"[^0-9A-Za-z]+", "_", name) + ".nii.gz"
            nib.save(nib.Nifti1Image(mask, affine, header), out_dir / fname)
            print("✔", fname)


if __name__ == "__main__":
    split_atlas(
        atlas_nii="AAL.nii.gz",      # path to the combined atlas NIfTI file (integer valued)
        labels_txt="AAL.txt",        # path to a .txt file with labels, e.g. AAL.txt
        out_dir="aal_masks"          # path to output dir. will be created if absent
    )
