from pathlib import Path
import shutil

def organize_dcm_to_bids(src_root: str | Path, bids_root: str | Path, dcm_pattern_list=['T1']) -> None:
    """
    Copy dcm scan folders into a BIDS-style layout.

    Creates:
        <bids_root>/<patient>/ses-01/DCM/<AX_T1_scan_dir>/*
        <bids_root>/<patient>/ses-01/ANAT/

    Args
    ----
    src_root : str | Path
        Directory whose immediate subdirectories are patient folders.
    bids_root : str | Path
        Destination directory; “BIDS” level will be created if absent.
    dcm_pattern_list : list of str
        List of patterns to match scan directories. Default is ['T1'].
    """
    src_root = Path(src_root).expanduser().resolve()
    bids_root = Path(bids_root).expanduser().resolve()
    unmatched = set()
    for patient_dir in filter(Path.is_dir, src_root.iterdir()):
        unmatched.add(patient_dir.name)
        dcm_dst  = bids_root / ('sub-' + patient_dir.name) / "ses-01" / "DCM"
        anat_dst = bids_root / ('sub-' + patient_dir.name) / "ses-01" / "ANAT"
        dcm_dst.mkdir(parents=True, exist_ok=True)
        anat_dst.mkdir(parents=True, exist_ok=True)

        for scan_dir in filter(Path.is_dir, patient_dir.iterdir()):
            if any(x in scan_dir.name.upper() for x in dcm_pattern_list):
                shutil.copytree(scan_dir, dcm_dst / scan_dir.name, dirs_exist_ok=True)
                unmatched.discard(patient_dir.name)
    print("Unmatched files: ", unmatched)