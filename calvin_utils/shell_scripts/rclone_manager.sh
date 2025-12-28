#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ----
REMOTE="dropbox"   # your rclone remote name (change if different)
BASE="AD_dataset"
DEST="/Volumes/HowExp/datasets/03h_ADVANCE_Alzheimer_DBS/subjects"

# List immediate subfolders on Dropbox under AD_dataset (e.g., 0001/, 0002/)
mapfile -t SUBJECTS < <(rclone lsf -d "${REMOTE}:${BASE}")

for s in "${SUBJECTS[@]}"; do
  subj="${s%/}"                                    # strip trailing slash
  src="${REMOTE}:${BASE}/${subj}"                  # e.g., dropbox:/.../AD_dataset/0001
  dst="${DEST}/sub-${subj}"                        # e.g., .../AD_dataset/sub-0001

  mkdir -p "${dst}"

  # --- files
  for f in ea_reconstruction.mat; do
    if rclone lsf "${src}/${f}" >/dev/null 2>&1; then
      rclone copy "${src}/${f}" "${dst}/" --progress
    else
      echo "SKIP: ${src}/${f} not found"
    fi
  done

done