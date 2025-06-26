#!/bin/bash

# Step-by-step Shell Script for Anonymizing Neuroimaging Data

# Required software:
# 1. MRICron (dcm2nii): https://www.nitrc.org/projects/mricron
# 2. FSL (BET): https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

# General instructions:
# Place your script at the same directory where you have the folder /dicom_data
# Store all the patient folders in /dicom_data
# Each patient folder should be labeled T1
# In each patient's T1 folder, store all their .dcm files. 
# If the patient already has T1 .nii files created:
#   1) place them in /nii_data/patient_xx/nii_data
#   2) comment out line 41 of this code (dcm2nii -o ./nii_data "$patient/T1/")
#   3) run the code

# Directory structure:
# DICOM files should be organized as follows:
# ./dicom_data/
# ├── patient_01/
# │   └── T1/
# │       └── *.dcm
# ├── patient_02/
# │   └── T1/
# │       └── *.dcm
# └── patient_N/
#     └── T1/
#         └── *.dcm

# Create output directories
mkdir -p ./nii_data
mkdir -p ./nii_brain_extracted

# Loop through each patient directory
for patient in ./dicom_data/*; do
  patient_id=$(basename "$patient")
  echo "Processing $patient_id"

  # Convert DICOM to NIFTI (anonymizes data)
  dcm2nii -o ./nii_data "$patient/T1/"

  # Assume single T1 image, move it to standardized naming
  nifti_file=$(ls ./nii_data/*.nii.gz | head -n 1)
  mv "$nifti_file" ./nii_data/${patient_id}_T1.nii.gz
done

# After completing, pair each brain-extracted file with ADAS-Cog scores
# Ensure filenames match the patient identifier used in your clinical data

# Final verification and transfer to MGB cluster can be done by contacting:
# Calvin Howard
# e: choward12@bwh.harvard.edu