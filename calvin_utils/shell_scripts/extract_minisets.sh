#!/bin/bash

#extract_minisets.sh

### Variables ###
ORIG_DIR="/Volumes/Expansion 1/MasterDataset/HornDatasets/derivatives/leaddbs"
DEST_DIR="/Volumes/Expansion/datasets/DBS_minisets"
JSON_LOC="/Volumes/Expansion 1/MasterDataset/HornDatasets/participants.json"

# Declare associative array properly
declare -A FOLDERS_TO_EXTRACT=(
    ["clinical"]="ses-01/clinical"
    ["reconstruction"]="ses-01/reconstruction"
    ["stimulations"]="ses-01/stimulations"
    ["preprocessing_anat"]="ses-01/anat/*T1w.nii*"
)

### Business Logic ###
mkdir -p "$DEST_DIR"
shopt -s nullglob

for SUBFOLDER in "$ORIG_DIR"/*; do
    SUBFOLDER_NAME=$(basename "$SUBFOLDER")

    # Get metadata from JSON safely
    METADATA_TSV=$(jq -r --arg id "$SUBFOLDER_NAME" '
        .[] | select(.id == $id) | [
            (.id                       // "unknown"),
            (.Age                      // "unknown"),
            (.elmodel                  // "unknown"),
            (.Condition                // "unknown"),
            (."Years since diagnosis"  // "unknown"),
            (.Target                   // "unknown"),
            (.DOIs                     // "unknown"),
            (.City                     // "Misc")
        ] | @tsv
    ' "$JSON_LOC")

    # Skip if no metadata found
    if [ -z "$METADATA_TSV" ]; then
        echo "No metadata found for $SUBFOLDER_NAME, skipping."
        continue
    fi

    # Read TSV into variables
    IFS=$'\t' read -r ID AGE ELMODEL CONDITION YEARS_SINCE_DX TARGET DOIS CITY <<< "$METADATA_TSV"

    echo "Processing subject: $ID"

    DEST_SUB="${CITY}_${CONDITION}_DBS"
    DIR="$DEST_DIR/$DEST_SUB"

    if [ -d "$DIR" ]; then
        echo "Directory $DIR already exists. Skipping..."
        continue
    fi

    mkdir -p "$DIR/metadata"
    mkdir -p "$DIR/data/ses-01"

    # Add metadata to master CSV
    metadata_file="$DIR/metadata/master_list.csv"
    if [ ! -f "$metadata_file" ]; then
        echo "ID,Age,ElModel,Condition,YearsSinceDx,Target,DOIs" > "$metadata_file"
    fi
    echo "$ID,$AGE,$ELMODEL,$CONDITION,$YEARS_SINCE_DX,$TARGET,\"$DOIS\"" >> "$metadata_file"

    # Copy required folders/files
    for key in "${!FOLDERS_TO_EXTRACT[@]}"; do
        src_pattern="$SUBFOLDER/${FOLDERS_TO_EXTRACT[$key]}"
        dst="$DIR/data/$key"
        mkdir -p "$dst"

        files=( $src_pattern )
        if [ ${#files[@]} -eq 0 ]; then
            echo "Warning: No files matched pattern $src_pattern."
            continue
        fi

        for file in "${files[@]}"; do
            if [ -d "$file" ]; then
                cp -r "$file" "$dst"
            elif [ -f "$file" ]; then
                cp "$file" "$dst"
            else
                echo "Warning: Source $file is neither file nor directory."
            fi
        done
    done
done