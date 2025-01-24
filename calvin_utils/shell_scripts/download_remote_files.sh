# Define your password
PASSWORD="@BoxBro916598670"

subfolder=$'reran_subjects/'
# Read the file list and copy each file to the local machine
sshpass -p "$PASSWORD" ssh cu135@dna005.research.partners.org "find /data/nimlab/PPMI_Luo/reran_subjects -type f -name 'T1.mgz'" | while read -r file; do
    folder=$(echo "$file" | awk -F$subfolder '{print $2}' | cut -d '/' -f 1)
    local_dir="/Volumes/OneTouch/datasets/PPMI_PD/derivatives/$folder/mri"
    mkdir -p "$local_dir"

    echo "copying: $file to $local_dir"
    sshpass -p "$PASSWORD" rsync -avz cu135@dna005.research.partners.org:"$file" $local_dir
done
