# Define the local machine's username and IP address
LOCAL_USER="cu135"
LOCAL_IP="your_local_ip_address"
LOCAL_BASE_DIR="/Volumes/OneTouch/datasets/PPMI_PD/derivatives"

# Find the files and transfer them to the local machine
find /data/nimlab/PPMI_Luo/reran_subjects -type f -name 'T1.mgz' | while read -r file; do
    folder=$(echo "$file" | awk -F'reran_subjects/' '{print $2}' | cut -d '/' -f 1)
    local_dir="$LOCAL_BASE_DIR/$folder/mri"
    
    # Create the local directory if it doesn't exist
    ssh $LOCAL_USER@$LOCAL_IP "mkdir -p $local_dir"
    
    echo "copying: $file to $local_dir"
    
    # Use rsync to transfer the file to the local machine
    rsync -avz "$file" $LOCAL_USER@$LOCAL_IP:"$local_dir"
done