tar -cvzf /data/nimlab/dl_archive/adni_calvin/csf_conn.tar.gz --dereference /data/nimlab/USERS/ahg26/Collaborators/Calvin/18012025_request/seeds_csf-gm_atrophy/derivatives/*/connectivity/*stat-t_conn.nii.gz


find /data/nimlab/dl_archive/MS_auto948/ -type f \( -path "*/connectivity/*stat-t_conn.nii.gz" -o -path "*/roi/*nii.gz" \) -print0 | tar --null -cvzf /PHShome/cu135/tars/ms_tar.tar.gz --dereference --files-from=-