tar -cvzf /home1/cu135/elec_coords_fixed.tar.gz --dereference /data/nimlab/dl_archive/MS_auto948/*/connectivity/*stat-t_conn.nii.gz


find /data/nimlab/dl_archive/MS_auto948/ -type f \( -path "*/connectivity/*stat-t_conn.nii.gz" -o -path "*/roi/*nii.gz" \) -print0 | tar --null -cvzf /PHShome/cu135/tars/ms_tar.tar.gz --dereference --files-from=-