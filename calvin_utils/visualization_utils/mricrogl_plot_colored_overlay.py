import gl
import sys
overlay_image_path = '/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/resources/datasets/Kahana_iEEG/derivatives/generated_spherical_rois/all_rois_overlaid/generated_nifti.nii'
gl.overlayload(overlay_image_path)
gl.coloreditor(1)
gl.colorname(1,'3blue')
gl.colornode(1,0,0,4, 43, 83,0) # replace 4,43,83 w/ your color
gl.colornode(1,1,100,4, 43, 83,94) # replace 4,43,83 w/ your color
gl.colornode(1,2,255,4, 43, 83,164) # replace 4,43,83 w/ your color
gl.coloreditorclose(1)

