import gl
import sys
overlay_image_path = '/Volumes/Expansion/datasets/Manitoba_PET/_averaged_summed_connectivity.nii'

gl.overlayload(overlay_image_path)
gl.colorname(1, '6bluegrn')
gl.minmax(1, -1, -0.00000001)  # Set the color range for the first overlay

gl.overlayload(overlay_image_path)
gl.colorname(2, '8redyell')
gl.minmax(2, 0.000001, 1)  # Set the color range for the second overlay
