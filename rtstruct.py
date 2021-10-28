import pydicom
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
import os
from scipy import stats

root = "/mnt/c/Users/James/Documents/MPhys-Project"

# Load existing RT Struct. Requires the series path and existing RT Struct path
rtstruct = RTStructBuilder.create_from(
  dicom_series_path="/mnt/c/Users/James/Documents/MPhys-Project/Sorted-data/HN-CHUM-003-CT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111", 
  rt_struct_path="/mnt/c/Users/James/Documents/MPhys-Project/Sorted-data/HN-CHUM-003-RTSTRUCT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111/1_RTstruct_CTsim-_CT_PET-CT_-UnknownInstanceNumber.dcm"
)

# View all of the ROI names from within the image
print(rtstruct.get_roi_names())

# Loading the 3D Mask from within the RT Struct
mask_3d = rtstruct.get_roi_mask_by_name("GTV")

# Display one slice of the region
first_mask_slice = mask_3d[:, :, 40]
plt.imshow(first_mask_slice)
plt.show()
plt.savefig("/mnt/c/Users/James/Documents/MPhys-Project/testRT.png")

# Converting array to sitk
print(mask_3d[:,200,40])
mask_3d_bin = mask_3d.astype(int)
print(mask_3d_bin[:,200,40])
img = sitk.GetImageFromArray(mask_3d_bin)



def resample_volume(volume, interpolator = sitk.sitkLinear):
  new_size = [512, 512, 256]
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(interpolator)
  resample.SetOutputDirection(volume.GetDirection())
  resample.SetOutputOrigin(volume.GetOrigin())
  resample.SetSize(new_size)
  resample.SetOutputSpacing([1, 1, 3])
  resample.SetDefaultPixelValue(-1024)

  return resample.Execute(volume)

x = resample_volume(img)
print(x)
sitk.WriteImage(x, f"{root}/test.nii")
