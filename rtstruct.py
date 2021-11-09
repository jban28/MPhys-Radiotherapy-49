import pydicom
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import sys
import os
from scipy import stats

# root = "/mnt/c/Users/James/Documents/MPhys-Project"
root = "/mnt/C:/Users/annaw/Documents/MPhys_Project"
# Load existing RT Struct. Requires the series path and existing RT Struct path
rtstruct = RTStructBuilder.create_from(
  # dicom_series_path="/mnt/c/Users/James/Documents/MPhys-Project/Sorted-data/HN-CHUM-003-CT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111", 
  # rt_struct_path="/mnt/c/Users/James/Documents/MPhys-Project/Sorted-data/HN-CHUM-003-RTSTRUCT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111/1_RTstruct_CTsim-_CT_PET-CT_-UnknownInstanceNumber.dcm"
  dicom_series_path = "/mnt/c/Users/annaw/Documents/MPhys_Project/Sorted_data2/HN-CHUM-003-CT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111",
  rt_struct_path = "/mnt/c/Users/annaw/Documents/MPhys_Project/Sorted_data2/HN-CHUM-003-RTSTRUCT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111/1_RTstruct_CTsim-_CT_PET-CT_-UnknownInstanceNumber.dcm"
) #/1_RTstruct_CTsim-_CT_PET-CT_-UnknownInstanceNumber.dcm

# View all of the ROI names from within the image
print(rtstruct.get_roi_names())

# Loading the 3D Mask from within the RT Struct
mask_3d = rtstruct.get_roi_mask_by_name("GTV")



# Converting array to sitk

mask_3d_bin = mask_3d.astype(float)
mask_3d_bin = np.rot90(mask_3d_bin, 1, (0,2))
print(mask_3d_bin.dtype)
img = sitk.GetImageFromArray(mask_3d_bin)

print(img.GetSize())
print('Pixel ID Value is ')
#print(img.GetSpacing())
#ogsizes = img.GetSize()
#newsizes = 



# img is the original mask
# volume is the original CT image
def resample_volume(volume, spacing, direction, origin, interpolator = sitk.sitkLinear, value=-1024):
  new_size = [512, 512, 512]
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(interpolator)
  resample.SetOutputDirection(volume.GetDirection())
  resample.SetOutputOrigin(volume.GetOrigin())
  resample.SetSize(new_size)
  resample.SetOutputSpacing(spacing)
  resample.SetDefaultPixelValue(value)
# problem is that we dont know the mask pixel size, so it assumes pixel sizes in 1 by 1 by 1
# so when it resamples, it doesn't do anything
# think of images as objects
  return resample.Execute(volume)

reader = sitk.ImageSeriesReader()
dcm_paths = reader.GetGDCMSeriesFileNames('/mnt/c/Users/annaw/Documents/MPhys_Project/Sorted_data2/HN-CHUM-003-CT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111')
reader.SetFileNames(dcm_paths)
volume = reader.Execute()

print(volume.GetSpacing()[1])
voxel_spacing = [(1/volume.GetSpacing()[0]), (1/volume.GetSpacing()[1]), (1/volume.GetSpacing()[2])]

#RESAMPLED CT   x is resampled CT image
x = resample_volume(volume, [1,1,1], volume.GetDirection(), volume.GetOrigin()) #mask
#print("THis is x")
#print(x)
sitk.WriteImage(x, f"./test.nii")
#print("IMAGE IS...............")
#print(img)

# RESAMPLED MASK     image is resampled mask
image = resample_volume(img, voxel_spacing ,volume.GetDirection(), volume.GetOrigin(), interpolator=sitk.sitkNearestNeighbor, value=0)
sitk.WriteImage(image, f"./test_mask.nii")
#print(type(img))
# image_nii = sitk.GetImageFromArray(img)
# Display one slice of the region
first_mask_slice = sitk.GetArrayFromImage(image)[:, :, 40]
plt.imshow(first_mask_slice)
# plt.savefig("/mnt/c/Users/James/Documents/MPhys-Project/testRT.png")
plt.savefig("testRT.png")

sitk.WriteImage(volume, f"./CToriginal.nii")
sitk.WriteImage(img, f"./maskOG.nii")

arrayRM = sitk.GetArrayFromImage(image)[:,:,213]
arrayRCT = sitk.GetArrayFromImage(x)[:,:,213]
plt.imshow(arrayRM)
plt.imshow(arrayRCT)
plt.savefig("Stacked")