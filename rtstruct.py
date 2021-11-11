from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# image_in is the original CT image
# image_out is resampled CT image
# mask_in is the original mask
# mask_out is resampled mask

def resample_volume(volume, spacing, interpolator = sitk.sitkLinear, value=-1024):
  new_size = [512, 512, 512]
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(interpolator)
  resample.SetOutputDirection(volume.GetDirection())
  resample.SetOutputOrigin(volume.GetOrigin())
  resample.SetSize(new_size)
  resample.SetOutputSpacing(spacing)
  resample.SetDefaultPixelValue(value)
  return resample.Execute(volume)

def full_resample(root, rtstruct):
  # View all of the ROI names from within the image
  print(rtstruct.get_roi_names())

  # Loading the 3D Mask from within the RT Struct
  mask_3d = rtstruct.get_roi_mask_by_name("GTV")

  # Converting array to sitk
  # mask_3d_bin = mask_3d.astype(float)
  mask_3d_bin = mask_3d.astype(np.float32)
  mask_3d_bin = np.rot90(mask_3d_bin, 1, (0,2))
  mask_in = sitk.GetImageFromArray(mask_3d_bin)

  # Reads in image to SimpleITK
  reader = sitk.ImageSeriesReader()
  dcm_paths = reader.GetGDCMSeriesFileNames(root + 'Sorted_data/HN-CHUM-003-CT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111')
  reader.SetFileNames(dcm_paths)
  image_in = reader.Execute()
    
  image_out = resample_volume(image_in, [1,1,1]) #mask

  # voxel_spacing = [(1/image_in.GetSpacing()[0]), (1/image_in.GetSpacing()[1]), (1/image_in.GetSpacing()[2])]
  print(mask_in.GetDirection(), mask_in.GetOrigin(), mask_in.GetSpacing())
  mask_in.SetDirection(image_out.GetDirection())
  mask_in.SetOrigin(image_out.GetOrigin())
  mask_in.SetSpacing(image_in.GetSpacing())
  print(mask_in.GetDirection(), mask_in.GetOrigin(), mask_in.GetSpacing())
  mask_out = resample_volume(mask_in, [1,1,1], interpolator=sitk.sitkNearestNeighbor, value=0)

  # Saves images and masks as nii
  sitk.WriteImage(image_out, root + "test.nii")
  sitk.WriteImage(mask_out, root + "test_mask.nii")
  sitk.WriteImage(image_in, root + "CToriginal.nii")
  sitk.WriteImage(mask_in, root + "maskOG.nii")

  # Plots overlay of mask on image for one slice
  arrayRM = sitk.GetArrayFromImage(mask_out)[280, :, :]
  arrayRCT = sitk.GetArrayFromImage(image_out)[280, :, :]
  plt.imshow(arrayRCT, cmap = 'gray')
  plt.imshow(arrayRM, cmap = 'Reds', alpha = 0.5) # alpha sets opacity
  plt.savefig("Stacked")



# root = "/mnt/c/Users/James/Documents/MPhys-Project/"
project_folder = "/mnt/c/Users/James/Documents/MPhys-Project/"
# project_folder "/mnt/C:/Users/annaw/Documents/MPhys_Project/"

path_list = [["HN-CHUM-003-CT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111","HN-CHUM-003-RTSTRUCT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111/1_RTstruct_CTsim-_CT_PET-CT_-UnknownInstanceNumber.dcm"],
["HN-CHUS-001-CT-CA_ORL-18850827-111111","HN-CHUS-001-RTSTRUCT-UnknownStudyDescription-18850827-111111/None_Pinnacle_POI-UnknownInstanceNumber.dcm"]]


# Load existing RT Struct. Requires the series path and existing RT Struct path
patient_number = 0
image_builder = RTStructBuilder.create_from(
  dicom_series_path = project_folder + "Sorted_data/" + path_list[patient_number][0],
  rt_struct_path = project_folder + "Sorted_data/"+ path_list[patient_number][1]
) 

full_resample(project_folder, image_builder)