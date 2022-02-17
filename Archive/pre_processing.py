import SimpleITK as sitk
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats

# root = sys.argv[1]
# root = "/mnt/c/Users/James/Documents/MPhys-Project/Sorted-data"
root = "/mnt/c/Users/annaw/Documents/MPhys_Project/Sorted_data2"

def CT_list(root_file):
  # find list of all directories which are 3D CTs
  all_folders = os.listdir(root_file)
  CT_folders = []
  for folder in all_folders:
    folder_path = root_file+"/"+folder
    if (len(os.listdir(folder_path)) > 2) and ("PT" not in folder_path) and ("PET" not in folder_path) and ("TEP" not in folder_path):
      CT_folders.append(root_file+"/"+folder)
  return CT_folders

def find_image_size(source):
  # Finds the dimensions of a single CT
  reader = sitk.ImageSeriesReader()
  dicom_names = reader.GetGDCMSeriesFileNames(source)
  reader.SetFileNames(dicom_names)
  image = reader.Execute()
  return image.GetSize()

def view_image(image):
  # Saves a png of a single Dicom
  data = sitk.ReadImage(image)
  data_2d = sitk.GetArrayViewFromImage(data)[0,:,:]
  #return data_2d
  plt.imshow(data_2d)
  plt.show()
  # plt.savefig("/mnt/c/Users/James/Documents/MPhys-Project/test.png")
  plt.savefig("/mnt/c/Users/annaw/Documents/MPhys_Project/test.png")

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

def WL_norm(img, window=40, level=80):
  """
  Apply window and level to image
  """

  maxval = level + window/2
  minval = level - window/2
  wl = sitk.IntensityWindowingImageFilter()
  wl.SetWindowMaximum(maxval)
  wl.SetWindowMinimum(minval)
  out = wl.Execute(img)
  return out

CT_folders = CT_list(root)

# image_size = find_image_size(CT_folders[0])
# print(image_size)

for filename in CT_folders:
  reader = sitk.ImageSeriesReader()
  dcm_paths = reader.GetGDCMSeriesFileNames(filename)  
  reader.SetFileNames(dcm_paths)
  volume = sitk.ReadImage(dcm_paths)
  x = resample_volume(volume)
  x = WL_norm(x)
  sitk.WriteImage(x, f"{root}/test.nii")
  break


mask_3d = rtstruct.get_roi_mask_by_name(root + "/" + "HN-CHUM-003-RTSTRUCT-PANC__avec_C_A___PRIMAIRE_-TP-18850827-111111/" + "1_RTstruct_CTsim-_CT_PET-CT_-UnknownInstanceNumber" )
plt.imshow(mask_3d)
plt.show()
