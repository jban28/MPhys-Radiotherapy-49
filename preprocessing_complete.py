from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
import csv

# Specify root folder for the project, where all images and data will be stored 
# and where the TCIA file is downloaded to
project_folder = "/mnt/f/MPhys"

# Specify the name of the manifest file where the dicoms are extracted to
manifest = "manifest-VpKfQUDr2642018792281691204"

def permute_axes(volume, permutation_order) :
  permute = sitk.PermuteAxesImageFilter()
  permute.SetOrder(permutation_order)
  return permute.Execute(volume)

def resample_volume(volume, direction, origin, spacing = [1,1,1], 
interpolator = sitk.sitkLinear, value=-1024):
  new_size = [512, 512, 512]
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(interpolator)
  resample.SetOutputDirection(direction)
  resample.SetOutputOrigin(origin)
  resample.SetSize(new_size)
  resample.SetOutputSpacing(spacing)
  resample.SetDefaultPixelValue(value)
  return resample.Execute(volume)

def crop(array, CoM, size):
  # Define dimensions to crop to
  min_ = [int(CoM[0]-size), int(CoM[1]-size), int(CoM[2]-size)]
  max_ = [int(CoM[0]+size), int(CoM[1]+size), int(CoM[2]+size)]

  # Crop array
  array = array[min_[0]:max_[0],min_[1]:max_[1],min_[2]:max_[2]]

  # Return cropped array
  return array

# Open the metadata.csv file, convert to an array, and remove column headers
metadata_file = open(project_folder + "/metadata.csv")
metadata = np.loadtxt(metadata_file, dtype="str", delimiter=",")
metadata = metadata[1:][:]

# Creates new path for resampled images if one does not already exist
if not os.path.exists(project_folder+"/resample"):
  os.makedirs(project_folder+"/resample")

# Creates new path for cropped images if one does not already exist
if not os.path.exists(project_folder+"/crop"):
  os.makedirs(project_folder+"/crop")

# Create an empty list to store patient data in. This will eventually be a copy 
# of the metadata array, but with extra elements for the tumour CoM and maximum 
# distance of tumour from CoM
patient_data = []

# Create an empty array to log errors
errors = []

for patient in metadata:
  print("Resampling patient " + patient[0])
  # Convert patient from np array to list
  patient = patient.tolist()

  # Build image and mask from their respective dicom series
  try: 
    image_builder = RTStructBuilder.create_from(
      dicom_series_path = project_folder + "/" + manifest + patient[2],
      rt_struct_path = project_folder + "/" + manifest + patient[3] + "/" +
      os.listdir(project_folder + "/" + manifest + patient[3])[0]
    )
  except:
    print("Unable to build image for " + patient[0])
    errors.append([patient[0], "Unable to build image"])
    continue
  
  # Read in image to SimpleITK
  try: 
    reader = sitk.ImageSeriesReader()
    dcm_paths = reader.GetGDCMSeriesFileNames(project_folder + "/" + manifest + 
    "/" + patient[2])
    reader.SetFileNames(dcm_paths)
    image_in = reader.Execute()
  except:
    print("Could not extract SimpleITK image from dicom series for patient " + 
    patient[0])
    errors.append([patient[0], 
    "Could not extract SimpleITK image from dicom series"])
    continue

  # Read in mask as an array and convert to SimpleITK image
  try: 
    mask_3d = image_builder.get_roi_mask_by_name(patient[4])
    mask_3d_bin = mask_3d.astype(np.float32)
    mask_in = sitk.GetImageFromArray(mask_3d_bin)
  except:
    print("Failed to read in mask for patient " + patient[0])
    errors.append([patient[0], "Failed to read mask"])
    continue

  # Set parameters of SimpleITK mask
  mask_in = permute_axes(mask_in, [1,2,0])
  mask_in.SetSpacing(image_in.GetSpacing())
  mask_in.SetDirection(image_in.GetDirection())
  mask_in.SetOrigin(image_in.GetOrigin())

  # Resample the image and mask
  try:
    image_out = resample_volume(image_in, image_in.GetDirection(), 
    image_in.GetOrigin(), [1,1,1])  
    mask_out = resample_volume(mask_in, mask_in.GetDirection(), 
    mask_in.GetOrigin(), [1,1,1], interpolator=sitk.sitkNearestNeighbor, value=0)
  except:
    print("Resample failed for patient " + patient[0])
    errors.append([patient[0], "Resample failed"])
    continue
    
  # Convert mask back to array, find CoM of tumour and add to metadata
  mask_array = sitk.GetArrayFromImage(mask_out)
  non_zeros = np.argwhere(mask_array)
  CoM_x = np.average(non_zeros[:,0])
  CoM_y = np.average(non_zeros[:,1])
  CoM_z = np.average(non_zeros[:,2])
  CoM = [CoM_x, CoM_y, CoM_z]
  patient.append(CoM)

  # Find maximum distance of tumour from CoM and add to metadata
  max_distance = 0
  for vox in non_zeros:
    displacement = [abs(vox[0]-CoM_x), abs(vox[1]-CoM_y), abs(vox[2]-CoM_z)]
    if max(displacement) > max_distance:
      max_distance = max(displacement)
  patient.append(max_distance)

  # Write image and mask to resample folder
  try:
    if not os.path.exists(project_folder + "/resample/" + patient[0]):
      os.makedirs(project_folder + "/resample/" + patient[0])
    sitk.WriteImage(image_out, project_folder + "/resample/" + patient[0] + 
    "/image.nii")
    sitk.WriteImage(mask_out, project_folder + "/resample/" + patient[0] + 
    "/mask.nii")
  except:
    print("Could not write to file for patient " + patient[0])
    errors.append([patient[0], "File write failed"])
    continue

  # Adds all metadata (inlcuding newly calculated CoM and max_distance) to list 
  # of patient metadata
  patient_data.append(patient)

# Define size of crop based on maximum tumour size
cube_size = 0
for patient in patient_data:
  if patient[10] > cube_size:
    cube_size = int(patient[10])

# Add padding to crop of 15 pixels
cube_size += 15

# Crop each image and mask to above size and output masked image
for patient in patient_data:
  print("Cropping patient " + patient[0])
  # Read in as SimpleITK
  image_in = sitk.ReadImage(project_folder + "/resample/" + patient[0] + 
  "/image.nii")
  mask_in = sitk.ReadImage(project_folder + "/resample/" + patient[0] + 
  "/mask.nii")

  # Convert to arrays
  image_array = sitk.GetArrayFromImage(image_in)
  mask_array = sitk.GetArrayFromImage(mask_in)

  # Crop image and mask
  try:
    image_array = crop(image_array, patient[9], cube_size)
    mask_array = crop(mask_array, patient[9], cube_size)
  except:
    print("Cropping failed for patient " + patient[0])
    errors.append([patient[0], "Cropping failed"])
    continue

  # Produce array for masked image and convert back to SimpleITK
  masked_image_array = np.multiply(image_array+1024, mask_array)-1024
  masked_image_out = sitk.GetImageFromArray(masked_image_array)

  # Define image properties based on input image
  masked_image_out.SetDirection(mask_in.GetDirection())
  masked_image_out.SetOrigin(mask_in.GetOrigin())
  masked_image_out.SetSpacing(mask_in.GetSpacing())

  # Write masked image to file
  try:
    sitk.WriteImage(masked_image_out, project_folder + "/crop/" + patient[0] + 
    ".nii")
  except:
    print("Could not write cropped image for patient " + patient[0])
    errors.append([patient[0], "Failed to write cropped image"])
    continue


print("Processing completed with ", len(errors), "errors")
with open(project_folder + '/Error Log.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(errors) 