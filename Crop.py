from curses import meta
import os
import sys
import csv
import scipy
import numpy as np
import SimpleITK as sitk
import scipy.ndimage.morphology as pad_mask


from datetime import datetime

# Specify root folder for the project, where all images and data will be stored 
# and where the TCIA file is downloaded to
project_folder = sys.argv[1]
padding = int(sys.argv[2])

date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

def crop(array, CoM, size):
  # Define dimensions to crop to
  min_ = [int(CoM[0]-size), int(CoM[1]-size), int(CoM[2]-size)]
  max_ = [int(CoM[0]+size), int(CoM[1]+size), int(CoM[2]+size)]

  # Crop array
  array = array[min_[0]:max_[0],min_[1]:max_[1],min_[2]:max_[2]]

  # Return cropped array
  return array

# Open the metadata.csv file, convert to an array, and remove column headers
metadata_file = open(project_folder + "/metadata_resample.csv")
metadata = np.loadtxt(metadata_file, dtype="str", delimiter=",")
metadata = metadata[1:][:]


# Creates new path for cropped images if one does not already exist
if not os.path.exists(project_folder+"/crop_"+str(date)):
  os.makedirs(project_folder+"/crop_"+str(date))
  os.makedirs(project_folder+"/crop_"+str(date)+"/Images")


patient_data = []
errors = []

# Define size of crop based on maximum tumour size
cube_size = 0
for patient in metadata:
  if float(patient[12]) > 90:
    continue
  else:
    patient_data.append(patient)
  if float(patient[12]) > cube_size:
    cube_size = int(float(patient[12]))

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

  # Add padding to mask
  while padding > 0:
    mask_array = pad_mask.binary_dilation(mask_array)
    padding -= 1

  # Crop image and mask
  try:
    CoM = (float(patient[9]),float(patient[10]),float(patient[11]))
    image_array = crop(image_array, CoM, cube_size)
    mask_array = crop(mask_array, CoM, cube_size)
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
    sitk.WriteImage(masked_image_out, project_folder + "/crop_" + str(date) 
    + "/Images/" + patient[0] + ".nii")
  except:
    print("Could not write cropped image for patient " + patient[0])
    errors.append([patient[0], "Failed to write cropped image"])
    continue

print("Cropping completed with ", len(errors), "errors")
with open(project_folder + "crop_errors_" + str(date) + ".csv", "w") as f:
    write = csv.writer(f) 
    write.writerows(errors) 