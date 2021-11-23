import os
import numpy as np
import SimpleITK as sitk

def permute_axes(volume, permutation_order) :
  # This function permutes the axes of the input volume.
  # It will be used on the mask because SimpleITK seems to flip the axes
  # at some stage in this process.

  permute = sitk.PermuteAxesImageFilter()
  permute.SetOrder(permutation_order)
  return permute.Execute(volume)

def cube_crop(folder, image_path, mask_path, CoM, cube_size):
  print("cropping")
  image_in = sitk.ReadImage(image_path)
  mask_in = sitk.ReadImage(mask_path)

  image_array = sitk.GetArrayFromImage(image_in)
  mask_array = sitk.GetArrayFromImage(mask_in)

  min_ = [int(CoM[0]-cube_size), int(CoM[1]-cube_size), int(CoM[2]-cube_size)]
  max_ = [int(CoM[0]+cube_size), int(CoM[1]+cube_size), int(CoM[2]+cube_size)]
  
  image_array = image_array[min_[0]:max_[0],min_[1]:max_[1],min_[2]:max_[2]]
  mask_array = mask_array[min_[0]:max_[0],min_[1]:max_[1],min_[2]:max_[2]]
  
  image_out = sitk.GetImageFromArray(image_array)
  mask_out = sitk.GetImageFromArray(mask_array)

  mask_out.SetDirection(mask_in.GetDirection())
  mask_out.SetOrigin(mask_in.GetOrigin())
  mask_out.SetSpacing(mask_in.GetSpacing())

  image_out.SetDirection(image_in.GetDirection())
  image_out.SetOrigin(image_in.GetOrigin())
  image_out.SetSpacing(image_in.GetSpacing())

  output_path = project_folder + "Nifti_crop/" + folder
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  sitk.WriteImage(image_out, output_path + "/image_crop.nii")
  sitk.WriteImage(mask_out, output_path + "/mask_crop.nii")
  

project_folder = "/mnt/c/Users/James/Documents/MPhys-Project/"
#project_folder = "/mnt/c/Users/annaw/Documents/MPhys_Project/"

image_folders = os.listdir(project_folder + "Nifti/")

image_list = []

max_distance = 0

for folder in image_folders:
  mask_path = project_folder + "Nifti/" + folder + "/mask.nii"
  image_path = project_folder + "Nifti/" + folder + "/image.nii"

  # read in mask as array
  mask = sitk.ReadImage(mask_path)
  image = sitk.ReadImage(image_path)
  

  mask_array = sitk.GetArrayFromImage(mask)
  # find location of all non zero points in mask, and find average position in x, y, and z, which is CoM
  non_zeros = np.argwhere(mask_array)
  CoM_x = np.average(non_zeros[:,0])
  CoM_y = np.average(non_zeros[:,1])
  CoM_z = np.average(non_zeros[:,2])
  CoM = [CoM_x, CoM_y, CoM_z]


  # find maximum distance in single dimension of mask point from CoM
  for voxel in non_zeros:
    displacement = [abs(voxel[0]-CoM_x), abs(voxel[1]-CoM_y), abs(voxel[2]-CoM_z)]
    if max(displacement) > max_distance:
      max_distance = max(displacement)
      

  # Add CoM to list of masks
  image_list.append([folder, image_path, mask_path, CoM])

i = 0
cube_crop(image_list[i][0], image_list[i][1], image_list[i][2], image_list[i][3], int(max_distance+15))
