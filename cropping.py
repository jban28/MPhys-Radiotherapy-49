import os
import numpy as np
import SimpleITK as sitk

project_folder = "/mnt/c/Users/James/Documents/MPhys-Project/"
# project_folder "/mnt/C:/Users/annaw/Documents/MPhys_Project/"

image_folders = os.listdir(project_folder + "Nifti/")

mask_list = []

max_distance = 0

for folder in image_folders:
  mask_path = project_folder + "Nifti/" + folder + "/mask.nii"
  mask_list.append([mask_path])

n=0

for mask_path in mask_list:
  # read in mask as array, and remove trvial 4th dimension (SimpleITK effectively reads a 3d nii as a single 4d slice)
  mask = sitk.ReadImage(mask_path)
  mask_array = sitk.GetArrayFromImage(mask)
  mask_array = mask_array[0]

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
  mask_list[n].append(CoM)
  n = n + 1

print(max_distance)