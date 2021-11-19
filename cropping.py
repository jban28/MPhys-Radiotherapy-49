import os
import numpy as np
import SimpleITK as sitk

<<<<<<< HEAD
def cube_crop(folder, image_path, mask_path, CoM, cube_size):
  print("cropping")
  image = sitk.ReadImage(image_path)
  mask = sitk.ReadImage(mask_path)
  image_array = sitk.GetArrayFromImage(image)
  mask_array = sitk.GetArrayFromImage(mask)
  min_ = [int(CoM[0]-cube_size), int(CoM[1]-cube_size), int(CoM[2]-cube_size)]
  max_ = [int(CoM[0]+cube_size), int(CoM[1]+cube_size), int(CoM[2]+cube_size)]
  
  image_array = image_array[min_[0]:max_[0],min_[1]:max_[1],min_[2]:max_[2]]
  mask_array = mask_array[min_[0]:max_[0],min_[1]:max_[1],min_[2]:max_[2]]
  
  image = sitk.GetImageFromArray(image_array)
  mask = sitk.GetImageFromArray(mask_array)

  output_path = project_folder + "Nifti_crop/" + folder
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  sitk.WriteImage(image, output_path + "/image_crop.nii")
  sitk.WriteImage(mask, output_path + "/mask_crop.nii")
  

=======
>>>>>>> e004e2c1b649c4e8e8c11472692d2124703fc5d1
project_folder = "/mnt/c/Users/James/Documents/MPhys-Project/"
# project_folder "/mnt/C:/Users/annaw/Documents/MPhys_Project/"

image_folders = os.listdir(project_folder + "Nifti/")

<<<<<<< HEAD
image_list = []
=======
mask_list = []
>>>>>>> e004e2c1b649c4e8e8c11472692d2124703fc5d1

max_distance = 0

for folder in image_folders:
  mask_path = project_folder + "Nifti/" + folder + "/mask.nii"
<<<<<<< HEAD
  image_path = project_folder + "Nifti/" + folder + "/image.nii"

  # read in mask as array, and remove trvial 4th dimension (SimpleITK effectively reads a 3d nii as a single 4d slice)
  mask = sitk.ReadImage(mask_path)
  mask_array = sitk.GetArrayFromImage(mask)
=======
  mask_list.append([mask_path])

n=0

for mask_path in mask_list:
  # read in mask as array, and remove trvial 4th dimension (SimpleITK effectively reads a 3d nii as a single 4d slice)
  mask = sitk.ReadImage(mask_path)
  mask_array = sitk.GetArrayFromImage(mask)
  mask_array = mask_array[0]

>>>>>>> e004e2c1b649c4e8e8c11472692d2124703fc5d1
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
<<<<<<< HEAD
  image_list.append([folder, image_path, mask_path, CoM])

i = 0
cube_crop(image_list[i][0], image_list[i][1], image_list[i][2], image_list[i][3], int(max_distance+15))
=======
  mask_list[n].append(CoM)
  n = n + 1

print(max_distance)
>>>>>>> e004e2c1b649c4e8e8c11472692d2124703fc5d1
