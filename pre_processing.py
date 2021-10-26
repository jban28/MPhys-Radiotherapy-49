# %% 

import SimpleITK as sitk
import matplotlib.pyplot as plt
import sys
import os

def CT_list(root_file):
  # find list of all directories which are 3D CTs
  all_folders = os.listdir(root_file)
  CT_folders = []
  for folder in all_folders:
    folder_path = root_file+"/"+folder
    if (len(os.listdir(folder_path)) > 2) and ("PT" not in folder_path) and ("PET" not in folder_path) and ("TEP" not in folder_path):
      CT_folders.append(folder)
  return CT_folders

def find_image_size(source):
  reader = sitk.ImageSeriesReader()
  dicom_names = reader.GetGDCMSeriesFileNames(source)
  reader.SetFileNames(dicom_names)
  image = reader.Execute()
  print(image.GetSize())

def view_image(image):
  data = sitk.ReadImage(image)
  data_2d = sitk.GetArrayViewFromImage(data)[0,:,:]
  #return data_2d
  plt.imshow(data_2d)
  plt.show()
  plt.savefig("/mnt/c/Users/James/Documents/MPhys-Project/test.png")

"""
# CT_folders = CT_list(sys.argv[1])
CT_folders = [sys.argv[1]]

# finds sizes of each CT image
for folder in CT_folders:
  folder_path = folder
  find_image_size(folder_path)
"""

# plt.imshow(view_image(sys.argv[1]))
# plt.show()

view_image(sys.argv[1])
# %%
