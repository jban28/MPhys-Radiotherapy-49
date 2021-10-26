import SimpleITK as sitk
import sys
import os

# find list of all directories which are 3D CTs
all_folders = os.listdir(sys.argv[1])
CT_folders = []
for folder in all_folders:
  folder_path = sys.argv[1]+"/"+folder
  if (len(os.listdir(folder_path)) > 2) and ("PT" not in folder_path) and ("PET" not in folder_path) and ("TEP" not in folder_path):
    CT_folders.append(folder)

print(CT_folders)

"""
print("Reading Dicom directory:", sys.argv[1])
reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames(sys.argv[1])
reader.SetFileNames(dicom_names)


image = reader.Execute()

size = image.GetSize()

print(size)
"""