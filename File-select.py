import os
import pydicom

project_folder = "/mnt/c/Users/James/Documents/MPhys-Project/"
images = []
# chum_list = []
# chus_list = []
# hgj_list = []
# hmr_list = []

path_list = os.listdir(project_folder+"Sorted_data")

for path in path_list:
  dicom = pydicom.dcmread(project_folder+"Sorted_data/"+path+"/"+os.listdir(project_folder+"Sorted_data/"+path)[0])
  print(dicom.Modality)
  # if (dicom.Modality() != "CT") and (dicom.Modality() != "RTSTRUCT"):
  #   path_list.remove(path)

print(path_list)

# for file in paths:
#   if file[:7] == "HN-CHUM":
#     chum_list.append(file)
#   elif file[:7] == "HN-CHUS":
#     chus_list.append(file)
#   elif file[:6] == "HN-HGJ":
#     hgj_list.append(file)
#   elif file[:6] == "HN-HMR":
#     hmr_list.append(file)

# for chum in chum_list:

