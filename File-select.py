import os

project_folder = "/mnt/c/Users/James/Documents/MPhys-Project/"
images = []
chum_list = []
chus_list = []
hgj_list = []
hmr_list = []

paths = os.listdir(project_folder+"Sorted_data")

for file in paths:
  if file[:7] == "HN-CHUM":
    chum_list.append(file)
  elif file[:7] == "HN-CHUS":
    chus_list.append(file)
  elif file[:6] == "HN-HGJ":
    hgj_list.append(file)
  elif file[:6] == "HN-HMR":
    hmr_list.append(file)

for chum in chum_list:
  
