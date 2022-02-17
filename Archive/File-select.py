import os
import pydicom

# Creates a blank array which will be used to organise image file paths in the format [StudyUID, CT path, RTStruct Path]
images = []

root = "/mnt/c/Users/James/Documents/MPhys-Project/"
path_list = os.listdir(root+"Sorted_data")
path_list = path_list[:100]

for path in path_list:
  # Extract the first dicom file from each directory in the data folder
  filename = os.listdir(root+"Sorted_data/"+path)[0]
  dicom = pydicom.dcmread(root+"Sorted_data/"+path+"/"+filename)
  
  # Checks if the dicom is a CT or RTStruct, then adds those to the images array
  if (dicom.Modality == "CT") or (dicom.Modality == "RTSTRUCT"):
    if dicom.StudyInstanceUID not in [i[0] for i in images] and dicom.Modality == "CT":
      images.append([dicom.StudyInstanceUID, path])
    elif dicom.StudyInstanceUID not in [i[0] for i in images] and dicom.Modality == "RTSTRUCT":
      images.append([dicom.StudyInstanceUID, path+"/"+filename])
    elif dicom.StudyInstanceUID in [i[0] for i in images] and dicom.Modality == "CT":
      index = [i[0] for i in images].index(dicom.StudyInstanceUID)
      images[index].insert(1,path+"/"+filename)
    elif dicom.StudyInstanceUID in [i[0] for i in images] and dicom.Modality == "RTSTRUCT":
      index = [i[0] for i in images].index(dicom.StudyInstanceUID)
      images[index].append(path+"/"+filename)
    else:
      print("Error at Study "+dicom.StudyInstanceUID)


print(images)

