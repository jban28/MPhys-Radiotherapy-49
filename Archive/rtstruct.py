from matplotlib import image
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
import pydicom
import csv

# image_in is the original CT image
# image_out is resampled CT image
# mask_in is the original mask
# mask_out is resampled mask

def find_image_paths(root):
  # Creates a blank array which will be used to organise image file paths in the format [StudyUID, CT path, RTStruct Path]
  images = []
  path_list = []
  original_path_list = os.listdir(root+"Sorted_data")
  for patient in original_path_list:
    if 'HMR' in patient:
      path_list.append(patient)

  #path_list = path_list[0:9]
  print(path_list)

  for path in path_list:
    # Extract the first dicom file from each directory in the data folder
    filename = os.listdir(root+"Sorted_data/"+path)[0]
    dicom = pydicom.dcmread(root+"Sorted_data/"+path+"/"+filename)
    
    # Checks if the dicom is a CT or RTStruct, then adds those to the images array
    if (dicom.Modality == "CT") or (dicom.Modality == "RTSTRUCT"):
      if dicom.StudyInstanceUID not in [i[1] for i in images] and dicom.Modality == "CT":
        images.append([dicom.PatientID, dicom.StudyInstanceUID, path])
      elif dicom.StudyInstanceUID not in [i[1] for i in images] and dicom.Modality == "RTSTRUCT":
        images.append([dicom.PatientID, dicom.StudyInstanceUID, path+"/"+filename])
      elif dicom.StudyInstanceUID in [i[1] for i in images] and dicom.Modality == "CT":
        index = [i[1] for i in images].index(dicom.StudyInstanceUID)
        images[index].insert(2,path+"/"+filename)
      elif dicom.StudyInstanceUID in [i[1] for i in images] and dicom.Modality == "RTSTRUCT":
        index = [i[1] for i in images].index(dicom.StudyInstanceUID)
        images[index].append(path+"/"+filename)
      else:
        print("Error at Study "+dicom.StudyInstanceUID)

  return(images)


def resample_volume(volume, direction, origin, spacing, interpolator = sitk.sitkLinear, value=-1024):
  new_size = [512, 512, 512]
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(interpolator)
  resample.SetOutputDirection(direction)
  resample.SetOutputOrigin(origin)
  resample.SetSize(new_size)
  resample.SetOutputSpacing(spacing)
  resample.SetDefaultPixelValue(value)
  return resample.Execute(volume)

def permute_axes(volume, permutation_order) :
  # This function permutes the axes of the input volume.
  # It will be used on the mask because SimpleITK seems to flip the axes
  # at some stage in this process.

  permute = sitk.PermuteAxesImageFilter()
  permute.SetOrder(permutation_order)
  return permute.Execute(volume)

def full_resample(root, rtstruct, study_paths):
  
  # Loading the 3D Mask from within the RT Struct
  try:  
    mask_3d = rtstruct.get_roi_mask_by_name("GTV")
  except:
    try:
      mask_3d = rtstruct.get_roi_mask_by_name("GTV primaire")
    except:
      try:
        mask_3d = rtstruct.get_roi_mask_by_name("GTV p")
      except:
        try:
          mask_3d = rtstruct.get_roi_mask_by_name("GTVt")
        except:
          try:
            mask_3d = rtstruct.get_roi_mask_by_name("GTV Primaire 70")
          except:
            try:
              mask_3d = rtstruct.get_roi_mask_by_name("GTV T irm")
            except:
              return 'No GTV found'

  # Converting array to sitk
  mask_3d_bin = mask_3d.astype(np.float32)
  mask_in = sitk.GetImageFromArray(mask_3d_bin)
  mask_in = permute_axes(mask_in, [1,2,0])

  

  # Reads in image to SimpleITK
  reader = sitk.ImageSeriesReader()
  dcm_paths = reader.GetGDCMSeriesFileNames(root + "Sorted_data/"+study_paths[2])
  reader.SetFileNames(dcm_paths)
  image_in = reader.Execute()  
  image_out = resample_volume(image_in, image_in.GetDirection(), image_in.GetOrigin(), [1,1,1])

  mask_in.SetSpacing(image_in.GetSpacing())
  mask_in.SetDirection(image_in.GetDirection())
  mask_in.SetOrigin(image_in.GetOrigin())
  
  mask_out = resample_volume(mask_in, mask_in.GetDirection(), mask_in.GetOrigin(), [1,1,1], interpolator=sitk.sitkNearestNeighbor, value=0)

  # Creates new path for study
  output_path = root+"Nifti/"+study_paths[1]
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  # Saves images and masks as nii
  sitk.WriteImage(image_out, output_path + "/" + "image.nii")
  sitk.WriteImage(mask_out, output_path + "/" + "mask.nii")
  sitk.WriteImage(image_in, output_path + "/CToriginal.nii")
  sitk.WriteImage(mask_in, output_path + "/maskOG.nii")

  # Plots overlay of mask on image for one slice
  arrayRM = sitk.GetArrayFromImage(mask_out)[280, :, :]
  arrayRCT = sitk.GetArrayFromImage(image_out)[280, :, :]
  plt.imshow(arrayRCT, cmap = 'gray')
  plt.imshow(arrayRM, cmap = 'Reds', alpha = 0.5) # alpha sets opacity
  plt.savefig("Stacked")




#project_folder = "/mnt/c/Users/James/Documents/MPhys-Project/"
project_folder = "/mnt/c/Users/annaw/Documents/MPhys_Project/"

failed_resamples = []

print("Locating CT images")
path_list = find_image_paths(project_folder)
print(path_list)
x=0
failed_resamples = 0
for study in path_list:
  print("Resampling image ", (x+1), " of ", len(path_list))
  # Load existing RT Struct. Requires the series path and existing RT Struct path
  try: 
    image_builder = RTStructBuilder.create_from(
      dicom_series_path = project_folder + "Sorted_data/" + path_list[x][2],
      rt_struct_path = project_folder + "Sorted_data/"+ path_list[x][3]
    )
    try: 
      full_resample(project_folder, image_builder, study)
      print("Image resampled successfully")
      path_list[x].append("Succeeded")
      print(study)
    except Exception as e:
      path_list[x].append("Failed")
      failed_resamples += 1
      print("Resample failed")
      print(study)
      print(e)
  except Exception as e:
    path_list[x].append("Failed")
    failed_resamples += 1
    print("Resample failed")
    print(study)
    print(e)
  x = x + 1

print("Resampling successful on", len(path_list)-failed_resamples, "out of", len(path_list))

headings = ["Patient ID", "Study UID", "CT folder", "RTStruct file path", "Resample"] 
rows = path_list
with open(project_folder+'studies.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerow(headings) 
    write.writerows(rows) 

