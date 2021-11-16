from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
import pydicom

# image_in is the original CT image
# image_out is resampled CT image
# mask_in is the original mask
# mask_out is resampled mask

def find_image_paths(root):
  # Creates a blank array which will be used to organise image file paths in the format [StudyUID, CT path, RTStruct Path]
  images = []

  path_list = os.listdir(root+"Sorted_data")
  path_list = path_list[0:90]

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

  return(images)


def resample_volume(volume, spacing, interpolator = sitk.sitkLinear, value=-1024):
  new_size = [512, 512, 512]
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(interpolator)
  resample.SetOutputDirection(volume.GetDirection())
  resample.SetOutputOrigin(volume.GetOrigin())
  resample.SetSize(new_size)
  resample.SetOutputSpacing(spacing)
  resample.SetDefaultPixelValue(value)
  return resample.Execute(volume)

def full_resample(root, rtstruct, study_paths):
  # View all of the ROI names from within the image
  # print(rtstruct.get_roi_names())

  # Loading the 3D Mask from within the RT Struct
  mask_3d = rtstruct.get_roi_mask_by_name("GTV")

  # Converting array to sitk
  # mask_3d_bin = mask_3d.astype(float)
  mask_3d_bin = mask_3d.astype(np.float32)
  mask_in = sitk.GetImageFromArray(mask_3d_bin)

  # Reads in image to SimpleITK
  reader = sitk.ImageSeriesReader()
  dcm_paths = reader.GetGDCMSeriesFileNames(root + "Sorted_data/"+study_paths[1])
  reader.SetFileNames(dcm_paths)
  image_in = reader.Execute()
    
  image_out = resample_volume(image_in, [1,1,1]) #mask

  # voxel_spacing = [(1/image_in.GetSpacing()[0]), (1/image_in.GetSpacing()[1]), (1/image_in.GetSpacing()[2])]
  # print(mask_in.GetDirection(), mask_in.GetOrigin(), mask_in.GetSpacing())
  mask_in.SetDirection(image_out.GetDirection())
  mask_in.SetOrigin(image_out.GetOrigin())
  mask_in.SetSpacing(image_in.GetSpacing())
  # print(mask_in.GetDirection(), mask_in.GetOrigin(), mask_in.GetSpacing())
  mask_out = resample_volume(mask_in, [1,1,1], interpolator=sitk.sitkNearestNeighbor, value=0)

  # Creates new path for study
  output_path = root+"Nifti/"+study_paths[0]
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  # Saves images and masks as nii
  sitk.WriteImage(image_out, output_path + "/" + "image.nii")
  sitk.WriteImage(mask_out, output_path + "/" + "mask.nii")
  # sitk.WriteImage(image_in, root + "CToriginal.nii")
  # sitk.WriteImage(mask_in, root + "maskOG.nii")

  # Plots overlay of mask on image for one slice
  arrayRM = sitk.GetArrayFromImage(mask_out)[280, :, :]
  arrayRCT = sitk.GetArrayFromImage(image_out)[280, :, :]
  plt.imshow(arrayRCT, cmap = 'gray')
  plt.imshow(arrayRM, cmap = 'Reds', alpha = 0.5) # alpha sets opacity
  plt.savefig("Stacked")




project_folder = "/mnt/c/Users/James/Documents/MPhys-Project/"
# project_folder "/mnt/C:/Users/annaw/Documents/MPhys_Project/"

failed_resamples = []

print("Locating CT images")
path_list = find_image_paths(project_folder)

x=0
for study in path_list:
  print("Resampling image ", (x+1), " of ", len(path_list))
  # Load existing RT Struct. Requires the series path and existing RT Struct path
  try: 
    image_builder = RTStructBuilder.create_from(
      dicom_series_path = project_folder + "Sorted_data/" + path_list[x][1],
      rt_struct_path = project_folder + "Sorted_data/"+ path_list[x][2]
    )
    try: 
      full_resample(project_folder, image_builder, study)
      print("Image resampled successfully")
    except:
      failed_resamples.append(path_list[x][0])
      print("Resample failed")
  except:
    failed_resamples.append(path_list[x][0])
    print("Resample failed")
  
  x = x + 1

print("Resampling successful on", len(path_list)-len(failed_resamples), "out of", len(path_list))