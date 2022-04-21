import SimpleITK as sitk
from Outcomes import load_metadata
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

project_folder = "/data/James_Anna"
subfolder = "crop_2022_03_29-16_08_02"

metadata = load_metadata(project_folder, subfolder)

# Find the size of the images being read in
image_sitk = sitk.ReadImage(project_folder + "/" + subfolder + "/Images/" + 
metadata[0][0] + ".nii")
image = sitk.GetArrayFromImage(image_sitk)
image_dimension = image.shape[0]

print(metadata[0][0])
patient_to_load = [[metadata[0][0], 1]]

# original
data = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=False, 
shift_augment=False, cube_size=image_dimension)
dataloader = DataLoader(data, 1, shuffle=False)
for (X,y, patient) in dataloader:
  plt.imshow(X[0,:,102,:], cmap='gray')
  plt.savefig("Augmentations/original.png")

# Rotate
data = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=True, scale_augment=False, flip_augment=False, 
shift_augment=False, cube_size=image_dimension)
dataloader = DataLoader(data, 1, shuffle=False)
for (X,y, patient) in dataloader:
  plt.imshow(X[0,:,102,:], cmap='gray')
  plt.savefig("Augmentations/rotate.png")

# Scale
data = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=True, flip_augment=False, 
shift_augment=False, cube_size=image_dimension)
dataloader = DataLoader(data, 1, shuffle=False)
for (X,y, patient) in dataloader:
  plt.imshow(X[0,:,102,:], cmap='gray')
  plt.savefig("Augmentations/scale.png")

# Flip
data = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=True, 
shift_augment=False, cube_size=image_dimension)
dataloader = DataLoader(data, 1, shuffle=False)
for (X,y, patient) in dataloader:
  plt.imshow(X[0,:,102,:], cmap='gray')
  plt.savefig("Augmentations/flip.png")

# shift
data = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=False, 
shift_augment=True, cube_size=image_dimension)
dataloader = DataLoader(data, 1, shuffle=False)
for (X,y, patient) in dataloader:
  plt.imshow(X[0,:,102,:], cmap='gray')
  plt.savefig("Augmentations/shift.png")