import SimpleITK as sitk
from Outcomes import load_metadata
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = "serif"

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

print(image_dimension)


index = 0

# original
dataset = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=False, 
shift_augment=False, cube_size=image_dimension)

print(dataset[index][0].numpy())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
array = dataset[index][0].numpy()
x,y,z = np.where(array > 0)
ax.scatter(x, y, z, c=z, alpha=1)
ax.set_xlim(0,204)
ax.set_ylim(0,204)
ax.set_zlim(0,204)
plt.savefig("Augmentations/original.png",bbox_inches='tight')

# Rotate
dataset = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=True, scale_augment=False, flip_augment=False, 
shift_augment=False, cube_size=image_dimension)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
array = dataset[index][0].numpy()
x,y,z = np.where(array > 0)
ax.scatter(x, y, z, c=z, alpha=1)
ax.set_xlim(0,204)
ax.set_ylim(0,204)
ax.set_zlim(0,204)
plt.savefig("Augmentations/rotate.png",bbox_inches='tight')

# Scale
dataset = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=True, flip_augment=False, 
shift_augment=False, cube_size=image_dimension)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
array = dataset[index][0].numpy()
x,y,z = np.where(array > 0)
ax.scatter(x, y, z, c=z, alpha=1)
ax.set_xlim(0,204)
ax.set_ylim(0,204)
ax.set_zlim(0,204)
plt.savefig("Augmentations/scale.png",bbox_inches='tight')

# Flip
dataset = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=True, 
shift_augment=False, cube_size=image_dimension)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
array = dataset[index][0].numpy()
x,y,z = np.where(array > 0)
ax.scatter(x, y, z, c=z, alpha=1)
ax.set_xlim(0,204)
ax.set_ylim(0,204)
ax.set_zlim(0,204)
plt.savefig("Augmentations/flip.png", bbox_inches='tight')

# shift
dataset = ImageDataset(patient_to_load, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=False, 
shift_augment=True, cube_size=image_dimension)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
array = dataset[index][0].numpy()
x,y,z = np.where(array > 0)
ax.scatter(x, y, z, c=z, alpha=1)
ax.set_xlim(0,204)
ax.set_ylim(0,204)
ax.set_zlim(0,204)
plt.savefig("Augmentations/shift.png", bbox_inches='tight')