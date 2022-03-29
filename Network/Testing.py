import os
import sys
import torch
import pickle
import numpy as np
import SimpleITK as sitk

from Results import Results
from Networks import CNN, ResNet
from Network_Loops import test_loop
from Tensorboard import customWriter
from pytorch_grad_cam import GradCAM
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from pytorch_grad_cam.utils.image import show_cam_on_image

project_folder = sys.argv[1] 
subfolder = sys.argv[2] 
date = sys.argv[3]
batch_size = int(sys.argv[4])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = CNN().to(device)
model.load_state_dict(torch.load(f'models/{date}'))

# target_layers = [model.layer4[-1]]

open_file = open(f"test_data/{date}.pkl", "rb")
test_outcomes = pickle.load(open_file)
open_file.close()

# Find the original metadata for the patients 
# Open the metadata.csv file, convert to an array, and remove column headers
metadata_file = open(project_folder + "/metadata.csv")
metadata = np.loadtxt(metadata_file, dtype="str", delimiter=",")
metadata = metadata[1:30][:]

# Find the size of the images being read in
image_sitk = sitk.ReadImage(project_folder + "/" + subfolder + "/Images/" + 
metadata[0][0] + ".nii")
image = sitk.GetArrayFromImage(image_sitk)
image_dimension = image.shape[0]

test_data = ImageDataset(test_outcomes, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=False, 
cube_size=image_dimension)

test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

# data = next(test_dataloader)
# input_tensor = data[0]

# with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
#   targets = None
#   grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#   grayscale_cam = grayscale_cam[0,:]

# input = input_tensor
# grad_cam = grayscale_cam

# Testing
print("Testing")
test_predictions, test_targets = test_loop(test_dataloader, model, device, 
image_dimension)

test_results = Results(test_predictions,test_targets)
# writer.plot_confusion_matrix(test_results.conf_matrix(), 
# ["No Recurrence", "Recurrence"], "Conf. matrix, testing")

# writer.add_text("Test Results", test_results.results_string())
print(test_results.results_string())
test_results.accuracy()
# writer.close()