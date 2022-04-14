import os
import csv
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from Results import Results
from Networks import CNN, ResNet
from Network_Loops import test_loop
from Tensorboard import customWriter
from pytorch_grad_cam import GradCAM
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from medcam import medcam

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#===============================================================================
# Set-up
#===============================================================================
project_folder = "/data/James_Anna"
subfolder = "crop_2022_03_29-16_08_02"
date = "2022_04_13_22_15_20"

# model = CNN().to(device)
model = ResNet.generate_model(10).to(device)
#===============================================================================

# Open test image data
open_file = open(f"test_data/{date}.pkl", "rb")
test_outcomes = pickle.load(open_file)
open_file.close()

# Load model state for testing
model.load_state_dict(torch.load(f'models/{date}'))

# Create file for test results if one does not exist
if not os.path.exists(f"test_runs/{date}"):
    os.mkdir(f"test_runs/{date}")

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

# Create dataset and dataloader
test_data = ImageDataset(test_outcomes, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=False, 
shift_augment=False, cube_size=image_dimension)
test_dataloader = DataLoader(test_data, 1, shuffle=False)

# Testing
print("Testing")
test_predictions, test_targets = test_loop(test_dataloader, model, device, 
image_dimension)

# Define test results
test_results = Results(test_predictions,test_targets)
accuracy = test_results.accuracy()
sensitivity = test_results.sensitivity
precision = test_results.precision
F1_measure = test_results.F1_measure
tn = test_results.tn
tp = test_results.tp
fn = test_results.fn
fp = test_results.fp
specificity = test_results.specificity
G_mean = test_results.G_mean

# Append test results to csv
if os.path.isfile('test_runs/Results.csv'):
    # Append row
    with open('test_runs/Results.csv', 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow([date, accuracy, sensitivity, precision, F1_measure,
        tp, tn, fp, fn, specificity, G_mean])
else:
    # Create and then add row
    with open('test_runs/Results.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['Date/time', 'Accuracy', 'Sensitivity','Precision',
        'F1 measure', 'True positive', 'True negative', 'False positive', 
        'False negative', 'Specificity', 'G mean'])
        filewriter.writerow([date, accuracy, sensitivity, precision, F1_measure,
        tp, tn, fp, fn, specificity, G_mean])

# Save medcam images for each layer
layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'conv2']
for layer in layer_names:
    x = medcam.inject(model, output_dir="medcam_test", 
        save_maps=False, layer=layer, replace=True) # save_maps=True for nii
    # print(medcam.get_layers(model))
    x.eval()
    image, label, pid = next(iter(test_dataloader))
    filename = pid[0][0]
    image = image[None].to(device, torch.float)
    attn = x(image)

    attn = np.squeeze(attn.cpu().numpy())
    img = np.squeeze(image.cpu().numpy())
    # print(img.shape, attn.shape)
    slice_num = 102
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    im = img[..., slice_num]
    attn = attn[..., slice_num]
    # print(pid)
    # print(attn.max(), attn.min())
    ax.imshow(im, cmap='gray')
    ax.imshow(attn, cmap='jet', alpha=0.5)
    fig.savefig(f'test_runs/{date}/{layer}.png')
