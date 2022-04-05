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
subfolder = "crop_2022_03_01-12_00_12"
date = "2022_03_29_15_46_01"

# model = CNN().to(device)
model = ResNet.generate_model(10).to(device)
model.load_state_dict(torch.load(f'models/{date}'))
#===============================================================================

# target_layers = [model.layer4[-1]]

open_file = open(f"test_data/{date}.pkl", "rb")
test_outcomes = pickle.load(open_file)
open_file.close()

#print(test_outcomes)

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
shift_augment=False, cube_size=image_dimension)

test_dataloader = DataLoader(test_data, 1, shuffle=False)

# Testing
print("Testing")
test_predictions, test_targets = test_loop(test_dataloader, model, device, 
image_dimension)

test_results = Results(test_predictions,test_targets)

# check to see if csv file exists and append test results

exists = os.path.isfile('Results.csv')
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
if exists:
    # Append row
    with open('Results.csv', 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow([date, accuracy, sensitivity, precision, F1_measure, tp, tn, fp, fn, 
                            specificity, G_mean])
else:
    # Create and then add row
    with open('Results.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['Date/time', 'Accuracy', 'Sensitivity', 'Precision', 'F1 measure',
                            'True positive', 'True negative', 'False positive', 
                            'False negative', 'Specificity', 'G mean'])
        filewriter.writerow([now, accuracy, sensitivity, precision, F1_measure, tp, tn, fp, fn, 
                            specificity, G_mean])





# GradCAM
# layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'conv2']
# for i in range(6):
#     layers = layer_names[i]
#     print(layers)
#     model = medcam.inject(model, output_dir="medcam_test", 
#         save_maps=True, layer=layers, replace=True)
#     #print(medcam.get_layers(model))
#     model.eval()
#     image, label, pid = next(iter(test_dataloader))
#     filename = pid[0][0]
#     image = image[None].to(device, torch.float)
#     attn = model(image)
#     attn = np.squeeze(attn.cpu().numpy())
#     img = np.squeeze(image.cpu().numpy())
#     print(img.shape, attn.shape)
#     slice_num = 102
#     fig, ax = plt.subplots(1,1, figsize=(10,10))
#     im = img[..., slice_num]
#     attn = attn[..., slice_num]
#     print(pid)
#     print(attn.max(), attn.min())
#     ax.imshow(im, cmap='gray')
#     ax.imshow(attn, cmap='jet', alpha=0.5)
#     fig.savefig('./GradCAM_'+ layer_names[i] +'.png')

# for looking at the convolutional layers conv1 and conv2

# for i in range(2):
#     layer = f'conv{i+1}'
#     print(layer)
#     model = medcam.inject(model, output_dir="medcam_test", 
#         save_maps=True, layer=layer, replace=True)
#     #print(medcam.get_layers(model))
#     model.eval()
#     image, label, pid = next(iter(test_dataloader))
#     filename = pid[0][0]
#     image = image[None].to(device, torch.float)
#     attn = model(image)

#     attn = np.squeeze(attn.cpu().numpy())
#     img = np.squeeze(image.cpu().numpy())
#     print(img.shape, attn.shape)
#     slice_num = 102
#     fig, ax = plt.subplots(1,1, figsize=(10,10))
#     im = img[..., slice_num]
#     attn = attn[..., slice_num]
#     print(pid)
#     print(attn.max(), attn.min())
#     ax.imshow(im, cmap='gray')
#     ax.imshow(attn, cmap='jet', alpha=0.5)
#     fig.savefig(f'./GradCAM_conv{i+1}.png')

# layer = 'layer1'
# #print(layer)
# model = medcam.inject(model, output_dir="medcam_test", 
#     save_maps=True, layer=layer, replace=True)
# print(medcam.get_layers(model))
# model.eval()
# image, label, pid = next(iter(test_dataloader))
# filename = pid[0][0]
# image = image[None].to(device, torch.float)
# attn = model(image)

# attn = np.squeeze(attn.cpu().numpy())
# img = np.squeeze(image.cpu().numpy())
# print(img.shape, attn.shape)
# slice_num = 102
# fig, ax = plt.subplots(1,1, figsize=(10,10))
# im = img[..., slice_num]
# attn = attn[..., slice_num]
# print(pid)
# print(attn.max(), attn.min())
# ax.imshow(im, cmap='gray')
# ax.imshow(attn, cmap='jet', alpha=0.5)
# fig.savefig('./GradCAM_layer1.png')


layer = 'conv2'
#print(layer)
model = medcam.inject(model, output_dir="medcam_test", 
    save_maps=True, layer=layer, replace=True)
print(medcam.get_layers(model))
model.eval()
image, label, pid = next(iter(test_dataloader))
filename = pid[0][0]
image = image[None].to(device, torch.float)
attn = model(image)

attn = np.squeeze(attn.cpu().numpy())
img = np.squeeze(image.cpu().numpy())
print(img.shape, attn.shape)
slice_num = 102
fig, ax = plt.subplots(1,1, figsize=(10,10))
im = img[..., slice_num]
attn = attn[..., slice_num]
print(pid)
print(attn.max(), attn.min())
ax.imshow(im, cmap='gray')
ax.imshow(attn, cmap='jet', alpha=0.5)
fig.savefig('./GradCAM_conv2.png')
