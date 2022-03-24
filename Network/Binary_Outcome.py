# commands to run from (py39) jd_bannister28_gmail_com@mphys-vm-http:~/MPhys-Radiotherapy-49$
# python Binary_Outcome.py /data/James_Anna crop_2022_03_01-12_00_12 {outcome type} {day to check for outcome} {epochs} {batch size} {learning rate}

import os
import sys
import torch
import shutil
import numpy as np
import torchvision
import SimpleITK as sitk
import matplotlib.pyplot as plt

from torch import nn
from Networks import CNN
from torchinfo import summary
from datetime import datetime
from ImageDataset import ImageDataset
from Tensorboard import customWriter
from torch.utils.data import DataLoader
from Results import log, Results, loss_plot
from outcomes import split, outcome_str_from_int
from Network_Loops import train_loop, validate_loop, test_loop

tag = 0

# Define the location of the data using system inputs
project_folder = sys.argv[1] 
subfolder = sys.argv[2] 
outcome_type = int(sys.argv[3])
check_day = int(sys.argv[4])
epochs = int(sys.argv[5])
batch_size = int(sys.argv[6])
learning_rate = float(sys.argv[7])

# Connect to GPU if available and move model and loss function across
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

# Define empty arrays for positive and negative examples
positives = []
negatives = []

# Filter through each patient and assign outcome 
for patient in metadata:
  # Check image exists for patient and skip patient if not
  if not os.path.exists(
    project_folder + "/" + subfolder + "/Images/" + patient[0] + ".nii"
    ):
    # No image file found for patient
    continue
  if (patient[(5+outcome_type)] == "") and (int(patient[5]) >= check_day):
    # Last follow up after check day, no event
    outcome = 0
  elif (patient[(5+outcome_type)] == "") and (int(patient[5]) < check_day):
    # Last follow up before check day, event unknown
    continue
  elif int(patient[(5+outcome_type)]) <= check_day:
    # Event occurred before or on check day
    outcome = 1
  else:
    # Event occurred after check day
    outcome = 0

  # Append patient name and outcome to arrays
  if outcome == 1:
    positives.append([patient[0], outcome])
  else:
    negatives.append([patient[0], outcome])

# Split outcomes into train, validation and test
tr_pos, val_pos, test_pos = split(outcome_list=positives, train_ratio=0.7)
tr_neg, val_neg, test_neg = split(outcome_list=negatives, train_ratio=0.7)

# Construct outcome variables
train_outcomes = tr_pos + tr_neg
validation_outcomes = val_pos + val_neg
test_outcomes = test_pos + test_neg

# Define variable for the weight of positive examples
pos_weights = len(tr_neg)/len(tr_pos)

# Build Datasets
training_data = ImageDataset(train_outcomes, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=True, scale_augment=True, flip_augment=True, 
cube_size=image_dimension)
validation_data = ImageDataset(validation_outcomes, project_folder + "/" + 
subfolder + "/Images/", rotate_augment=False, scale_augment=False, 
flip_augment=False, cube_size=image_dimension)
test_data = ImageDataset(test_outcomes, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=False, scale_augment=False, flip_augment=False, 
cube_size=image_dimension)

# Define model and send to device
model = CNN().to(device)

# Define loss function and optimizer and send to device
loss_fn = nn.BCEWithLogitsLoss(torch.tensor([(1/pos_weights), 
pos_weights])).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Build Dataloaders
train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
validate_dataloader = DataLoader(validation_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

writer = customWriter("/data/James_Anna/Tensorboard/", 2, 0, 1, 
train_dataloader, image_dimension)

writer.add_text("Input", " ".join(sys.argv))

# Define string for outcome description
outcome_description = ("Binary outcome; " + outcome_str_from_int(outcome_type) +   
" on/before day " + str(check_day))

writer.add_text("Outcome", outcome_description)
writer.add_text("Network", model.name)

# Training
train_losses = [[],[]]
validate_losses = [[],[]]
for t in range(epochs):
  # plot 3d plots here
  writer.epoch = t+1
  # writer.plot_tumour(dataloader = train_dataloader, tag=tag)
  print(f"Epoch {t+1}")
  print("    Training")
  train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, 
  device, image_dimension, batch_size)
  print("    Validation")
  validate_loss, predictions, targets = validate_loop(validate_dataloader, 
  model, loss_fn, device,
  image_dimension)

  val_results = Results(predictions,targets)
  writer.plot_confusion_matrix(val_results.conf_matrix(), 
  ["No Recurrence", "Recurrence"], f"Conf. matrix, epoch {t+1}, validation")

  train_losses[0].append(t)
  train_losses[1].append(train_loss)
  validate_losses[0].append(t)
  validate_losses[1].append(validate_loss)

  # Plot the losses and save the plot in the results folder
  loss = loss_plot(train_losses, validate_losses)
  # plt.savefig(project_folder + "/" + subfolder + "/Results/" + str(date) + 
  # "/loss.png")
  writer.add_scalar("Train Loss", train_loss, t)
  writer.add_scalar("Validate Loss", validate_loss, t)
  writer.add_scalar("Validation Accuracy", val_results.accuracy(), t)
  writer.add_scalar("Validation Sensitivity", val_results.sensitivity, t)
  writer.add_scalar("Validation Specificity", val_results.specificity, t)
  writer.add_scalar("Validation Precision", val_results.precision, t)
  writer.add_scalar("Validation G-mean", val_results.G_mean, t)
  writer.add_scalar("Validation F1 score", val_results.F1_measure, t)

writer.close()

# Testing
print("Testing")
test_loss, test_predictions, test_targets = test_loop(test_dataloader, model, 
loss_fn, device, image_dimension)

test_results = Results(test_predictions,test_targets)
writer.plot_confusion_matrix(test_results.conf_matrix(), 
["No Recurrence", "Recurrence"], "Conf. matrix, testing")

writer.add_text("Test Results", test_results.results_string())
writer.add_hparam(test_results.results_dict())