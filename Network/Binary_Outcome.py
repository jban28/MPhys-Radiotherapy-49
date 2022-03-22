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
from cmath import sqrt
from Networks import CNN
from torchinfo import summary
from datetime import datetime
from Results import log, Results
from ImageDataset import ImageDataset
from Tensorboard import customWriter
from torch.utils.data import DataLoader
from outcomes import split, outcome_str_from_int
from Network_Loops import train_loop, validate_loop, test_loop

tag = 0
logger = ""
underline = ("-----------------------------------------------------------------"
"-------------------------")
double_underline = ("=========================================================="
"================================")

logger = log(" ".join(sys.argv), logger)

# Define the location of the data using system inputs
project_folder = sys.argv[1] 
subfolder = sys.argv[2] 
outcome_type = int(sys.argv[3])
check_day = int(sys.argv[4])
epochs = int(sys.argv[5])
batch_size = int(sys.argv[6])
learning_rate = float(sys.argv[7])

# Create a folder for the results if one does not already exist
if not os.path.exists(project_folder + "/" + subfolder + "/Results"):
  os.makedirs(project_folder + "/" + subfolder + "/Results")

# Make a subfolder within the Results folder to store the current set of results
date = datetime.now().strftime("%Y_%m_%d/%H_%M_%S")
os.makedirs(project_folder + "/" + subfolder + "/Results/" + date)

logger = log(double_underline, logger)
logger = log(date, logger)

# Connect to GPU if available and move model and loss function across
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logger = log(f'Using {device} device', logger)

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

# Define string for outcome description
outcome_description = ("Binary outcome; " + outcome_str_from_int(outcome_type) +   
" on/before day " + str(check_day))

logger = log(outcome_description, logger)

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
logger = log(model.__repr__(), logger)
logger = log(str(summary(model, (batch_size, 1, image_dimension, 
image_dimension, image_dimension), verbose=0)), logger)
logger = log(double_underline, logger)

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

# Training
train_losses = [[],[]]
validate_losses = [[],[]]

logger = log("Training", logger)
logger = log(double_underline, logger)
for t in range(epochs):
  logger = log(f"Epoch {t+1}\n" + underline, logger)
  train_loss, logger = train_loop(train_dataloader, model, loss_fn, optimizer, 
  device, image_dimension, logger, batch_size)
  validate_loss, predictions, targets = validate_loop(validate_dataloader, 
  model, loss_fn, device,
  image_dimension)

  val_results = Results(predictions,targets)

  logger = log(underline, logger)
  logger = log(val_results.results_string(), logger)
  logger = log(double_underline, logger)

  train_losses[0].append(t)
  train_losses[1].append(train_loss)
  validate_losses[0].append(t)
  validate_losses[1].append(validate_loss)

  # Plot the losses and save the plot in the results folder
  loss = loss_plot(train_losses, validate_losses)
  plt.savefig(project_folder + "/" + subfolder + "/Results/" + str(date) + 
  "/loss.png")
  writer.add_scalar('Train Loss', train_loss, t)
  writer.add_scalar('Validate Loss', validate_loss, t)
  # plot 3d plots here
  # writer.plot_tumour(dataloader = train_dataloader, tag=tag)
writer.close()

# Testing
test_loss, test_predictions, test_targets = test_loop(test_dataloader, model, 
device, image_dimension)

test_results = Results(test_predictions,test_targets)

logger = log(double_underline + "\nTesting\n" + double_underline, logger)
logger = log(test_results.results_string(), logger)
logger = log(double_underline, logger)

with open(project_folder + "/" + subfolder + "/Results/" + date +
'/Summary.txt', 'w') as f:
  f.write(logger)
  f.close()

# Move runs file to main storage then delete original
if not os.path.exists(project_folder + "/" + subfolder + "/Tensorboard"):
  os.makedirs(project_folder + "/" + subfolder + "/Tensorboard")

runs_files = os.listdir("runs")
for each in runs_files:
  if not os.path.exists(project_folder + "/" + subfolder + "/Tensorboard/" + 
  each):
    shutil.copytree("runs/"+each, project_folder + "/" + subfolder + 
    "/Tensorboard/" + each)