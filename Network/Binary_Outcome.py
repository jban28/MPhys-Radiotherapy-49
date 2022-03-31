import os
import sys
import torch
import pickle
import shutil
import numpy as np
import torchvision
import SimpleITK as sitk
import matplotlib.pyplot as plt

from torch import nn
from torchinfo import summary
from datetime import datetime
from Results import log, Results
from Networks import CNN, ResNet
from Tensorboard import customWriter
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from Outcomes import outcomes, split, load_metadata
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Network_Loops import train_loop, validate_loop, test_loop

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
notes = input("Add any notes for this run")
date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

#===============================================================================
# Set-up
#===============================================================================
project_folder = "/data/James_Anna"
subfolder = "crop_2022_03_01-12_00_12"
check_day = 1000
epochs = 100
batch_size = 4
learning_rate = 0.001
metadata = load_metadata(project_folder, subfolder)
patient_outcomes = outcomes(metadata, check_day)
positives, negatives = patient_outcomes.lr_dm_binary()
# model = CNN().to(device)
model = ResNet.generate_model(10).to(device)
#loss_fn = nn.BCEWithLogitsLoss(torch.tensor([(len(negatives)/len(positives)), 
#(len(positives)/len(negatives))])).to(device)
loss_fn = nn.BCEWithLogitsLoss(torch.tensor([(len(negatives)/len(positives)), 
1])).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)
#===============================================================================

# Split outcomes into train, validation and test
tr_pos, val_pos, test_pos = split(outcome_list=positives, train_ratio=0.7)
tr_neg, val_neg, test_neg = split(outcome_list=negatives, train_ratio=0.7)

# Construct outcome variables
train_outcomes = tr_pos + tr_neg
validation_outcomes = val_pos + val_neg
test_outcomes = test_pos + test_neg

if not os.path.exists("test_data"):
  os.mkdir("test_data")
if not os.path.exists("models"):
  os.mkdir("models")

test_data_file = open(f"test_data/{date}.pkl", "wb")
pickle.dump(test_outcomes, test_data_file)
test_data_file.close()

# Find the size of the images being read in
image_sitk = sitk.ReadImage(project_folder + "/" + subfolder + "/Images/" + 
metadata[0][0] + ".nii")
image = sitk.GetArrayFromImage(image_sitk)
image_dimension = image.shape[0]

# Build Datasets
training_data = ImageDataset(train_outcomes, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=True, scale_augment=True, flip_augment=True, 
shift_augment=True, cube_size=image_dimension)
validation_data = ImageDataset(validation_outcomes, project_folder + "/" + 
subfolder + "/Images/", rotate_augment=False, scale_augment=False, 
flip_augment=False, shift_augment=False, cube_size=image_dimension)


# Create Weighted Random Sampler to feed into dataloader

#class_sample_count = [, 1] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
# weights = 1 / torch.Tensor(class_sample_count)
#sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
#trainloader = data_utils.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, sampler = sampler)

# Build Dataloaders
train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
validate_dataloader = DataLoader(validation_data, batch_size, shuffle=False)

writer = customWriter("/data/James_Anna/Tensorboard/", 2, 0, 1, 
train_dataloader, image_dimension)

info_string = (f"Date: {date}  \nNetwork: {model.name}  \nBinary Outcome: "
f"{patient_outcomes.name}  \nCheck for outcome on day: {check_day}"
f"  \n  Batch Size: {batch_size}  \nLearning Rate: {learning_rate}")

writer.add_text("Info", info_string)
# print("Plotting Images")
# writer.plot_tumour(dataloader = train_dataloader)

writer.add_text("Notes", notes)

# Training
train_losses = [[],[]]
validate_losses = [[],[]]
min_val_loss = 999999999
for t in range(epochs):

  writer.epoch = t+1
  print(f"Epoch {t+1}")
  print("    Training")
  train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, 
  device, image_dimension, batch_size)
  print("    Validation")
  validate_loss, predictions, targets = validate_loop(validate_dataloader, 
  model, loss_fn, device, image_dimension)
  scheduler.step(validate_loss)

  val_results = Results(predictions,targets)
  writer.plot_confusion_matrix(val_results.conf_matrix(), 
  ["No Recurrence", "Recurrence"], f"Conf. matrix, epoch {t+1}, validation")

  train_losses[0].append(t)
  train_losses[1].append(train_loss)
  validate_losses[0].append(t)
  validate_losses[1].append(validate_loss)

  # Plot the losses and save the plot in the results folder
  writer.add_scalar("Train Loss", train_loss, t)
  writer.add_scalar("Validate Loss", validate_loss, t)
  writer.add_scalar("Validation Accuracy", val_results.accuracy(), t)
  writer.add_scalar("Validation Sensitivity", val_results.sensitivity, t)
  writer.add_scalar("Validation Specificity", val_results.specificity, t)
  writer.add_scalar("Validation Precision", val_results.precision, t)
  writer.add_scalar("Validation G-mean", val_results.G_mean, t)
  writer.add_scalar("Validation F1 score", val_results.F1_measure, t)

  # saves the 'best' model i.e. the model that 
  if validate_loss < min_val_loss:
    min_val_loss = validate_loss
    torch.save(model.state_dict(), f'models/{date}')
  else:
    continue

writer.close()