import os
import sys
import torch
import random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from torch import nn
from torch import reshape
from torchinfo import summary
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def find_metadata(project_folder):
  # Open the metadata.csv file, convert to an array, and remove column headers
  metadata_file = open(project_folder + "/metadata.csv")
  metadata = np.loadtxt(metadata_file, dtype="str", delimiter=",")
  metadata = metadata[1:24][:]
  return metadata

def split(outcome_list, train_ratio):
  train = int(train_ratio * len(outcome_list))
  validate = int(0.5 * (len(outcome_list) - train))

  train_labels = random.sample(outcome_list, train)
  for item in train_labels:
    outcome_list.remove(item)

  validate_labels = random.sample(outcome_list, validate)
  for item in validate_labels:
    outcome_list.remove(item)

  test_labels = random.sample(outcome_list, validate)

  return train_labels, validate_labels, test_labels

def binary_outcome(project_folder, subfolder, metadata, outcome_type, check_day):
  positives = []
  negatives = []
  # Loop through each patient and identify whether they are true or false for the specified outcome from above
  for patient in metadata:
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

    if not os.path.exists(project_folder + "/" + subfolder + "/Images/" + patient[0] + ".nii"):
      # No image file found for patient
      continue
    
    # Append patient name and outcome to arrays
    if outcome == 1:
      positives.append([patient[0], outcome])
    else:
      negatives.append([patient[0], outcome])
  
  tr_pos, val_pos, test_pos = split(outcome_list=positives, train_ratio=0.7)
  tr_neg, val_neg, test_neg = split(outcome_list=negatives, train_ratio=0.7)

  description = "Binary outcome type ", outcome_type, " day ", check_day

  return tr_pos + tr_neg, val_pos + val_neg, test_pos + test_neg, description

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = reshape(X, (X.shape[0],1,246,246,246))
        X = X.float()
        X = X.to(device)
        y = y.to(device)
        #y = reshape(y, (y.shape[0],1))
        hot_y = torch.empty((X.shape[0],2)).to(device)
        for index in range(len(y)):
          if y[index] == 0:
            hot_y[index,0] = 1
            hot_y[index,1] = 0
          elif y[index] == 1:
            hot_y[index,0] = 0
            hot_y[index,1] = 1
      
        pred = model(X)
        torch.squeeze(pred)
        loss = loss_fn(pred, hot_y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print results after each batch        
        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss

def validate_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validate_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = reshape(X, (X.shape[0],1,246,246,246))
            X = X.float()
            X = X.to(device)
            y = y.to(device)
            #y = reshape(y, (y.shape[0],1))
            hot_y = torch.empty((X.shape[0],2)).to(device)
            for index in range(len(y)):
              if y[index] == 0:
                hot_y[index,0] = 1
                hot_y[index,1] = 0
              elif y[index] == 1:
                hot_y[index,0] = 0
                hot_y[index,1] = 1
            
            pred = model(X)
            # print(f'pred: {pred}')
            # print(f'hot_y: {hot_y}')
            _,predictions = torch.max(pred , 1)
            _,targets = torch.max(hot_y, 1)
            # print(f'predictions: {predictions}')
            # print(f'targets: {targets}')
            print(f'Correct this batch = {(predictions == targets).sum().item()}')

            torch.squeeze(pred)
            validate_loss += loss_fn(pred, hot_y.float()).item()
            # correct += (pred.argmax(1) == hot_y).type(torch.float).sum().item()
            correct += (predictions == targets).sum().item()

    validate_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print(f"Validate Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {validate_loss:>8f} \n")
    return validate_loss, accuracy

def test_loop(dataloader, model, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = reshape(X, (X.shape[0],1,246,246,246))
            X = X.float()
            X = X.to(device)
            y = y.to(device)
            #y = reshape(y, (y.shape[0],1))
            hot_y = torch.empty((X.shape[0],2)).to(device)
            for index in range(len(y)):
              if y[index] == 0:
                hot_y[index,0] = 1
                hot_y[index,1] = 0
              elif y[index] == 1:
                hot_y[index,0] = 0
                hot_y[index,1] = 1
                
            pred = model(X)
            _,predictions = torch.max(pred , 1)
            _,targets = torch.max(hot_y, 1)
            torch.squeeze(pred)
            #test_loss += loss_fn(pred, y.float()).item()
            test_loss += loss_fn(pred, hot_y.float()).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += (predictions == targets).sum().item()

    test_loss /= num_batches
    accuracy = correct/size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy

def loss_plot(train_losses, validation_losses):
  fig = plt.figure()
  ax = plt.axes()
  ax.plot(train_losses[0], train_losses[1], label="Train Loss")
  ax.plot(validation_losses[0], validation_losses[1], label="Validate Loss")
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Loss')
  ax.legend()
  return fig

def Run(project_folder, subfolder, train_outcomes, validation_outcomes, test_outcomes, model, loss_fn, learning_rate, epochs, batch_size):
  # Connect to GPU if available
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Using {device} device')

  model.to(device)
  loss_fn.to(device)

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  # Build Datasets
  training_data = ImageDataset(train_outcomes, project_folder + "/" + subfolder + "/Images/")
  validation_data = ImageDataset(validation_outcomes, project_folder + "/" + subfolder + "/Images/")
  test_data = ImageDataset(test_outcomes, project_folder + "/" + subfolder + "/Images/")

  # Build Dataloaders
  train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
  validate_dataloader = DataLoader(validation_data, batch_size, shuffle=True)
  test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

  # Training
  train_losses = [[],[]]
  validate_losses = [[],[]]
  validate_accuracies = [[],[]]
  for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
      validate_loss = validate_loop(validate_dataloader, model, loss_fn, device)

      train_losses[0].append(t)
      train_losses[1].append(train_loss)
      validate_losses[0].append(t)
      validate_losses[1].append(validate_loss[0])
      validate_accuracies[0].append(t)
      validate_accuracies[1].append(validate_loss[1])
  print("Done!")

  # Testing
  accuracy = test_loop(test_dataloader, model, device)

  parameters = str(summary(model, (batch_size, 1, 246, 246, 246), verbose=0))


  return accuracy, train_losses, validate_losses, model, parameters

class ImageDataset(Dataset):
  def __init__(self, annotations, img_dir, transform=None, target_transform=None):
    self.img_labels = annotations
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
      return len(self.img_labels)

  def __getitem__(self, idx):
      img_path = os.path.join(self.img_dir, self.img_labels[idx][0]+".nii")
      image_sitk = sitk.ReadImage(img_path)
      image = sitk.GetArrayFromImage(image_sitk)
      label = self.img_labels[idx][1]
      if self.transform:
        image = self.transform(image)
      if self.target_transform:
        label = self.target_transform(label)
      return image, label

class CNN(nn.Module):
  def __init__(self):
    self.name = "CNN"
    super(CNN, self).__init__()
    out1 = 4
    out2 = 4
    out3 = 2
    self.cnn_layers = nn.Sequential(
      # Layer 1
      nn.Conv3d(1,out1,4,1,1),
      nn.BatchNorm3d(out1),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      # Layer 2
      nn.Conv3d(out1, out2, 4, 1, 1),
      nn.BatchNorm3d(out2),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      # Layer 3
      nn.Conv3d(out2, out3, 4, 1, 1),
      nn.BatchNorm3d(out3),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
    )
    self.linear_layers = nn.Sequential(
      nn.Linear(48778, 2)
    )
  def forward(self, x):
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)
    x = self.linear_layers(x)
    return x

project_folder = sys.argv[1] # "/content/drive/My Drive/Degree/MPhys/Data"
subfolder = sys.argv[2] # "crop"

if not os.path.exists(project_folder + "/" + subfolder + "/Results"):
  os.makedirs(project_folder + "/" + subfolder + "/Results")

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
os.makedirs(project_folder + "/" + subfolder + "/Results/" + date)

metadata = find_metadata(project_folder)

if int(sys.argv[3]) == 1:
  train_outcomes, validation_outcomes, test_outcomes, outcome_description = binary_outcome(project_folder, subfolder, metadata, 1, 500)

if int(sys.argv[4]) == 1:
  model = CNN()

if int(sys.argv[5]) == 1:
  loss_fn = nn.BCEWithLogitsLoss(torch.tensor(len(train_outcomes)/len(metadata)))

epochs = sys.argv[6]
batch_size = sys.argv[7]
repeats = sys.argv[8]
learning_rate = sys.argv[9]

accuracies = []

counter = 0

while counter < repeats:
  counter += 1
  accuracy, train_losses, validation_losses, model, parameters = Run(project_folder, subfolder, train_outcomes, validation_outcomes, test_outcomes, model,loss_fn,learning_rate,epochs,batch_size)
  accuracies.append(accuracy)
  loss = loss_plot(train_losses, validation_losses)
  plt.savefig(project_folder + "/" + subfolder + "/Results/" + str(date) + "/loss" + str(counter) + ".png")

average_test_accuracy = np.average(accuracies)

results = "".join("Date/time: " + str(date) + "\n" + "Average test accuracy: " + str(100*average_test_accuracy) + "%\n" + "Epochs: " + str(epochs) +"\n" + "Batch size: " + str(batch_size) + "\n" + "Repeats: " + str(repeats) + "\n" + "Learning rate: " + str(learning_rate) + "\n" + "Model: " + model.__repr__() + "\n" + parameters)

print(results)

with open(project_folder + "/" + subfolder + "/Results/" + "/" + date +'/Summary.txt', 'w') as f:
    f.write(results)
    f.close()