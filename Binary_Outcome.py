from cmath import sqrt
import os
import sys
import torch
import random
import shutil
import numpy as np
import torchvision
import SimpleITK as sitk
import matplotlib.pyplot as plt


from torch import nn
from torch import reshape
from torchinfo import summary
from datetime import datetime
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

tag = 0
data_transform = transforms.Compose([transforms.ToTensor()])
logger = ""
underline = ("-----------------------------------------------------------------"
"-------------------------")
double_underline = ("=========================================================="
"================================")

def log(item, log_string):
  print(item)
  log_string += "\n"
  log_string += str(item)
  return log_string

def outcome_str_from_int(outcome_int):
  if outcome_int == 1:
    outcome_str = "Locoregional recurrence"
  elif outcome_int == 2:
    outcome_str = "Distant Metastasis"
  elif outcome_int == 3:
    outcome_str = "Death"
  return outcome_str

def split(outcome_list, train_ratio):
  # Split a list of patient outcomes into a train, validation and test set. No.
  # of training images set by the train_ratio, then validation and testing split 
  # equally from remaining images

  # Set the number of patients to assign to the train and validation sets
  train = int(train_ratio * len(outcome_list))
  validate = int(0.5 * (len(outcome_list) - train))

  # Define train images from random sample without replacement of all outcomes
  train_labels = random.sample(outcome_list, train)
  for item in train_labels:
    outcome_list.remove(item)

  # Define the validation images in the same way as above
  validate_labels = random.sample(outcome_list, validate)
  for item in validate_labels:
    outcome_list.remove(item)

  # Assign all remaining images to the test set and return the label lists
  test_labels = random.sample(outcome_list, validate)
  return train_labels, validate_labels, test_labels

def loss_plot(train_losses, validation_losses):
  fig = plt.figure()
  ax = plt.axes()
  ax.plot(train_losses[0], train_losses[1], label="Train Loss")
  ax.plot(validation_losses[0], validation_losses[1], label="Validate Loss")
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Loss')
  ax.legend()
  return fig

class Results:
  def __init__(self, predictions, targets):
    self.predictions = predictions.cpu().numpy()
    self.targets = targets.cpu().numpy()
    # Compute the numbers of true/false postives/negatives
    self.tp = 0
    self.fp = 0
    self.tn = 0
    self.fn = 0
    i = 0
    while i < len(self.predictions):
      if (predictions[i] == targets[i] and predictions[i] == 1):
        self.tp += 1
      elif (predictions[i] == targets[i] and predictions[i] == 0):
        self.tn += 1
      elif (predictions[i] != targets[i] and predictions[i] == 1):
        self.fp += 1
      elif (predictions[i] != targets[i] and predictions[i] == 0):
        self.fn += 1
      i += 1
    
    if self.tp != 0:
      self.sensitivity = self.tp/(self.tp+self.fn)
      self.precision = self.tp/(self.fp+self.tp)
      self.F1_measure = ((2*self.precision*self.sensitivity)/
                      (self.precision+self.sensitivity))
    else:
      self.sensitivity = 0
      self.precision = 0
      self.F1_measure = 0
    
    if self.tn != 0:
      self.specificity = self.tn/(self.fp+self.tn)
    else:
      self.specificity = 0

    self.G_mean = sqrt(self.sensitivity*self.specificity)

  def accuracy(self):
    # Compute accuracy of predictions as fraction
    correct = (self.predictions == self.targets).sum().item()
    correct /= len(self.predictions)
    return correct
  
  def conf_matrix(self):
    matrix = [[self.tp,self.fn],[self.fp, self.tn]]
    return matrix
  
  def conf_matrix_str(self):
    matrix_string = str(
    "                   Target   \n"
    "                | Pos | Neg \n"+
    "           -----------------\n"+
    "            Pos | "+str(self.tp).rjust(2," ")+" | "+str(self.fn).rjust(2," ")+ " \n"+
    "Prediction -----------------\n"+
    "            Neg | "+str(self.fp).rjust(2," ")+" | "+str(self.tn).rjust(2," ")+ " ")
    return matrix_string
  
  def results_string(self):
    return str(
    f"Accuracy:    {100*self.accuracy():>.2f}%" +  
    f"\nSensitivity: {self.sensitivity:>.2f}" +
    f"\nSpecificity: {self.specificity:>.2f}" + 
    f"\nPrecision:   {self.precision:>.2f}" + 
    f"\nG-mean:      {self.G_mean:>.2f}" + 
    f"\nF1 score:    {self.F1_measure:>.2f}" + "\n" +
    self.conf_matrix_str()
    )



def one_hot_vector_labels(scalar_labels):
  vector_labels = torch.empty((len(scalar_labels),2))
  for index in range(len(scalar_labels)):
    if scalar_labels[index] == 0:
      vector_labels[index,0] = 1
      vector_labels[index,1] = 0
    elif scalar_labels[index] == 1:
      vector_labels[index,0] = 0
      vector_labels[index,1] = 1
  vector_labels = vector_labels.float()
  return vector_labels

def train_loop(dataloader, model, loss_fn, optimizer, device, cube_size, logger):
  size = len(dataloader.dataset)
  sum_loss = 0
  batches = 0
  for batch, (X, y) in enumerate(dataloader):
    batches += 1
    # Compute prediction and loss
    X = reshape(X, (X.shape[0],1,cube_size,cube_size,cube_size))
    X = X.float().to(device)
    y = one_hot_vector_labels(y).to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    sum_loss += loss.item()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print results after each batch        
    if batch % 1 == 0:
      loss, current = loss.item(), batch * len(X)
      logger = log(f"Training Batch {batches:>} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", 
      logger)
  return sum_loss/batches, logger

def validate_loop(dataloader, model, loss_fn, device, cube_size):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  validate_loss= 0
  all_predictions = torch.zeros(size).to(device)
  all_targets = torch.zeros(size).to(device)

  with torch.no_grad():
    i = 0
    for X, y in dataloader:
      X = reshape(X, (X.shape[0],1,cube_size,cube_size,cube_size))
      X = X.float().to(device)
      y = one_hot_vector_labels(y).to(device)
      
      # Find outputs from model for each image in batch X
      pred = model(X)

      # Convert model predictions to from vector of 2 floats to 1 or 0 and
      # targets from 1 hot vector to 1 or 0
      _,predictions = torch.max(pred , 1)
      _,targets = torch.max(y, 1)

      # Add loss for this batch (divided by number of batches later to return
      # average)
      validate_loss += loss_fn(pred, y.float()).item()
      
      # Loop through predictions and targets for the batch and add to array for 
      # all batches. Brings all results together so they can be returned as 
      # arrays
      j = 0
      while j < len(predictions):
        all_predictions[i+j] = predictions[j]
        all_targets[i+j] = targets[j]
        j += 1
      i += 1

  validate_loss /= num_batches
  return validate_loss, all_predictions, all_targets

def test_loop(dataloader, model, device, cube_size):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss = 0
  all_predictions = torch.zeros(size).to(device)
  all_targets = torch.zeros(size).to(device)

  with torch.no_grad():
    i = 0
    for X, y in dataloader:
      X = reshape(X, (X.shape[0],1,cube_size,cube_size,cube_size))
      X = X.float().to(device)
      y = one_hot_vector_labels(y).to(device)

      pred = model(X)
      _,predictions = torch.max(pred , 1)
      _,targets = torch.max(y, 1)

      test_loss += loss_fn(pred, y.float()).item()

      j = 0
      while j < len(predictions):
        all_predictions[i+j] = predictions[j]
        all_targets[i+j] = targets[j]
        j += 1
      i += 1

  test_loss /= num_batches
  return test_loss, all_predictions, all_targets

class ImageDataset(Dataset):
  def __init__(self, annotations, img_dir, transform=data_transform, 
  target_transform=None, rotate_augment=True, scale_augment=True, 
  flip_augment=True, shift_augment=True, cube_size=246):
    self.img_labels = annotations
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    self.flips = flip_augment
    self.rotations = rotate_augment
    self.scaling = scale_augment
    self.shifts = shift_augment
    self.cube_size = cube_size

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels[idx][0]+".nii")
    image_sitk = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(image_sitk)
    label = self.img_labels[idx][1]
        
    if self.target_transform:
      label = self.target_transform(label)

    if self.shifts and random.random()<0.5:
      mx_x, mx_yz = 10, 10
      # find shift values
      cc_shift, ap_shift, lr_shift = (random.randint(-mx_x,mx_x), 
      random.randint(-mx_yz,mx_yz), random.randint(-mx_yz,mx_yz))
      
      # pad for shifting into
      image = np.pad(image, pad_width=((mx_x,mx_x),(mx_yz,mx_yz),(mx_yz,mx_yz)),
      mode='constant', constant_values=-1024)

      # crop to complete shift
      image = image[mx_x+cc_shift:self.cube_size+mx_x+cc_shift, mx_yz+ap_shift:self.cube_size+mx_yz+
      ap_shift, mx_yz+lr_shift:self.cube_size+mx_yz+lr_shift]

    if self.rotations and random.random()<0.5:
      # taking implementation from my 3DSegmentationNetwork which can be applied
      #  -> rotations in the axial plane only I should think? -10->10 degrees?
      # make -10,10
      roll_angle = np.clip(np.random.normal(loc=0,scale=3), -10, 10) 
      # (1,2) originally
      image = self.rotation(image, roll_angle, rotation_plane=(1,2)) 
        
    if self.scaling and random.random()<0.5:
      # same here -> zoom between 80-120%
      # original scale = 0.05
      scale_factor = np.clip(np.random.normal(loc=1.0,scale=0.5), 0.8, 1.2)
      image = self.scale(image, scale_factor)
        
    if self.flips and random.random()<0.5:
      image = self.flip(image)
    
    if self.transform:
      image = self.transform(image)
    # window and levelling
    image = self.windowLevelNormalize(image, level=40, window=50)

    return image, label

  def windowLevelNormalize(self, image, level, window):
    minval = level - window/2
    maxval = level + window/2
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / window)
    return wld

  def scale(self, image, scale_factor):
    # scale the image or mask using scipy zoom function
    order, cval = (3, 0) # changed from -1024 to 0
    height, width, depth = image.shape
    zheight = int(np.round(scale_factor*height))
    zwidth = int(np.round(scale_factor*width))
    zdepth = int(np.round(scale_factor*depth))
    # zoomed out
    if scale_factor < 1.0:
      new_image = np.full_like(image, cval)
      ud_buffer = (height-zheight) // 2
      ap_buffer = (width-zwidth) // 2
      lr_buffer = (depth-zdepth) // 2
      new_image[ud_buffer:ud_buffer+zheight, ap_buffer:ap_buffer+zwidth,
       lr_buffer:lr_buffer+zdepth] = zoom(input=image, zoom=scale_factor, 
       order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
      return new_image
    elif scale_factor > 1.0:
      new_image = zoom(input=image, zoom=scale_factor, order=order, 
      mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]

      ud_extra = (new_image.shape[0] - height) // 2
      ap_extra = (new_image.shape[1] - width) // 2
      lr_extra = (new_image.shape[2] - depth) // 2
      new_image = new_image[ud_extra:ud_extra+height, ap_extra:ap_extra+width, 
      lr_extra:lr_extra+depth]

      return new_image
    return image
  
  def rotation(self, image, rotation_angle, rotation_plane):
    # rotate the image using scipy rotate function
    order, cval = (3, -1024) # changed from -1024 to 0
    return rotate(input=image, angle=rotation_angle, axes=rotation_plane, 
    reshape=False, order=order, mode='constant', cval=cval)

  def flip(self, image):
    image = np.flip(image).copy()
    return image

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    out1 = 32
    out2 = 64
    out3 = 128
    out4 = 64
    out5 = 16
    out6 = 2
    self.cnn_layers = nn.Sequential(
      # Layer 1
      nn.Conv3d(1,out1,2,2),
      nn.BatchNorm3d(out1),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      # Layer 2
      nn.Conv3d(out1, out2, 2, 2),
      nn.BatchNorm3d(out2),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      # Layer 3
      nn.Conv3d(out2, out3, 2, 2),
      nn.BatchNorm3d(out3),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool3d(kernel_size=2, stride=2),
      # Layer 4
      nn.Conv3d(out3, out4, 1, 1),
      nn.BatchNorm3d(out4),
      nn.LeakyReLU(inplace=True),
      # Layer 5
      nn.Conv3d(out4, out5, 1, 1),
      nn.BatchNorm3d(out5),
      nn.LeakyReLU(inplace=True),
      # Layer 6
      nn.Conv3d(out5, out6, 1, 1),
      nn.BatchNorm3d(out6),
      nn.LeakyReLU(inplace=True),
      nn.AvgPool3d(2)
    )

  def forward(self, x):
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)
    return x

class customWriter(SummaryWriter):
  # Custom tensorboard writer class
  def __init__(self, log_dir, batch_size, epoch, num_classes, dataloader, cube_size):
    super(customWriter, self).__init__()
    self.log_dir = log_dir
    self.batch_size = batch_size
    self.epoch = epoch
    self.num_classes = num_classes
    self.train_loss = []
    self.val_loss = []
    self.class_loss = {n: [] for n in range(num_classes+1)}
    self.dataloader = dataloader
    self.cube_size = cube_size
    
  @staticmethod
  def sigmoid(x):
      return 1/(1+torch.exp(-x))

  def reset_losses(self):
      self.train_loss, self.val_loss, self.class_loss = [], [], {
          n: [] for n in range(self.num_classes+1)}

  def plot_batch(self, tag, images):
    """
    Plot batches in grid
​
    Args: tag = identifier for plot (string)
          images = input batch (torch.tensor)
    """
    img_grid = torchvision.utils.make_grid(images, nrow=self.batch_size // 2)
    self.add_image(tag, img_grid)

  def plot_pred(self, tag, prediction):
    """
    Plot predictions vs target segmentation.
    Args: tag = identifier for plot (string)
          prediction = batch output of trained model (torch.tensor)
          target = batch ground-truth segmentations (torch.tensor)
    """
    fig = plt.figure(figsize=(24, 24))#changed from (24,24)
    for idx in np.arange(self.batch_size):
      ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2, self.batch_size // 2,
                          idx+1, label='images')
      ax.imshow(prediction[idx, 0].cpu().numpy(
      ), cmap='viridis')
      
      ax.set_title('prediction @ epoch: {} - idx: {}'.format(self.epoch, idx))
    self.add_figure(tag, fig)

  def plot_tumour(self, tag, dataloader):
    fig = plt.figure(figsize=(24, 24))
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):
      X = reshape(X, (X.shape[0],1,self.cube_size,self.cube_size,self.cube_size))
      X = X.float()
      X = X.to(device)
      X = X.cpu()
      X = X.detach().numpy()
      for i in range(X.shape[0]):
        Xbig = X[i,0,:,:,:]
        Xsmall = Xbig[:,:,123]
        ax = fig.add_subplot()
        ax.imshow(Xsmall, cmap='viridis')
        self.add_figure(str(tag), fig)
        tag += 1

  def plot_histogram(self, tag, prediction):
    print('Plotting histogram')
    fig = plt.figure(figsize=(24, 24))
    for idx in np.arange(self.batch_size):
      ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                            idx+1, yticks=[], label='histogram')
      pred_norm = (prediction[idx, 0]-prediction[idx, 0].min())/(
          prediction[idx, 0].max()-prediction[idx, 0].min())
      ax.hist(pred_norm.cpu().flatten(), bins=100)
      ax.set_title(
          f'Prediction histogram @ epoch: {self.epoch} - idx: {idx}')
    self.add_figure(tag, fig)

  def per_class_loss(self, prediction, target, criterion, alpha=None):
    # Predict shape: (4, 1, 512, 512)
    # Target shape: (4, 1, 512, 512)
    #pred, target = prediction.cpu().numpy(), target.cpu().numpy()
    pred, target = prediction, target
    for class_ in range(self.num_classes + 1):
      class_pred, class_tgt = torch.where(
        target == class_, pred, torch.tensor([0], dtype=torch.float32).cuda()),  torch.where(target == class_, target, torch.tensor([0], dtype=torch.float32).cuda())

      #class_pred, class_tgt = pred[target == class_], target[target == class_] 
      if alpha is not None:
          loss = criterion(class_pred, class_tgt, alpha)
          #bce_loss, dice_loss = criterion(class_pred, class_tgt, alpha)
      else:
          loss = criterion(class_pred, class_tgt)
          #bce_loss, dice_loss = criterion(class_pred, class_tgt)
      #loss = bce_loss + dice_loss
      self.class_loss[class_].append(loss.item())

  def write_class_loss(self):
    for class_ in range(self.num_classes+1):
      self.add_scalar(f'Per Class loss for class {class_}', np.mean(self.class_loss[class_]), self.epoch)

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
logger = log(str(summary(model, (batch_size, 1, image_dimension, image_dimension, 
image_dimension), verbose=0)), logger)
logger = log(double_underline, logger)

# Define loss function and optimizer and send to device
loss_fn = nn.BCEWithLogitsLoss(torch.tensor([(1/pos_weights), 
pos_weights])).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Build Dataloaders
train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
validate_dataloader = DataLoader(validation_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

writer = customWriter("/data/James_Anna/Tensorboard/", 2, 0, 1, train_dataloader, image_dimension)

# Training
train_losses = [[],[]]
validate_losses = [[],[]]

logger = log("Training", logger)
logger = log(double_underline, logger)
for t in range(epochs):
  logger = log(f"Epoch {t+1}\n" + underline, logger)
  train_loss, logger = train_loop(train_dataloader, model, loss_fn, optimizer, 
  device, image_dimension, logger)
  validate_loss, predictions, targets = validate_loop(validate_dataloader, model, loss_fn, device,
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
  # writer.add_scalar('Train Loss', train_loss, t)
  # writer.add_scalar('Validate Loss', validate_loss, t)
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

if not os.path.exists(project_folder + "/" + subfolder + "/Tensorboard"):
  os.makedirs(project_folder + "/" + subfolder + "/Tensorboard")

runs_files = os.listdir("runs")
for each in runs_files:
  if not os.path.exists(project_folder + "/" + subfolder + "/Tensorboard/" + 
  each):
    shutil.copytree("runs/"+each, project_folder + "/" + subfolder + 
    "/Tensorboard/" + each)

shutil.rmtree("runs")