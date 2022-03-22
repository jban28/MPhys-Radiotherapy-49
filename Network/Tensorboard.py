import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class customWriter(SummaryWriter):
  # Custom tensorboard writer class
  def __init__(self, log_dir, batch_size, epoch, num_classes, dataloader, 
  cube_size):
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
      ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2, 
      self.batch_size // 2, idx+1, label='images')
      ax.imshow(prediction[idx, 0].cpu().numpy(
      ), cmap='viridis')
      
      ax.set_title('prediction @ epoch: {} - idx: {}'.format(self.epoch, idx))
    self.add_figure(tag, fig)

  def plot_tumour(self, tag, dataloader):
    fig = plt.figure(figsize=(24, 24))
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):
      X = reshape(X, (X.shape[0],1,self.cube_size,self.cube_size,
      self.cube_size))
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
      class_pred = torch.where(target == class_, pred, torch.tensor([0], 
      dtype=torch.float32).cuda())
      class_tgt = torch.where(target == class_, target, torch.tensor([0], 
        dtype=torch.float32).cuda())

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
      self.add_scalar(f'Per Class loss for class {class_}', 
      np.mean(self.class_loss[class_]), self.epoch)

  def plot_confusion_matrix(self, cm, class_names):
    #function taken from https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    print(type(cm))
    figure = plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    #Normalize confusion matrix
    cm = np.around(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis],decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      color = "white" if cm[i, j] > threshold else "black"
      plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    self.add_figure(f"Confusion Matrix at epoch {self.epoch}", figure)