import torch
import torchvision
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch import reshape


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

  def plot_batch(self, tag, images):
    """
    Plot batches in grid
​
    Args: tag = identifier for plot (string)
          images = input batch (torch.tensor)
    """
    # img_grid = torchvision.utils.make_grid(images, nrow=self.batch_size // 2)
    img_grid = torchvision.utils.make_grid(images, nrow=3)
    self.add_image(tag, img_grid)

  def plot_tumour(self, tag, dataloader):
    fig = plt.figure(figsize=(24, 24))
    size = len(dataloader.dataset)
    
    for batch, (X, y, patient) in enumerate(dataloader):
      X = reshape(X, (X.shape[0],1,self.cube_size,self.cube_size,
      self.cube_size))
      X = X.float()
      #X = X.to(device)
      X = X.cpu()
      plot_slice = int(self.cube_size/2)
      X_test = X[:,:,:,:,plot_slice]
      self.plot_batch('tag', X_test)
      X = X.detach().numpy()
      for i in range(X.shape[0]):
        Xbig = X[i,0,:,:,:]
        Xsmall = Xbig[:,:,plot_slice]
        ax = fig.add_subplot()
        ax.imshow(Xsmall.T, cmap='gray')
        self.add_figure(patient[i], fig)

  def plot_confusion_matrix(self, cm, class_names, label):
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
    self.add_figure(label, figure)