import sklearn
from cmath import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def log(item, log_string):
  print(item)
  log_string += "\n"
  log_string += str(item)
  return log_string

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

    self.G_mean = sqrt(self.sensitivity*self.specificity).real

  def accuracy(self):
    # Compute accuracy of predictions as fraction
    correct = (self.predictions == self.targets).sum().item()
    correct /= len(self.predictions)
    return correct
  
  def conf_matrix(self):
    return sklearn.metrics.confusion_matrix(self.targets, self.predictions)
  
  def results_string(self):
    return str(
    f"Accuracy:    {100*self.accuracy():>.2f}%" +  
    f"  \nSensitivity: {self.sensitivity:>.2f}" +
    f"  \nSpecificity: {self.specificity:>.2f}" + 
    f"  \nPrecision:   {self.precision:>.2f}" + 
    f"  \nG-mean:      {self.G_mean:>.2f}" + 
    f"  \nF1 score:    {self.F1_measure:>.2f}"
    )