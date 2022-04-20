import os
import random
import numpy as np

class outcomes():
  def __init__(self, data_array, check_day=100, censoring=True):
    # data_array form: [patient, check day, LR, DM, death]
    self.check_day = check_day
    self.censoring = censoring
    self.metadata = data_array
    if self.censoring == True:
      self.name = "Censored"
    else:
      self.name = "Uncensored"

  def lr_binary(self):
    positives = []
    negatives = []
    for patient in self.metadata:
      if ((patient[2] == "") and self.censoring and (int(patient[1]) < 
      self.check_day)):
        continue 
      elif (patient[2] == "") or (int(patient[2]) > self.check_day):
        negatives.append([patient[0], 0])
      elif (int(patient[2]) <= self.check_day):
        positives.append([patient[0], 1])
    self.name += ", binary locoregional recurrence"
    return positives, negatives
  
  def dm_binary(self):
    positives = []
    negatives = []
    for patient in self.metadata:
      if (patient[3] == "") and self.censoring and (int(patient[1]) < 
      self.check_day):
        continue 
      elif (patient[3] == "") or (int(patient[3]) > self.check_day):
        negatives.append([patient[0], 0])
      elif (int(patient[3]) <= self.check_day):
        positives.append([patient[0], 1])
    self.name += ", binary distant metastasis"
    return positives, negatives

  def lr_dm_binary(self):
    positives = []
    negatives = []
    for patient in self.metadata:
      if ((patient[2] == "") and (patient[3] == "") and self.censoring and 
      (int(patient[1]) < self.check_day)):
        continue 
      if (patient[2] == "") and (patient[3] == ""):
        negatives.append([patient[0], 0])
      elif (patient[2] == "") and (patient[3] != ""):
        if (int(patient[3]) <= self.check_day):
          positives.append([patient[0], 1])
        elif (int(patient[3]) > self.check_day):
          negatives.append([patient[0], 0])
      elif (patient[2] != "") and (patient[3] == ""):
        if (int(patient[2]) <= self.check_day):
          positives.append([patient[0], 1])
        elif (int(patient[2]) > self.check_day):
          negatives.append([patient[0], 0])
      elif (patient[2] != "") and (patient[3] != ""):
        if ((int(patient[2]) <= self.check_day) or (int(patient[3]) <= 
        self.check_day)):
          positives.append([patient[0], 1])
      else:
        negatives.append([patient[0], 1])
    self.name += ", binary locoregional recurrence or distant metastasis"
    return positives, negatives

def split(outcome_list, train_ratio):
  # Split a list of patient outcomes into a train, validation and test set. No.
  # of training images set by the train_ratio, then validation and testing split 
  # equally from remaining images

  # Set the number of patients to assign to the train and validation sets
  train = int(train_ratio * len(outcome_list))
  validate = int(0.5 * (len(outcome_list) - train))

  # Define train images from random sample without replacement of all outcomes
  random.seed(0) # Use random.seed(0) for the train dataset
  train_labels = random.sample(outcome_list, train)
  for item in train_labels:
    outcome_list.remove(item)

  # Define the validation images in the same way as above
  random.seed(1) # Use random.seed(1) for the validation dataset
  validate_labels = random.sample(outcome_list, validate)
  for item in validate_labels:
    outcome_list.remove(item)

  # Assign all remaining images to the test set and return the label lists
  random.seed(2) # Use random.seed(2) for the test dataset
  test_labels = random.sample(outcome_list, validate)
  return train_labels, validate_labels, test_labels

def load_metadata(project_folder, subfolder):
  metadata_file = open(project_folder + "/metadata.csv")
  metadata = np.loadtxt(metadata_file, dtype="str", delimiter=",")
  metadata = metadata[1:][:]
  new_metadata = []
  for patient in metadata:
    if not os.path.exists(
      project_folder + "/" + subfolder + "/Images/" + patient[0] + ".nii"
      ):
      continue
    new_metadata.append([patient[0], patient[5], patient[6], patient[7], 
    patient[8]])
  return new_metadata
