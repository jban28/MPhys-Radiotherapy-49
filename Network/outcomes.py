import random
import numpy as np

class outcomes():
  def __init__(self, data_array, check_day=100, censoring=True):
    # data_array form: [patient, check day, LR, DM, death]
    self.check_day = check_day
    if censoring == True:
      self.metadata = self.censor(data_array)
    else:
      self.metadata = data_array

  def censor(self, array):
    new_array = []
    for patient in array:
      if (int(patient[1]) < self.check_day):
        continue
      else:
        new_array.append(patient)
    return new_array

  def lr_binary(self):
    positives = []
    negatives = []
    for patient in self.metadata:
      if (patient[2] == "") or (int(patient[2]) > self.check_day):
        negatives.append([patient[0], 0])
      elif (int(patient[2]) < self.check_day):
        positives.append([patient[0], 1])
    return positives, negatives
  
  def dm_binary(self):
    positives = []
    negatives = []
    for patient in self.metadata:
      if (patient[3] == "") or (int(patient[3]) > self.check_day):
        negatives.append([patient[0], 0])
      elif (int(patient[3]) < self.check_day):
        positives.append([patient[0], 1])
    return positives, negatives

  def lr_dm_binary(self):
    positives = []
    negatives = []
    for patient in self.metadata:
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
    return positives, negatives

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