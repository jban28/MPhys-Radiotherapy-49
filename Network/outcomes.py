import random

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