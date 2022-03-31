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





  
  