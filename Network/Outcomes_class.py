test = [
  ["1", "100", "", "", ""],
  ["2", "200", "", "", ""],
  ["1", "100", "", "", ""],
  ["2", "200", "", "", ""]
]


class outcomes:
  def __init__(data_array):
    # data_array form: [patient, check day, LR, DM, death]
    self.metadata = data_array

  def censor(metadata):
    new_array = []
    for patient in metadata:
      if (int(patient[5]) < check_day):
        continue
      else:
        new_array.append(patient)
    return new_array

test_outcomes = outcomes(test)
print(test_outcomes)


  
  