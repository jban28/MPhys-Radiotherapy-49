from Outcomes_class import outcomes
import numpy as np
import matplotlib.pyplot as plt

project_folder = "/data/James_Anna"

metadata_file = open(project_folder + "/metadata.csv")
metadata = np.loadtxt(metadata_file, dtype="str", delimiter=",")
metadata = metadata[1:][:]
new_metadata = []
max_follow_up_day = 0
for patient in metadata:
  new_metadata.append([patient[0], patient[5], patient[6], patient[7], patient[8]])
  max_follow_up_day = max(max_follow_up_day, int(patient[5]))

total_patients = []
pos_fractions_lr = []
pos_fractions_dm = []
pos_fractions_lr_dm = []
days = []
day = 1
while day <= max_follow_up_day:
  test_outcomes = outcomes(new_metadata, day)
  days.append(day)

  positives, negatives = test_outcomes.lr_binary()
  pos_fractions_lr.append(len(positives)/len(negatives))

  positives, negatives = test_outcomes.dm_binary()
  pos_fractions_dm.append(len(positives)/len(negatives))

  positives, negatives = test_outcomes.lr_dm_binary()
  pos_fractions_lr_dm.append(len(positives)/len(negatives))

  total_patients.append(len(positives)+len(negatives))
  day += 1


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Day')
ax1.set_ylabel('Fraction of patients with positive outcome', color=color)
ax1.plot(days, pos_fractions_lr, color=color, linestyle=':', label='LR')
ax1.plot(days, pos_fractions_dm, color=color, linestyle='--', label='DM')
ax1.plot(days, pos_fractions_lr_dm, color=color, linestyle='-', label='LR+DM')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc=2,bbox_to_anchor=(0, 0.9, 0, 0))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Patients remaining in study', color=color)  # we already handled the x-label with ax1
ax2.plot(days, total_patients, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.savefig('Outcomes.png', dpi=300)