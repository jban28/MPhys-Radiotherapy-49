from Outcomes import outcomes
import numpy as np
import matplotlib.pyplot as plt

metadata_file = open("metadata.csv")
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
pos_lr = []
pos_dm = []
pos_lr_dm = []
days = []
day = 1
while day <= max_follow_up_day:
  outcome = outcomes(new_metadata, day, True)
  days.append(day)

  positives, negatives = outcome.lr_binary()
  pos_fractions_lr.append(len(positives)/len(outcome.metadata))
  pos_lr.append(len(positives))

  total_patients.append(len(positives) + len(negatives))

  # positives, negatives = outcome.dm_binary()
  # pos_fractions_dm.append(len(positives)/len(outcome.metadata))
  # pos_dm.append(len(positives))

  # positives, negatives = outcome.lr_dm_binary()
  # pos_fractions_lr_dm.append(len(positives)/len(outcome.metadata))
  # pos_lr_dm.append(len(positives))
  day += 1

plt.rcParams['font.family'] = "serif"
"""
fig, ax1 = plt.subplots()
ax1.set_xlabel('Day')
ax1.set_ylabel('Patients with positive outcome', color="red")
ax1.plot(days, pos_lr, color="red", linestyle='--', label='Patients with locoregional recurrence')
ax1.tick_params(axis='y', labelcolor="red")

ax2 = ax1.twinx()
ax2.set_ylabel('Patients remaining in study', color="blue") 
ax2.plot(days, total_patients, color="blue", label='Total patients remaining')
ax2.tick_params(axis='y', labelcolor="blue")
plt.savefig('Outcomes1.png')
"""
"""
color = 'tab:red'
ax1.set_xlabel('Day')
ax1.set_ylabel('Patients with positive outcome', color=color)
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
plt.savefig('Outcomes.pdf', dpi=300)
"""



plt.plot(total_patients, color="blue", label="Total")
plt.plot(pos_lr,linestyle="dashed", color="red", label="Locoregional recurrence")
# plt.plot(pos_dm, linestyle="dotted", color="red", label="Distant metastasis")
plt.xlabel("Day")
plt.ylabel("Number of patients remaining in study")
# plt.yscale("log")
plt.legend(loc=1)
plt.savefig('Outcomes.pdf')
