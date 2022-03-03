import sys
import numpy as np
import matplotlib.pyplot as plt

project_folder = sys.argv[1]

# Open the metadata.csv file, convert to an array, and remove column headers
metadata_file = open(project_folder + "/metadata_resample.csv")
metadata = np.loadtxt(metadata_file, dtype="str", delimiter=",")
metadata = metadata[1:][:]

plt.hist(metadata[:,12].astype(np.float32), bins=50)
plt.xlabel("Max tumour size from CoM (mm)")
plt.ylabel("Number of patients")

plt.savefig(project_folder + "/histogram.png")