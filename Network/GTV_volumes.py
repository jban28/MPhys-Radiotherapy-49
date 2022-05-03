from Outcomes import outcomes, split, load_metadata
import SimpleITK as sitk
from ImageDataset import ImageDataset
import numpy as np
import matplotlib.pyplot as plt

project_folder = "/mnt/f/MPhys"
subfolder = "crop_2022_03_01-12_00_12"
check_day = 1000

metadata = load_metadata(project_folder, subfolder)
patient_outcomes = outcomes(metadata, check_day)
positives, negatives = patient_outcomes.lr_binary()

tr_pos, val_pos, test_pos = split(outcome_list=positives, train_ratio=0.7)
tr_neg, val_neg, test_neg = split(outcome_list=negatives, train_ratio=0.7)

train_outcomes = tr_pos + tr_neg
validation_outcomes = val_pos + val_neg
test_outcomes = test_pos + test_neg

# Find the size of the images being read in
image_sitk = sitk.ReadImage(project_folder + "/" + subfolder + "/Images/" + 
metadata[0][0] + ".nii")
image = sitk.GetArrayFromImage(image_sitk)
image_dimension = image.shape[0]

# Build Datasets
training_data = ImageDataset(train_outcomes, project_folder + "/" + subfolder + 
"/Images/", rotate_augment=True, scale_augment=True, flip_augment=True, 
shift_augment=True, cube_size=image_dimension)
validation_data = ImageDataset(validation_outcomes, project_folder + "/" + 
subfolder + "/Images/", rotate_augment=False, scale_augment=False, 
flip_augment=False, shift_augment=False, cube_size=image_dimension)
testing_data = ImageDataset(test_outcomes, project_folder + "/" + 
subfolder + "/Images/", rotate_augment=False, scale_augment=False, 
flip_augment=False, shift_augment=False, cube_size=image_dimension)

training_gtv_vols = []
validation_gtv_vols = []
testing_gtv_vols = []
"""
for patient in training_data:
  array = patient[0].numpy()
  training_gtv_vols.append(np.count_nonzero(array))

print(training_gtv_vols)

for patient in validation_data:
  array = patient[0].numpy()
  validation_gtv_vols.append(np.count_nonzero(array))

print(validation_gtv_vols)

for patient in testing_data:
  array = patient[0].numpy()
  testing_gtv_vols.append(np.count_nonzero(array))

print(testing_gtv_vols)
"""

training_gtv_vols = [71847, 43918, 35326, 1248, 39144, 59297, 7351, 14572, 
16298, 56759, 17532, 19268, 29942, 46381, 27997, 69082, 24272, 18321, 5855, 
36623, 32349, 11213, 11837, 8472, 24893, 17423, 33899, 51591, 32369, 14526, 
40972, 31315, 14748, 22367, 8321, 19900, 11937, 27094, 43162, 6648, 10281, 
76074, 46588, 31921, 30559, 35433, 6349, 27771, 48826, 32174, 16261, 10882, 
7813, 19027, 15467, 25946, 10632, 38061, 20560, 19848, 27575, 3188, 42241, 
26920, 22799, 9145, 6444, 95043, 9212, 5256, 1256, 39262, 2012, 24272, 29690, 
20536, 6523, 2858, 7566, 23389, 3461, 15036, 46118, 6500, 21424, 15865, 1759, 
27750, 88989, 65427, 26019, 8085, 72585, 13878, 72867, 25157, 5372, 5053, 8009, 
92855, 25173, 4738, 12294, 30298, 6797, 16478, 31592, 37804, 21031, 29560, 
13630, 1335, 24069, 16448, 19095, 30436, 39715, 35831, 11504, 1677, 26335, 
4692, 4894, 6453, 9365, 32430, 67290, 9961, 22876, 25339, 16802, 7666, 4720, 
9487, 76282, 6832, 20896, 77030, 3852, 9988, 69058, 27559, 1307, 22057, 21644, 
17281]
validation_gtv_vols = [23383, 25849, 4829, 28348, 32323, 12910, 48128, 14575, 
6600, 1954, 36055, 36287, 42519, 7813, 103789, 5855, 29289, 22799, 18692, 18779,
 47870, 38587, 8528, 123819, 19957, 2935, 34624, 10601, 1844, 28343, 25838]
testing_gtv_vols = [491, 19408, 37386, 8077, 36297, 61997, 26514, 7251, 4311, 
22370, 19431, 48309, 19154, 20881, 17215, 32115, 9310, 7826, 36079, 4787, 32002,
 21120, 5114, 28114, 25086, 17928, 26774, 34087, 34383, 55497, 35377]

plt.rcParams['font.family'] = "serif"
plt.hist(training_gtv_vols, bins=30)
plt.xlabel("Tumour volume", )
plt.ylabel("Number of patients")
plt.savefig("train_hist.png")

plt.rcParams['font.family'] = "serif"
plt.hist(validation_gtv_vols, bins=10)
plt.xlabel("Tumour volume", )
plt.ylabel("Number of patients")
plt.savefig("val_hist.png")

plt.rcParams['font.family'] = "serif"
plt.hist(testing_gtv_vols, bins=30)
plt.xlabel("Tumour volume", )
plt.ylabel("Number of patients")
plt.savefig("test_hist.png")