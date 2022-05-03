from Outcomes import outcomes, split, load_metadata
import SimpleITK as sitk
from ImageDataset import ImageDataset

project_folder = "/data/James_Anna"
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

for patient in training_data:
  print(patient[0])