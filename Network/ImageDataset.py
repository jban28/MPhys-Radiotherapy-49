import os
import random
import numpy as np
import SimpleITK as sitk
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate

data_transform = transforms.Compose([transforms.ToTensor()])

class ImageDataset(Dataset):
  def __init__(self, annotations, img_dir, transform=data_transform, 
  target_transform=None, rotate_augment=True, scale_augment=True, 
  flip_augment=True, shift_augment=True, cube_size=246):
    self.img_labels = annotations
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    self.flips = flip_augment
    self.rotations = rotate_augment
    self.scaling = scale_augment
    self.shifts = shift_augment
    self.cube_size = cube_size

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels[idx][0]+".nii")
    image_sitk = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(image_sitk)
    label = self.img_labels[idx][1]
    patient = self.img_labels[idx][0]
        
    if self.target_transform:
      label = self.target_transform(label)

    if self.shifts and random.random()<0.5:
      mx_x, mx_yz = 10, 10
      # find shift values
      cc_shift, ap_shift, lr_shift = (random.randint(-mx_x,mx_x), 
      random.randint(-mx_yz,mx_yz), random.randint(-mx_yz,mx_yz))
      
      # pad for shifting into
      image = np.pad(image, pad_width=((mx_x,mx_x),(mx_yz,mx_yz),(mx_yz,mx_yz)),
      mode='constant', constant_values=-1024)

      # crop to complete shift
      image = image[mx_x+cc_shift:self.cube_size+mx_x+cc_shift, 
      mx_yz+ap_shift:self.cube_size+mx_yz+ap_shift, mx_yz+
      lr_shift:self.cube_size+mx_yz+lr_shift]

    if self.rotations and random.random()<0.5:
      # taking implementation from my 3DSegmentationNetwork which can be applied
      #  -> rotations in the axial plane only I should think? -10->10 degrees?
      # make -10,10
      roll_angle = np.clip(np.random.normal(loc=0,scale=3), -10, 10) 
      # (1,2) originally
      image = self.rotation(image, roll_angle, rotation_plane=(1,2)) 
        
    if self.scaling and random.random()<0.5:
      # same here -> zoom between 80-120%
      # original scale = 0.05
      scale_factor = np.clip(np.random.normal(loc=1.0,scale=0.5), 0.8, 1.2)
      image = self.scale(image, scale_factor)
        
    if self.flips and random.random()<0.5:
      image = self.flip(image)
    
    if self.transform:
      image = self.transform(image)
    # window and levelling
    image = self.windowLevelNormalize(image, level=40, window=50)

    return image, label, patient

  def windowLevelNormalize(self, image, level, window):
    minval = level - window/2
    maxval = level + window/2
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / window)
    return wld

  def scale(self, image, scale_factor):
    # scale the image or mask using scipy zoom function
    order, cval = (3, 0) # changed from -1024 to 0
    height, width, depth = image.shape
    zheight = int(np.round(scale_factor*height))
    zwidth = int(np.round(scale_factor*width))
    zdepth = int(np.round(scale_factor*depth))
    # zoomed out
    if scale_factor < 1.0:
      new_image = np.full_like(image, cval)
      ud_buffer = (height-zheight) // 2
      ap_buffer = (width-zwidth) // 2
      lr_buffer = (depth-zdepth) // 2
      new_image[ud_buffer:ud_buffer+zheight, ap_buffer:ap_buffer+zwidth,
       lr_buffer:lr_buffer+zdepth] = zoom(input=image, zoom=scale_factor, 
       order=order, mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]
      return new_image
    elif scale_factor > 1.0:
      new_image = zoom(input=image, zoom=scale_factor, order=order, 
      mode='constant', cval=cval)[0:zheight, 0:zwidth, 0:zdepth]

      ud_extra = (new_image.shape[0] - height) // 2
      ap_extra = (new_image.shape[1] - width) // 2
      lr_extra = (new_image.shape[2] - depth) // 2
      new_image = new_image[ud_extra:ud_extra+height, ap_extra:ap_extra+width, 
      lr_extra:lr_extra+depth]

      return new_image
    return image
  
  def rotation(self, image, rotation_angle, rotation_plane):
    # rotate the image using scipy rotate function
    order, cval = (3, -1024) # changed from -1024 to 0
    return rotate(input=image, angle=rotation_angle, axes=rotation_plane, 
    reshape=False, order=order, mode='constant', cval=cval)

  def flip(self, image):
    image = np.flip(image).copy()
    return image