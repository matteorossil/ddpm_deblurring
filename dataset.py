# Matteo Rossi

import os

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.io import read_image

from PIL import Image
import random
import torch

class Data(Dataset):

    def __init__(self, path, mode='train', size=(128,128)):
        
        # store img path
        self.path = path

        # store mode type
        self.mode = mode

        #size of the crop
        self.size = size

        # used in tranformations
        self.angles = [90,180,270]

        # stores paths
        sharp_folder = os.path.join(path+mode, "sharp")
        blur_folder = os.path.join(path+mode, "blur")

        self.sharp_imgs = os.listdir(sharp_folder)
        self.sharp_imgs = [os.path.join(sharp_folder, img) for img in self.sharp_imgs]

        self.blur_imgs = os.listdir(blur_folder)
        self.blur_imgs = [os.path.join(blur_folder, img) for img in self.blur_imgs]

    def __len__(self):
        return len(self.sharp_imgs)
    
    def __getitem__(self, idx):

        sharp = Image.open(self.sharp_imgs[idx])
        blur = Image.open(self.blur_imgs[idx])
        
        if self.mode == 'train':
            return self.transform_train(sharp, blur)
        else:
            return self.transform_val(sharp, blur)

    def transform_train(self, sharp, blur):

        """
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(sharp, output_size=self.size)
        sharp = TF.crop(sharp, i, j, h, w)
        blur = TF.crop(blur, i, j, h, w)

        """
         # random horizontal flip
        if random.random() > 0.5:
            sharp = TF.hflip(sharp)
            blur = TF.hflip(blur)

        # Random vertical flip
        if random.random() > 0.5:
            sharp = TF.vflip(sharp)
            blur = TF.vflip(blur)

        # random rotation
        if random.random() > 0.5:
            angle = random.choice(self.angles)
            sharp = TF.rotate(sharp, angle)
            blur = TF.rotate(blur, angle)

        return TF.to_tensor(sharp), TF.to_tensor(blur)

    def transform_train2(self, sharp, blur):

        return TF.to_tensor(sharp), TF.to_tensor(blur)
    
    def transform_val(self, sharp, blur):

        # convert to tensors
        return TF.to_tensor(sharp), TF.to_tensor(blur)