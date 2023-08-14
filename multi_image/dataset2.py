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

    def __init__(self, path, mode='train', crop_eval=False, size=(128,128), multiplier=1):
        
        # store img path
        self.path = path

        # store mode type
        self.mode = mode

        #size of the crop
        self.size = size

        # crop validation set
        self.crop_eval = crop_eval

        # used in tranformations
        self.angles = [90.,180.,270.]

        # stores paths
        self.sharp_folder = os.path.join(path+mode, "sharp")
        self.blur_folder = os.path.join(path+mode, "blur")

        self.sharp_imgs = os.listdir(self.sharp_folder)
        self.sharp_imgs.sort()
        self.blur_imgs = os.listdir(self.blur_folder)
        self.blur_imgs.sort()

        self.data_multiplicator = multiplier

        self.l = len(self.sharp_imgs)

        self.i = 0
        self.j = 0
        self.h = 0
        self.w = 0

        self.hflip = False
        self.vflip = False
        self.rotate = False
        self.angle = 0.

    def __len__(self):
        assert len(self.sharp_imgs) == len(self.blur_imgs)
        return len(self.sharp_imgs) * self.data_multiplicator
    
    def __getitem__(self, idx):

        idx = idx % (self.__len__() // self.data_multiplicator) # left frame

        left = idx - 1
        center = idx
        right = idx + 1
        
        if idx == 0: # if left endpoint, drop it
            left = center
            center = right
            right = right + 1
        
        if idx == self.l - 1: # if right endpoint, drop it
            right = center
            center = left
            left = left - 1
        
        sharp_left = Image.open(os.path.join(self.sharp_folder, self.sharp_imgs[left])).convert('RGB')
        blur_left = Image.open(os.path.join(self.blur_folder, self.blur_imgs[left])).convert('RGB')

        sharp = Image.open(os.path.join(self.sharp_folder, self.sharp_imgs[center])).convert('RGB')
        blur = Image.open(os.path.join(self.blur_folder, self.blur_imgs[center])).convert('RGB')

        sharp_right = Image.open(os.path.join(self.sharp_folder, self.sharp_imgs[right])).convert('RGB')
        blur_right = Image.open(os.path.join(self.blur_folder, self.blur_imgs[right])).convert('RGB')

        if self.mode == 'train':
            # Random crop
            self.i, self.j, self.h, self.w = transforms.RandomCrop.get_params(sharp, output_size=self.size)
            self.hflip = random.random() > 0.5
            self.vflip = random.random() > 0.5
            self.rotate = random.random() > 0.5
            if self.rotate: self.angle = random.choice(self.angles)
            return self.transform_train(sharp_left, blur_left), self.transform_train(sharp, blur), self.transform_train(sharp_right, blur_right)
        else:
            if self.crop_eval:
                self.i, self.j, self.h, self.w = transforms.RandomCrop.get_params(sharp, output_size=self.size)
                return self.transform_val2(sharp_left, blur_left), self.transform_val2(sharp, blur), self.transform_val2(sharp_right, blur_right)
            else:
                return self.transform_val(sharp_left, blur_left), self.transform_val(sharp, blur), self.transform_val(sharp_right, blur_right)

    def transform_train(self, sharp, blur):

        sharp = TF.crop(sharp, self.i, self.j, self.h, self.w)
        blur = TF.crop(blur, self.i, self.j, self.h, self.w)

        # random horizontal flip
        if self.hflip:
            sharp = TF.hflip(sharp)
            blur = TF.hflip(blur)

        # Random vertical flip
        if self.vflip:
            sharp = TF.vflip(sharp)
            blur = TF.vflip(blur)

        # random rotation
        if self.rotate:
            sharp = TF.rotate(sharp, self.angle)
            blur = TF.rotate(blur, self.angle)

        return TF.to_tensor(sharp), TF.to_tensor(blur)

    def transform_val(self, sharp, blur):

        # convert to tensors
        return TF.to_tensor(sharp), TF.to_tensor(blur)
    
    def transform_val2(self, sharp, blur):

        sharp = TF.crop(sharp, self.i, self.j, self.h, self.w)
        blur = TF.crop(blur, self.i, self.j, self.h, self.w)

        # convert to tensors
        return TF.to_tensor(sharp), TF.to_tensor(blur)