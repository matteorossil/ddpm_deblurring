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
        self.blur_imgs = os.listdir(self.blur_folder)

        self.data_multiplicator = multiplier

    def __len__(self):
        assert len(self.sharp_imgs) == len(self.blur_imgs)
        return len(self.sharp_imgs) * self.data_multiplicator
    
    def __getitem__(self, idx):

        idx = idx % (self.__len__() // self.data_multiplicator)

        sharp = Image.open(os.path.join(self.sharp_folder, self.sharp_imgs[idx])).convert('RGB')
        blur = Image.open(os.path.join(self.blur_folder, self.blur_imgs[idx])).convert('RGB')

        if self.mode == 'train':
            return self.transform_train(sharp, blur)
        else:
            if self.crop_eval:
                return self.transform_val2(sharp, blur)
            else:
                return self.transform_val(sharp, blur)

    def transform_train(self, sharp, blur):

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(sharp, output_size=self.size)
        sharp = TF.crop(sharp, i, j, h, w)
        blur = TF.crop(blur, i, j, h, w)

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
    
    def transform_val(self, sharp, blur):

        # convert to tensors
        return TF.to_tensor(sharp), TF.to_tensor(blur)
    
    def transform_val2(self, sharp, blur):

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(sharp, output_size=self.size)
        sharp = TF.crop(sharp, i, j, h, w)
        blur = TF.crop(blur, i, j, h, w)

        # convert to tensors
        return TF.to_tensor(sharp), TF.to_tensor(blur)