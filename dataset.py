# Matteo Rossi

import os

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

from PIL import Image
import random
import torch

class Data(Dataset):

    def __init__(self, path, mode='train', size=(128,128)):

        # only for validation
        #if mode == 'val':
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)

        self.dataset_name = {
            'train': path + "train",
            'val': path + "val",
        }

        #size of the crop
        self.size = size

        #store mode
        self.mode = mode

        # stores paths
        self.sharp = os.path.join(self.dataset_name[mode], "sharp")
        self.blur = os.path.join(self.dataset_name[mode], "blur")

        self.sharp_imgs = os.listdir(self.sharp)
        self.sharp_imgs.sort()

        self.blur_imgs = os.listdir(self.blur)
        self.blur_imgs.sort()

        assert len(self.sharp_imgs) == len(self.blur_imgs)

    def __len__(self):
        return len(self.sharp_imgs)
    
    def __getitem__(self, idx):

        sharp = Image.open(os.path.join(self.sharp, self.sharp_imgs[idx])).convert('RGB')
        blur = Image.open(os.path.join(self.blur, self.blur_imgs[idx])).convert('RGB')
        
        '''
        if self.mode == 'train':
            return self.transform_train(sharp, blur)
        else: # do not apply trainsfomation to validation set
            return self.transform_val(sharp, blur)
        '''

        return TF.to_tensor(sharp), TF.to_tensor(blur)

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

        # convert to tensors
        sharp = TF.to_tensor(sharp)
        blur = TF.to_tensor(blur)

        return sharp, blur
    
    def transform_val(self, sharp, blur):

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(sharp, output_size=self.size)
        sharp = TF.crop(sharp, i, j, h, w)
        blur = TF.crop(blur, i, j, h, w)

        # convert to tensors
        sharp = TF.to_tensor(sharp)
        blur = TF.to_tensor(blur)

        return sharp, blur