# Matteo Rossi

import os

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms as T

from PIL import Image
import random
import torch

class Data(Dataset):

    def __init__(self, path, mode='train', size=(128,128)):

        # only for validation
        #if mode == 'val':
            #torch.manual_seed(1)
            #torch.cuda.manual_seed_all(1)
            #random.seed(1)

        self.dataset_name = {
            'train': path + "train",
            'val': path + "val",
        }

        # size of the crop
        self.size = size

        # store mode
        self.mode = mode

        # angles for tranformations
        self.angles = [90,180,270]

        # stores paths
        if mode == "train":
            self.imgs_dir = os.path.join(self.dataset_name[mode], "sharp")
            self.imgs = os.listdir(self.imgs_dir)
            #self.imgs.sort()

            self.imgs_dir_blur = os.path.join(self.dataset_name[mode], "blur")
            self.imgs_blur = os.listdir(self.imgs_dir_blur)
            #self.imgs_blur.sort()

        else:
            self.imgs_dir_blur = os.path.join(self.dataset_name[mode], "blur")
            self.imgs_blur = os.listdir(self.imgs_dir_blur)

            self.imgs_dir = os.path.join(self.dataset_name[mode], "sharp")
            self.imgs = os.listdir(self.imgs_dir)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        
        if self.mode == 'train':
            sharp = Image.open(os.path.join(self.imgs_dir, self.imgs[idx])).convert('RGB')
            blur = Image.open(os.path.join(self.imgs_dir_blur, self.imgs_blur[idx])).convert('RGB')
            #return self.transform_train(sharp), self.transform_train(blur)
            return self.transform_train2(sharp, blur)
        
        else: # do not apply trainsfomation to validation set
            blur = Image.open(os.path.join(self.imgs_dir_blur, self.imgs_blur[idx])).convert('RGB')
            sharp = Image.open(os.path.join(self.imgs_dir, self.imgs[idx])).convert('RGB')
            return self.transform_val(sharp), self.transform_val(blur)
        
    def transform_train(self, sharp):

        i, j, h, w = T.RandomCrop.get_params(sharp, output_size=self.size)
        sharp = TF.crop(sharp, i, j, h, w)

        # random horizontal flip
        if random.random() > 0.5:
            sharp = TF.hflip(sharp)
        
        # random vertical flip
        if random.random() > 0.5:
            sharp = TF.vflip(sharp)
        
        # random rotation
        if random.random() > 0.5:
            angle = random.choice(self.angles)
            sharp = TF.rotate(sharp, angle)

        return TF.to_tensor(sharp)
    
    def transform_train2(self, sharp, blur):

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(sharp, output_size=self.size)
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
                
    def transform_val(self, img):

        return TF.to_tensor(img)