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
            #torch.manual_seed(1)
            #torch.cuda.manual_seed_all(1)
            #random.seed(1)

        self.dataset_name = {
            'train': path + "train",
            'val': path + "val",
        }

        #size of the crop
        self.size = size

        #store mode
        self.mode = mode

        # stores paths
        if mode == "train":
            self.imgs_dir = os.path.join(self.dataset_name[mode], "sharp")
        else:
            self.imgs_dir = os.path.join(self.dataset_name[mode], "blur")

        self.imgs = os.listdir(self.imgs_dir)
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        
        if self.mode == 'train':
            sharp = Image.open(os.path.join(self.imgs_dir, self.imgs[idx])).convert('RGB')
            return TF.to_tensor(sharp)
        else: # do not apply trainsfomation to validation set
            blur = Image.open(os.path.join(self.imgs_dir, self.imgs[idx])).convert('RGB')
            return TF.to_tensor(blur)

    def transform_train(self, sharp):

        # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(sharp, output_size=self.size)
        # sharp = TF.crop(sharp, i, j, h, w)

        # random horizontal flip
        if random.random() > 0.5:
            sharp = TF.hflip(sharp)

        # Random vertical flip
        if random.random() > 0.5:
            sharp = TF.vflip(sharp)

        return sharp