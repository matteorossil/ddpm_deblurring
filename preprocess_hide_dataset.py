# Matteo Rossi

import os

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from PIL import Image
import random
import torch

class Data(Dataset):

    def __init__(self, path, mode='train', size=(128,128)):

        self.dataset_name = {
            'train': path + "train",
            'val': path + "val",
        }

        #size of the crop
        self.size = size

        #store mode
        self.mode = mode

        # stores paths
        self.blur = self.dataset_name[self.mode]

        self.blur_imgs = os.listdir(self.blur)
        self.blur_imgs.sort()

    #def __len__(self):
        #return len(self.sharp_imgs)
    
    def __getitem__(self, idx):

        blur = Image.open(os.path.join(self.blur, self.blur_imgs[idx])).convert('RGB')
        
        save_image(blur, os.path.join('/scratch/mr6744/pytorch/HIDE/val', str(idx) + '.png'))

        blur = self.transform(blur)

        save_image(blur, os.path.join('/scratch/mr6744/pytorch/HIDE_128/val', str(idx) + '.png'))

    def transform(self, blur):

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(blur, output_size=self.size)
        #sharp = TF.crop(sharp, i, j, h, w)
        blur = TF.crop(blur, i, j, h, w)

        '''

        # random horizontal flip
        if random.random() > 0.5:
            sharp = TF.hflip(sharp)
            blur = TF.hflip(blur)

        # Random vertical flip
        if random.random() > 0.5:
            sharp = TF.vflip(sharp)
            blur = TF.vflip(blur)
        
        '''

        # convert to tensors
        blur = TF.to_tensor(blur)

        return blur

dataset = Data(path='/scratch/mr6744/pytorch/HIDE_dataset/', mode="val", size=(128,128))
dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, drop_last=False)

for _ in enumerate(dataloader):
    pass

