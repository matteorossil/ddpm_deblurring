# Matteo Rossi


from typing import List
import os
import torch
import torch.utils.data
from eps_models.initial_predictor import UNet
import torch.nn.functional as F

from dataset import Data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import gather
from metrics import *
from numpy import savetxt
import numpy as np

import sys

class Trainer():
    """
    ## Configurations
    """
    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 128
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 4, 8]

    # Define sampling

    # Number of sample images
    n_samples: int = 4
    # checkpoint path
    epoch: int = 5600
    #checkpoint: str = f'/scratch/mr6744/pytorch/checkpoints_init_predictor/06182023_103900/checkpoint_{epoch}.pt'
    checkpoint = f'/home/mr6744/checkpoints_init_predictor/checkpoint_{epoch}.pt'
    # store sample
    #sampling_path: str = f'/scratch/mr6744/pytorch/checkpoints_init_predictor/sample/'
    sampling_path = '/home/mr6744/checkpoints_init_predictor/sample3/'
    # dataset
    #dataset: str = '/scratch/mr6744/pytorch/gopro/'
    dataset: str = '/home/mr6744/gopro/'

    def init(self):
        # device
        self.device: torch.device = 'cuda' # change to 'cuda'

        # Create $\epsilon_\theta(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
        ).to(self.device)
        
        #load cfrom checkpoint
        checkpoint_ = torch.load(self.checkpoint)
        self.eps_model.load_state_dict(checkpoint_)

        dataset_train = Data(path=self.dataset, mode="train", size=(self.image_size,self.image_size))
        dataset_val = Data(path=self.dataset, mode="val", size=(self.image_size,self.image_size))

        self.dataloader_train = DataLoader(dataset=dataset_train,
                                    batch_size=self.n_samples, 
                                    num_workers=0,
                                    drop_last=True, 
                                    shuffle=True, 
                                    pin_memory=False)

        self.dataloader_val = DataLoader(dataset=dataset_val,
                                    batch_size=self.n_samples, 
                                    num_workers=0,
                                    drop_last=True, 
                                    shuffle=True, 
                                    pin_memory=False)

    def test(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            
            # training dataset
            sharp_train, blur_train = next(iter(self.dataloader_train))
            sharp_train = sharp_train.to(self.device)
            blur_train = blur_train.to(self.device)

            save_image(sharp_train, os.path.join(self.sampling_path, f"sharp_train.png"))
            save_image(blur_train, os.path.join(self.sampling_path, f"blur_train.png"))

            deblurred_train = self.eps_model(blur_train)
            save_image(deblurred_train, os.path.join(self.sampling_path, f"deblurred_train_epoch{self.epoch}.png"))

            # compute psnr for train
            psnr_train1 = psnr(sharp_train, blur_train)
            savetxt(os.path.join(self.sampling_path, f"psnr_train_blur_epoch{self.epoch}.txt"), psnr_train1)
            savetxt(os.path.join(self.sampling_path, f"psnr_train_blur_epoch{self.epoch}_avg.txt"), np.array([np.mean(psnr_train1)]))
            psnr_train2 = psnr(sharp_train, deblurred_train)
            savetxt(os.path.join(self.sampling_path, f"psnr_train_deblurred_epoch{self.epoch}.txt"), psnr_train2)
            savetxt(os.path.join(self.sampling_path, f"psnr_train_deblurred_epoch{self.epoch}_avg.txt"), np.array([np.mean(psnr_train2)]))

            # compute ssim for train
            ssim_train1 = ssim(sharp_train, blur_train)
            savetxt(os.path.join(self.sampling_path, f"ssim_train_blur_epoch{self.epoch}.txt"), ssim_train1)
            savetxt(os.path.join(self.sampling_path, f"ssim_train_blur_epoch{self.epoch}_avg.txt"), np.array([np.mean(ssim_train1)]))
            ssim_train2 = ssim(sharp_train, deblurred_train)
            savetxt(os.path.join(self.sampling_path, f"ssim_train_deblurred_epoch{self.epoch}.txt"), ssim_train2)
            savetxt(os.path.join(self.sampling_path, f"ssim_train_deblurred_epoch{self.epoch}_avg.txt"), np.array([np.mean(ssim_train2)]))

            # validation dataset
            sharp_val, blur_val = next(iter(self.dataloader_val))
            sharp_val = sharp_val.to(self.device)
            blur_val = blur_val.to(self.device)

            save_image(sharp_val, os.path.join(self.sampling_path, f"sharp_val.png"))
            save_image(blur_val, os.path.join(self.sampling_path, f"blur_val.png"))

            deblurred_val = self.eps_model(blur_val)
            save_image(deblurred_val, os.path.join(self.sampling_path, f"deblurred_val_epoch{self.epoch}.png"))

            # compute psnr for val
            psnr_val1 = psnr(sharp_val, blur_val)
            savetxt(os.path.join(self.sampling_path, f"psnr_val_blur_epoch{self.epoch}.txt"), psnr_val1)
            savetxt(os.path.join(self.sampling_path, f"psnr_val_blur_epoch{self.epoch}_avg.txt"), np.array([np.mean(psnr_val1)]))
            psnr_val2 = psnr(sharp_val, deblurred_val)
            savetxt(os.path.join(self.sampling_path, f"psnr_val_deblurred_epoch{self.epoch}.txt"), psnr_val2)
            savetxt(os.path.join(self.sampling_path, f"psnr_val_deblurred_epoch{self.epoch}_avg.txt"), np.array([np.mean(psnr_val2)]))

            # compute ssim for val
            ssim_val1 = ssim(sharp_val, blur_val)
            savetxt(os.path.join(self.sampling_path, f"ssim_val_blur_epoch{self.epoch}.txt"), ssim_val1)
            savetxt(os.path.join(self.sampling_path, f"ssim_val_blur_epoch{self.epoch}_avg.txt"), np.array([np.mean(ssim_val1)]))
            ssim_val2 = ssim(sharp_val, deblurred_val)
            savetxt(os.path.join(self.sampling_path, f"ssim_val_deblurred_epoch{self.epoch}.txt"), ssim_val2)
            savetxt(os.path.join(self.sampling_path, f"ssim_val_deblurred_epoch{self.epoch}_avg.txt"), np.array([np.mean(ssim_val2)]))


def main():
    trainer = Trainer()
    trainer.init()
    trainer.test()

if __name__ == "__main__":
    main()
