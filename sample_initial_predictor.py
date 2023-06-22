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
    n_samples: int = int(sys.argv[2])
    #n_samples: int = 1
    # checkpoint path
    epoch: int = int(sys.argv[3])
    checkpoint: str = f'/scratch/mr6744/pytorch/checkpoints_init_predictor/06202023_123619/checkpoint_{epoch}.pt'
    #checkpoint = f'/home/mr6744/checkpoints_distributed/checkpoint_{epoch}.pt'
    # store sample
    sampling_path: str = '/scratch/mr6744/pytorch/checkpoints_init_predictor/sampling/'
    #sampling_path = '/home/mr6744/checkpoints_init_predictor/sampling/'
    # dataset
    dataset: str = '/scratch/mr6744/pytorch/gopro_128/'
    #dataset: str = '/home/mr6744/gopro_ALL_128/'

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

        dataset = Data(path=self.dataset, mode="val", size=(self.image_size,self.image_size))

        self.dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.n_samples, 
                                    num_workers=0,
                                    drop_last=True, 
                                    shuffle=True, 
                                    pin_memory=False)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():

            sharp, blur = next(iter(self.dataloader))
            sharp = sharp.to(self.device)
            blur = blur.to(self.device)

            save_image(sharp, os.path.join(self.sampling_path, f"sharp.png"))
            save_image(blur, os.path.join(self.sampling_path, f"blur.png"))

            deblurred = self.eps_model(blur)
            save_image(deblurred, os.path.join(self.sampling_path, f"deblurred_epoch{self.epoch}.png"))

def main():
    trainer = Trainer()
    trainer.init()
    trainer.sample()

if __name__ == "__main__":
    main()
