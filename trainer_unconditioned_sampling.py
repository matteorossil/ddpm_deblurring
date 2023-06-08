# Matteo Rossi


from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm_unconditioned import DenoiseDiffusion
from eps_models.unet_unconditioned import UNet
from pathlib import Path
from datetime import datetime
import wandb
import torch.nn.functional as F

from dataset_unconditioned import Data
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

class Trainer():
    """
    ## Configurations
    """
    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 128
    # Number of channels in the initial feature map
    n_channels: int = 32
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 3, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, False]
    attention_middle: List[int] = [False]
    # noise scheduler
    beta_0 = 1e-6 # 0.000001
    beta_T = 1e-2 # 0.01

    # Define sampling

    # Number of time steps $T$
    n_steps: int = 2_000
    # Number of sample images
    n_samples: int = 1
    # checkpoint path
    checkpoint = '/Users/m.rossi/Desktop/research/results/unconditioned/checkpoint_1150.pt'
    # store sample
    sample = '/Users/m.rossi/Desktop/research/results/unconditioned/'

    def init(self):
        # device
        self.device: torch.device = 'cuda' # change to 'cuda'

        # Create $\epsilon_\theta(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
            attn_middle=self.attention_middle
        ).to(self.device)
        
        #load cfrom checkpoint
        checkpoint_ = torch.load(self.checkpoint, map_location=torch.device(self.device))
        self.eps_model.load_state_dict(checkpoint_)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
            beta_0=self.beta_0,
            beta_T=self.beta_T
        )

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():

            # Sample Initial Image (Random Gaussian Noise)
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

            print(x)
            
            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):

                print(t_)

                t = self.n_steps - t_ - 1

                # Sample
                t_vec = x.new_full((self.n_samples,), t, dtype=torch.long)
                x = self.diffusion.p_sample(x, t_vec)

                #if ((t_+1) % 1800 == 0) or ((t_+1) % 1900 == 0) or ((t_+1) % 2000 == 0):
                    # save sampled images
                    #save_image(x, self.sample + )
                    #torch.save(x, os.path.join(self.exp_path, f'epoch{epoch}_gpu{self.gpu_id}_t{t_+1}.pt'))

            return x

def main():
    trainer = Trainer()
    trainer.init()
    trainer.sample()

if __name__ == "__main__":
    main()