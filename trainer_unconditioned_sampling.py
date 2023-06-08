# Matteo Rossi


from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm_unconditioned import DenoiseDiffusion
from eps_models.unet_unconditioned import UNet
import torch.nn.functional as F

from dataset_unconditioned import Data
from torchvision.utils import save_image

import torch.multiprocessing as mp

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
    n_samples: int = int(sys.argv[1])
    # checkpoint path
    epoch = int(sys.argv[2])
    checkpoint = f'/home/mr6744/checkpoints_distributed/06082023_001509/checkpoint_{epoch}.pt'
    # store sample
    sampling_path = '/home/mr6744/checkpoints_distributed/06082023_001509/sampling/'

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
        checkpoint_ = torch.load(self.checkpoint)
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

            # Set seed for replicability
            torch.cuda.manual_seed(0)

            # Sample Initial Image (Random Gaussian Noise)
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

            #print(x)
            
            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):

                print(t_)

                t = self.n_steps - t_ - 1

                # Sample
                t_vec = x.new_full((self.n_samples,), t, dtype=torch.long)
                x = self.diffusion.p_sample(x, t_vec)

                # Normalize img
            min_val = x.min(-1)[0].min(-1)[0]
            max_val = x.max(-1)[0].max(-1)[0]
            x_norm = (x-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])

            # save sampled images
            #if ((t_+1) % 2000 == 0):
            save_image(x, os.path.join(self.sampling_path, f"epoch{self.epoch}_t{t_+1}.png"))
            save_image(x_norm, os.path.join(self.sampling_path, f"epoch{self.epoch}_t{t_+1}_norm.png"))
            #torch.save(x, os.path.join(self.exp_path, f'epoch{epoch}_gpu{self.gpu_id}_t{t_+1}.pt'))

            return x

def main():
    trainer = Trainer()
    trainer.init()
    trainer.sample()

if __name__ == "__main__":
    main()
