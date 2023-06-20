# Matteo Rossi


from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm_unconditioned import DenoiseDiffusion
from eps_models.unet_unconditioned import UNet
import torch.nn.functional as F

from dataset import Data
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
    n_steps: int = int(sys.argv[1])
    # Number of sample images
    n_samples: int = int(sys.argv[2])
    # checkpoint path
    epoch = int(sys.argv[3])
    #checkpoint = f'/scratch/mr6744/pytorch/checkpoints_distributed/06132023_202606/checkpoint_{epoch}.pt'
    checkpoint = f'/home/mr6744/checkpoints_distributed/checkpoint_{epoch}.pt'
    # store sample
    #sampling_path = '/scratch/mr6744/pytorch/checkpoints_distributed/06132023_202606/sampling/'
    sampling_path = '/home/mr6744/checkpoints_distributed/sampling/'
    # dataset
    #dataset: str = '/scratch/mr6744/pytorch/gopro_128/'
    dataset: str = '/home/mr6744/gopro_ALL_128/'

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

        dataset = Data(path=self.dataset, mode="train", size=(self.image_size,self.image_size))
        self.dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.n_samples, 
                                    num_workers=0,
                                    drop_last=True, 
                                    shuffle=False, 
                                    pin_memory=False)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():

            # Set seed for replicability
            torch.cuda.manual_seed_all(0)

            # Sample Initial Image (Random Gaussian Noise)
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

            _, (sharp, blur) = next(iter(self.dataloader))
            save_image(sharp, os.path.join(self.sampling_path, f"sharp.png"))
            save_image(blur, os.path.join(self.sampling_path, f"blur.png"))
            print("done")
            
            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):

                print(t_)

                t = self.n_steps - t_ - 1

                # Sample
                t_vec = x.new_full((self.n_samples,), t, dtype=torch.long)
                x = self.diffusion.p_sample(x, t_vec)

                # Normalize img
                #min_val = x.min(-1)[0].min(-1)[0]
                #max_val = x.max(-1)[0].max(-1)[0]
                #x_norm = (x-min_val[:,:,None,None])/(max_val[:,:,None,None]-min_val[:,:,None,None])

                # save sampled images
                if ((t_+1) % self.n_steps == 0):
                    pass
                    #save_image(x, os.path.join(self.sampling_path, f"epoch{self.epoch}_t{t_+1}.png"))
                    #save_image(x_norm, os.path.join(self.sampling_path, f"epoch{self.epoch}_t{t_+1}_norm.png"))

            return x

def main():
    trainer = Trainer()
    trainer.init()
    trainer.sample()

if __name__ == "__main__":
    main()
