# Matteo Rossi


from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm_conditioned import DenoiseDiffusion
from eps_models.denoiser import UNet as Denoiser # conditioned
from eps_models.initial_predictor import UNet as InitP # simple Unet (doesn't take t as param)
from pathlib import Path
from datetime import datetime
import wandb
import torch.nn.functional as F

from dataset import Data
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def get_exp_path(path=''):
    exp_path = os.path.join(path, datetime.now().strftime("%m%d%Y_%H%M%S"))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    return exp_path

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
    channel_multipliers: List[int] = [1, 2, 4, 8]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, False]
    # Number of time steps $T$
    n_steps: int = 2_000
    # noise scheduler Beta_0
    beta_0 = 1e-6 # 0.000001
    # noise scheduler Beta_T
    beta_T = 1e-2 # 0.01
    
    # Number of sample images
    n_samples: int = 4
    # where to store the checkpoints
    #store_checkpoints: str = '/scratch/mr6744/pytorch/checkpoints_conditioned/'
    store_checkpoints: str = '/home/mr6744/checkpoints_conditioned/'
    # where to training and validation data is stored
    #dataset: str = '/scratch/mr6744/pytorch/gopro/'
    dataset: str = '/home/mr6744/gopro/'
    # load from a checkpoint
    checkpoint_denoiser_epoch: int = 2740
    checkpoint_init_epoch: int = 2740
    checkpoint_denoiser: str = f'/home/mr6744/checkpoints_conditioned/checkpoint_denoiser_{checkpoint_denoiser_epoch}.pt'
    checkpoint_init: str = f'/home/mr6744/checkpoints_conditioned/checkpoint_initpr_{checkpoint_init_epoch}.pt'
    #checkpoint: str = f'/home/mr6744/checkpoints_conditioned/06022023_001525/checkpoint_{checkpoint_epoch}.pt'
    sampling_path = '/home/mr6744/checkpoints_conditioned/sample/'

    def init(self):
        # gpu id
        self.device: torch.device = 'cuda' # change to 'cuda'

        # Create $\epsilon_\theta(x_t, t)$ model
        self.denoiser = Denoiser(
            image_channels=self.image_channels*2, # *2 because we concatenate xt with y
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers
        ).to(self.device)

        # initial prediction x_init
        self.init_predictor = InitP(
            image_channels=self.image_channels,
            n_channels=self.n_channels*2,
            ch_mults=self.channel_multipliers
        ).to(self.device)

        # only loads checkpoint if model is trained
        checkpoint_ = torch.load(self.checkpoint_denoiser)
        self.denoiser.load_state_dict(checkpoint_)
        
        checkpoint_2 = torch.load(self.checkpoint_init)
        self.init_predictor.load_state_dict(checkpoint_2)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.denoiser,
            predictor=self.init_predictor,
            n_steps=self.n_steps,
            device=self.device,
            beta_0=self.beta_0,
            beta_T=self.beta_T
        )
        # Create dataloader (shuffle False for validation)
        dataset_train = Data(path=self.dataset, mode="train", size=(self.image_size,self.image_size))
        dataset_val = Data(path=self.dataset, mode="val", size=(self.image_size,self.image_size))

        self.data_loader_train = DataLoader(dataset=dataset_train,
                                            batch_size=self.n_samples, 
                                            num_workers=0,
                                            drop_last=True, 
                                            shuffle=False, 
                                            pin_memory=False)
        
        self.data_loader_val = DataLoader(dataset=dataset_val, 
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

            sharp, blur = next(iter(self.data_loader_val))
            # push to device
            sharp = sharp.to(self.device)
            blur = blur.to(self.device)
            init = self.init_predictor(blur)

            #condition = blur # or condition = init 

            # Sample Initial Image (Random Gaussian Noise)
            torch.cuda.manual_seed(0)
            z = torch.randn([self.n_samples, self.image_channels, blur.shape[2], blur.shape[3]],
                            device=self.device)
            
            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $p_\theta(x_{t-1}|x_t)$
                t_vec = z.new_full((self.n_samples,), t, dtype=torch.long)
                z = self.diffusion.p_sample(z, blur, t_vec)

            # save sharp images
            save_image(sharp, os.path.join(self.sampling_path, f'epoch_{self.epoch}_sharp.png'))

            # save blur images
            save_image(blur, os.path.join(self.sampling_path, f'epoch_{self.epoch}_blur.png'))

            # sharp - blur
            save_image(sharp - blur, os.path.join(self.sampling_path, f'epoch_{self.epoch}_sharp-blur.png'))

            # sampled residual
            save_image(z, os.path.join(self.sampling_path, f'epoch_{self.epoch}_residual.png'))

            # prediction for sharp image
            save_image(init + z, os.path.join(self.sampling_path, f'epoch_{self.epoch}_final.png'))
            
            # initial predictor
            save_image(init, os.path.join(self.sampling_path, f'epoch_{self.epoch}_init.png'))

            # correct residual
            save_image(sharp - init, os.path.join(self.sampling_path, f'epoch_{self.epoch}_residual_correct.png'))

            return z

def main():
    trainer = Trainer()
    trainer.init()
    trainer.sample()

if __name__ == "__main__":
    main()