# Matteo Rossi


from typing import List
import os
import sys
import torch
import torch.utils.data
from diffusion.ddpm_unconditioned import DenoiseDiffusion
from eps_models.unet_unconditioned import UNet
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
    channel_multipliers: List[int] = [1, 2, 3, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]
    attention_middle: List[int] = [True]
    # Number of time steps $T$
    n_steps: int = 1_000
    # noise scheduler
    beta_0 = 1e-6 # 0.000001
    beta_T = 1e-2 # 0.01
    # Batch size
    batch_size: int = 4
    # Learning rate
    learning_rate: float = 1e-4
    # Weight decay rate
    weight_decay_rate: float = 1e-3
    # ema decay
    betas = (0.9, 0.999)
    # Number of training epochs
    epochs: int = 100_000
    # Number of sample images
    n_samples: int = 4
    # Use wandb
    wandb: bool = False
    # where to store the checkpoints
    store_checkpoints: str = '/home/mr6744/checkpoints_unconditioned/'
    #store_checkpoints: str = '/home/mr6744/checkpoints_unconditioned/'
    # where to training and validation data is stored
    dataset: str = '/home/mr6744/gopro_128/'
    #dataset: str = '/home/mr6744/gopro/'
    # load from a checkpoint
    checkpoint_epoch: int = 0
    checkpoint: str = f'/scratch/mr6744/pytorch/checkpoints_unconditioned/06292023_221044/checkpoint_{checkpoint_epoch}.pt'
    #checkpoint: str = f'/home/mr6744/checkpoints_distributed/06092023_132041/checkpoint_{checkpoint_epoch}.pt'

    def init(self, rank: int):
        # gpu id
        self.gpu_id = rank

        # Create $\epsilon_\theta(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
            attn_middle=self.attention_middle
        )

        self.eps_model = self.eps_model.to(self.gpu_id)
        self.eps_model = DDP(self.eps_model, device_ids=[self.gpu_id])

        # only load checpoint if model is trained
        if self.checkpoint_epoch != 0:
            #map_location = {'cuda:%d' % 0: 'cuda:%d' % self.gpu_id}
            #checkpoint_ = torch.load(self.checkpoint, map_location=map_location)
            checkpoint_ = torch.load(self.checkpoint)
            self.eps_model.module.load_state_dict(checkpoint_)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.gpu_id,
            beta_0=self.beta_0,
            beta_T=self.beta_T
        )
        # Create dataloader
        dataset = Data(path=self.dataset, mode="train", size=(self.image_size,self.image_size))
        self.dataloader_train = DataLoader(dataset=dataset,
                                    batch_size=self.batch_size,
                                    num_workers=0,
                                    drop_last=False,
                                    shuffle=False,
                                    pin_memory=False,
                                    sampler=DistributedSampler(dataset, shuffle=False)) # assures no overlapping samples

        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.eps_model.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)
        self.step = 0
        self.exp_path = get_exp_path(path=self.store_checkpoints)

    def sample(self, n_samples, epoch):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            # Sample Initial Image (Random Gaussian Noise)
            #torch.cuda.manual_seed(0)
            x = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.gpu_id)
            # Remove noise for $T$ steps
            #for t_ in range(self.n_steps):
            for t_ in range(self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $p_\theta(x_{t-1}|x_t)$
                t_vec = x.new_full((n_samples,), t, dtype=torch.long)
                x = self.diffusion.p_sample(x, t_vec)
            
            save_image(x, os.path.join(self.exp_path, f'epoch_{epoch}_xt_sample.png'))

            # Log samples
            #if self.wandb:
                #wandb.log({'samples': wandb.Image(x)}, step=self.step)

            return x

    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset
        sharp, blur = next(iter(self.dataloader_train))
        if self.step == 0:
            save_image(sharp, os.path.join(self.exp_path, f'epoch_{self.step}_sharp_train.png'))

        # Increment global step
        self.step += 1
        # Move data to device
        sharp = sharp.to(self.gpu_id)
        # Make the gradients zero
        self.optimizer.zero_grad()
        # Calculate loss
        loss = self.diffusion.loss(sharp)
        print("loss:", loss.item())
        print("epoch:", self.step)
        # Compute gradients
        loss.backward()
        # Take an optimization step
        self.optimizer.step()
        # Track the loss
        #print('loss:', loss, 'step:', self.step)
        if self.wandb and self.gpu_id == 0:
            wandb.log({'loss': loss}, step=self.step)

    def run(self):
        """
        ### Training loop
        """
        for epoch in range(self.epochs):
            if (epoch == 0) and (self.gpu_id == 0):
                self.sample(self.n_samples, epoch=0)
            # Train the model
            self.train()
            if ((epoch+1) % 500 == 0) and (self.gpu_id == 0):
                # Save the eps model
                self.sample(self.n_samples, self.checkpoint_epoch+epoch+1)
                #### torch.save(self.denoiser.module.state_dict(), os.path.join(self.exp_path, f'checkpoint_denoiser_{self.checkpoint_denoiser_epoch+epoch+1}.pt'))
                #### torch.save(self.init_predictor.module.state_dict(), os.path.join(self.exp_path, f'checkpoint_initpr_{self.checkpoint_denoiser_epoch+epoch+1}.pt'))

def ddp_setup(rank, world_size):
    """
    Args:
        rank: unique identifier of each process
        world_size: total number of processes
    """ 
    # IP address of machine running rank 0 process
    # master: machine coordinates communication across processes
    os.environ["MASTER_ADDR"] = "localhost" # we assume a single machine setup)
    os.environ["MASTER_PORT"] = "12357" # any free port on machine
    # nvidia collective comms library (comms across CUDA GPUs)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size:int):
    ddp_setup(rank=rank, world_size=world_size)
    trainer = Trainer()
    trainer.init(rank) # initialize trainer class

    params = sum(p.numel() for p in trainer.eps_model.parameters() if p.requires_grad)
    print("denoiser params:", params)
    
    if trainer.wandb and rank == 0:

        params = sum(p.numel() for p in trainer.eps_model.parameters() if p.requires_grad)
        
        wandb.init(
            project="deblurring",
            name=f"unconditioned",
            config=
            {
            "GPUs": world_size,
            "GPU Type": torch.cuda.get_device_name(rank),
            "denoiser # params": params,
            "dataset": trainer.dataset,
            "loaded from checkpoint": trainer.checkpoint,
            "checkpoints saved at": trainer.exp_path
            }
        )

    trainer.run() # perform training
    destroy_process_group()

if __name__ == "__main__":
    #world_size = torch.cuda.device_count() # how many GPUs available in the machine
    world_size = 1
    mp.spawn(main, args=(world_size,), nprocs=world_size)