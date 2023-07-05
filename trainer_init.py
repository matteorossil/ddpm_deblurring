# Matteo Rossi


from typing import List
import os
import sys
import torch
import torch.utils.data
from eps_models.initial_predictor import UNet
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
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 4, 8]
    # The list of booleans that indicate whether to use attention at each resolution
    # Batch size
    batch_size: int = 32
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
    wandb: bool = True
    # where to store the checkpoints
    store_checkpoints: str = '/scratch/mr6744/pytorch/checkpoints_init_predictor/'
    #store_checkpoints: str = '/home/mr6744/checkpoints_init_predictor/'
    #store_checkpoints: str = '/Users/m.rossi/Desktop/research/'
    # where to training and validation data is stored
    dataset: str = '/scratch/mr6744/pytorch/gopro/'
    #dataset: str = '/Users/m.rossi/Desktop/research/ddpm_deblurring/dataset/'
    # load from a checkpoint
    checkpoint_epoch: int = 12480
    checkpoint: str = f'/scratch/mr6744/pytorch/checkpoints_init_predictor/06302023_103807/checkpoint_{checkpoint_epoch}.pt'
    #checkpoint: str = f'/home/mr6744/checkpoints_init_predictor/06152023_200330/checkpoint_{checkpoint_epoch}.pt'

    def init(self, rank: int):
        # gpu id
        self.gpu_id = rank

        # Create $\epsilon_\theta(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers
        )

        self.eps_model = self.eps_model.to(self.gpu_id)
        self.eps_model = DDP(self.eps_model, device_ids=[self.gpu_id])

        # only load checpoint if model is trained
        if self.checkpoint_epoch != 0:
            checkpoint_ = torch.load(self.checkpoint)
            self.eps_model.module.load_state_dict(checkpoint_)

        # Create dataloader
        dataset_train = Data(path=self.dataset, mode="train", size=(self.image_size,self.image_size))
        dataset_val = Data(path=self.dataset, mode="val", size=(self.image_size,self.image_size))

        self.data_loader_train = DataLoader(dataset=dataset_train,
                                            batch_size=self.batch_size, 
                                            num_workers=20,
                                            #num_workers=os.cpu_count() // 4, 
                                            drop_last=True, 
                                            shuffle=False, 
                                            pin_memory=False,
                                            sampler=DistributedSampler(dataset_train))
        
        self.data_loader_val = DataLoader(dataset=dataset_val, 
                                          batch_size=self.n_samples, 
                                          num_workers=0, 
                                          #num_workers=os.cpu_count() // 4, 
                                          drop_last=True, 
                                          shuffle=False, 
                                          pin_memory=False,
                                          sampler=DistributedSampler(dataset_val))

        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.eps_model.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)
        self.step = 0
        self.exp_path = get_exp_path(path=self.store_checkpoints)

    def val(self, epoch):
        """
        ### Sample images
        """
        with torch.no_grad():

            sharp, blur = next(iter(self.data_loader_val))
            # push to device
            sharp = sharp.to(self.gpu_id)
            blur = blur.to(self.gpu_id)

            if epoch == 0:
                # save sharp images
                save_image(sharp, os.path.join(self.exp_path, f'epoch_{epoch}_sharp.png'))

                # save blur images
                save_image(blur, os.path.join(self.exp_path, f'epoch_{epoch}_blur.png'))

            # predicted
            save_image(self.eps_model(blur), os.path.join(self.exp_path, f'epoch_{epoch}_deblurred.png'))

    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset
        for batch_idx, (sharp, blur) in enumerate(self.data_loader_train):
            # Increment global step
            self.step += 1
            # Move data to device
            sharp = sharp.to(self.gpu_id)
            blur = blur.to(self.gpu_id)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = F.mse_loss(sharp, self.eps_model(blur))
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            if self.wandb and self.gpu_id == 0:
                wandb.log({'loss': loss}, step=self.step)

    def run(self):
        """
        ### Training loop
        """
        for epoch in range(self.epochs):
            if (epoch == 0) and (self.gpu_id == 0):
                self.val(epoch)
            # Train the model
            self.train()
            if ((epoch+1) % 20 == 0) and (self.gpu_id == 0):
                # Save the eps model
                self.val(self.checkpoint_epoch+epoch+1)
                torch.save(self.eps_model.module.state_dict(), os.path.join(self.exp_path, f'checkpoint_{self.checkpoint_epoch+epoch+1}.pt'))

def ddp_setup(rank, world_size):
    """
    Args:
        rank: unique identifier of each process
        world_size: total number of processes
    """ 
    # IP address of machine running rank 0 process
    # master: machine coordinates communication across processes
    os.environ["MASTER_ADDR"] = "localhost" # we assume a single machine setup)
    os.environ["MASTER_PORT"] = "12355" # any free port on machine
    # nvidia collective comms library (comms across CUDA GPUs)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size:int):
    ddp_setup(rank=rank, world_size=world_size)
    trainer = Trainer()
    trainer.init(rank) # initialize trainer class

    #### Track Hyperparameters ####
    if trainer.wandb and rank == 0:

        params = sum(p.numel() for p in trainer.eps_model.parameters() if p.requires_grad)

        wandb.init(
                project="deblurring",
                name=f"initial predictor",
                config=
                {
                "GPUs": world_size,
                "GPU Type": torch.cuda.get_device_name(rank),
                "initial predictor # params": params,
                "dataset": trainer.dataset,
                "loaded from checkpoint": trainer.checkpoint,
                "checkpoints saved at": trainer.exp_path
                }
            )
    
    trainer.run() # perform training
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() # how many GPUs available in the machine
    #world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size)