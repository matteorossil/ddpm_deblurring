# Matteo Rossi


from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm_conditioned import DenoiseDiffusion
#from eps_models.denoiser import UNet as Denoiser # conditioned
from eps_models.unet_conditioned import UNet as Denoiser # conditioned
from eps_models.initial_predictor import UNet as InitP # simple Unet (doesn't take t as param)
from pathlib import Path
from datetime import datetime
import wandb
import torch.nn.functional as F
from metrics import *
import numpy as np

from dataset import Data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from numpy import savetxt

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import matplotlib.pyplot as plt
import itertools

def get_exp_path(path=''):
    exp_path = os.path.join(path, datetime.now().strftime("%m%d%Y_%H%M%S"))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    return exp_path

def plot(steps, R, G, B, path):

    print("steps", steps[-1])

    plt.plot(steps, R, label='red', color='r')
    plt.plot(steps, G, label='green', color='g')
    plt.plot(steps, B, label='blu', color='b')

    plt.xlabel("training steps")
    plt.ylabel("channel average")
    plt.legend()
    plt.title('channel averages over training time')
    #plt.show()
    plt.savefig(path + f'{steps[-1]}.png')


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
    channel_multipliers2: List[int] = [1, 2, 3, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, False]
    attention_middle: List[int] = [False]
    # Number of time steps $T$
    n_steps: int = 1_000
    # noise scheduler Beta_0
    beta_0 = 1e-6 # 0.000001
    # noise scheduler Beta_T
    beta_T = 1e-2 # 0.01
    # Batch size
    batch_size: int = 6
    # Learning rate
    #learning_rate: float = 1e-4
    learning_rate: float = 2e-5
    # Weight decay rate
    weight_decay_rate: float = 1e-3
    # ema decay
    betas = (0.9, 0.999)
    # Number of training epochs
    epochs: int = 100_000
    # Number of sample images
    n_samples: int = 8
    # Use wandb
    wandb: bool = False
    # where to store the checkpoints
    #store_checkpoints: str = '/scratch/mr6744/pytorch/checkpoints_conditioned/'
    store_checkpoints: str = '/home/mr6744/checkpoints_conditioned/'
    # where to training and validation data is stored
    #dataset: str = '/scratch/mr6744/pytorch/gopro/'
    dataset: str = '/home/mr6744/gopro_128/'
    # load from a checkpoint
    checkpoint_denoiser_epoch: int = 0
    checkpoint_init_epoch: int = 16880 #0
    checkpoint_denoiser: str = f'/home/mr6744/checkpoints_conditioned/06302023_192836/checkpoint_denoiser_{checkpoint_denoiser_epoch}.pt'
    #checkpoint_init: str = f'/scratch/mr6744/pytorch/checkpoints_conditioned/06292023_100717/checkpoint__initpr_{checkpoint_init_epoch}.pt'
    checkpoint_init: str = f'/home/mr6744/checkpoints_init_predictor/checkpoint_{checkpoint_init_epoch}.pt'
    #checkpoint: str = f'/home/mr6744/checkpoints_conditioned/06022023_001525/checkpoint_{checkpoint_epoch}.pt'

    def init(self, rank: int):
        # gpu id
        self.gpu_id = rank

        self.denoiser = Denoiser(
            image_channels=self.image_channels*2,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers2,
            is_attn=self.is_attention,
            attn_middle=self.attention_middle
        ).to(self.gpu_id)

        '''
        self.denoiser = Denoiser(
            image_channels=self.image_channels*2,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers
            #is_attn=self.is_attention,
            #attn_middle=self.attention_middle
        ).to(self.gpu_id)
        '''

        # initial prediction x_init
        self.init_predictor = InitP(
            image_channels=self.image_channels,
            n_channels=self.n_channels*2,
            ch_mults=self.channel_multipliers
        ).to(self.gpu_id)

        self.denoiser = DDP(self.denoiser, device_ids=[self.gpu_id])
        self.init_predictor = DDP(self.init_predictor, device_ids=[self.gpu_id])

        # only loads checkpoint if model is trained
        if self.checkpoint_denoiser_epoch != 0:
            checkpoint_ = torch.load(self.checkpoint_denoiser)
            self.denoiser.module.load_state_dict(checkpoint_)
        
        if self.checkpoint_init_epoch != 0:
            checkpoint_2 = torch.load(self.checkpoint_init)
            self.init_predictor.module.load_state_dict(checkpoint_2)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.denoiser,
            predictor=self.init_predictor,
            n_steps=self.n_steps,
            device=self.gpu_id,
            beta_0=self.beta_0,
            beta_T=self.beta_T
        )
        # Create dataloader (shuffle False for validation)
        dataset_train = Data(path=self.dataset, mode="val", size=(self.image_size,self.image_size))
        dataset_val = Data(path=self.dataset, mode="val", size=(self.image_size,self.image_size))

        self.data_loader_train = DataLoader(dataset=dataset_train,
                                            batch_size=self.batch_size, 
                                            num_workers=0,
                                            #num_workers=os.cpu_count() // 4, 
                                            drop_last=True, 
                                            shuffle=False, 
                                            pin_memory=False,
                                            sampler=DistributedSampler(dataset_train, shuffle=False))
        
        self.data_loader_val = DataLoader(dataset=dataset_val, 
                                          batch_size=self.n_samples, 
                                          num_workers=0,
                                          #num_workers=os.cpu_count() // 4, 
                                          drop_last=True, 
                                          shuffle=False, 
                                          pin_memory=False,
                                          sampler=DistributedSampler(dataset_val, shuffle=False))

        # Create optimizer
        self.params_denoiser = list(self.denoiser.parameters())
        self.params_init = list(self.init_predictor.parameters())

        self.optimizer = torch.optim.AdamW(self.params_denoiser + self.params_init, lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)
        
        self.step = 0
        self.exp_path = get_exp_path(path=self.store_checkpoints)

    def sample(self, n_samples, epoch):
        """
        ### Sample images
        """
        with torch.no_grad():

            sharp, blur = next(iter(self.data_loader_val))
            # push to device
            sharp = sharp.to(self.gpu_id)
            blur = blur.to(self.gpu_id)
            init = self.diffusion.predictor(blur)
            residual = sharp - init
            #### init = self.init_predictor(blur)

            #condition = blur # or condition = init 

            # Sample Initial Image (Random Gaussian Noise)
            #torch.cuda.manual_seed(0)
            z = torch.randn([n_samples, self.image_channels, blur.shape[2], blur.shape[3]],device=self.gpu_id)
            #### z = blur


            '''
            t_step = torch.randint(0, self.n_steps, (self.batch_size,), device=sharp.device, dtype=torch.long)
            print("t_step:", t_step.item())
            noise = torch.randn_like(sharp)
            z = self.diffusion.q_sample(sharp, t_step, eps=noise)
            save_image(z, os.path.join(self.exp_path, f'epoch_{epoch}_xt.png'))
            xt_ = torch.cat((z, blur), dim=1)
            eps_theta = self.denoiser(xt_, t_step)
            loss = F.mse_loss(noise, eps_theta)
            print("val loss:", loss)
            '''

            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
            #### for t_ in range(t_step.item()):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $p_\theta(x_{t-1}|x_t)$
                t_vec = z.new_full((n_samples,), t, dtype=torch.long)
                z = self.diffusion.p_sample(z, blur, t_vec)

                #xt_ = torch.cat((z, blur), dim=1)
                #eps_theta = self.denoiser(xt_, t_step)
                #loss = F.mse_loss(noise, eps_theta)


            # Log samples
            #if self.wandb:
                #wandb.log({'samples': wandb.Image(x)}, step=self.step)

            if epoch == 0:
                # save sharp images
                save_image(sharp, os.path.join(self.exp_path, f'epoch_{epoch}_sharp_val.png'))

                # save blur images
                save_image(blur, os.path.join(self.exp_path, f'epoch_{epoch}_blur_val.png'))

                psnr_sharp_blur = psnr(sharp, blur)
                ssim_sharp_blur = ssim(sharp, blur)
                savetxt(os.path.join(self.exp_path, f"psnr_sharp_blur_epoch{epoch}.txt"), psnr_sharp_blur)
                savetxt(os.path.join(self.exp_path, f"ssim_sharp_blur_epoch{epoch}.txt"), ssim_sharp_blur)

            # residual
            save_image(residual, os.path.join(self.exp_path, f'epoch_{epoch}_residual_true.png'))

            # sharp - blur
            #### save_image(sharp - blur, os.path.join(self.exp_path, f'epoch_{epoch}_sharp-blur.png'))

            # sampled residual
            save_image(z, os.path.join(self.exp_path, f'epoch_{epoch}_residual_sample.png'))

            # sampled sharp
            save_image(init + z, os.path.join(self.exp_path, f'epoch_{epoch}_xt_sample.png'))
            psnr_sharp_deblurred = psnr(sharp, init + z)
            ssim_sharp_deblurred = ssim(sharp, init + z)
            savetxt(os.path.join(self.exp_path, f"psnr_sharp_deblurred_epoch{epoch}.txt"), psnr_sharp_deblurred)
            savetxt(os.path.join(self.exp_path, f"ssim_sharp_deblurred_epoch{epoch}.txt"), ssim_sharp_deblurred)

            save_image(init, os.path.join(self.exp_path, f'epoch_{epoch}_init.png'))
            psnr_sharp_init = psnr(sharp, init)
            ssim_sharp_init = ssim(sharp, init)
            savetxt(os.path.join(self.exp_path, f"psnr_sharp_init_epoch{epoch}.txt"), psnr_sharp_init)
            savetxt(os.path.join(self.exp_path, f"ssim_sharp_init_epoch{epoch}.txt"), ssim_sharp_init)

            # prediction for sharp image
            ### save_image(init + z, os.path.join(self.exp_path, f'epoch_{epoch}_final.png'))
            
            # initial predictor
            ### save_image(init, os.path.join(self.exp_path, f'epoch_{epoch}_init.png'))

            # correct residual
            ### save_image(sharp - init, os.path.join(self.exp_path, f'epoch_{epoch}_residual_correct.png'))

            return z

    def train(self, steps, R, G, B):
        """
        ### Train
        """
        # Iterate through the dataset
        #for batch_idx, (sharp, blur) in enumerate(self.data_loader_train):
        sharp, blur = next(iter(self.data_loader_train))

        # Move data to device
        sharp = sharp.to(self.gpu_id)
        blur = blur.to(self.gpu_id)
        init = self.diffusion.predictor(blur)
        residual = sharp - init

        # store mean value of channels
        r = torch.mean(init[:,0,:,:])
        R.append(r.item())
        g = torch.mean(init[:,1,:,:])
        G.append(g.item())
        b = torch.mean(init[:,2,:,:])
        B.append(b.item())
        print()

        steps.append(self.step)

        #if self.step == 0:
            #save_image(sharp, os.path.join(self.exp_path, f'epoch_{self.step}_sharp_train.png'))
            #save_image(blur, os.path.join(self.exp_path, f'epoch_{self.step}_blur_train.png'))
            #save_image(init, os.path.join(self.exp_path, f'epoch_{self.step}_init_train.png'))
            #save_image(residual, os.path.join(self.exp_path, f'epoch_{self.step}_residual_train.png'))

        # Increment global step
        self.step += 1
        steps.append(self.step)
        # Make the gradients zero
        self.optimizer.zero_grad()
        # Calculate loss
        loss = self.diffusion.loss(residual, blur) #+ F.mse_loss(sharp, init)
        print("loss:", loss.item())
        print("epoch:", self.step)
        # Compute gradients
        loss.backward()
        #print("############ GRAD OUTPUT ############")
        #print(self.denoiser.module.final.bias.grad)
        #print(self.init_predictor.module.final.bias.grad)

        # Take an optimization step
        self.optimizer.step()
        #self.optimizer2.step()
        # Track the loss
        if self.wandb and self.gpu_id == 0:
            wandb.log({'loss': loss}, step=self.step)

    def run(self):
        """
        ### Training loop
        """
        steps = []
        R = []
        G = []
        B = []
        
        for epoch in range(self.epochs):
            if (epoch == 0) and (self.gpu_id == 0):
                pass
                #self.sample(self.n_samples, epoch=0)
            # Train the model
            self.train(steps, R, G, B)

            if ((epoch+1) % 20 == 0) and (self.gpu_id == 0):
                plot(steps, R, G, B, self.exp_path)

            if ((epoch+1) % 2000 == 0) and (self.gpu_id == 0):
                # Save the eps model
                self.sample(self.n_samples, self.checkpoint_denoiser_epoch+epoch+1)
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
    os.environ["MASTER_PORT"] = "12358" # any free port on machine
    # nvidia collective comms library (comms across CUDA GPUs)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size:int):
    ddp_setup(rank=rank, world_size=world_size)
    trainer = Trainer()
    trainer.init(rank) # initialize trainer class
    #print(trainer.init_predictor)

    params_denoiser = sum(p.numel() for p in trainer.params_denoiser if p.requires_grad)
    print("denoiser params:", params_denoiser)
    init_denoiser = sum(p.numel() for p in trainer.params_init if p.requires_grad)
    print("init predictor params:", init_denoiser)

    #### Track Hyperparameters ####
    if trainer.wandb and rank == 0:

        params_denoiser = sum(p.numel() for p in trainer.params_denoiser if p.requires_grad)
        init_denoiser = sum(p.numel() for p in trainer.params_init if p.requires_grad)
        
        wandb.init(
            project="deblurring",
            name=f"conditioned p x|y",
            config=
            {
            "GPUs": world_size,
            "GPU Type": torch.cuda.get_device_name(rank),
            "freeze init": False,
            "pretrained init": trainer.checkpoint_init_epoch > 0,
            "conditioning": "blurred image",
            "dataset": trainer.dataset,
            "denoiser # params": params_denoiser,
            "init # params": init_denoiser,
            "loaded from checkpoint": trainer.checkpoint_init,
            "checkpoints saved at": trainer.exp_path
            }
        )
    ##### ####
    trainer.run() # perform training
    destroy_process_group()

if __name__ == "__main__":
    #world_size = torch.cuda.device_count() # how many GPUs available in the machine
    world_size = 1
    mp.spawn(main, args=(world_size,), nprocs=world_size)