## Matteo Rossi

# Modules
from dataset import Data
from metrics import psnr, ssim
from eps_models.unet_conditioned import UNet as Denoiser
from eps_models.init_predictor_new import UNet as Init
from diffusion.ddpm_conditioned import DenoiseDiffusion


# Torch
import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Numpy
import numpy as np
from numpy import savetxt

# Other
import os
from typing import List
from pathlib import Path
from datetime import datetime
import wandb
import matplotlib.pyplot as plt

# DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def get_exp_path(path=''):
    exp_path = os.path.join(path, datetime.now().strftime("%m%d%Y_%H%M%S"))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    return exp_path

def plot_channels(steps, R, G, B, path, title):

    plt.plot(steps, R, label='red', color='r')
    plt.plot(steps, G, label='green', color='g')
    plt.plot(steps, B, label='blu', color='b')

    plt.xlabel("training steps")
    plt.ylabel("channel average")
    plt.legend()
    plt.title(title)
    #plt.show()
    plt.savefig(path + f'/channel_avgs_steps{steps[-1]}.png')
    plt.figure().clear()
    plt.close('all')

def plot_loss(steps, loss, path, title):

    plt.plot(steps, loss, label='red', color='r')

    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.legend()
    plt.title(title)
    #plt.show()
    plt.savefig(path + f'/loss_steps{steps[-1]}.png')
    plt.figure().clear()
    plt.close('all')

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
    # Number of time steps $T$
    n_steps: int = 1_000
    # noise scheduler Beta_0
    beta_0 = 1e-6 # 0.000001
    # noise scheduler Beta_T
    beta_T = 1e-2 # 0.01
    # Batch size
    batch_size: int = 1
    # Learning rate
    learning_rate: float = 1e-4
    # Weight decay rate
    weight_decay_rate: float = 1e-3
    # ema decay
    betas = (0.9, 0.999)
    # Number of training epochs
    epochs: int = 100_000
    # Number of samples (evaluation)
    n_samples: int = 1
    # Use wandb
    wandb: bool = False
    # checkpoints path
    store_checkpoints: str = '/home/mr6744/ckpts/'
    #store_checkpoints: str = '/scratch/mr6744/pytorch/checkpoints_conditioned/'
    # dataset path
    dataset: str = '/home/mr6744/gopro_128/'
    #dataset: str = '/scratch/mr6744/pytorch/gopro/'
    # load from a checkpoint
    ckpt_denoiser_epoch: int = 0
    ckpt_initP_epoch: int = 0
    ckpt_denoiser: str = f'/home/mr6744/ckpts/06302023_192836/checkpoint_denoiser_{ckpt_denoiser_epoch}.pt'
    #checkpoint_init: str = f'/scratch/mr6744/pytorch/checkpoints_conditioned/06292023_100717/checkpoint__initpr_{checkpoint_init_epoch}.pt'
    ckpt_init: str = f'/home/mr6744/checkpoints_init_predictor/checkpoint_{ckpt_initP_epoch}.pt'
    #checkpoint: str = f'/home/mr6744/checkpoints_conditioned/06022023_001525/checkpoint_{checkpoint_epoch}.pt'

    def init(self, rank: int):
        # gpu id
        self.gpu_id = rank

        self.denoiser = Denoiser(
            image_channels=self.image_channels*2,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention
        ).to(self.gpu_id)
        
        self.initP = Init(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention
        ).to(self.gpu_id)

        self.denoiser = DDP(self.denoiser, device_ids=[self.gpu_id])
        self.initP = DDP(self.initP, device_ids=[self.gpu_id])

        # only loads checkpoint if model is trained
        if self.ckpt_denoiser_epoch != 0:
            checkpoint_d = torch.load(self.ckpt_denoiser)
            self.denoiser.module.load_state_dict(checkpoint_d)
        
        if self.ckpt_initP_epoch != 0:
            checkpoint_i = torch.load(self.ckpt_init)
            self.initP.module.load_state_dict(checkpoint_i)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.denoiser,
            predictor=self.initP,
            n_steps=self.n_steps,
            beta_0=self.beta_0,
            beta_T=self.beta_T,
            device=self.gpu_id
        )

        # Create dataloader (shuffle False for validation)
        dataset_train = Data(path=self.dataset, mode="val", size=(self.image_size,self.image_size))
        dataset_val = Data(path=self.dataset, mode="val", size=(self.image_size,self.image_size))

        self.dataloader_train = DataLoader(dataset=dataset_train,
                                            batch_size=self.batch_size, 
                                            num_workers=0, # os.cpu_count() // 4,
                                            drop_last=True, 
                                            shuffle=False, 
                                            pin_memory=False,
                                            sampler=DistributedSampler(dataset_train, shuffle=False))
        
        self.dataloader_val = DataLoader(dataset=dataset_val, 
                                          batch_size=self.n_samples, 
                                          num_workers=0, # os.cpu_count() // 4,
                                          drop_last=True, 
                                          shuffle=False, 
                                          pin_memory=False,
                                          sampler=DistributedSampler(dataset_val, shuffle=False))

        # Create optimizer
        self.params_denoiser = list(self.denoiser.parameters())
        self.num_params_denoiser = sum(p.numel() for p in self.params_denoiser if p.requires_grad)

        self.params_init = list(self.initP.parameters())
        self.num_params_init = sum(p.numel() for p in self.params_init if p.requires_grad)

        params = self.params_denoiser + self.params_init
        self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)
        
        # path 
        self.step = 0
        self.exp_path = get_exp_path(path=self.store_checkpoints)

        #sigmoid
        self.sigmoid = nn.Sigmoid()

    def sample(self, epoch):

        with torch.no_grad():

            sharp, blur = next(iter(self.dataloader_val))
            
            sharp = sharp.to(self.gpu_id)
            blur = blur.to(self.gpu_id)

            # compute initial predictor
            init = self.diffusion.predictor(blur)
            # get true residual
            X_true = sharp - init

            # Sample X from Gaussian Noise
            #torch.cuda.manual_seed(0)
            X = torch.randn([self.n_samples, self.image_channels, blur.shape[2], blur.shape[3]],device=self.gpu_id)

            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                
                # e.g. t_ from 999 to 0 for 1_000 time steps
                t = self.n_steps - t_ - 1

                # create a t for every sample in batch
                t_vec = X.new_full((self.n_samples,), t, dtype=torch.long)

                # take one denoising step
                X = self.diffusion.p_sample(X, blur, t_vec)

            if epoch == 0:
                # save images blur and sharp image pairs
                save_image(sharp, os.path.join(self.exp_path, f'sharp_val.png'))
                save_image(blur, os.path.join(self.exp_path, f'blur_val.png'))
                
                # compute metrics for blur sharp pairs
                psnr_sharp_blur = psnr(sharp, blur)
                ssim_sharp_blur = ssim(sharp, blur)
                savetxt(os.path.join(self.exp_path, f"psnr_sharp_blur_avg.txt"), np.array([np.mean(psnr_sharp_blur)]))
                savetxt(os.path.join(self.exp_path, f"ssim_sharp_blur_avg.txt"), np.array([np.mean(ssim_sharp_blur)]))
                #savetxt(os.path.join(self.exp_path, f"psnr_sharp_blur_epoch{epoch}.txt"), psnr_sharp_blur)
                #savetxt(os.path.join(self.exp_path, f"ssim_sharp_blur_epoch{epoch}.txt"), ssim_sharp_blur)


            # save initial predictor
            save_image(init, os.path.join(self.exp_path, f'init_epoch{epoch}.png'))
            # save true residual
            save_image(X_true, os.path.join(self.exp_path, f'residual_true_epoch{epoch}.png'))
            # save sampled residual
            save_image(X, os.path.join(self.exp_path, f'residual_sampled_epoch{epoch}.png'))
            # save sampled deblurred
            save_image(init + X, os.path.join(self.exp_path, f'deblurred_epoch{epoch}.png'))

            # compute metrics (sharp, init)
            psnr_sharp_init = psnr(sharp, init)
            ssim_sharp_init = ssim(sharp, init)
            savetxt(os.path.join(self.exp_path, f"psnr_sharp_init_avg_epoch{epoch}.txt"), np.array([np.mean(psnr_sharp_init)]))
            savetxt(os.path.join(self.exp_path, f"ssim_sharp_init_avg_epoch{epoch}.txt"), np.array([np.mean(ssim_sharp_init)]))

            # compute metrics (sharp, deblurred)
            psnr_sharp_deblurred = psnr(sharp, init + X)
            ssim_sharp_deblurred = ssim(sharp, init + X)
            savetxt(os.path.join(self.exp_path, f"psnr_sharp_deblurred_avg_epoch{epoch}.txt"), np.array([np.mean(psnr_sharp_deblurred)]))
            savetxt(os.path.join(self.exp_path, f"ssim_sharp_deblurred_avg_epoch{epoch}.txt"), np.array([np.mean(ssim_sharp_deblurred)]))

    def train(self, epoch, steps, R, G, B, loss_, ch_blur):
        """
        ### Train
        """
        # Iterate through the dataset

        # Increment global step
        self.step += 1

        # Iterate through the dataset
        #for batch_idx, (sharp, blur) in enumerate(self.data_loader_train):
        sharp, blur = next(iter(self.dataloader_train))

        # Move data to device
        sharp = sharp.to(self.gpu_id)
        blur = blur.to(self.gpu_id)

        if epoch == 0:
            # save images blur and sharp image pairs
            save_image(sharp, os.path.join(self.exp_path, f'sharp_train.png'))
            save_image(blur, os.path.join(self.exp_path, f'blur_train.png'))
            # get avg channels for blur dataset
            ch_blur.append(round(torch.mean(blur[:,0,:,:]).item(), 2))
            ch_blur.append(round(torch.mean(blur[:,1,:,:]).item(), 2))
            ch_blur.append(round(torch.mean(blur[:,2,:,:]).item(), 2))

        # get initial prediction
        init = self.diffusion.predictor(blur)

        # compute residual
        residual = sharp - init

        # store mean value of channels (RED, GREEN, BLUE)
        steps.append(self.step)

        r = torch.mean(init[:,0,:,:])
        R.append(r.item())

        g = torch.mean(init[:,1,:,:])
        G.append(g.item())

        b = torch.mean(init[:,2,:,:])
        B.append(b.item())

        # Make the gradients zero
        self.optimizer.zero_grad()

        # Calculate loss
        loss = self.diffusion.loss(residual, blur) #+ F.mse_loss(sharp, init)
        print(f"epoch: {self.step}, loss: {loss.item()}")
        loss_.append(loss.item())

        # Compute gradients
        loss.backward()

        #print("############ GRAD OUTPUT ############")
        #print(self.denoiser.module.final.bias.grad)
        #print(self.init_predictor.module.final.bias.grad)

        # clip gradients
        nn.utils.clip_grad_norm_(self.params_denoiser, 0.1)
        nn.utils.clip_grad_norm_(self.params_init, 0.1)

        # Take an optimization step
        self.optimizer.step()

        # Track the loss with WANDB
        if self.wandb and self.gpu_id == 0:
            wandb.log({'loss': loss}, step=self.step)

    def run(self):

        # used to plot channel averages
        R = []
        G = []
        B = []
        steps = []
        loss_ = []
        ch_blur = []

        for epoch in range(self.epochs):

            # sample at epoch 0
            if (epoch == 0) and (self.gpu_id == 0):
                self.sample(epoch=0)

            # train
            self.train(epoch, steps, R, G, B, loss_, ch_blur)

            # plot graph every 20 epochs
            if ((epoch + 1) % 100 == 0) and (self.gpu_id == 0):
                title = f"D:{self.num_params_denoiser//1_000_000}M, G:{self.num_params_init//1_000_000}M, G_pre:No, Lr:{'{:.0e}'.format(self.learning_rate)}, Tr_set:{self.batch_size}, Ch_blur:{ch_blur}"
                plot_channels(steps, R, G, B, self.exp_path, title=title)
                plot_loss(steps, loss_, self.exp_path, title=title)

            # sample at 2000's epoch
            if ((epoch + 1) % 500 == 0) and (self.gpu_id == 0):
                # Save the eps model
                self.sample(self.ckpt_denoiser_epoch + epoch + 1)
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
    os.environ["MASTER_PORT"] = "12356" # any free port on machine
    # nvidia collective comms library (comms across CUDA GPUs)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size:int):
    ddp_setup(rank=rank, world_size=world_size)
    trainer = Trainer()
    trainer.init(rank) # initialize trainer class
    #print(trainer.init_predictor)

    if rank == 0:
        print("Denoiser params:", trainer.num_params_denoiser)
        print("Initial Predictor params:", trainer.num_params_init)
        print("Learning rate:", trainer.learning_rate)
        print("Channel multipliers", trainer.channel_multipliers)
        print()

    #### Track Hyperparameters with WANDB####
    if trainer.wandb and rank == 0:
        
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
            "denoiser # params": trainer.num_params_denoiser,
            "init # params": trainer.num_init_denoiser,
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