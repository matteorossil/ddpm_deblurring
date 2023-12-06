### Matteo Rossi

# Modules
from dataset import Data
from metrics import psnr, ssim
from eps_models.unet_conditioned import UNet as Denoiser #
from eps_models.init_reps import UNet as Init
from diffusion.ddpm_conditioned import DenoiseDiffusion #

# Torch
import torch
from torch import nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# Numpy
import numpy as np
#from numpy import savetxt

# Other
import os
from typing import List
from pathlib import Path
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
import argparse
import pickle

# DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def get_exp_path(path=''):
    exp_path = os.path.join(path, datetime.now().strftime("%m%d%Y_%H%M%S"))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    return exp_path

def plot_metrics(steps, ylabel, label_init_t, label_deblur_t, label_init_v, label_deblur_v, metric_init_t, metric_deblur_t, metric_init_v, metric_deblur_v, path, title):

    plt.plot(steps, metric_init_t, label=label_init_t, color='b')
    plt.plot(steps, metric_deblur_t, label=label_deblur_t, color='r')
    plt.plot(steps, metric_init_v, label=label_init_v, color='b', linestyle='dashed')
    plt.plot(steps, metric_deblur_v, label=label_deblur_v, color='r', linestyle='dashed')

    plt.xlabel("training steps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title + ylabel)
    #plt.show()
    plt.savefig(path + f'/{ylabel}_step{steps[-1]}.png')
    plt.figure().clear()
    plt.close('all')

def save_metrics(metrics, name):
    file = open(name, 'wb')
    pickle.dump(metrics, file)
    file.close()

def load_metrics(name):
    file = open(name, 'rb')
    metrics = pickle.load(file)
    file.close()
    return metrics

class Trainer():
    """
    ## Configurations
    """
    def __init__(self, argv):
        # Number of channels in the image. 3 for RGB.
        self.image_channels: int = 3
        # Image size
        self.image_size: int = 224
        # Number of channels in the initial feature map
        self.n_channels: int = 32
        # The list of channel numbers at each resolution.
        # The number of channels is `channel_multipliers[i] * n_channels`
        self.channel_multipliers: List[int] = [1, 2, 2, 3]
        # The list of booleans that indicate whether to use attention at each resolution
        self.is_attention: List[int] = [False, False, False, False]
        # Number of time steps $T$
        self.n_steps: int = 1_000
        # noise scheduler Beta_0
        self.beta_0 = 1e-6 # 0.000001
        # noise scheduler Beta_T
        self.beta_T = 1e-2 # 0.01
        # Batch size
        self.batch_size: int = argv.batch_size
        # L2 loss
        self.alpha = argv.l2_loss
        # Threshold Regularizer
        self.threshold = argv.threshold
        # Learning rate D
        self.learning_rate: float = argv.d_lr
        # Learning rate G
        self.learning_rate_init: float = argv.g_lr
        # Weight decay rate
        self.weight_decay_rate: float = 1e-3
        # ema decay
        self.betas = (0.9, 0.999)
        # Number of training epochs
        self.epochs: int = 1_000_000
        # Number of samples (evaluation)
        self.n_samples: int = argv.sample_size
        # Use wandb
        self.wandb: bool = argv.wandb
        # load from a checkpoint
        self.ckpt_step: int = argv.ckpt_step
        # paths
        if  argv.hpc: 
            self.store_checkpoints: str = '/scratch/mr6744/pytorch/ckpts/'
            self.dataset_t: str = f'/scratch/mr6744/pytorch/{argv.dataset_t}/'
            self.dataset_v: str = f'/scratch/mr6744/pytorch/{argv.dataset_v}/'
            self.ckpt_denoiser: str = f'/scratch/mr6744/pytorch/ckpts/{argv.ckpt_path}/ckpt_denoiser_{self.ckpt_step}.pt'
            self.ckpt_initp: str = f'/scratch/mr6744/pytorch/ckpts/{argv.ckpt_path}/ckpt_initp_{self.ckpt_step}.pt'
            self.ckpt_metrics_: str = f'/scratch/mr6744/pytorch/ckpts/{argv.ckpt_path}/metrics_step{self.ckpt_step}.p'
        else:
            self.store_checkpoints: str = '/home/mr6744/diff_rep/'
            self.dataset_t: str = f'/home/mr6744/{argv.dataset_t}/'
            self.dataset_v: str = f'/home/mr6744/{argv.dataset_v}/'
            self.ckpt_denoiser: str = f'/home/mr6744/diff_rep/{argv.ckpt_path}/ckpt_denoiser_{self.ckpt_step}.pt'
            self.ckpt_initp: str = f'/home/mr6744/diff_rep/{argv.ckpt_path}/ckpt_initp_{self.ckpt_step}.pt'
            self.ckpt_metrics_: str = f'/home/mr6744/diff_rep/{argv.ckpt_path}/metrics_step{self.ckpt_step}.p'
        # multiplier for virtual dataset
        self.multiplier = argv.multiplier
        # dataloader workers
        self.num_workers = argv.num_workers
        # how often to sample
        self.sampling_interval = argv.sampling_interval
        # random seed for evaluation
        self.seed = argv.random_seed
        # whether to sample or not
        self.sample = argv.sample
        # import metrics from chpt
        self.ckpt_metrics = argv.ckpt_metrics
        # perform crops on eval
        self.crop_eval = argv.crop_eval
        # training step start
        self.step = self.ckpt_step
        # path
        self.exp_path = get_exp_path(path=self.store_checkpoints)

    def init_train(self, rank: int, world_size: int):
        # gpu id
        self.gpu_id = rank
        # world_size
        self.world_size = world_size

        self.denoiser = Denoiser(
            image_channels=self.image_channels+1, #self.image_channels*2,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention
        ).to(self.gpu_id)

        self.initp = Init(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention
        ).to(self.gpu_id)

        self.denoiser = DDP(self.denoiser, device_ids=[self.gpu_id])
        self.initp = DDP(self.initp, device_ids=[self.gpu_id])

        # load checkpoints
        if self.ckpt_step != 0:
            checkpoint_d = torch.load(self.ckpt_denoiser)
            self.denoiser.module.load_state_dict(checkpoint_d)
            checkpoint_i = torch.load(self.ckpt_initp)
            self.initp.module.load_state_dict(checkpoint_i)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.denoiser,
            predictor=self.initp,
            n_steps=self.n_steps,
            device=self.gpu_id,
            beta_0=self.beta_0,
            beta_T=self.beta_T
        )

        # simple augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), interpolation=3), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.transform_val = transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        dataset_train = ImageFolder(self.dataset_t, transform=transform)

        # Create dataloader (shuffle False for validation)
        #dataset_train = Data(path=self.dataset_t, mode="train", size=(self.image_size,self.image_size), multiplier=self.multiplier)

        self.dataloader_train = DataLoader(dataset=dataset_train,
                                            batch_size=self.batch_size // self.world_size, 
                                            num_workers=self.num_workers, #os.cpu_count() // 2,
                                            drop_last=True,
                                            shuffle=False, 
                                            pin_memory=False,
                                            sampler=DistributedSampler(dataset_train))

        # Num params of models
        params_denoiser = list(self.denoiser.parameters())
        params_init = list(self.initp.parameters())
        num_params_denoiser = sum(p.numel() for p in params_denoiser if p.requires_grad)
        self.num_params_init = sum(p.numel() for p in params_init if p.requires_grad)
        print(num_params_denoiser)

        # Create optimizers
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)
        self.optimizer2 = torch.optim.AdamW(self.initp.parameters(), lr=self.learning_rate_init, weight_decay= self.weight_decay_rate, betas=self.betas)

    def sample_(self):
        
        dataset_val = ImageFolder(self.dataset_v, transform=self.transform_val)
        dataloader = DataLoader(dataset=dataset_val, batch_size=self.n_samples, num_workers=0, drop_last=False, shuffle=True, pin_memory=False)

        with torch.no_grad():

            torch.manual_seed(self.seed)
            sharp = next(iter(dataloader))
            
            sharp = sharp[0].to(self.gpu_id)

            print(sharp.shape)

            # compute initial predictor
            init = self.diffusion.predictor(sharp)

            # Sample X from Gaussian Noise
            X = torch.randn([self.n_samples, self.image_channels, sharp.shape[2], sharp.shape[3]], device=self.gpu_id)

            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                    
                # e.g. t_ from 999 to 0 for 1_000 time steps
                t = self.n_steps - t_ - 1
                print(t)

                # create a t for every sample in batch
                t_vec = X.new_full((self.n_samples,), t, dtype=torch.long)

                # take one denoising step
                X = self.diffusion.p_sample(X, init, t_vec)
            
            #save initial image
            save_image(sharp, os.path.join(self.exp_path, f'sharp_step{self.step}.png'))
            # save initial predictor
            save_image(init, os.path.join(self.exp_path, f'init_step{self.step}.png'))
            # save sampled residual
            save_image(X, os.path.join(self.exp_path, f'sampled_step{self.step}.png'))


    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset

        # Iterate through the dataset
        for _, sharp in enumerate(self.dataloader_train):
        #sharp, blur = next(iter(self.dataloader_train))

            # Increment global step
            self.step += 1

            # Move data to device
            sharp = sharp[0].to(self.gpu_id)

            # save images blur and sharp image pairs
            #save_image(sharp, os.path.join(self.exp_path, f'sharp_train_step{self.step}.png'))
            #save_image(blur, os.path.join(self.exp_path, f'blur_train_step{self.step}.png'))

            # get initial prediction
            init = self.diffusion.predictor(sharp)
            #save_image(init, os.path.join(self.exp_path, f'init_step{self.step}.png'))


            # Make the gradients zero
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()

            #### DENOISER LOSS ####
            denoiser_loss = self.diffusion.loss(sharp, init)

            # final loss
            loss = denoiser_loss

            if self.gpu_id == 0:
                print('Step: {:4d}, Loss: {:.4f}, D_loss: {:.4f}'.format(self.step, loss.item(), denoiser_loss.item()))

            # Compute gradients
            loss.backward()

            #print("############ GRAD OUTPUT ############")
            #print("Grad bias denoiser:", self.denoiser.module.final.bias.grad)
            #print("Grad bias init:", self.initp.module.final.bias.grad)

            # clip gradients
            nn.utils.clip_grad_norm_(self.denoiser.parameters(), 0.01)
            nn.utils.clip_grad_norm_(self.initp.parameters(), 0.01)

            # Take an optimization step
            self.optimizer.step()
            self.optimizer2.step()

            # Track the loss with WANDB
            if self.wandb and self.gpu_id == 0:
                wandb.log({'loss': loss}, step=self.step)

    def run(self):

        for _ in range(self.epochs):

            # sample at step 0
            if (self.sample) and (self.step == 0) and (self.gpu_id == 0):
                self.sample_()

            # train
            self.train()

            if (self.sample) and (((self.step - self.ckpt_step) % self.sampling_interval) == 0) and (self.gpu_id == 0):
                self.sample_()

def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: unique identifier of each process
        world_size: total number of processes
    """ 
    # IP address of machine running rank 0 process
    # master: machine coordinates communication across processes
    os.environ["MASTER_ADDR"] = "localhost" # we assume a single machine setup)
    os.environ["MASTER_PORT"] = "123" + port # any free port on machine
    # nvidia collective comms library (comms across CUDA GPUs)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size:int, argv):
    ddp_setup(rank=rank, world_size=world_size, port=argv.port)
    trainer = Trainer(argv)
    trainer.init_train(rank, world_size) # initialize trainer class

    #### Track Hyperparameters with WANDB####
    if trainer.wandb and rank == 0:
        
        wandb.init(
            project="deblurring",
            name=argv.name,
            config=
            {
            "GPUs": world_size,
            "GPU Type": torch.cuda.get_device_name(rank),
            "Denoiser params": trainer.num_params_denoiser,
            "Initial Predictor params": trainer.num_params_init,
            "Denoiser LR": trainer.learning_rate,
            "Init Predictor LR": trainer.learning_rate_init,
            "Batch size": trainer.batch_size,
            "Sample size": trainer.n_samples,
            "L2 Loss": trainer.alpha > 0,
            "L2 param": trainer.alpha,
            "Regularizer": trainer.threshold <= 0.5,
            "Regularizer Threshold": trainer.threshold,
            "Dataset_t": trainer.dataset_t,
            "Dataset_v": trainer.dataset_v,
            "Path": trainer.exp_path,
            "Port": argv.port,
            "Ckpt step": trainer.ckpt_step,
            "Ckpt path": argv.ckpt_path,
            "Ckpt metrics": trainer.ckpt_metrics,
            "Workers": trainer.num_workers,
            "Dataset multiplier": trainer.multiplier,
            "Sampling interval": trainer.sampling_interval,
            "Random seed eval": trainer.seed,
            "Sampling": trainer.sample,
            "Crop eval": trainer.crop_eval
            }
        )
    ##### ####
    trainer.run() # perform training
    destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_size', type=int, default=32)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--l2_loss', type=float, default=0.)
    parser.add_argument('--dataset_t', type=str, default="gopro")
    parser.add_argument('--dataset_v', type=str, default="gopro_128")
    parser.add_argument('--ckpt_step', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--multiplier', type=int, default=1)
    parser.add_argument('--sampling_interval', type=int, default=10_000)
    parser.add_argument('--random_seed', type=int, default=7)
    parser.add_argument('--name', type=str, default="conditioned")
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--hpc', action="store_true")
    parser.add_argument('--sample', action="store_true")
    parser.add_argument('--ckpt_metrics', action="store_true")
    parser.add_argument('--crop_eval', action="store_true")
    argv = parser.parse_args()

    print('port:', argv.port, type(argv.port))
    print('batch_size:', argv.batch_size, type(argv.batch_size))
    print('sample_size:', argv.sample_size, type(argv.sample_size))
    print('d_lr:', argv.d_lr, type(argv.d_lr))
    print('g_lr:', argv.g_lr, type(argv.g_lr))
    print('threshold:', argv.threshold, type(argv.threshold))
    print('l2_loss:', argv.l2_loss, type(argv.l2_loss))
    print('dataset_t:', argv.dataset_t, type(argv.dataset_t))
    print('dataset_v:', argv.dataset_v, type(argv.dataset_v))
    print('ckpt_step:', argv.ckpt_step, type(argv.ckpt_step))
    print('ckpt_path:', argv.ckpt_path, type(argv.ckpt_path))
    print('num_workers:', argv.num_workers, type(argv.num_workers))
    print('multiplier:', argv.multiplier, type(argv.multiplier))
    print('sampling_interval:', argv.sampling_interval, type(argv.sampling_interval))
    print('random_seed:', argv.random_seed, type(argv.random_seed))
    print('name:', argv.name, type(argv.name))
    print('wandb:', argv.wandb, type(argv.wandb))
    print('hpc:', argv.hpc, type(argv.hpc))
    print('sample:', argv.sample, type(argv.sample))
    print('ckpt_metrics:', argv.ckpt_metrics, type(argv.ckpt_metrics))
    print('crop_eval:', argv.crop_eval, type(argv.crop_eval))

    world_size = torch.cuda.device_count() # how many GPUs available in the machine
    mp.spawn(main, args=(world_size,argv), nprocs=world_size)


#CUDA_VISIBLE_DEVICES=0 python trainer_reps.py --batch_size 16 --dataset_t SAYCAM_200K_deblur_crop --dataset_v val_saycam --sample --sample_size 1