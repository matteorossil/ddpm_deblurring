### Matteo Rossi

# Modules
from dataset import Data
from metrics import psnr, ssim
from eps_models.unet_conditioned import UNet as Denoiser #
from eps_models.init_predictor_new import UNet as Init
from diffusion.ddpm_conditioned import DenoiseDiffusion #

# Torch
import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F

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

# DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def get_exp_path(path=''):
    exp_path = os.path.join(path, datetime.now().strftime("%m%d%Y_%H%M%S"))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    return exp_path

def plot_channels(steps, R, G, B, path, title, ext=""):

    plt.plot(steps, R, label='red', color='r')
    plt.plot(steps, G, label='green', color='g')
    plt.plot(steps, B, label='blu', color='b')

    plt.xlabel("training steps")
    plt.ylabel("channel average")
    plt.legend()
    plt.title(title)
    #plt.show()
    plt.savefig(path + f'/channel_{ext}step{steps[-1]+1}.png')
    plt.figure().clear()
    plt.close('all')

def plot(steps, Y, path, title, ext=""):

    plt.plot(steps, Y, label="means", color='r')

    plt.xlabel("training steps")
    plt.ylabel(ext)
    plt.legend()
    plt.title(title)
    #plt.show()
    plt.savefig(path + f'/{ext}_step{steps[-1]+1}.png')
    plt.figure().clear()
    plt.close('all')

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

class Trainer():
    """
    ## Configurations
    """
    def __init__(self, argv):
        # Number of channels in the image. 3 for RGB.
        self.image_channels: int = 3
        # Image size
        self.image_size: int = 128
        # Number of channels in the initial feature map
        self.n_channels: int = 32
        # The list of channel numbers at each resolution.
        # The number of channels is `channel_multipliers[i] * n_channels`
        self.channel_multipliers: List[int] = [1, 2, 3, 4]
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
        # Learning rate
        self.learning_rate: float = argv.d_lr
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
        #self.store_checkpoints: str = '/home/mr6744/ckpts/'
        self.store_checkpoints: str = '/scratch/mr6744/pytorch/ckpts/'
        #self.dataset_t: str = f'/home/mr6744/{argv.dataset_t}/'
        self.dataset_t: str = f'/scratch/mr6744/pytorch/{argv.dataset_t}/'
        #self.dataset_v: str = f'/home/mr6744/{argv.dataset_v}/'
        self.dataset_v: str = f'/scratch/mr6744/pytorch/{argv.dataset_v}/'
        # load from a checkpoint
        self.ckpt_step: int = argv.ckpt_step
        #self.ckpt_denoiser: str = f'/home/m6744/ckpts/{argv.ckpt_path}/ckpt_denoiser_{self.ckpt_step}.pt'
        self.ckpt_denoiser: str = f'/scratch/mr6744/pytorch/ckpts/{argv.ckpt_path}/ckpt_denoiser_{self.ckpt_step}.pt'
        #self.ckpt_initp: str = f'/home/mr6744/ckpts/{argv.ckpt_path}/ckpt_initp_{self.ckpt_step}.pt'
        self.ckpt_initp: str = f'/scratch/mr6744/pytorch/ckpts/{argv.ckpt_path}/ckpt_initp_{self.ckpt_step}.pt'
        self.multiplier = argv.multiplier
        self.num_workers = argv.num_workers
        self.sampling_interval = argv.sampling_interval
        self.seed = argv.random_seed

    def init(self, rank: int, world_size: int):
        # gpu id
        self.gpu_id = rank
        # world_size
        self.world_size = world_size

        self.denoiser = Denoiser(
            image_channels=self.image_channels*2,
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

        # only loads checkpoint if model is trained
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

        # Create dataloader (shuffle False for validation)
        dataset_train = Data(path=self.dataset_t, mode="train", size=(self.image_size,self.image_size), multiplier=self.multiplier)

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
        self.num_params_denoiser = sum(p.numel() for p in params_denoiser if p.requires_grad)
        self.num_params_init = sum(p.numel() for p in params_init if p.requires_grad)

        # Create optimizers
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)
        self.optimizer2 = torch.optim.AdamW(self.initp.parameters(), lr=self.learning_rate_init, weight_decay= self.weight_decay_rate, betas=self.betas)

        # training steps
        self.step = self.ckpt_step

        # path
        self.exp_path = get_exp_path(path=self.store_checkpoints)

    def sample(self, mode, path, psnr_init, ssim_init, psnr_deblur, ssim_deblur):

        dataset = Data(path=path, mode=mode, size=(self.image_size,self.image_size))
        dataloader = DataLoader(dataset=dataset, batch_size=self.n_samples, num_workers=0, drop_last=False, shuffle=True, pin_memory=False)

        with torch.no_grad():

            torch.manual_seed(self.seed)
            sharp, blur = next(iter(dataloader))
            
            sharp = sharp.to(self.gpu_id)
            blur = blur.to(self.gpu_id)

            if self.step == 0:
                # save images blur and sharp image pairs
                save_image(sharp, os.path.join(self.exp_path, f'{mode}__sharp.png'))
                save_image(blur, os.path.join(self.exp_path, f'{mode}__blur.png'))

            # compute initial predictor
            init = self.diffusion.predictor(blur)

            # get true residual
            X_true = sharp - init

            # Sample X from Gaussian Noise
            #torch.cuda.manual_seed(0)
            X = torch.randn([self.n_samples, self.image_channels, blur.shape[2], blur.shape[3]], device=self.gpu_id)

            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                
                # e.g. t_ from 999 to 0 for 1_000 time steps
                t = self.n_steps - t_ - 1

                # create a t for every sample in batch
                t_vec = X.new_full((self.n_samples,), t, dtype=torch.long)

                # take one denoising step
                X = self.diffusion.p_sample(X, blur, t_vec)

            # save initial predictor
            save_image(init, os.path.join(self.exp_path, f'{mode}_init_step{self.step}.png'))
            # save true residual
            save_image(X_true, os.path.join(self.exp_path, f'{mode}_residual_true_step{self.step}.png'))
            # save sampled residual
            save_image(X, os.path.join(self.exp_path, f'{mode}_residual_sampled_step{self.step}.png'))
            # save sampled deblurred
            save_image(init + X, os.path.join(self.exp_path, f'{mode}_deblurred_step{self.step}.png'))

            # compute metrics (sharp, init)
            psnr_sharp_init = psnr(sharp, init)
            ssim_sharp_init = ssim(sharp, init)
            #savetxt(os.path.join(self.exp_path, f"psnr_sharp_init_avg_step{self.step}.txt"), np.array([np.mean(psnr_sharp_init)]))
            #savetxt(os.path.join(self.exp_path, f"ssim_sharp_init_avg_step{self.step}.txt"), np.array([np.mean(ssim_sharp_init)]))
            psnr_init.append(psnr_sharp_init)
            ssim_init.append(ssim_sharp_init)

            # compute metrics (sharp, deblurred)
            psnr_sharp_deblurred = psnr(sharp, init + X)
            ssim_sharp_deblurred = ssim(sharp, init + X)
            #savetxt(os.path.join(self.exp_path, f"psnr_sharp_deblurred_avg_step{self.step}.txt"), np.array([np.mean(psnr_sharp_deblurred)]))
            #savetxt(os.path.join(self.exp_path, f"ssim_sharp_deblurred_avg_step{self.step}.txt"), np.array([np.mean(ssim_sharp_deblurred)]))
            psnr_deblur.append(psnr_sharp_deblurred)
            ssim_deblur.append(ssim_sharp_deblurred)

    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset

        # Iterate through the dataset
        for batch_idx, (sharp, blur) in enumerate(self.dataloader_train):
        #sharp, blur = next(iter(self.dataloader_train))

            # Increment global step
            self.step += 1

            # Move data to device
            sharp = sharp.to(self.gpu_id)
            blur = blur.to(self.gpu_id)

            # save images blur and sharp image pairs
            #save_image(sharp, os.path.join(self.exp_path, f'sharp_train_step{self.step}.png'))
            #save_image(blur, os.path.join(self.exp_path, f'blur_train_step{self.step}.png'))

            # get avg channels for blur dataset
            #if self.step == 0:
                #pass
                # save images blur and sharp image pairs
                #save_image(sharp, os.path.join(self.exp_path, f'sharp_train.png'))
                #save_image(blur, os.path.join(self.exp_path, f'blur_train.png'))
                #ch_blur.append(round(torch.mean(blur[:,0,:,:]).item(), 2))
                #ch_blur.append(round(torch.mean(blur[:,1,:,:]).item(), 2))
                #ch_blur.append(round(torch.mean(blur[:,2,:,:]).item(), 2))

            # get initial prediction
            init = self.diffusion.predictor(blur)
            #save_image(init, os.path.join(self.exp_path, f'init_step{self.step}.png'))

            # compute residual
            residual = sharp - init
            #save_image(residual, os.path.join(self.exp_path, f'residual_step{self.step}.png'))

            # store mean value of channels (RED, GREEN, BLUE)
            #steps.append(self.step)

            r = torch.mean(init[:,0,:,:])
            #R.append(r.item())

            g = torch.mean(init[:,1,:,:])
            #G.append(g.item())

            b = torch.mean(init[:,2,:,:])
            #B.append(b.item())

            # Make the gradients zero
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()

            #### REGULARIZER ####

            # Compute regularizer 1 (std dev)
            #rgb = torch.tensor([r, g, b], device=self.gpu_id, requires_grad=True)
            #regularizer = torch.std(rgb) * 10
            #regularizer = torch.tensor([0.], device=self.gpu_id, requires_grad=False)

            #### REGULARIZER INIT ####
            r_blur = torch.mean(blur[:,0,:,:])
            g_blur = torch.mean(blur[:,1,:,:])
            b_blur = torch.mean(blur[:,2,:,:])
            regularizer_init = (F.l1_loss(r, r_blur) + F.l1_loss(g, g_blur)+ F.l1_loss(b, b_blur))
            regularizer_init = F.threshold(regularizer_init, self.threshold, 0.)
            #regularizer_init = torch.tensor([0.], device=self.gpu_id, requires_grad=False)

            #### DENOISER LOSS ####
            #denoiser_loss, reg_denoiser_mean, reg_denoiser_std, mean_r, mean_g, mean_b, std_r, std_g, std_b = self.diffusion.loss(residual, blur)
            denoiser_loss = self.diffusion.loss(residual, blur)

            #### REGRESSION LOSS INIT ####
            if self.alpha > 0: regression_loss = self.alpha * F.mse_loss(sharp, init)
            else: regression_loss = torch.tensor([0.], device=self.gpu_id, requires_grad=False)

            # final loss
            loss = denoiser_loss + regression_loss + regularizer_init #+ regularizer_denoiser_mean + regularizer_denoiser_std

            #print('Epoch: {:4d}, Step: {:4d}, TOT_loss: {:.4f}, D_loss: {:.4f}, G_loss: {:.4f}, reg_G: {:.4f}, reg_D_mean: {:.4f}, reg_D_std: {:.4f}, D_mean_r: {:+.4f}, D_mean_g: {:+.4f}, D_mean_b: {:+.4f}, D_std_r: {:.4f}, D_std_r: {:.4f}, D_std_r: {:.4f}'.format(epoch, self.step, loss.item(), denoiser_loss.item(), regression_loss.item(), regularizer_init.item(), reg_denoiser_mean.item(), reg_denoiser_std.item(), mean_r.item(), mean_g.item(), mean_b.item(), std_r.item(), std_g.item(), std_b.item()))
            if self.gpu_id == 0:
                print('Step: {:4d}, Loss: {:.4f}, D_loss: {:.4f}, G_loss: {:.4f}, G_reg: {:.4f}'.format(self.step, loss.item(), denoiser_loss.item(), regression_loss.item(), regularizer_init.item()))

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

        # used to plot channel averages
        R = []
        G = []
        B = []
        steps = []
        ch_blur = []

        sample_steps= [] # stores the step at which you sample
        psnr_init_t = []
        ssim_init_t = []
        psnr_deblur_t = []
        ssim_deblur_t = []
        psnr_init_v = []
        ssim_init_v = []
        psnr_deblur_v = []
        ssim_deblur_v = []

        for epoch in range(self.epochs):

            # sample at step 0
            if (self.step == 0) and (self.gpu_id == 0):
                self.sample("train2", self.dataset_v, psnr_init_t, ssim_init_t, psnr_deblur_t, ssim_deblur_t)
                self.sample("val", self.dataset_v, psnr_init_v, ssim_init_v, psnr_deblur_v, ssim_deblur_v)
                sample_steps.append(self.step + self.ckpt_step)

            # train
            #self.train(epoch, steps, R, G, B, ch_blur)
            self.train()

            #if ((epoch+1) % 10 == 0) and (self.gpu_id == 0):
                #title = f"Init - D:{self.num_params_denoiser//1_000_000}M, G:{self.num_params_init//1_000_000}M, Pre:No, D:{'{:.0e}'.format(self.learning_rate)}, G:{'{:.0e}'.format(self.learning_rate_init)}, B:{self.batch_size}, RGB:{ch_blur}"
                #title = f"Init - D:{self.num_params_denoiser//1_000_000}M, G:{self.num_params_init//1_000_000}M, Pre:No, D:{'{:.0e}'.format(self.learning_rate)}, G:{'{:.0e}'.format(self.learning_rate_init)}, B:{self.batch_size}"
                #plot_channels(steps, R, G, B, self.exp_path, title=title, ext="init_")

            if ((self.step % self.sampling_interval) == 0) and (self.gpu_id == 0):
                self.sample("train2", self.dataset_v, psnr_init_t, ssim_init_t, psnr_deblur_t, ssim_deblur_t)
                self.sample("val", self.dataset_v, psnr_init_v, ssim_init_v, psnr_deblur_v, ssim_deblur_v)
                sample_steps.append(self.step + self.ckpt_step)
                title = f"eval:train,val - metric:"
                plot_metrics(sample_steps, ylabel="psnr", label_init_t="init train", label_deblur_t="deblur train", label_init_v="init val", label_deblur_v="deblur val", metric_init_t=psnr_init_t, metric_deblur_t=psnr_deblur_t, metric_init_v=psnr_init_v, metric_deblur_v=psnr_deblur_v, path=self.exp_path, title=title)
                plot_metrics(sample_steps, ylabel="ssim", label_init_t="init train", label_deblur_t="deblur train", label_init_v="init val", label_deblur_v="deblur val", metric_init_t=ssim_init_t, metric_deblur_t=ssim_deblur_t, metric_init_v=ssim_init_v, metric_deblur_v=ssim_deblur_v, path=self.exp_path, title=title)
                torch.save(self.denoiser.module.state_dict(), os.path.join(self.exp_path, f'ckpt_denoiser_{self.ckpt_step+self.step}.pt'))
                torch.save(self.initp.module.state_dict(), os.path.join(self.exp_path, f'ckpt_initp_{self.ckpt_step+self.step}.pt'))

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
    trainer.init(rank, world_size) # initialize trainer class

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
            "L2 Loss": trainer.alpha > 0,
            "L2 param": trainer.alpha,
            "Regularizer": True,
            "Regularizer Threshold": trainer.threshold,
            "Dataset_t": trainer.dataset_t,
            "Dataset_v": trainer.dataset_v,
            "Path": trainer.exp_path,
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

    world_size = torch.cuda.device_count() # how many GPUs available in the machine
    mp.spawn(main, args=(world_size,argv), nprocs=world_size)