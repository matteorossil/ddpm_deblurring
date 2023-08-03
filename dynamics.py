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
    plt.savefig(path + f'/channel_{ext}_step{steps[-1]}.png')
    plt.figure().clear()
    plt.close('all')

def plot(steps, Y, path, title, ext="", l="means", c='r'):

    plt.plot(steps, Y, label=l, color=c)

    plt.xlabel("training steps")
    plt.ylabel(ext)
    plt.legend()
    plt.title(title)
    #plt.show()
    plt.savefig(path + f'/{ext}_step{steps[-1]}.png')
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
    batch_size: int = 32
    # L2 loss
    alpha = 0.
    # Threshold Regularizer
    threshold = 10.
    # Learning rate
    learning_rate: float = 1e-4
    learning_rate_init: float = 1e-4
    # Weight decay rate
    weight_decay_rate: float = 1e-3
    # ema decay
    betas = (0.9, 0.999)
    # Number of training epochs
    epochs: int = 1_000_000
    # Number of samples (evaluation)
    n_samples: int = 32
    # Use wandb
    wandb: bool = False
    # checkpoints path
    store_checkpoints: str = '/home/mr6744/ckpts/'
    #store_checkpoints: str = '/scratch/mr6744/pytorch/ckpts/'
    # dataset path
    dataset_t: str = '/home/mr6744/gopro_small/'
    #dataset_t: str = '/scratch/mr6744/pytorch/gopro_small/'
    dataset_v: str = '/home/mr6744/gopro_small/'
    #dataset_v: str = '/scratch/mr6744/pytorch/gopro_small/'
    # load from a checkpoint
    ckpt_denoiser_step: int = 0
    ckpt_initp_step: int = 0
    ckpt_denoiser: str = f'/home/mr6744/ckpts/07272023_182450/ckpt_denoiser_{ckpt_denoiser_step}.pt'
    #checkpoint_init: str = f'/scratch/mr6744/pytorch/checkpoints_conditioned/06292023_100717/checkpoint__initpr_{checkpoint_init_epoch}.pt'
    ckpt_initp: str = f'/home/mr6744/ckpts/07272023_182450/ckpt_initp_{ckpt_initp_step}.pt'
    #checkpoint: str = f'/home/mr6744/checkpoints_conditioned/06022023_001525/checkpoint_{checkpoint_epoch}.pt'

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
        if self.ckpt_denoiser_step != 0:
            checkpoint_d = torch.load(self.ckpt_denoiser)
            self.denoiser.module.load_state_dict(checkpoint_d)
        
        if self.ckpt_initp_step != 0:
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
        dataset_train = Data(path=self.dataset_t, mode="train2", size=(self.image_size,self.image_size), multiplier=100)

        self.dataloader_train = DataLoader(dataset=dataset_train,
                                            batch_size=self.batch_size // self.world_size, 
                                            num_workers=8, #os.cpu_count() // 2,
                                            drop_last=False,
                                            shuffle=False, 
                                            pin_memory=False,
                                            sampler=DistributedSampler(dataset_train, shuffle=False))

        # Num params of models
        params_denoiser = list(self.denoiser.parameters())
        params_init = list(self.initp.parameters())
        self.num_params_denoiser = sum(p.numel() for p in params_denoiser if p.requires_grad)
        self.num_params_init = sum(p.numel() for p in params_init if p.requires_grad)

        # Create optimizers
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay_rate, betas=self.betas)
        self.optimizer2 = torch.optim.AdamW(self.initp.parameters(), lr=self.learning_rate_init, weight_decay= self.weight_decay_rate, betas=self.betas)

        # training steps
        self.step = 0

        # path
        self.exp_path = get_exp_path(path=self.store_checkpoints)

    def sample(self, mode, path, psnr_init, ssim_init, psnr_deblur, ssim_deblur):

        dataset = Data(path=path, mode=mode, size=(self.image_size,self.image_size))
        dataloader = DataLoader(dataset=dataset, batch_size=self.n_samples, num_workers=0, drop_last=False, shuffle=True, pin_memory=False)

        with torch.no_grad():

            torch.manual_seed(7)
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

    def train(self, steps, R, G, B, ch_blur, grad_bias_init_R, grad_bias_init_G, grad_bias_init_B, grad_bias_denoiser_R, grad_bias_denoiser_G, grad_bias_denoiser_B):
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
            #save_image(sharp, os.path.join(self.exp_path, f'sharp_train_{self.step}.png'))
            #save_image(blur, os.path.join(self.exp_path, f'blur_train_{self.step}.png'))

            # get avg channels for blur dataset
            if self.step == 1:
                # save images blur and sharp image pairs
                save_image(sharp, os.path.join(self.exp_path, f'sharp_train_{self.step}.png'))
                save_image(blur, os.path.join(self.exp_path, f'blur_train_{self.step}.png'))
                ch_blur.append(round(torch.mean(blur[:,0,:,:]).item(), 2))
                ch_blur.append(round(torch.mean(blur[:,1,:,:]).item(), 2))
                ch_blur.append(round(torch.mean(blur[:,2,:,:]).item(), 2))

            # get initial prediction
            init = self.diffusion.predictor(blur)
            #save_image(init, os.path.join(self.exp_path, f'init_step{self.step}.png'))

            # compute residual
            residual = sharp - init
            #save_image(residual, os.path.join(self.exp_path, f'residual_step{self.step}.png'))

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
            #print("Grad bias denoiser:", self.denoiser.module.final.bias.grad[0].item())
            #print("Grad bias init:", self.initp.module.final.bias.grad[0].item())

            self.initp.module.final.bias.grad[0] = 0.
            self.initp.module.final.bias.grad[1] = 0.
            self.initp.module.final.bias.grad[2] = 0.

            self.denoiser.module.final.bias.grad[0] = 0.
            self.denoiser.module.final.bias.grad[1] = 0.
            self.denoiser.module.final.bias.grad[2] = 0.

            grad_bias_denoiser_R.append(self.denoiser.module.final.bias.grad[0].item())
            grad_bias_denoiser_G.append(self.denoiser.module.final.bias.grad[1].item())
            grad_bias_denoiser_B.append(self.denoiser.module.final.bias.grad[2].item())

            grad_bias_init_R.append(self.initp.module.final.bias.grad[0].item())
            grad_bias_init_G.append(self.initp.module.final.bias.grad[1].item())
            grad_bias_init_B.append(self.initp.module.final.bias.grad[2].item())

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
        grad_bias_denoiser_R = []
        grad_bias_denoiser_G = []
        grad_bias_denoiser_B = []
        grad_bias_init_R = []
        grad_bias_init_G = []
        grad_bias_init_B = []

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
            '''
            if (self.step == 0) and (self.gpu_id == 0):
                self.sample("train2", self.dataset_v, psnr_init_t, ssim_init_t, psnr_deblur_t, ssim_deblur_t)
                self.sample("val", self.dataset_v, psnr_init_v, ssim_init_v, psnr_deblur_v, ssim_deblur_v)
                sample_steps.append(self.step + self.ckpt_denoiser_step)
            '''

            # train
            #self.train(epoch, steps, R, G, B, ch_blur)
            self.train(steps, R, G, B, ch_blur, grad_bias_init_R, grad_bias_init_G, grad_bias_init_B, grad_bias_denoiser_R, grad_bias_denoiser_G, grad_bias_denoiser_B)

            if ((self.step % 100) == 0) and (self.gpu_id == 0):

                title = f"Init - D:{'{:.0e}'.format(self.learning_rate)}, G:{'{:.0e}'.format(self.learning_rate_init)}, B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot_channels(steps, R, G, B, self.exp_path, title=title, ext="init")

                title = f"Grad Bias Init - D:{'{:.0e}'.format(self.learning_rate)}, G:{'{:.0e}'.format(self.learning_rate_init)}, B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot_channels(steps, grad_bias_init_R, grad_bias_init_G, grad_bias_init_B, self.exp_path, title=title, ext="grad_bias_init")

                title = f"Grad Bias Denoiser - D:{'{:.0e}'.format(self.learning_rate)}, G:{'{:.0e}'.format(self.learning_rate_init)}, B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot_channels(steps, grad_bias_denoiser_R, grad_bias_denoiser_G, grad_bias_denoiser_B, self.exp_path, title=title, ext="grad_bias_denoiser")

                title = f"Denoiser Mean Red - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.R, self.exp_path, title=title, ext="denoiser_mean_red", l='Mean Red', c='r')

                title = f"Denoiser Mean Green - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.G, self.exp_path, title=title, ext="denoiser_mean_green", l='Mean Green', c='g')

                title = f"Denoiser Mean Blue - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.B, self.exp_path, title=title, ext="denoiser_mean_blue", l='Mean Blue', c='b')

                title = f"Denoiser Std Red - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.R_std, self.exp_path, title=title, ext="denoiser_std_red", l='Std Red', c='r')

                title = f"Denoiser Std Green - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.G_std, self.exp_path, title=title, ext="denoiser_std_green", l='Std Green', c='g')

                title = f"Denoiser Std Blue - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.B_std, self.exp_path, title=title, ext="denoiser_std_blue", l='Std Blue', c='b')

                title = f"Denoiser Min Red - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.R_min, self.exp_path, title=title, ext="denoiser_min_red", l='Min Red', c='r')

                title = f"Denoiser Max Red - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.R_max, self.exp_path, title=title, ext="denoiser_max_red", l='Max Red', c='r')

                title = f"Denoiser Min Green - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.G_min, self.exp_path, title=title, ext="denoiser_min_green", l='Min Green', c='g')

                title = f"Denoiser Max Green - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.G_max, self.exp_path, title=title, ext="denoiser_max_green", l='Max Green', c='g')

                title = f"Denoiser Min Blue - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.B_min, self.exp_path, title=title, ext="denoiser_min_blue", l='Min Blue', c='b')

                title = f"Denoiser Max Blue - B:{self.batch_size}, RGB:{ch_blur}, Reg:{self.threshold}"
                plot(steps, self.diffusion.B_max, self.exp_path, title=title, ext="denoiser_max_blue", l='Max Blue', c='b')

            '''
            if ((self.step % 1_000) == 0) and (self.gpu_id == 0):
                self.sample("train2", self.dataset_v, psnr_init_t, ssim_init_t, psnr_deblur_t, ssim_deblur_t)
                self.sample("val", self.dataset_v, psnr_init_v, ssim_init_v, psnr_deblur_v, ssim_deblur_v)
                sample_steps.append(self.step + self.ckpt_denoiser_step)
                title = f"eval:train,val - metric:"
                plot_metrics(sample_steps, ylabel="psnr", label_init_t="init train", label_deblur_t="deblur train", label_init_v="init val", label_deblur_v="deblur val", metric_init_t=psnr_init_t, metric_deblur_t=psnr_deblur_t, metric_init_v=psnr_init_v, metric_deblur_v=psnr_deblur_v, path=self.exp_path, title=title)
                plot_metrics(sample_steps, ylabel="ssim", label_init_t="init train", label_deblur_t="deblur train", label_init_v="init val", label_deblur_v="deblur val", metric_init_t=ssim_init_t, metric_deblur_t=ssim_deblur_t, metric_init_v=ssim_init_v, metric_deblur_v=ssim_deblur_v, path=self.exp_path, title=title)
                torch.save(self.denoiser.module.state_dict(), os.path.join(self.exp_path, f'ckpt_denoiser_{self.ckpt_denoiser_step+self.step}.pt'))
                torch.save(self.initp.module.state_dict(), os.path.join(self.exp_path, f'ckpt_initp_{self.ckpt_initp_step+self.step}.pt'))
            '''

def ddp_setup(rank, world_size):
    """
    Args:
        rank: unique identifier of each process
        world_size: total number of processes
    """ 
    # IP address of machine running rank 0 process
    # master: machine coordinates communication across processes
    os.environ["MASTER_ADDR"] = "localhost" # we assume a single machine setup)
    os.environ["MASTER_PORT"] = "12354" # any free port on machine
    # nvidia collective comms library (comms across CUDA GPUs)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size:int):
    ddp_setup(rank=rank, world_size=world_size)
    trainer = Trainer()
    trainer.init(rank, world_size) # initialize trainer class

    #### Track Hyperparameters with WANDB####
    if trainer.wandb and rank == 0:
        
        wandb.init(
            project="deblurring",
            name=f"conditioned",
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
    world_size = torch.cuda.device_count() # how many GPUs available in the machine
    #world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size)