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
from torchvision.transforms.functional import gaussian_blur

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

def save_metrics(metrics, name):
    file = open(name, 'wb')
    pickle.dump(metrics, file)
    file.close()

def load_metrics(name):
    file = open(name, 'rb')
    metrics = pickle.load(file)
    file.close()
    return metrics

class Evaluator():
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
        # Number of samples (evaluation)
        self.n_samples: int = argv.sample_size
        # load from a checkpoint
        self.ckpt_step: int = argv.ckpt_step
        # paths
        if  argv.hpc:
            self.store_checkpoints: str = '/scratch/mr6744/pytorch/ckpts/'
            self.dataset: str = f'/scratch/mr6744/pytorch/{argv.dataset}/'
            self.ckpt_denoiser: str = f'/scratch/mr6744/pytorch/ckpts/{argv.ckpt_path}/ckpt_denoiser_{self.ckpt_step}.pt'
            self.ckpt_initp: str = f'/scratch/mr6744/pytorch/ckpts/{argv.ckpt_path}/ckpt_initp_{self.ckpt_step}.pt'
        else:
            self.store_checkpoints: str = '/home/mr6744/ckpts/'
            self.dataset: str = f'/home/mr6744/{argv.dataset}/'
            self.ckpt_denoiser: str = f'/home/mr6744/ckpts/{argv.ckpt_path}/ckpt_denoiser_{self.ckpt_step}.pt'
            self.ckpt_initp: str = f'/home/mr6744/ckpts/{argv.ckpt_path}/ckpt_initp_{self.ckpt_step}.pt'
        # dataloader workers
        self.num_workers = argv.num_workers
        # random seed for evaluation
        self.seed = argv.random_seed
        # perform crops on eval
        self.crop_eval = argv.crop_eval
        # perform eval on training set
        self.train = argv.train
        # sample average
        self.sa = argv.sa
        # training step start
        self.step = self.ckpt_step
        # path
        self.exp_path = get_exp_path(path=self.store_checkpoints)
        # device
        self.device = torch.device('cuda')

        self.denoiser = Denoiser(
            image_channels=self.image_channels*2,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention
        ).to(self.device)

        self.initp = Init(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention
        ).to(self.device)

        # load checkpoints
        if self.ckpt_step != 0:
            checkpoint_d = torch.load(self.ckpt_denoiser)
            self.denoiser.load_state_dict(checkpoint_d)
            checkpoint_i = torch.load(self.ckpt_initp)
            self.initp.load_state_dict(checkpoint_i)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.denoiser,
            predictor=self.initp,
            n_steps=self.n_steps,
            device=self.device,
            beta_0=self.beta_0,
            beta_T=self.beta_T
        )

    def sample_(self, mode, path, psnr_init, ssim_init, psnr_deblur, ssim_deblur):

        dataset = Data(path=path, mode=mode, crop_eval=self.crop_eval, size=(self.image_size,self.image_size))
        dataloader = DataLoader(dataset=dataset, batch_size=self.n_samples, num_workers=self.num_workers, drop_last=False, shuffle=False, pin_memory=False)

        with torch.no_grad():
            
            torch.manual_seed(self.seed)
            for idx, (sharp, blur) in enumerate(dataloader):
            
                sharp = sharp.to(self.device)
                blur = blur.to(self.device)

                # compute initial predictor
                init = self.diffusion.predictor(blur)

                # get true residual
                X_true = sharp - init

                # Sample X from Gaussian Noise
                residuals = torch.zeros_like(sharp)

                for _ in range(self.sa):

                    X = torch.randn([self.n_samples, self.image_channels, blur.shape[2], blur.shape[3]], device=self.device)

                    # Remove noise for $T$ steps
                    for t_ in range(self.n_steps):
                        
                        # e.g. t_ from 999 to 0 for 1_000 time steps
                        t = self.n_steps - t_ - 1

                        # create a t for every sample in batch
                        t_vec = X.new_full((self.n_samples,), t, dtype=torch.long)

                        # take one denoising step
                        X = self.diffusion.p_sample(X, blur, t_vec)

                    residuals += X

                X = residuals / self.sa

                # save initial predictor
                save_image(sharp, os.path.join(self.exp_path, f'{mode}_sharp_{idx+1}.png'))
                # save initial predictor
                save_image(blur, os.path.join(self.exp_path, f'{mode}_blur_{idx+1}.png'))
                # save initial predictor
                save_image(init, os.path.join(self.exp_path, f'{mode}_init_{idx+1}.png'))
                # save true residual
                save_image(X_true, os.path.join(self.exp_path, f'{mode}_residual_true_{idx+1}.png'))
                # save sampled residual
                save_image(X, os.path.join(self.exp_path, f'{mode}_residual_sampled_{idx+1}.png'))
                # save sampled deblurred
                save_image(init + X, os.path.join(self.exp_path, f'{mode}_deblurred_{idx+1}.png'))

                # compute metrics (sharp, init)
                psnr_sharp_init = psnr(sharp, init)
                ssim_sharp_init = ssim(sharp, init)
                savetxt(os.path.join(self.exp_path, f"{mode}_psnr_sharp_init_{idx+1}.txt"), np.array([psnr_sharp_init]))
                savetxt(os.path.join(self.exp_path, f"{mode}_ssim_sharp_init_{idx+1}.txt"), np.array([ssim_sharp_init]))
                psnr_init.append(psnr_sharp_init)
                ssim_init.append(ssim_sharp_init)

                print('Eval: {:6s} Samples: {:4d} PSRN S-I: {:.6f}'.format(mode, idx+1, sum(psnr_init)/len(psnr_init)))
                print('Eval: {:6s} Samples: {:4d} SSIM S-I: {:.6f}'.format(mode, idx+1, sum(ssim_init)/len(ssim_init)))

                # compute metrics (sharp, deblurred)
                psnr_sharp_deblurred = psnr(sharp, init + X)
                ssim_sharp_deblurred = ssim(sharp, init + X)
                savetxt(os.path.join(self.exp_path, f"{mode}_psnr_sharp_deblurred_{idx+1}.txt"), np.array([psnr_sharp_deblurred]))
                savetxt(os.path.join(self.exp_path, f"{mode}_ssim_sharp_deblurred_{idx+1}.txt"), np.array([ssim_sharp_deblurred]))
                psnr_deblur.append(psnr_sharp_deblurred)
                ssim_deblur.append(ssim_sharp_deblurred)

                print('Eval: {:6s} Samples: {:4d} PSRN S-D: {:.6f}'.format(mode, idx + 1, sum(psnr_deblur)/len(psnr_deblur)))
                print('Eval: {:6s} Samples: {:4d} SSIM S-D: {:.6f}\n'.format(mode, idx + 1, sum(ssim_deblur)/len(ssim_deblur)))

    def run_eval(self):

        metrics = {"sample_steps":[], "psnr_init_t":[], "ssim_init_t":[], "psnr_deblur_t":[], "ssim_deblur_t":[], "psnr_init_v":[], "ssim_init_v":[], "psnr_deblur_v":[], "ssim_deblur_v":[]}
        
        self.sample_("val", self.dataset, metrics["psnr_init_v"], metrics["ssim_init_v"], metrics["psnr_deblur_v"], metrics["ssim_deblur_v"])
        save_metrics(metrics, os.path.join(self.exp_path, f"metrics.p"))
        if self.train: 
            self.sample_("train2", self.dataset, metrics["psnr_init_t"], metrics["ssim_init_t"], metrics["psnr_deblur_t"], metrics["ssim_deblur_t"])
            save_metrics(metrics, os.path.join(self.exp_path, f"metrics.p"))

def main(argv):
    trainer = Evaluator(argv)
    trainer.run_eval() # perform training

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="gopro")
    parser.add_argument('--ckpt_step', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=7)
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hpc', action="store_true")
    parser.add_argument('--crop_eval', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--sa', type=int, default=1)
    argv = parser.parse_args()

    print('sample_size:', argv.sample_size, type(argv.sample_size))
    print('dataset:', argv.dataset, type(argv.dataset))
    print('ckpt_step:', argv.ckpt_step, type(argv.ckpt_step))
    print('random_seed:', argv.random_seed, type(argv.random_seed))
    print('ckpt_path:', argv.ckpt_path, type(argv.ckpt_path))
    print('num_workers:', argv.num_workers, type(argv.num_workers))
    print('hpc:', argv.hpc, type(argv.hpc))
    print('crop_eval:', argv.crop_eval, type(argv.crop_eval))
    print('train:', argv.train, type(argv.train))
    print('sample_average:', argv.sa, type(argv.sa))

    main(argv)