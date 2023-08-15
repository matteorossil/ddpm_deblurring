### Matteo Rossi

# Modules
from dataset_multi import Data
from metrics2 import psnr, ssim
from denoiser2 import UNet as Denoiser #
from init2 import UNet as Init
from ddpm2 import DenoiseDiffusion #

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

# Flow
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

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
            self.store_checkpoints: str = '/home/mr6744/ckpts/'
            self.dataset_t: str = f'/home/mr6744/{argv.dataset_t}/'
            self.dataset_v: str = f'/home/mr6744/{argv.dataset_v}/'
            self.ckpt_denoiser: str = f'/home/mr6744/ckpts/{argv.ckpt_path}/ckpt_denoiser_{self.ckpt_step}.pt'
            self.ckpt_initp: str = f'/home/mr6744/ckpts/{argv.ckpt_path}/ckpt_initp_{self.ckpt_step}.pt'
            self.ckpt_metrics_: str = f'/home/mr6744/ckpts/{argv.ckpt_path}/metrics_step{self.ckpt_step}.p'
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
            image_channels=self.image_channels*2+6, #+2 for optical flow #+6 for left and right concatenation
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

        self.flow = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.gpu_id)

        self.denoiser = DDP(self.denoiser, device_ids=[self.gpu_id])
        self.initp = DDP(self.initp, device_ids=[self.gpu_id])
        self.flow = DDP(self.flow, device_ids=[self.gpu_id])
        self.flow  = self.flow.eval()

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

    def sample_(self, mode, path, psnr_init, ssim_init, psnr_deblur, ssim_deblur):

        dataset = Data(path=path, mode=mode, crop_eval=self.crop_eval, size=(self.image_size,self.image_size))
        dataloader = DataLoader(dataset=dataset, batch_size=self.n_samples, num_workers=0, drop_last=False, shuffle=True, pin_memory=False)

        with torch.no_grad():

            torch.manual_seed(self.seed)
            (sharp_left, blur_left), (sharp, blur), (sharp_right, blur_right) = next(iter(dataloader))

            # Move data to device
            sharp_left = sharp_left.to(self.gpu_id)
            blur_left = blur_left.to(self.gpu_id)

            sharp = sharp.to(self.gpu_id)
            blur = blur.to(self.gpu_id)

            sharp_right = sharp_right.to(self.gpu_id)
            blur_right = blur_right.to(self.gpu_id)

            if self.step == 0:
                # save images blur and sharp image pairs
                save_image(sharp_left, os.path.join(self.exp_path, f'{mode}__sharp_left.png'))
                save_image(sharp, os.path.join(self.exp_path, f'{mode}__sharp.png'))
                save_image(sharp_right, os.path.join(self.exp_path, f'{mode}__sharp_right.png'))
                save_image(blur_left, os.path.join(self.exp_path, f'{mode}__blur_left.png'))
                save_image(blur, os.path.join(self.exp_path, f'{mode}__blur.png'))
                save_image(blur_right, os.path.join(self.exp_path, f'{mode}__blur_right.png'))

            # compute initial predictor
            init = self.diffusion.predictor(blur)

            # concatenate conditions on channel dimension
            concat_ = torch.cat((blur_left, blur), dim=1)
            concat_ = torch.cat((concat_, blur_right), dim=1)

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
                X = self.diffusion.p_sample(X, concat_, t_vec)

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
        for batch_idx, ((sharp_left, blur_left), (sharp, blur), (sharp_right, blur_right)) in enumerate(self.dataloader_train):
        #sharp, blur = next(iter(self.dataloader_train))

            # Increment global step
            self.step += 1

            # Move data to device
            sharp_left = sharp_left.to(self.gpu_id)
            blur_left = blur_left.to(self.gpu_id)

            sharp = sharp.to(self.gpu_id)
            blur = blur.to(self.gpu_id)

            sharp_right = sharp_right.to(self.gpu_id)
            blur_right = blur_right.to(self.gpu_id)

            # save images blur and sharp image pairs
            #save_image(sharp, os.path.join(self.exp_path, f'sharp_train_step{self.step}.png'))
            #save_image(blur, os.path.join(self.exp_path, f'blur_train_step{self.step}.png'))

            # get initial prediction
            init = self.diffusion.predictor(blur)
            #save_image(init, os.path.join(self.exp_path, f'init_step{self.step}.png'))

            ### PREDICT FLOW ###
            
            #flow_sharp = self.flow(sharp_left, sharp_right)[-1]
            #imgs_sharp = flow_to_image(flow_sharp)
            #save_image(imgs_sharp.to(torch.float)/255., os.path.join(self.exp_path, f'flow_sharp_step{self.step}.png'))
            #save_image(sharp_left, os.path.join(self.exp_path, f'sharp_left_step{self.step}.png'))
            #save_image(sharp, os.path.join(self.exp_path, f'sharp_step{self.step}.png'))
            #save_image(sharp_right, os.path.join(self.exp_path, f'sharp_right_step{self.step}.png'))
            #flow_blur = self.flow(blur_left, blur_right)[-1]
            #imgs_blur = flow_to_image(flow_blur)
            #save_image(imgs_blur.to(torch.float)/255., os.path.join(self.exp_path, f'flow_blur_step{self.step}.png'))
            #save_image(blur_left, os.path.join(self.exp_path, f'blur_left_step{self.step}.png'))
            #save_image(blur, os.path.join(self.exp_path, f'blur_step{self.step}.png'))
            #save_image(blur_right, os.path.join(self.exp_path, f'blur_right_step{self.step}.png'))

            # compute residual
            residual = sharp - init
            #save_image(residual, os.path.join(self.exp_path, f'residual_step{self.step}.png'))

            # Make the gradients zero
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()

            #### REGULARIZER INIT ####
            if self.threshold <= 0.5: # if threshold <= 0.5 regularizer is applied, otherwise no 
                r = torch.mean(init[:,0,:,:])
                g = torch.mean(init[:,1,:,:])
                b = torch.mean(init[:,2,:,:])
                r_blur = torch.mean(blur[:,0,:,:])
                g_blur = torch.mean(blur[:,1,:,:])
                b_blur = torch.mean(blur[:,2,:,:])
                regularizer_init = (F.l1_loss(r, r_blur) + F.l1_loss(g, g_blur)+ F.l1_loss(b, b_blur))
                regularizer_init = F.threshold(regularizer_init, self.threshold, 0.)
            else:
                regularizer_init = torch.tensor([0.], device=self.gpu_id, requires_grad=False)

            #### DENOISER LOSS ####
            concat_ = torch.cat((blur_left, blur), dim=1)
            concat_ = torch.cat((concat_, blur_right), dim=1)
            denoiser_loss = self.diffusion.loss(residual, concat_)

            #### REGRESSION LOSS INIT ####
            if self.alpha > 0: regression_loss = self.alpha * F.mse_loss(sharp, init)
            else: regression_loss = torch.tensor([0.], device=self.gpu_id, requires_grad=False)

            # final loss
            loss = denoiser_loss + regression_loss + regularizer_init #+ regularizer_denoiser_mean + regularizer_denoiser_std

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

        if self.ckpt_metrics:
            metrics = load_metrics(self.ckpt_metrics_)
        else:
            metrics = {"sample_steps":[], "psnr_init_t":[], "ssim_init_t":[], "psnr_deblur_t":[], "ssim_deblur_t":[], "psnr_init_v":[], "ssim_init_v":[], "psnr_deblur_v":[], "ssim_deblur_v":[]}

        for _ in range(self.epochs):

            # sample at step 0
            if (self.sample) and (self.step == 0) and (self.gpu_id == 0):
                self.sample_("train2", self.dataset_v, metrics["psnr_init_t"], metrics["ssim_init_t"], metrics["psnr_deblur_t"], metrics["ssim_deblur_t"])
                self.sample_("val", self.dataset_v, metrics["psnr_init_v"], metrics["ssim_init_v"], metrics["psnr_deblur_v"], metrics["ssim_deblur_v"])
                metrics["sample_steps"].append(self.step)
                save_metrics(metrics, os.path.join(self.exp_path, f"metrics_step{self.step}.p"))

            # train
            self.train()

            if (self.sample) and (((self.step - self.ckpt_step) % self.sampling_interval) == 0) and (self.gpu_id == 0):
                self.sample_("train2", self.dataset_v, metrics["psnr_init_t"], metrics["ssim_init_t"], metrics["psnr_deblur_t"], metrics["ssim_deblur_t"])
                self.sample_("val", self.dataset_v, metrics["psnr_init_v"], metrics["ssim_init_v"], metrics["psnr_deblur_v"], metrics["ssim_deblur_v"])
                metrics["sample_steps"].append(self.step)
                title = f"eval:train,val - metric:"
                plot_metrics(metrics["sample_steps"], ylabel="psnr", label_init_t="init train", label_deblur_t="deblur train", label_init_v="init val", label_deblur_v="deblur val", metric_init_t=metrics["psnr_init_t"], metric_deblur_t=metrics["psnr_deblur_t"], metric_init_v=metrics["psnr_init_v"], metric_deblur_v=metrics["psnr_deblur_v"], path=self.exp_path, title=title)
                plot_metrics(metrics["sample_steps"], ylabel="ssim", label_init_t="init train", label_deblur_t="deblur train", label_init_v="init val", label_deblur_v="deblur val", metric_init_t=metrics["ssim_init_t"], metric_deblur_t=metrics["ssim_deblur_t"], metric_init_v=metrics["ssim_init_v"], metric_deblur_v=metrics["ssim_deblur_v"], path=self.exp_path, title=title)
                torch.save(self.denoiser.module.state_dict(), os.path.join(self.exp_path, f'ckpt_denoiser_{self.step}.pt'))
                torch.save(self.initp.module.state_dict(), os.path.join(self.exp_path, f'ckpt_initp_{self.step}.pt'))
                save_metrics(metrics, os.path.join(self.exp_path, f"metrics_step{self.step}.p"))

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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_size', type=int, default=16)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--l2_loss', type=float, default=0.)
    parser.add_argument('--dataset_t', type=str, default="gopro2")
    parser.add_argument('--dataset_v', type=str, default="gopro2")
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