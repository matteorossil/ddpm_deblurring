# Matteo Rossi


from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm import DenoiseDiffusion
from eps_models.unet import UNet
from pathlib import Path
from datetime import datetime
import wandb
import torch.nn.functional as F

from dataset import Data
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def get_exp_path(path=''):
    exp_path = os.environ.get('EXP') or os.path.join(path, 'checkpoints')
    exp_path = os.path.join(exp_path, datetime.now().strftime("%m%d%Y_%H%M%S"))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    return exp_path


class Trainer():
    """
    ## Configurations
    """
    # Device to train the model on.
    device: torch.device = 'cuda' # change to 'cuda'
    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 256
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]
    # Number of time steps $T$
    n_steps: int = 1_000
    # Batch size
    batch_size: int = 16
    # Learning rate
    learning_rate: float = 2e-5
    # Number of training epochs
    epochs: int = 1_000
    # Number of sample images
    n_samples: int = 4
    # Use wandb
    wandb: bool = False
    # where to store the checkpoints
    ckp_path: str = '/home/mr6744/'
    #ckp_path: str = '/Users/m.rossi/Desktop/research/'
    # where to training and validation data is stored
    dataset = '/home/mr6744/gopro/'
    #dataset = '/Users/m.rossi/Desktop/research/ddpm_deblurring/dataset/'
    # where to store image samples
    samples = '/home/mr6744/ddpm_deblurring/samples'
    #samples = '/Users/m.rossi/Desktop/research/ddpm_deblurring/samples/'

    def init(self):
        # Create $\epsilon_\theta(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)
        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )
        # Create dataloader
        self.data_loader = DataLoader(dataset=Data(path=self.dataset, mode="train", size=(self.image_size,self.image_size)), batch_size=self.batch_size, num_workers=0, drop_last=True, shuffle=True, pin_memory=True)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)
        self.step = 0
        self.exp_path = get_exp_path(path=self.ckp_path)

    def sample(self, n_samples=64):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            # Sample Initial Image (Random Gaussian Noise)
            x = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)
            # Remove noise for $T$ steps
            for t_ in range(self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $p_\theta(x_{t-1}|x_t)$
                t_vec = x.new_full((n_samples,), t, dtype=torch.long)
                x = self.diffusion.p_sample(x, t_vec)
            # Log samples
            if self.wandb:
                wandb.log({'samples': wandb.Image(x)}, step=self.step)
            return x

    def train(self):
        """
        ### Train
        """
        # Iterate through the dataset
        for batch_idx, (sharp, blur) in enumerate(self.data_loader):
            # Increment global step
            self.step += 1
            # Move data to device
            sharp = sharp.to(self.device)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(sharp)
            print(loss)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            #print('loss:', loss, 'step:', self.step)
            wandb.log({'loss': loss}, step=self.step)

    def run(self):
        """
        ### Training loop
        """
        for epoch in range(self.epochs):
            if epoch % 10 == 0:
                # Sample some images
                s = self.sample(self.n_samples)
                save_image(s, os.path.join(self.samples, f'epoch_{epoch}.png'))
            # Train the model
            self.train()
            if (epoch+1) % 10 == 0:
                # Save the eps model
                torch.save(self.eps_model.state_dict(), os.path.join(self.exp_path, f'checkpoint_{epoch+1}.pt'))


def main():
    wandb.init()
    train = Trainer()
    train.init() # initialize trainer class
    train.run() # perform training

if __name__ == "__main__":
    main()