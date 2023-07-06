# Matteo Rossi


from typing import List
import os
import torch
import torch.utils.data
from diffusion.ddpm_unconditioned import DenoiseDiffusion
from eps_models.unet_unconditioned import UNet

from dataset_unconditioned import Data
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class Trainer():
    """
    ## Configurations
    """
    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size_h: int = 128
    image_size_w: int = 128
    # Number of channels in the initial feature map
    n_channels: int = 32
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 3, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]
    attention_middle: List[int] = [True]
    # noise scheduler
    beta_0: int = 1e-6 # 0.000001
    beta_T: int = 1e-2 # 0.01

    # Define sampling

    # Number of time steps $T$
    n_steps: int = 2000
    # Number of sample images
    n_samples: int = 64
    # checkpoint path
    epoch: int = 14940
    #checkpoint = f'/scratch/mr6744/pytorch/checkpoints_distributed/06132023_202606/checkpoint_{epoch}.pt'
    checkpoint: str = f'/home/mr6744/checkpoints_unconditioned/checkpoint_{epoch}.pt'
    # store sample
    #sampling_path = '/scratch/mr6744/pytorch/checkpoints_distributed/06132023_202606/sampling/'
    sampling_path: str = '/home/mr6744/checkpoints_unconditioned/sample2/'

    dataset: str = '/home/mr6744/gopro/'

    def init(self):
        # device
        self.device: torch.device = 'cuda' # change to 'cuda'

        # Create $\epsilon_\theta(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
            attn_middle=self.attention_middle
        ).to(self.device)
        
        #load from checkpoint
        checkpoint_ = torch.load(self.checkpoint)
        self.eps_model.load_state_dict(checkpoint_)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
            beta_0=self.beta_0,
            beta_T=self.beta_T
        )

        dataset = Data(path=self.dataset, mode="val", size=(self.image_size_h,self.image_size_w))
        self.dataloader_val = DataLoader(dataset=dataset,
                                    batch_size=self.n_samples,
                                    num_workers=0,
                                    drop_last=False,
                                    shuffle=False)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():

            # Set seed for replicability
            #torch.cuda.manual_seed_all(0)
            blur = next(iter(self.dataloader_val))
            blur = blur.to(self.device)

            save_image(blur, os.path.join(self.sampling_path, f"blur.png"))

            # Sample Initial Image (Random Gaussian Noise)
            # x = torch.randn([self.n_samples, self.image_channels, self.image_size_h, self.image_size_w], device=self.device)

            #x = torch.load('xt.pt')
            #x = x.to(self.device)

            t_seq = torch.floor(torch.linspace(25, 500 - 1, 20)).type(torch.long).unsqueeze(-1)

            for t_i in t_seq:

                print("running for t:", t_i.item()+1)

                noise = torch.randn_like(blur, device=self.device)
                #noise = torch.zeros(blur.shape, device=self.device)
                blur_noise = self.diffusion.q_sample(blur, t_i.repeat(blur.shape[0]), eps=noise)

                for t_ in range(t_i.item()):

                    print(t_)

                    t = t_i.item() - t_ - 1

                    # Sample
                    t_vec = blur_noise.new_full((self.n_samples,), t, dtype=torch.long)
                    blur_noise = self.diffusion.p_sample(blur_noise, t_vec)

                    # save sampled images
                    if ((t_+1) % t_i.item() == 0):
                        save_image(blur_noise, os.path.join(self.sampling_path, f"deblurred_{t_i.item()+1}.png"))


def main():
    trainer = Trainer()
    trainer.init()
    trainer.sample()

if __name__ == "__main__":
    main()