import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        #
        # where $d$ is `half_dim`
        
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb

class ResidualBlockDown(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, noise_channels: int):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        """
        super().__init__()

        self.conv1_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.downsample1 = nn.Conv2d(in_channels, in_channels, (3, 3), (2, 2), (1, 1))

        self.act1 = Swish()
        self.conv2_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))

        self.act2 = Swish()
        self.noise_emb = nn.Linear(noise_channels, in_channels)

        self.act3 = Swish()
        self.dropout = nn.Dropout(p=0.2)
        self.conv3_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.downsample2 = nn.Conv2d(in_channels, in_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h1 = self.downsample1(self.conv1_1x1(x))

        h2 = self.conv2_3x3(self.act1(x)) + self.noise_emb(self.act2(t))[:, :, None, None]

        h3 = self.downsample2(self.conv3_3x3(self.dropout(self.act3(h2))))

        return h1 + h3

class ResidualBlockUp(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, noise_channels: int):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        """
        super().__init__()


        self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels, (4, 4), (2, 2), (1, 1))
        self.conv1_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.act1 = Swish()
        self.upsample2 = nn.ConvTranspose2d(in_channels, in_channels, (4, 4), (2, 2), (1, 1))
        self.conv2_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))

        self.act2 = Swish()
        self.noise_emb = nn.Linear(noise_channels, in_channels)

        self.act3 = Swish()
        self.dropout = nn.Dropout(p=0.2)
        self.conv3_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h1 = self.conv1_1x1(self.upsample1(x))

        h2 = self.conv2_3x3(self.upsample2(self.act1(x))) + self.noise_emb(self.act2(t))[:, :, None, None]

        h3 = self.conv3_3x3(self.dropout(self.act3(h2)))

        return h1 + h3

class Intermediate(nn.Module):
    """
    ### Intermediate block

    This combines `ResidualBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.act1 = Swish()
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.act2 = Swish()

    def forward(self, x: torch.Tensor):

        x = self.dropout(self.act1(self.conv1(x)))
        x = self.act2(self.conv2(x))

        return x
    
class MiddleBlock(nn.Module):
    """
    ### Intermediate block

    This combines `ResidualBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block1 = Intermediate(in_channels, out_channels)
        self.block2 = Intermediate(out_channels, out_channels)

    def forward(self, x: torch.Tensor):

        return self.block2(self.block1(x))

class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, 
                image_channels: int = 6, 
                n_channels: int = 32,
                ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),
                n_blocks: int = 1):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Time embedding layer. Time embedding has `n_channels` channels
        time_embedding = n_channels * 4
        self.time_emb = TimeEmbedding(time_embedding)

        # Project image into feature map
        self.init = nn.Conv2d(image_channels, image_channels, kernel_size=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        in_channels = image_channels
        out_channels = n_channels
        # For each resolution
        for i in range(n_resolutions - 1):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            down.append(Intermediate(in_channels, out_channels))
            down.append(ResidualBlockDown(out_channels, time_embedding))
            in_channels = out_channels

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        out_channels = n_channels * ch_mults[-1]
        self.middle = MiddleBlock(in_channels, out_channels)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions - 1)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i]
            up.append(ResidualBlockUp(in_channels, time_embedding))
            up.append(Intermediate(in_channels + out_channels, out_channels))
            in_channels = out_channels

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.final = nn.Conv2d(n_channels, 3, kernel_size=(1, 1))

    def unet_forward(self, x: torch.Tensor, t: torch.Tensor):
        # Get image projection
        x = self.init(x)
        #print(x.shape)

        # `h` will store outputs at each resolution for skip connection
        h = []
        # First half of U-Net
        for m in self.down:
            if isinstance(m, Intermediate):
                x = m(x)
                h.append(x)
            else:
                x = m(x, t)

            #print(x.shape)

        # Middle (bottom)
        x = self.middle(x)
        #print(x.shape)
        
        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Intermediate):
                s = h.pop()
                s = s * (1 / 2**0.5)
                x = torch.cat((x, s), dim=1)
                x = m(x)
            else:
                x = m(x, t)
            #print(x.shape)

        # Final convolution
        x = self.final(x)
        #print(x.shape)

        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        # Get noise embeddings
        t = self.time_emb(t)

        return self.unet_forward(x, t)
