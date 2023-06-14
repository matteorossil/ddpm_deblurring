import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class NoiseEmbedding(nn.Module):
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
        self.lin1 = nn.Linear(1, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, a_bar: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        #
        # Transform with the MLP
        emb = self.act(self.lin1(a_bar))
        emb = self.lin2(emb)

        return emb

class ResidualBlockDown(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, noise_channels: int):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        """
        super().__init__()

        self.act1 = Swish()
        self.act2 = Swish()
        self.act3 = Swish()

        self.downsample1 = nn.Conv2d(in_channels, in_channels, (3, 3), (2, 2), (1, 1))
        self.downsample2 = nn.Conv2d(in_channels, in_channels, (3, 3), (2, 2), (1, 1))

        self.conv1_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv2_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv3_3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.dropout = nn.Dropout(p=0.2)

        self.noise_emb = nn.Linear(noise_channels, out_channels)

    def forward(self, x: torch.Tensor, a_bar: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h1 = self.conv1_1x1(self.downsample1(x))

        h2 = self.conv2_3x3(self.downsample2(self.act1(x))) + self.noise_emb(self.act2(a_bar))[:, :, None, None]

        h3 = self.conv3_3x3(self.dropout(self.act3(h2)))

        return h1 + h3
    

class ResidualBlockUp(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, noise_channels: int):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        self.act1 = Swish()
        self.act2 = Swish()
        self.act3 = Swish()

        self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels, (4, 4), (2, 2), (1, 1))
        self.upsample2 = nn.ConvTranspose2d(in_channels, in_channels, (4, 4), (2, 2), (1, 1))

        self.conv1_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv2_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv3_3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.dropout = nn.Dropout(p=0.2)

        self.noise_emb = nn.Linear(noise_channels, out_channels)

    def forward(self, x: torch.Tensor, a_bar: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h1 = self.conv1_1x1(self.upsample1(x))

        print((self.conv2_3x3(self.upsample2(self.act1(x)))).shape)
        print((self.act2(a_bar)).shape)

        print((self.noise_emb(self.act2(a_bar))).shape)

        h2 = self.conv2_3x3(self.upsample2(self.act1(x))) + self.noise_emb(self.act2(a_bar))[:, :, None, None]

        h3 = self.conv3_3x3(self.dropout(self.act3(h2)))

        return h1 + h3

class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, noise_channels: int):
        super().__init__()
        self.res = ResidualBlockDown(in_channels, out_channels, noise_channels)

    def forward(self, x: torch.Tensor, a_bar: torch.Tensor):
        x = self.res(x, a_bar)
        return x


class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, noise_channels: int):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlockUp(in_channels, out_channels, noise_channels)

    def forward(self, x: torch.Tensor, a_bar: torch.Tensor):
        x = self.res(x, a_bar)
        return x


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
        self.noise_emb = NoiseEmbedding(n_channels)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels * ch_mults[0], kernel_size=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels * ch_mults[0]
        # For each resolution
        for i in range(1, n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels))
                in_channels = out_channels

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(1, n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i-1]
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels))
                in_channels = out_channels

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.noise_proj = nn.Conv2d(in_channels, image_channels // 2, kernel_size=(1, 1))

    def unet_forward(self, x: torch.Tensor, a_bar: torch.Tensor):
        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, a_bar)
            h.append(x)

        # Middle (bottom)
        x = h.pop()

        # Second half of U-Net
        for m in self.up:
            x = m(x, a_bar)
            s = h.pop()
            x = x + s

        # Final normalization and convolution
        return self.noise_proj(x + a_bar)

    def forward(self, x: torch.Tensor, a_bar: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get noise embeddings
        a_bar = self.noise_emb(a_bar)

        return self.unet_forward(x, a_bar)
