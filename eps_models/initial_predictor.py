import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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
        self.dropout = nn.Dropout(p=0.2)
        self.conv3_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.downsample2 = nn.Conv2d(in_channels, in_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h1 = self.downsample1(self.conv1_1x1(x))

        h2 = self.conv2_3x3(self.act1(x))

        h3 = self.downsample2(self.conv3_3x3(self.dropout(self.act2(h2))))

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
        self.dropout = nn.Dropout(p=0.2)
        self.conv3_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h1 = self.conv1_1x1(self.upsample1(x))

        h2 = self.conv2_3x3(self.upsample2(self.act1(x)))

        h3 = self.conv3_3x3(self.dropout(self.act2(h2)))

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
                image_channels: int = 3, 
                n_channels: int = 64,
                ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4, 8)):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

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
            down.append(ResidualBlockDown(out_channels))
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
            up.append(ResidualBlockUp(in_channels))
            up.append(Intermediate(in_channels + out_channels, out_channels))
            in_channels = out_channels

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.final = nn.Conv2d(n_channels, 3, kernel_size=(1, 1))

    def unet_forward(self, x: torch.Tensor):
        # Get image projection
        x = self.init(x)
        print(x.shape)

        # `h` will store outputs at each resolution for skip connection
        h = []
        # First half of U-Net
        for m in self.down:
            x = m(x)
            if isinstance(m, Intermediate):
                h.append(x)

            print(x.shape)

        # Middle (bottom)
        x = self.middle(x)
        print(x.shape)
        
        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Intermediate):
                s = h.pop()
                s = s * (1 / 2**0.5)
                x = torch.cat((x, s), dim=1)
            x = m(x)
            print(x.shape)

        # Final convolution
        x = self.final(x)
        print(x.shape)

        return x

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        return self.unet_forward(x)
