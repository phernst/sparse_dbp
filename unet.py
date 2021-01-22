# Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 1,
                 depth: int = 5, wf: int = 6):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
        """
        super().__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(
                prev_channels,
                2**(wf+i),
                twice=i != (depth - 1)))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(
                prev_channels,
                2**(wf+i),
                twice=i == 0))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, *inputs):
        assert len(inputs) == 1
        x = inputs[0]

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, twice=True):
        super().__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=1, padding_mode='replicate'))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_size))

        if twice:
            block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                                   padding=1, padding_mode='replicate'))
            block.append(nn.ReLU())
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, *inputs):
        assert len(inputs) == 1
        x = inputs[0]

        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, twice=False):
        super().__init__()
        self.conv_block_lores = UNetConvBlock(in_size, in_size//2, twice=False)

        self.up = nn.Upsample(mode='bilinear',
                              scale_factor=2,
                              align_corners=False)

        self.conv_block_hires = UNetConvBlock(in_size, out_size, twice=twice)

    @staticmethod
    def center_crop(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]),
                     diff_x:(diff_x + target_size[1])]

    def forward(self, *inputs):
        assert len(inputs) == 2
        x, bridge = inputs

        x_lores = self.conv_block_lores(x)
        x_hires = self.up(x_lores)
        crop1 = UNetUpBlock.center_crop(bridge, x_hires.shape[2:])
        out = torch.cat([x_hires, crop1], 1)
        out = self.conv_block_hires(out)

        return out
