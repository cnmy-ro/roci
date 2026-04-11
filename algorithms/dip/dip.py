import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from tqdm import tqdm

np.random.seed(0)


def fft2c(image):
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fft2(image, dim=(-2, -1))
    kspace = torch.fft.fftshift(kspace, dim=(-2, -1))
    return kspace


def ifft2c(kspace):
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(kspace, dim=(-2, -1))
    image = torch.fft.fftshift(image, dim=(-2, -1))
    return image


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 2, 1))        
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1))
    def forward(self, input):
        return self.block(input)

class UNet(nn.Module):
    def __init__(self, num_filters=8, depth=3):
        super().__init__()

        self.num_filters = num_filters
        self.depth = depth        

        self.from_image = nn.Conv2d(2, num_filters, 3, 1, 1)
        
        self.down = nn.ModuleList()        
        for _ in range(depth):
            res_block = DownBlock(num_filters, num_filters*2)
            self.down.append(res_block)
            num_filters *= 2

        self.bottleneck = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, 1, 1), nn.LeakyReLU(),
                                        nn.Conv2d(num_filters, num_filters, 3, 1, 1), nn.LeakyReLU())

        self.up = nn.ModuleList()
        for _ in range(depth):
            res_block = UpBlock(num_filters*2, num_filters//2)
            self.up.append(res_block)
            num_filters //= 2

        self.to_image = nn.Conv2d(num_filters*2, 2, 3, 1, 1)

    def forward(self, image):
        x = self.from_image(image)
        skips = []
        for block in self.down:
            skips.append(x.clone())
            x = block(x)
        skips.append(x.clone())
        skips.reverse()
        x = self.bottleneck(x)        
        for i, block in enumerate(self.up):
            x = torch.cat([skips[i], x], dim=1)
            x = block(x)
        x = torch.cat([skips[i+1], x], dim=1)
        x = self.to_image(x)
        return x