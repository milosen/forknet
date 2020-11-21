""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Conv => BN => ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncodeModule(nn.Module):
    """Encoding with conv then maxpool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
            ConvModule(in_channels, out_channels),
            # 2x2 max pooling
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class DecodeModule(nn.Module):
    """Decode with deconv => bn => ReLU => conv module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),
            ConvModule(out_channels, out_channels)
        )

    def forward(self, x):
        return self.decode(x)


class Concat(nn.Module):
    """Tensor concatenation"""
    def forward(self, x1, x2):
        # input is DATAxCHANNELxHEIGHTxWIDTH
        return torch.cat([x2, x1], dim=1)


class Map(nn.Module):
    """Out convolution and logic sigmoid"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))
