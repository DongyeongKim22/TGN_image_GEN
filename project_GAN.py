import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Generator(nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.upsample = nn.Sequential(
            UpsampleBlock(64),
            # UpsampleBlock(64), 
        )
        
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)
    
    def forward(self, x):
        batch_size, num_nodes, feature_dim = x.size()

        side_length = int(sqrt(num_nodes))
        x = x.reshape(batch_size, feature_dim, side_length, side_length)
        x = self.prelu(self.conv1(x))
        x = self.conv2(x)
        x = self.upsample(x)
        
        x = torch.tanh(self.conv3(x))
        return x



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, image_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            ConvBlock(64, 64, stride=2),
            ConvBlock(64, 128),
            ConvBlock(128, 128, stride=2),
            ConvBlock(128, 256),
            ConvBlock(256, 256, stride=2),
            ConvBlock(256, 512),
            ConvBlock(512, 512, stride=2),
            nn.Flatten(),
            nn.Linear(512 * (image_size // 16) ** 2, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.net(x)




