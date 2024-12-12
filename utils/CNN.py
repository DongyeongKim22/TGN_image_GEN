import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, TopKPooling, GraphNorm
from torch_geometric.utils import grid
import torchvision.models as models
import math

class CNN_extract(nn.Module):
  def __init__(self, in_channels, seq_length):
      super().__init__()
      self.cnn = nn.Sequential(
            nn.Conv2d(in_channels*seq_length, 16*seq_length, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16*seq_length),
            nn.GELU(),
            nn.Conv2d(16*seq_length, 16*seq_length, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16*seq_length),
            nn.GELU(),
            
            nn.Conv2d(16*seq_length, 32*seq_length, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32*seq_length),
            nn.GELU(),
            nn.Conv2d(32*seq_length, 64*seq_length, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64*seq_length),
            nn.GELU(),

            nn.Conv2d(64*seq_length, 128*seq_length, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128*seq_length),
            nn.GELU(),
            nn.Conv2d(128*seq_length, 128*seq_length, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128*seq_length),
            nn.GELU(),

            nn.Conv2d(128*seq_length, 256*seq_length, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256*seq_length),
            nn.GELU(),
            nn.Conv2d(256*seq_length, 256*seq_length, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256*seq_length),
            nn.GELU(),

            nn.Conv2d(256*seq_length, 512*seq_length, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512*seq_length),
            nn.GELU(),
            nn.Conv2d(512*seq_length, 512*seq_length, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512*seq_length),
            nn.GELU()
            )
    
  def forward(self, x):
      return self.cnn(x)

class CNN_upsample(nn.Module):
  def __init__(self, hidden_channels):
      super().__init__()
      self.hidden_channels = hidden_channels
      self.upsample = self.upsample = nn.Sequential(
          nn.ConvTranspose2d(self.hidden_channels, int(self.hidden_channels/2), kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.GELU(),
          nn.Conv2d(int(self.hidden_channels/2), int(self.hidden_channels/2), kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(int(self.hidden_channels/2)),
          nn.GELU(),

          nn.ConvTranspose2d(int(self.hidden_channels/2), int(self.hidden_channels/4), kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.GELU(),
          nn.Conv2d(int(self.hidden_channels/4), int(self.hidden_channels/4), kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(int(self.hidden_channels/4)),
          nn.GELU(),

          nn.ConvTranspose2d(int(self.hidden_channels/4), int(self.hidden_channels/8), kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.GELU(),
          nn.Conv2d(int(self.hidden_channels/8), int(self.hidden_channels/8), kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(int(self.hidden_channels/8)),
          nn.GELU(),

          nn.ConvTranspose2d(int(self.hidden_channels/8), int(self.hidden_channels/16), kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.GELU(),
          nn.Conv2d(int(self.hidden_channels/16), int(self.hidden_channels/16), kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(int(self.hidden_channels/16)),
          nn.GELU(),

          nn.ConvTranspose2d(int(self.hidden_channels/16), int(self.hidden_channels/32), kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.GELU(),
          nn.Conv2d(int(self.hidden_channels/32), 3, kernel_size=3, stride=1, padding=1),
          nn.Tanh()
        )
    
  def forward(self, x):
      return self.upsample(x)