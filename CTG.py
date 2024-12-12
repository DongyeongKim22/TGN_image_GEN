import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, TopKPooling, GraphNorm
from torch_geometric.utils import grid
from utils.utils2 import process_batch_to_graph
import torchvision.models as models
import math
from utils.CNN import CNN_extract, CNN_upsample
import time

class FrameGraphGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_hops):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GCNConv(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_hops)
        ])
        self.num_hops = num_hops
        self.relu = nn.LeakyReLU()

    def forward(self, x, edge_index):
        x_orig = x
        for i in range(self.num_hops):
            x = self.gcn_layers[i](x, edge_index)
            x = self.relu(x)
        x = x + x_orig
        return x

class SpatioTemporalGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_hops, size):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GCNConv(in_channels if i == 0 else out_channels, out_channels) 
            for i in range(num_hops)
        ])
        height = width = size
        hidden_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_hops = num_hops
        self.relu = nn.LeakyReLU()
        self.rnn = nn.LSTM(
            input_size=hidden_channels * height * width, 
            hidden_size=hidden_channels, 
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 2, hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 4, hidden_channels * height * width)
        )

    def forward(self, x, edge_index, batch_size, seq_length, height, width):
      
        x = x.view(batch_size, seq_length, self.hidden_channels*height*width)
        _, (h_n, c_n) = self.rnn(x)
        h_n = h_n.squeeze(0)
        pred = self.mlp(h_n) 
        pred = pred.view(batch_size, self.hidden_channels, height, width)
        
        return pred

class CTG(nn.Module):
  def __init__(self, in_channels, hidden_channels, num_hops, seq_length, size):
      super().__init__()
      self.size = size
      flat = int(size/32)
      self.flat = flat
      self.CNN_extract = CNN_extract(in_channels=in_channels, seq_length=seq_length)
      self.TGCN = SpatioTemporalGCN(hidden_channels, hidden_channels, num_hops, flat)
      self.hidden_channels = hidden_channels

      self.upsample = CNN_upsample(hidden_channels=hidden_channels)
    
  def forward(self, x):
    #lower model
      batch_size, seq_length, channels, height, width = x.size()
      x = x.view(batch_size, seq_length * channels, height, width)

      x = self.CNN_extract(x)
      x = x.view(batch_size, seq_length, self.hidden_channels, self.flat, self.flat)

      # LSTM only
      start_time = time.time()
      mem_before = torch.cuda.memory_allocated()
      
      x = self.TGCN(x, 0 , batch_size, seq_length, self.flat, self.flat)

      mem_after = torch.cuda.memory_allocated()
      peak_mem = torch.cuda.max_memory_allocated()
      end_time = time.time()
      print("Memory Usage = ", peak_mem - mem_before)
      print("Processing time = ", end_time - start_time)
      
      x = self.upsample(x)
      return x






