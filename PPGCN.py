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

class SpatioTemporalGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_hops, seq_length_plus_one):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GCNConv(in_channels if i == 0 else out_channels, out_channels) 
            for i in range(num_hops)
        ])
        self.num_hops = num_hops
        self.relu = nn.LeakyReLU()
        self.seq_length_plus_one = seq_length_plus_one

    def forward(self, x, edge_index, batch_size, seq_length_plus_one, height, width):
        for i in range(self.num_hops):
            x = self.gcn_layers[i](x, edge_index)
            x = self.relu(x)

        #extract last frame (predicted data)
        last_t = seq_length_plus_one - 1
        nodes_per_frame = height * width
        out_channels = x.size(1)

        out = []
        for b in range(batch_size):
            start = b * seq_length_plus_one * nodes_per_frame + last_t * nodes_per_frame
            end = start + nodes_per_frame
            out_b = x[start:end, :] # (H*W, out_channels)
            out_b = out_b.view(height, width, out_channels).permute(2,0,1).contiguous() # (C,H,W)
            out.append(out_b)
        out = torch.stack(out, dim=0) # (batch_size, C, H, W)

        return out

class PPGCN(nn.Module):
  def __init__(self, in_channels, hidden_channels, num_hops, seq_length, size):
      super().__init__()
      self.size = size
      flat = int(size/32)
      self.flat = flat
      self.CNN_extract = CNN_extract(in_channels=in_channels, seq_length=seq_length)
      self.TGCN = SpatioTemporalGCN(hidden_channels, hidden_channels, num_hops, seq_length+1)
      self.hidden_channels = hidden_channels

      self.upsample = CNN_upsample(hidden_channels=hidden_channels)
    
  def forward(self, x):
      batch_size, seq_length, channels, height, width = x.size()
      x = x.view(batch_size, seq_length * channels, height, width)
      x = self.CNN_extract(x)

      x = x.view(batch_size, seq_length, self.hidden_channels, self.flat, self.flat)

      #feature map to graph with frame to frame edge and virtual future frame
      x, batch_size, seq_p_1, c, h, w = process_batch_to_graph(x)
      
      #No LSTM, GCN only prediction
      start_time = time.time()
      mem_before = torch.cuda.memory_allocated()
      
      x = self.TGCN(x.x, x.edge_index, batch_size, seq_p_1, h, w)
      mem_after = torch.cuda.memory_allocated()
      peak_mem = torch.cuda.max_memory_allocated()
      end_time = time.time()
      
      print("Memory Usage = ", peak_mem - mem_before)
      print("Processing time = ", end_time - start_time)

      x = self.upsample(x)
      return x






