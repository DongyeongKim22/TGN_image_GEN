import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), 
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.cnn(x)



class STGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, time_steps):
        super(STGN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.self_attention = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_channels, in_channels)
        self.time_steps = time_steps

    def forward(self, data_list_seq):
        batch_size = len(data_list_seq) // self.time_steps

        h = []
        for b in range(batch_size):
           
            batch_h = []
            for t in range(self.time_steps):
                idx = b * self.time_steps + t 
                x_t = data_list_seq[idx].x  
                edge_index_t = data_list_seq[idx].edge_index 
                x_t = self.gcn1(x_t, edge_index_t)
                x_t = self.gcn2(x_t, edge_index_t)
                batch_h.append(x_t)
            batch_h = torch.stack(batch_h, dim=0)  # (time_steps, num_nodes, hidden_channels)
            h.append(batch_h)
        
        h = torch.stack(h, dim=0)  # (batch_size, time_steps, num_nodes, hidden_channels)
        
        batch_size, time_steps, num_nodes, hidden_channels = h.size()
        h = h.view(batch_size, time_steps * num_nodes, hidden_channels)  # (batch_size, time_steps * num_nodes, hidden_channels)

        h, _ = self.self_attention(h, h, h)  # (batch_size, time_steps * num_nodes, hidden_channels)

        return h



class Upscaler(nn.Module):
    def __init__(self, in_channels, height, width):
        super(Upscaler, self).__init__()
        self.height = height
        self.width = width
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.permute(1, 0).view(1, -1, self.height, self.width)  # (batch_size, channels, height, width)
        return self.deconv(x)




