import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import grid
import matplotlib.pyplot as plt


def create_spatiotemporal_edge_index(height, width, seq_length, add_temporal_edges=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = []
    H, W = height, width
    # Spatial edges
    for t in range(seq_length):
        frame_offset = t * (H * W)
        for i in range(H):
            for j in range(W):
                idx = frame_offset + i * W + j
                # grid edge gen
                if j + 1 < W:
                    right = frame_offset + i * W + (j + 1)
                    edge_index.extend([[idx, right], [right, idx]])
                if i + 1 < H:
                    down = frame_offset + (i + 1) * W + j
                    edge_index.extend([[idx, down], [down, idx]])
    
    # Temporal edges (one direction: t -> t+1)
    if add_temporal_edges and seq_length > 1:
        for t in range(seq_length - 1):
            curr_offset = t * (H * W)
            next_offset = (t + 1) * (H * W)
            for i in range(H):
                for j in range(W):
                    # current frame to next frame
                    curr_idx = curr_offset + i * W + j
                    next_idx = next_offset + i * W + j
                    # erase inverse direction
                    edge_index.extend([[curr_idx, next_idx]])
                    

    edge_index = torch.tensor(edge_index).t().long().to(device)
    return edge_index


def process_batch_to_graph(batch, add_future_frame=True, add_temporal_edges=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = batch.to(device)
    batch_size, seq_length, channels, height, width = batch.size()
    if add_future_frame:
        # add one virtual future frame graph
        seq_length_plus_one = seq_length + 1
        future_frame = torch.zeros(batch_size, 1, channels, height, width, device=device)
        full_frames = torch.cat([batch, future_frame], dim=1) # (batch_size, seq_length+1, C, H, W)
        x = full_frames.permute(0,1,3,4,2).contiguous()
        x = x.view(batch_size * seq_length_plus_one * height * width, channels)

        edge_index_single = create_spatiotemporal_edge_index(height, width, seq_length_plus_one, add_temporal_edges)

        num_nodes_per_sequence = seq_length_plus_one * height * width
        edge_indices = []
        for b in range(batch_size):
            offset = b * num_nodes_per_sequence
            ei = edge_index_single + offset
            edge_indices.append(ei)
        edge_index = torch.cat(edge_indices, dim=1)

        data = Data(x=x, edge_index=edge_index)
        data = data.to(device)
        return data, batch_size, seq_length_plus_one, channels, height, width

    else:
        # No virtual frame graph
        x = batch.permute(0,1,3,4,2).contiguous()
        x = x.view(batch_size * seq_length * height * width, channels)

        edge_index_single = create_spatiotemporal_edge_index(height, width, seq_length, add_temporal_edges)

        num_nodes_per_sequence = seq_length * height * width
        edge_indices = []
        for b in range(batch_size):
            offset = b * num_nodes_per_sequence
            ei = edge_index_single + offset
            edge_indices.append(ei)
        edge_index = torch.cat(edge_indices, dim=1)

        data = Data(x=x, edge_index=edge_index)
        data = data.to(device)
        return data, batch_size, seq_length, channels, height, width