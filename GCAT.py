import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TopKPooling, GraphNorm
from torch_geometric.utils import grid
import math

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        # self.norm = nn.BatchNorm1d(out_channels)
        self.norm = GraphNorm(out_channels)
        self.relu = nn.LeakyReLU()
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.norm(x)
        x = self.relu(x)
        return x, edge_index


class FramePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_channels, hidden_channels))
        self.hidden_channels = hidden_channels
        # 어텐션 메커니즘 (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 픽셀 값을 [0, 1]로 정규화
        )
    
    def forward(self, graph_batches):
        batch_size = len(graph_batches[0])  # 배치 크기
        num_frames = len(graph_batches)     # 시퀀스 길이
        num_nodes = graph_batches[0][0].num_nodes
        hidden_channels = self.hidden_channels

        # 모든 프레임에서의 노드 특징을 저장할 리스트
        node_features = []

        for t in range(num_frames):
            graphs = graph_batches[t]
            x_list = []
            for data in graphs:
                x, edge_index = data.x, data.edge_index
                for gcn in self.gcn_layers:
                    x, _ = gcn(x, edge_index)
                x_list.append(x)  # x: (num_nodes, hidden_channels)
            x_batch = torch.stack(x_list, dim=0)  # (batch_size, num_nodes, hidden_channels)
            node_features.append(x_batch)

        # node_features: List of (batch_size, num_nodes, hidden_channels)
        # 텐서로 변환하여 형태 조정
        node_features = torch.stack(node_features, dim=1)  # (batch_size, num_frames, num_nodes, hidden_channels)

        # 노드 차원과 배치 차원을 병합하여 Transformer에 입력
        node_features = node_features.permute(2, 0, 1, 3)  # (num_nodes, batch_size, num_frames, hidden_channels)
        node_features = node_features.reshape(-1, num_frames, hidden_channels)  # (num_nodes * batch_size, num_frames, hidden_channels)
        node_features = node_features.permute(1, 0, 2)  # (num_frames, num_nodes * batch_size, hidden_channels)

        # Transformer 적용
        attention_output = self.transformer(node_features)  # (num_frames, num_nodes * batch_size, hidden_channels)

        # 마지막 타임스텝의 출력 사용
        last_output = attention_output[-1]  # (num_nodes * batch_size, hidden_channels)
        last_output = last_output.view(num_nodes, batch_size, hidden_channels)
        last_output = last_output.permute(1, 0, 2)  # (batch_size, num_nodes, hidden_channels)

        batch_size, num_nodes, hidden_channels = last_output.size()
        H = W = int(math.sqrt(num_nodes))  # 정사각형 이미지 크기 가정
        last_output = last_output.reshape(batch_size, hidden_channels, H, W)  # (batch_size, hidden_channels, H, W)

        # 디코더를 통해 이미지 복원
        generated_images = self.cnn(last_output)

        return generated_images






