import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import grid
import matplotlib.pyplot as plt

def visualize_predictions(model, dataloader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch_data, target_frames in dataloader:
            batch_data, target_frames = batch_data.to(device), target_frames.to(device)
            output = model(batch_data)

            batch_size, in_channels, H, W = output.size()

            # 3 set visualization
            max_sets_to_show = min(batch_size, 3)

            fig, axes = plt.subplots(max_sets_to_show * 2, 3, figsize=(15, 10 * max_sets_to_show))

            for i in range(max_sets_to_show):
                # input 3 images at top size
                for j in range(3):
                    axes[i*2, j].imshow(batch_data[i, j].permute(1, 2, 0).cpu().numpy())
                    axes[i*2, j].set_title(f"Input Frame {j+1} (Batch {i+1})")
                
                # last frame, target frame and predicted frame
                axes[i*2+1, 0].imshow(batch_data[i, -1].permute(1, 2, 0).cpu().numpy())
                axes[i*2+1, 0].set_title(f"Input Last Frame (Batch {i+1})")
                
                axes[i*2+1, 1].imshow(target_frames[i].permute(1, 2, 0).cpu().numpy())
                axes[i*2+1, 1].set_title(f"Target Frame (Batch {i+1})")
                
                axes[i*2+1, 2].imshow(output[i].permute(1, 2, 0).cpu().numpy())
                axes[i*2+1, 2].set_title(f"Predicted Frame (Batch {i+1})")

            plt.tight_layout()
            plt.show()
            break


