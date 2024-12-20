import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class KITTIDataset(Dataset):
    def __init__(self, image_dir, seq_length=5, transform=None, frame_step=5):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
        self.seq_length = seq_length
        self.transform = transform
        self.frame_step = frame_step

    def __len__(self):
        return len(self.image_files) - (self.seq_length - 1) * self.frame_step # 0.1 sec step between frames.

    def __getitem__(self, idx):
        frames = []
        for i in range(self.seq_length):
            img_idx = idx + i * self.frame_step
            img_path = os.path.join(self.image_dir, self.image_files[img_idx])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        frames = torch.stack(frames, dim=0)  # (seq_length, C, H, W)
        
        input_frames = frames[:-1]  # (seq_length - 1, C, H, W)
        target_frame = frames[-1]   # (C, H, W)
        
        return input_frames, target_frame




