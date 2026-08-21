import torch

from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features.iloc[idx].values.astype(np.float32)
        label = self.labels.iloc[idx].astype(np.int64)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        #print(f"DEBUG: label.dtype in dataset = {label.dtype}")
        return features, label
