import torch
from torch.utils.data import Dataset

class FoGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # CNN wants (C, T) not (T, C)
        x = self.X[idx].transpose(0, 1)  # (6, 384)
        y = self.y[idx]
        return x, y
