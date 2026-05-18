import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# ── 1. Updated 3-Class Focal Loss ──
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()

# ── 2. Your CNN + TCN Architecture ──
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.drop(self.relu(self.bn(self.conv(x))))
        return self.relu(out + self.skip(x))

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        # Note: Symmetric padding (Non-causal)
        pad = (kernel - 1) * dilation // 2   
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad))
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad))
        self.bn2   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop1(self.relu(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + x)

class FoGCNNTCN(nn.Module):
    def __init__(self, num_classes=3, dropout_cnn=0.3, dropout_tcn=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv_blocks = nn.Sequential(
            ConvBlock(64,  64,  kernel=7, dropout=dropout_cnn),
            nn.MaxPool1d(2),
            ConvBlock(64,  128, kernel=5, dropout=dropout_cnn),
            nn.MaxPool1d(2),
            ConvBlock(128, 128, kernel=3, dropout=dropout_cnn),
        )
        self.tcn_blocks = nn.Sequential(
            TCNBlock(128, kernel=3, dilation=1,  dropout=dropout_tcn),
            TCNBlock(128, kernel=3, dilation=2,  dropout=dropout_tcn),
            TCNBlock(128, kernel=3, dilation=4,  dropout=dropout_tcn),
            TCNBlock(128, kernel=3, dilation=8,  dropout=dropout_tcn),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_blocks(x)
        x = self.tcn_blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)

# ── 3. Update evaluation for 3 classes in your loso_cv function ──
# Change your model instantiation:
# model = FoGCNNTCN(num_classes=3).to(device)

# Provide class weights to FocalLoss to handle the 3-class imbalance:
# class_counts = np.bincount(train_labels)
# weights = 1.0 / class_counts
# weights = weights / weights.sum() # Normalize
# criterion = FocalLoss(alpha=weights, gamma=2.0)