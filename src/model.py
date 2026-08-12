"""1D-CNN for ECG heartbeat classification.

Matches the Week 6 plan card: Conv1d + BatchNorm + ReLU + MaxPool stacks,
global average pool, then a Linear head. Input: [batch, 1, 187].
"""

import torch.nn as nn


class ECG1DCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # global average pool -> [batch, 128, 1]
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        z = self.features(x).squeeze(-1)  # [batch, 128]
        return self.head(z)
