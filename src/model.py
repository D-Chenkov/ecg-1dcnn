"""1D-ResNet for ECG heartbeat classification.

Residual (skip) connections let gradients flow through deeper stacks (Course 3
Module 1). Same interface as before: class ECG1DCNN(n_classes=5), input
[batch, 1, 187] -> logits [batch, n_classes], so train/evaluate/serve are unchanged.
"""

import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Two conv layers + a skip connection (with a 1x1 projection when shapes change)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        # project the identity when channels/length change, so it can be added
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)   # the skip connection


class ECG1DCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.layer1 = ResidualBlock1D(32, 32)
        self.layer2 = ResidualBlock1D(32, 64, stride=2)
        self.layer3 = ResidualBlock1D(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)     # length-agnostic -> [batch, 128, 1]
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)            # [batch, 128]
        return self.head(x)
