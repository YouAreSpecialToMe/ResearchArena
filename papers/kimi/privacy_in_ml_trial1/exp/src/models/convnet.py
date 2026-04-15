"""
4-layer ConvNet for privacy benchmarking.
Architecture: 128-256-512-512 channels with BN, ReLU, MaxPool, FC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(512 * 2 * 2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 -> 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16 -> 8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8 -> 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4 -> 2
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
