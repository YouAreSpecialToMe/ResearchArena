from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ResNet9(nn.Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.res1 = ResidualBlock(128)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.res2 = ResidualBlock(embedding_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res2(x)
        embedding = self.pool(x).flatten(1)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits


class PurchaseMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.features(x)
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits
