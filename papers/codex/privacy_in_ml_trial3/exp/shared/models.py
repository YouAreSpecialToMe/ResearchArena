from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class PurchaseMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.features(x)
        logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits


class ResNet18WithFeatures(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = resnet18(num_classes=num_classes)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x)
        logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits
