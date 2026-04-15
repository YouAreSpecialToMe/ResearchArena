from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def feature_norm(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x).flatten(1)
        return feats.norm(dim=1)


class CIFARLinearProbe(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def feature_norm(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return feats.norm(dim=1)


def build_model(dataset: str) -> nn.Module:
    if dataset == "fashion_mnist":
        return FashionCNN()
    if dataset == "cifar10":
        return CIFARLinearProbe()
    raise ValueError(dataset)
