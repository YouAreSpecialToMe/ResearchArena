from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


@dataclass
class ModelBundle:
    backbone: nn.Module
    projection_head: nn.Module
    classifier: nn.Module


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class EncoderWithHead(nn.Module):
    def __init__(self, num_coarse_classes: int) -> None:
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=weights)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection_head = ProjectionHead(in_dim, 512, 128)
        self.classifier = nn.Linear(in_dim, num_coarse_classes)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return F.normalize(feats, dim=-1)

    def forward_views(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        stacked = torch.cat([x1, x2], dim=0)
        feats = self.forward_backbone(stacked)
        f1, f2 = feats[: x1.shape[0]], feats[x1.shape[0] :]
        p1 = self.projection_head(f1)
        p2 = self.projection_head(f2)
        logits = self.classifier(f1)
        return {
            "feat1": f1,
            "feat2": f2,
            "proj1": p1,
            "proj2": p2,
            "logits": logits,
        }


def create_model(num_coarse_classes: int) -> EncoderWithHead:
    return EncoderWithHead(num_coarse_classes)
