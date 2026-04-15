"""Model definitions for contrastive learning experiments."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """ResNet encoder modified for 32x32 input (CIFAR)."""

    def __init__(self, arch='resnet18'):
        super().__init__()
        if arch == 'resnet18':
            base = models.resnet18(weights=None)
            feat_dim = 512
        elif arch == 'resnet50':
            base = models.resnet50(weights=None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Modify for 32x32 input: replace 7x7 conv with 3x3, remove maxpool
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        h = self.features(x)
        return h.squeeze(-1).squeeze(-1)


class SupConModel(nn.Module):
    """Encoder + projection head for contrastive learning."""

    def __init__(self, arch='resnet18', proj_dim=128):
        super().__init__()
        self.encoder = ResNetEncoder(arch)
        feat_dim = self.encoder.feat_dim

        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        feat = self.encoder(x)
        z = self.head(feat)
        z = nn.functional.normalize(z, dim=1)
        return feat, z


class LinearClassifier(nn.Module):
    """Linear classifier for evaluation."""

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class CEModel(nn.Module):
    """Encoder + linear classifier for cross-entropy training."""

    def __init__(self, arch='resnet18', num_classes=100):
        super().__init__()
        self.encoder = ResNetEncoder(arch)
        self.fc = nn.Linear(self.encoder.feat_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.fc(feat)
        return feat, logits
