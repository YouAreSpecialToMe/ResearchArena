"""ResNet encoder adapted for CIFAR-100 (32x32 input) with projection head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck


# Architecture configurations
CONFIGS = {
    'resnet18': (BasicBlock, [2, 2, 2, 2], 512),
    'resnet50': (Bottleneck, [3, 4, 6, 3], 2048),
}


class ResNetCIFAR(nn.Module):
    """ResNet adapted for 32x32 CIFAR images.

    Changes from standard ResNet:
    - First conv: 3x3, stride=1, padding=1 (instead of 7x7, stride=2)
    - No max-pool layer
    """
    def __init__(self, arch='resnet18', num_classes=None):
        super().__init__()
        block, layers, self.feat_dim = CONFIGS[arch]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feat_dim, num_classes) if num_classes else None

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            downsample = None
            if s != 1 or self.in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, planes * block.expansion, 1, stride=s, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers.append(block(self.in_planes, planes, stride=s, downsample=downsample))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)

        if return_features or self.fc is None:
            return feat

        out = self.fc(feat)
        return out


def ResNet50CIFAR(num_classes=None):
    return ResNetCIFAR(arch='resnet50', num_classes=num_classes)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning."""
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ContrastiveModel(nn.Module):
    """Encoder + projection head for contrastive learning."""
    def __init__(self, arch='resnet18', proj_dim=128):
        super().__init__()
        self.encoder = ResNetCIFAR(arch=arch)
        self.projector = ProjectionHead(in_dim=self.encoder.feat_dim,
                                        hidden_dim=self.encoder.feat_dim,
                                        out_dim=proj_dim)

    def forward(self, x):
        feat = self.encoder(x, return_features=True)
        z = self.projector(feat)
        return feat, z
