"""Opacus-compatible model definitions - no inplace operations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet BasicBlock without any inplace operations."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(num_groups, planes), planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.relu(out + identity)  # non-inplace addition and relu
        return out


class ResNet18(nn.Module):
    """Opacus-compatible ResNet-18 with GroupNorm and no inplace ops."""

    def __init__(self, num_classes=10, in_channels=3, num_groups=32):
        super().__init__()
        self.in_planes = 64
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, 64), 64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(self.num_groups, planes), planes),
            )

        layers = [BasicBlock(self.in_planes, planes, stride, downsample, self.num_groups)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, num_groups=self.num_groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def get_resnet18(num_classes, input_channels=3):
    """Get Opacus-compatible ResNet-18 for 64x64 inputs."""
    return ResNet18(num_classes=num_classes, in_channels=input_channels)


def get_model(arch_name, num_classes, input_channels=3):
    """Factory function for models."""
    if arch_name == "resnet18":
        return get_resnet18(num_classes, input_channels)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
