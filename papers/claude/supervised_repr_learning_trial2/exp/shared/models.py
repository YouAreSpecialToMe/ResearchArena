"""ResNet models with feature extraction for neural collapse analysis."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetWithFeatures(nn.Module):
    """Wrapper around torchvision ResNet that returns both logits and penultimate features."""

    def __init__(self, arch='resnet18', num_classes=10, pretrained=False):
        super().__init__()
        if arch == 'resnet18':
            base = models.resnet18(weights=None)
            self.feat_dim = 512
        elif arch == 'resnet50':
            base = models.resnet50(weights=None)
            self.feat_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # For CIFAR (32x32): modify first conv and remove maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        # Skip maxpool for small images
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(self.feat_dim, num_classes)

        self._is_large_input = False

    def set_large_input(self, large=True):
        """Call this for TinyImageNet (64x64) to use standard conv1."""
        if large:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self._is_large_input = True

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self._is_large_input:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)

        if return_features:
            return logits, features
        return logits


def get_model(arch='resnet18', num_classes=10, dataset='cifar10'):
    model = ResNetWithFeatures(arch=arch, num_classes=num_classes)
    if dataset == 'tinyimagenet':
        model.set_large_input(True)
    return model
