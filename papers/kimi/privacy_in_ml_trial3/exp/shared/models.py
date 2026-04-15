"""
Model architectures for Federated Contrastive Learning.
Includes ResNet-18 encoder and SimCLR projection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic ResNet block for CIFAR."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR with modified first conv."""
    def __init__(self, block, num_blocks, num_classes=10, width=1):
        super(ResNet, self).__init__()
        self.in_planes = 64 * width

        # Modified first conv for CIFAR (kernel=3, stride=1 instead of kernel=7, stride=2)
        self.conv1 = nn.Conv2d(3, 64 * width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * width)
        self.layer1 = self._make_layer(block, 64 * width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512 * width * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        if return_features:
            return features
        return features


def ResNet18(num_classes=10):
    """ResNet-18 for CIFAR."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class ProjectionHead(nn.Module):
    """SimCLR-style projection head."""
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


class SimCLRModel(nn.Module):
    """SimCLR model with encoder and projection head."""
    def __init__(self, encoder, projection_head):
        super(SimCLRModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x, return_embedding=False):
        features = self.encoder(x, return_features=True)
        if return_embedding:
            return features
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)


class LinearClassifier(nn.Module):
    """Linear classifier for evaluation."""
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class MembershipClassifier(nn.Module):
    """Adversarial membership classifier for privacy regularization."""
    def __init__(self, feature_dim=128, hidden_dim=64):
        super(MembershipClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)


def create_simclr_model(width=1):
    """Create SimCLR model with ResNet-18 encoder."""
    encoder = ResNet(BasicBlock, [2, 2, 2, 2], width=width)
    projection_head = ProjectionHead(in_dim=encoder.feature_dim, hidden_dim=512, out_dim=128)
    return SimCLRModel(encoder, projection_head)
