"""
Model architectures for GC-SCL experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
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
    def __init__(self, block, num_blocks, num_classes=10, width=1):
        super(ResNet, self).__init__()
        self.in_planes = 64 * width

        # Modified first conv for CIFAR (smaller images)
        self.conv1 = nn.Conv2d(3, 64 * width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * width)
        self.layer1 = self._make_layer(block, 64 * width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * width * block.expansion, num_classes)
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
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        logits = self.linear(features)
        if return_features:
            return logits, features
        return logits

    def get_features(self, x):
        """Extract features without classification head."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        return features


def ResNet18(num_classes=10, width=1):
    """ResNet-18 for CIFAR datasets."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width=width)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, in_dim=512, hidden_dim=128, out_dim=128):
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


class ContrastiveModel(nn.Module):
    """Encoder + Projection Head for contrastive learning."""
    def __init__(self, encoder, projection_head):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        
    def forward(self, x, return_features=False):
        features = self.encoder.get_features(x)
        projections = self.projection_head(features)
        if return_features:
            return projections, features
        return projections


class LinearClassifier(nn.Module):
    """Linear classifier for evaluation."""
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.linear(features)
