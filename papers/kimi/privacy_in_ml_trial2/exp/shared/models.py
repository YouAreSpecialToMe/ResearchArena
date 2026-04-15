"""
Model definitions for PRISM experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR datasets."""
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None, num_classes=num_classes)
        # Adapt for CIFAR (32x32 images)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
    def forward(self, x):
        return self.model(x)
    
    def get_layers(self):
        """Return list of layers for layer-wise analysis."""
        layers = []
        # conv1 + bn1
        layers.append(('conv1', nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu)))
        # layer1 (2 blocks)
        layers.append(('layer1_block1', self.model.layer1[0]))
        layers.append(('layer1_block2', self.model.layer1[1]))
        # layer2 (2 blocks)
        layers.append(('layer2_block1', self.model.layer2[0]))
        layers.append(('layer2_block2', self.model.layer2[1]))
        # layer3 (2 blocks)
        layers.append(('layer3_block1', self.model.layer3[0]))
        layers.append(('layer3_block2', self.model.layer3[1]))
        # layer4 (2 blocks)
        layers.append(('layer4_block1', self.model.layer4[0]))
        layers.append(('layer4_block2', self.model.layer4[1]))
        # fc
        layers.append(('fc', self.model.fc))
        return layers


class VGG16(nn.Module):
    """VGG-16 adapted for CIFAR datasets."""
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 
                                           256, 256, 256, 'M', 512, 512, 512, 'M', 
                                           512, 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def get_layers(self):
        """Return list of layers for layer-wise analysis."""
        layers = []
        # Conv layers (13 conv layers with BN and ReLU)
        conv_idx = 0
        current_block = []
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                if current_block:
                    layers.append((f'conv_block_{conv_idx}', nn.Sequential(*current_block)))
                    current_block = []
                conv_idx += 1
            else:
                current_block.append(layer)
        if current_block:
            layers.append((f'conv_block_{conv_idx}', nn.Sequential(*current_block)))
        
        # FC layers (3 layers)
        layers.append(('fc1', nn.Sequential(self.classifier[0], self.classifier[1])))
        layers.append(('fc2', nn.Sequential(self.classifier[3], self.classifier[4])))
        layers.append(('fc3', self.classifier[6]))
        
        return layers


class SimpleCNN(nn.Module):
    """Simple CNN for tabular data (Purchase-100)."""
    def __init__(self, input_dim=600, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    
    def get_layers(self):
        """Return list of layers for layer-wise analysis."""
        layers = [
            ('fc1', self.fc1),
            ('fc2', self.fc2),
            ('fc3', self.fc3),
            ('fc4', self.fc4)
        ]
        return layers


def get_model(arch, num_classes, input_dim=None):
    """Factory function to get model by architecture name."""
    if arch == 'resnet18':
        return ResNet18(num_classes=num_classes)
    elif arch == 'vgg16':
        return VGG16(num_classes=num_classes)
    elif arch == 'simplecnn':
        return SimpleCNN(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
