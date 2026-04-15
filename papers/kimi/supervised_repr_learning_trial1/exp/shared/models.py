"""
Model architectures for contrastive learning experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    """ResNet-18 modified for CIFAR (32x32 images)."""
    
    def __init__(self, num_classes=None):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        
        # Modify for CIFAR: change first conv and remove maxpool
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        
        # Remove the final fc layer for feature extraction
        self.feature_dim = 512
        self.resnet.fc = nn.Identity()
        
    def forward(self, x):
        return self.resnet(x)


class MLPProjector(nn.Module):
    """2-layer MLP projector for contrastive learning."""
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return F.normalize(x, dim=1)


class ContrastiveModel(nn.Module):
    """Full model for contrastive learning: backbone + projector."""
    
    def __init__(self, backbone, projector):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        projections = self.projector(features)
        if return_features:
            return projections, features
        return projections


class LinearClassifier(nn.Module):
    """Linear classifier for evaluation."""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)


def create_resnet18_encoder(projector_dim=128):
    """Create ResNet-18 encoder with MLP projector for contrastive learning."""
    backbone = ResNet18Backbone()
    projector = MLPProjector(input_dim=512, hidden_dim=512, output_dim=projector_dim)
    return ContrastiveModel(backbone, projector)


def create_resnet18_classifier(num_classes):
    """Create linear classifier for ResNet-18 features."""
    return LinearClassifier(512, num_classes)
