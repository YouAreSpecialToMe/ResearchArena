"""
Model definitions for CAG-HNM experiments.
Includes ResNet-18 for CIFAR and ResNet-50 for fine-grained datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18_CIFAR(nn.Module):
    """ResNet-18 modified for CIFAR-100 (32x32 images)."""
    
    def __init__(self, num_classes=100, feature_dim=512):
        super(ResNet18_CIFAR, self).__init__()
        # Load standard ResNet-18
        base_model = models.resnet18(pretrained=False)
        
        # Modify first conv for CIFAR (32x32 instead of 224x224)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()  # Remove maxpool for small images
        
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC
        self.feature_dim = feature_dim
        
        # For supervised learning (cross-entropy baseline)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        if return_features:
            return features
        
        logits = self.fc(features)
        return logits
    
    def get_features(self, x):
        """Get feature representations."""
        return self.forward(x, return_features=True)


class ResNet50_FineGrained(nn.Module):
    """ResNet-50 for fine-grained datasets (CUB-200)."""
    
    def __init__(self, num_classes=200, pretrained=True):
        super(ResNet50_FineGrained, self).__init__()
        base_model = models.resnet50(pretrained=pretrained)
        
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_dim = 2048
        
        # For supervised learning
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        if return_features:
            return features
        
        logits = self.fc(features)
        return logits
    
    def get_features(self, x):
        """Get feature representations."""
        return self.forward(x, return_features=True)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning."""
    
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class SupConModel(nn.Module):
    """Supervised Contrastive Learning model with encoder + projection head."""
    
    def __init__(self, encoder, projection_head):
        super(SupConModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(self, x, return_features=False):
        """
        Returns projection head output for contrastive loss.
        If return_features=True, also returns encoder features.
        """
        features = self.encoder.get_features(x)
        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)
        
        if return_features:
            return projections, features
        return projections


class LinearClassifier(nn.Module):
    """Linear classifier for linear evaluation protocol."""
    
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features):
        return self.fc(features)


def create_model(model_name='resnet18_cifar', num_classes=100, projection_dim=128):
    """
    Create model based on name.
    
    Args:
        model_name: 'resnet18_cifar' or 'resnet50_fg'
        num_classes: number of output classes
        projection_dim: dimension of projection head output
    """
    if model_name == 'resnet18_cifar':
        encoder = ResNet18_CIFAR(num_classes=num_classes)
        projection_head = ProjectionHead(
            input_dim=512, hidden_dim=512, output_dim=projection_dim
        )
        return SupConModel(encoder, projection_head)
    
    elif model_name == 'resnet50_fg':
        encoder = ResNet50_FineGrained(num_classes=num_classes, pretrained=True)
        projection_head = ProjectionHead(
            input_dim=2048, hidden_dim=2048, output_dim=projection_dim
        )
        return SupConModel(encoder, projection_head)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_cross_entropy_model(model_name='resnet18_cifar', num_classes=100):
    """Create model for standard cross-entropy training."""
    if model_name == 'resnet18_cifar':
        return ResNet18_CIFAR(num_classes=num_classes)
    elif model_name == 'resnet50_fg':
        return ResNet50_FineGrained(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == '__main__':
    # Test models
    print("Testing ResNet-18 CIFAR...")
    model = create_model('resnet18_cifar', num_classes=100)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    print("\nTesting ResNet-50 Fine-Grained...")
    model = create_model('resnet50_fg', num_classes=200)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
