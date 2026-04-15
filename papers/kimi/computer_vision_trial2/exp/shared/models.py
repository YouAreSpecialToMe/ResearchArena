"""
Model definitions and utilities for TTA experiments.
Supports ResNet for CIFAR and ResNet-50 for ImageNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============== ResNet for CIFAR ==============

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Use 'downsample' instead of 'shortcut' to match pytorch-cifar-models format
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # Use 'fc' instead of 'linear' to match pytorch-cifar-models format
        self.fc = nn.Linear(64, num_classes)

        self.feature_dim = 64

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
        out = F.avg_pool2d(out, out.size()[3])
        features = out.view(out.size(0), -1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

    def get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        features = out.view(out.size(0), -1)
        return features


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


# ============== WideResNet for CIFAR (legacy support) ==============

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes,
                                                                  kernel_size=1, stride=stride,
                                                                  padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, drop_rate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.feature_dim = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        features = out.view(-1, self.nChannels)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

    def get_features(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        features = out.view(-1, self.nChannels)
        return features


def wideresnet28_10(num_classes=10):
    """WideResNet-28-10 for CIFAR-10/100"""
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=10)


# ============== ResNet for ImageNet ==============

def resnet50(num_classes=1000, pretrained=True):
    """ResNet-50 for ImageNet"""
    model = models.resnet50(pretrained=pretrained)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Modify forward to return features
    original_forward = model.forward
    
    def new_forward(x, return_features=False):
        # See https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = model.fc(features)
        
        if return_features:
            return logits, features
        return logits
    
    model.forward = new_forward
    model.get_features = lambda x: model.forward(x, return_features=True)[1]
    model.feature_dim = 2048
    return model


# ============== Model Loading ==============

def load_model(model_name, num_classes=10, device='cuda'):
    """Load a model by name"""
    if model_name == 'resnet20':
        model = resnet20(num_classes=num_classes)
    elif model_name == 'resnet32':
        model = resnet32(num_classes=num_classes)
    elif model_name == 'resnet44':
        model = resnet44(num_classes=num_classes)
    elif model_name == 'wideresnet28_10':
        model = wideresnet28_10(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def load_pretrained_cifar_model(model_name='resnet32', dataset='cifar10', device='cuda'):
    """
    Load a pre-trained model for CIFAR.
    Models are downloaded from pytorch-cifar-models repository.
    """
    num_classes = 10 if dataset == 'cifar10' else 100
    
    # Load model architecture
    if model_name == 'resnet20':
        model = resnet20(num_classes=num_classes)
    elif model_name == 'resnet32':
        model = resnet32(num_classes=num_classes)
    elif model_name == 'resnet44':
        model = resnet44(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Try to load pretrained weights
    checkpoint_path = f'models/{dataset}_{model_name}.pt'
    if checkpoint_path.startswith('models/'):
        checkpoint_path = checkpoint_path
    
    import os
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {checkpoint_path}: {e}")
            print("Using randomly initialized model (results will not be meaningful)")
    else:
        print(f"Warning: No pretrained model found at {checkpoint_path}")
        print("Using randomly initialized model (results will not be meaningful)")
    
    return model.to(device)


# ============== Meta-APN Model ==============

class MetaAPN(nn.Module):
    """
    Meta-Augmentation Policy Network.
    Predicts augmentation parameters based on prototype distances.
    """
    def __init__(self, feature_dim, num_classes, num_operations=8, hidden_dim=64):
        super(MetaAPN, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_operations = num_operations
        
        # Input: feature vector concatenated with prototype distances
        input_dim = feature_dim + num_classes
        
        # 2-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_operations + 2)  # operations + severity + num_aug
        
        self.relu = nn.ReLU()
        
    def forward(self, features, prototype_distances):
        """
        Args:
            features: [B, feature_dim]
            prototype_distances: [B, num_classes]
        Returns:
            policy_logits: [B, num_operations]
            severity_scale: [B, 1]
            num_aug_logits: [B, 1] (for selecting among {4, 8, 16})
        """
        x = torch.cat([features, prototype_distances], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        policy_logits = x[:, :self.num_operations]
        severity_scale = torch.sigmoid(x[:, self.num_operations:self.num_operations+1]) * 1.5 + 0.5  # [0.5, 2.0]
        num_aug_logits = x[:, self.num_operations+1:self.num_operations+2]
        
        return policy_logits, severity_scale, num_aug_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============== Utility Functions ==============

def compute_prototypes(model, dataloader, device='cuda', num_classes=10):
    """
    Compute class prototypes from source training data.
    Returns prototypes of shape [num_classes, feature_dim]
    """
    model.eval()
    features_by_class = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.get_features(images)
            
            for i, label in enumerate(labels):
                features_by_class[label.item()].append(features[i].cpu())
    
    # Compute mean feature for each class
    prototypes = []
    for c in range(num_classes):
        if len(features_by_class[c]) > 0:
            class_features = torch.stack(features_by_class[c])
            prototypes.append(class_features.mean(dim=0))
        else:
            prototypes.append(torch.zeros(model.feature_dim))
    
    prototypes = torch.stack(prototypes)
    return prototypes


def collect_params(model, adapt_bias=False):
    """Collect parameters for adaptation (BN stats and affine params)"""
    params = []
    names = []
    
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for np, p in m.named_parameters():
                if p.requires_grad:
                    names.append(f"{nm}.{np}")
                    params.append(p)
    
    return params, names


def get_prototype_distances(features, prototypes, metric='cosine'):
    """
    Compute distances between features and class prototypes.
    
    Args:
        features: [B, feature_dim]
        prototypes: [num_classes, feature_dim]
        metric: 'cosine' or 'euclidean'
    
    Returns:
        distances: [B, num_classes]
    """
    if metric == 'cosine':
        # Cosine similarity -> distance = 1 - similarity
        features_norm = F.normalize(features, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        similarity = torch.mm(features_norm, prototypes_norm.t())
        distances = 1 - similarity
    elif metric == 'euclidean':
        # Euclidean distance
        distances = torch.cdist(features, prototypes, p=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def enable_adaptation(model, adapt_bn=True):
    """Enable/disable gradient computation for adaptation parameters"""
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    if adapt_bn:
        # Enable only BN parameters
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True
                module.train()  # Set to train mode for adaptation
    
    return model


def copy_model_state(model):
    """Create a copy of model state dict for reset"""
    return {name: param.clone() for name, param in model.named_parameters()}


def reset_model_state(model, original_state):
    """Reset model to original state"""
    for name, param in model.named_parameters():
        if name in original_state:
            param.data.copy_(original_state[name])
