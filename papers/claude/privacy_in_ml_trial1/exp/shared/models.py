"""Model definitions: ResNet-18 and VGG-16 for CIFAR and ImageNet-100."""
import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes=10, dataset='cifar10'):
    """ResNet-18 adapted for CIFAR (32x32) or ImageNet-100 (224x224)."""
    if dataset in ('cifar10', 'cifar100'):
        # Modified ResNet-18 for 32x32 input
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        # Standard ResNet-18 for ImageNet-100
        model = models.resnet18(weights=None, num_classes=num_classes)
    return model


def get_vgg16_bn(num_classes=10):
    """VGG-16 with batch norm for CIFAR (32x32)."""
    model = models.vgg16_bn(weights=None, num_classes=num_classes)
    # Replace adaptive pool with (1,1) to reduce feature map to 512
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


def get_model(model_name, num_classes, dataset='cifar10'):
    if model_name == 'resnet18':
        return get_resnet18(num_classes, dataset)
    elif model_name == 'vgg16':
        return get_vgg16_bn(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count total and prunable parameters."""
    total = sum(p.numel() for p in model.parameters())
    prunable = sum(p.numel() for n, p in model.named_parameters()
                   if 'bn' not in n and 'bias' not in n and p.dim() >= 2)
    return total, prunable
