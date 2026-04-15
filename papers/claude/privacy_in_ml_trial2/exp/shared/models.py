import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18_cifar(num_classes=10):
    """ResNet-18 adapted for 32x32 CIFAR images."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    # Replace first conv: 3x3 stride 1 padding 1 (instead of 7x7 stride 2)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool
    model.maxpool = nn.Identity()
    return model


class PurchaseMLP(nn.Module):
    def __init__(self, n_features=600, n_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_model(dataset, device='cuda'):
    if dataset == 'cifar10':
        model = get_resnet18_cifar(num_classes=10)
    elif dataset == 'cifar100':
        model = get_resnet18_cifar(num_classes=100)
    elif dataset == 'purchase100':
        model = PurchaseMLP(n_features=600, n_classes=100)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return model.to(device)
