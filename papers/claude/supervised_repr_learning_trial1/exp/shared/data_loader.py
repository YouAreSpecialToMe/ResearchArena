"""Data loading utilities for contrastive learning experiments."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class TwoCropTransform:
    """Create two crops of the same image."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


# CIFAR-100 superclass mapping (fine class -> superclass index)
CIFAR100_SUPERCLASS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 17, 5, 0,
    2, 4, 6, 17, 10, 16, 12, 1, 9, 19,
    8, 8, 15, 13, 16, 15, 7, 12, 2, 4,
    6, 0, 10, 2, 14, 1, 10, 16, 19, 3,
    8, 4, 12, 5, 18, 19, 2, 17, 8, 2,
    1, 9, 19, 13, 17, 15, 5, 5, 1, 13,
    12, 9, 0, 19, 4, 13, 16, 18, 6, 9,
]


def get_cifar100_transforms(two_crop=True):
    """Get training and test transforms for CIFAR-100."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if two_crop:
        train_transform = TwoCropTransform(train_transform)

    return train_transform, test_transform


def get_cifar10_transforms(two_crop=True):
    """Get training and test transforms for CIFAR-10."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if two_crop:
        train_transform = TwoCropTransform(train_transform)

    return train_transform, test_transform


def get_dataloaders(dataset='cifar100', batch_size=512, num_workers=4,
                    two_crop=True, data_root='./data'):
    """Get train and test dataloaders."""
    if dataset == 'cifar100':
        train_transform, test_transform = get_cifar100_transforms(two_crop)
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=test_transform)
        num_classes = 100
    elif dataset == 'cifar10':
        train_transform, test_transform = get_cifar10_transforms(two_crop)
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)

    return train_loader, test_loader, num_classes
