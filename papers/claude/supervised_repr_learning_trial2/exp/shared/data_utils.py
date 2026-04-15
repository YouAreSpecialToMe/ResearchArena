"""Data loading utilities for CIFAR-10, CIFAR-100, and TinyImageNet."""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_cifar10(batch_size=256, num_workers=4, data_dir='./data'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 10


def get_cifar100(batch_size=256, num_workers=4, data_dir='./data'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 100


def get_tinyimagenet(batch_size=256, num_workers=4, data_dir='./data'):
    """Load TinyImageNet dataset."""
    tiny_dir = os.path.join(data_dir, 'tiny-imagenet-200')

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dir = os.path.join(tiny_dir, 'train')
    val_dir = os.path.join(tiny_dir, 'val')

    # Check if val has been reorganized into class folders
    val_images_dir = os.path.join(val_dir, 'images')
    if os.path.exists(val_images_dir):
        # Need to reorganize val set into class folders
        _reorganize_tinyimagenet_val(tiny_dir)

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(val_dir, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 200


def _reorganize_tinyimagenet_val(tiny_dir):
    """Reorganize TinyImageNet val set into class folders."""
    val_dir = os.path.join(tiny_dir, 'val')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')

    if not os.path.exists(val_annotations):
        return

    with open(val_annotations, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            fname, class_name = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(val_dir, 'images', fname)
            dst = os.path.join(class_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)

    # Remove empty images dir
    images_dir = os.path.join(val_dir, 'images')
    if os.path.exists(images_dir):
        try:
            os.rmdir(images_dir)
        except OSError:
            pass


def get_dataset(name, batch_size=256, num_workers=4, data_dir='./data'):
    """Get dataset by name."""
    if name == 'cifar10':
        return get_cifar10(batch_size, num_workers, data_dir)
    elif name == 'cifar100':
        return get_cifar100(batch_size, num_workers, data_dir)
    elif name == 'tinyimagenet':
        return get_tinyimagenet(batch_size, num_workers, data_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_val_split(dataset_name, val_fraction=0.1, seed=42, batch_size=256, num_workers=4, data_dir='./data'):
    """Get a train/val split for temperature scaling calibration."""
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        full_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        full_train = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'tinyimagenet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        tiny_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
        full_train = torchvision.datasets.ImageFolder(tiny_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n = len(full_train)
    indices = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    val_size = int(n * val_fraction)
    val_indices = indices[:val_size]

    val_set = Subset(full_train, val_indices)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return val_loader


def get_train_eval_loader(dataset_name, batch_size=256, num_workers=4, data_dir='./data'):
    """Get training set with test-time transforms (no augmentation) for NC metric computation."""
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'tinyimagenet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        tiny_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
        dataset = torchvision.datasets.ImageFolder(tiny_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader
