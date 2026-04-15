"""
Data loading utilities for CIFAR-100 with coarse/fine labels.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


# CIFAR-100 coarse to fine class mapping
CIFAR100_COARSE_FINE = {
    'aquatic_mammals': [4, 30, 55, 72, 95],
    'fish': [1, 32, 67, 73, 91],
    'flowers': [54, 62, 70, 82, 92],
    'food_containers': [9, 10, 16, 28, 61],
    'fruit_and_vegetables': [0, 51, 53, 57, 83],
    'household_electrical_devices': [22, 39, 40, 86, 87],
    'household_furniture': [5, 20, 25, 84, 94],
    'insects': [6, 7, 14, 18, 24],
    'large_carnivores': [3, 42, 43, 88, 97],
    'large_man-made_outdoor_things': [12, 17, 37, 68, 76],
    'large_natural_outdoor_scenes': [23, 33, 49, 60, 71],
    'large_omnivores_and_herbivores': [15, 19, 21, 31, 38],
    'medium_mammals': [34, 63, 64, 66, 75],
    'non-insect_invertebrates': [26, 45, 77, 79, 99],
    'people': [2, 11, 35, 46, 98],
    'reptiles': [27, 29, 44, 78, 93],
    'small_mammals': [36, 50, 65, 74, 80],
    'trees': [47, 52, 56, 59, 96],
    'vehicles_1': [8, 13, 48, 58, 90],
    'vehicles_2': [41, 69, 81, 85, 89],
}

# Create fine to coarse mapping
FINE_TO_COARSE = {}
for coarse_idx, (coarse_name, fine_classes) in enumerate(CIFAR100_COARSE_FINE.items()):
    for fine_idx in fine_classes:
        FINE_TO_COARSE[fine_idx] = coarse_idx


def get_cifar100_transforms(train=True):
    """Get CIFAR-100 data augmentation transforms."""
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def get_contrastive_transforms():
    """Get two-view augmentation for contrastive learning."""
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    return transform


class TwoViewDataset(Dataset):
    """Wrapper to create two augmented views of each sample."""
    
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, label


class CoarseLabelDataset(Dataset):
    """Wrapper to use coarse labels instead of fine labels."""
    
    def __init__(self, dataset, fine_to_coarse_map):
        self.dataset = dataset
        self.fine_to_coarse = fine_to_coarse_map
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, fine_label = self.dataset[idx]
        coarse_label = self.fine_to_coarse[fine_label]
        return img, coarse_label, fine_label


def get_cifar100_loaders(batch_size=256, num_workers=4, use_coarse_labels=False):
    """Get CIFAR-100 train and test dataloaders.
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_coarse_labels: If True, return coarse (20-class) labels for training
                          but keep fine labels for evaluation
    """
    # Standard transforms
    train_transform = get_cifar100_transforms(train=True)
    test_transform = get_cifar100_transforms(train=False)
    
    # Download datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    if use_coarse_labels:
        # Wrap to use coarse labels for training
        train_dataset = CoarseLabelDataset(train_dataset, FINE_TO_COARSE)
        # Test dataset should also provide both labels
        test_dataset = CoarseLabelDataset(test_dataset, FINE_TO_COARSE)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    num_classes = 20 if use_coarse_labels else 100
    return train_loader, test_loader, num_classes


def get_contrastive_cifar100_loaders(batch_size=256, num_workers=4, use_coarse_labels=False):
    """Get CIFAR-100 dataloaders for contrastive learning (two views).
    
    Args:
        batch_size: Batch size (total batch will be 2*batch_size with two views)
        num_workers: Number of data loading workers
        use_coarse_labels: If True, use coarse (20-class) labels
    """
    contrastive_transform = get_contrastive_transforms()
    
    # Base dataset (without transforms)
    base_train = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=None
    )
    base_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=None
    )
    
    if use_coarse_labels:
        base_train = CoarseLabelDataset(base_train, FINE_TO_COARSE)
        base_test = CoarseLabelDataset(base_test, FINE_TO_COARSE)
    
    # Wrap with two-view augmentation
    train_dataset = TwoViewDataset(base_train, contrastive_transform)
    
    # For test, use single view
    test_transform = get_cifar100_transforms(train=False)
    base_test_with_transform = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    if use_coarse_labels:
        base_test_with_transform = CoarseLabelDataset(base_test_with_transform, FINE_TO_COARSE)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        base_test_with_transform, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    num_classes = 20 if use_coarse_labels else 100
    return train_loader, test_loader, num_classes


def get_fine_labels_from_coarse_batch(coarse_labels, fine_labels):
    """Utility to get fine labels from a coarse label batch."""
    return fine_labels
