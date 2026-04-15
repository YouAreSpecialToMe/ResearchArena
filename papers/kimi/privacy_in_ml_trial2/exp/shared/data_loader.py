"""
Data loading utilities for PRISM experiments.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar_transforms(train=True):
    """Get CIFAR transforms."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def get_cifar10_loaders(data_dir='./data', batch_size=128, num_workers=4, seed=42):
    """Get CIFAR-10 train/val/test loaders."""
    train_transform = get_cifar_transforms(train=True)
    test_transform = get_cifar_transforms(train=False)
    
    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    
    # Split train into train/val (80/20 split)
    n_train = int(0.9 * len(full_train))
    n_val = len(full_train) - n_train
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_train


def get_cifar100_loaders(data_dir='./data', batch_size=128, num_workers=4, seed=42):
    """Get CIFAR-100 train/val/test loaders."""
    train_transform = get_cifar_transforms(train=True)
    test_transform = get_cifar_transforms(train=False)
    
    full_train = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    
    # Split train into train/val (90/10 split)
    n_train = int(0.9 * len(full_train))
    n_val = len(full_train) - n_train
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_train


def get_shadow_datasets(full_train, sizes=[0.05, 0.1, 0.2], num_shadows=4, seed=42):
    """
    Create shadow datasets of different sizes for MIA evaluation.
    Returns dict: {size_percent: [datasets]}
    """
    shadow_datasets = {}
    n_total = len(full_train)
    
    for size in sizes:
        n_samples = int(size * n_total)
        shadow_datasets[size] = []
        
        for i in range(num_shadows):
            # Use different seeds for different shadows
            g = torch.Generator().manual_seed(seed + i * 100)
            indices = torch.randperm(n_total, generator=g)[:n_samples].tolist()
            shadow_subset = Subset(full_train, indices)
            shadow_datasets[size].append(shadow_subset)
    
    return shadow_datasets


class PurchaseDataset(Dataset):
    """Purchase-100 dataset."""
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def download_purchase_dataset(data_dir='./data'):
    """Download and prepare Purchase-100 dataset."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'dataset_purchase')
    
    if not os.path.exists(filepath):
        print("Downloading Purchase-100 dataset...")
        import urllib.request
        url = "https://github.com/privacytrustlab/datasets/raw/master/purchase/dataset_purchase"
        urllib.request.urlretrieve(url, filepath)
    
    # Load data
    with open(filepath, 'r') as f:
        data = []
        labels = []
        for line in f:
            parts = line.strip().split(',')
            labels.append(int(parts[0]) - 1)  # 0-indexed
            features = [int(x) for x in parts[1:]]
            data.append(features)
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # Normalize features
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    return data, labels


def get_purchase100_loaders(data_dir='./data', batch_size=256, num_workers=4, seed=42):
    """Get Purchase-100 train/val/test loaders."""
    data, labels = download_purchase_dataset(data_dir)
    
    # Split: 60% train, 20% val, 20% test
    n_total = len(data)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    
    np.random.seed(seed)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_dataset = PurchaseDataset(data[train_idx], labels[train_idx])
    val_dataset = PurchaseDataset(data[val_idx], labels[val_idx])
    test_dataset = PurchaseDataset(data[test_idx], labels[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset


def get_data_loaders(dataset_name, data_dir='./data', batch_size=128, num_workers=4, seed=42):
    """Factory function to get data loaders by dataset name."""
    if dataset_name == 'cifar10':
        return get_cifar10_loaders(data_dir, batch_size, num_workers, seed)
    elif dataset_name == 'cifar100':
        return get_cifar100_loaders(data_dir, batch_size, num_workers, seed)
    elif dataset_name == 'purchase100':
        return get_purchase100_loaders(data_dir, batch_size, num_workers, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
