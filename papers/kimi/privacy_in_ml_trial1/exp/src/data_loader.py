"""
Data loaders for CIFAR-10, CIFAR-100, and Purchase-100 datasets.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader, TensorDataset


def get_cifar10_loaders(batch_size=256, data_dir='./data'):
    """Get CIFAR-10 train and test dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader


def get_cifar100_loaders(batch_size=256, data_dir='./data'):
    """Get CIFAR-100 train and test dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader


def download_purchase100(data_dir='./data'):
    """Download and prepare Purchase-100 dataset."""
    import urllib.request
    import tarfile
    
    url = "https://github.com/privacytrustlab/datasets/raw/master/dataset_purchase.tgz"
    tar_path = os.path.join(data_dir, "dataset_purchase.tgz")
    
    if not os.path.exists(os.path.join(data_dir, "dataset_purchase")):
        print("Downloading Purchase-100 dataset...")
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, tar_path)
        
        print("Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("Done.")
    
    return os.path.join(data_dir, "dataset_purchase")


def get_purchase100_loaders(batch_size=512, data_dir='./data'):
    """Get Purchase-100 train and test dataloaders."""
    data_path = download_purchase100(data_dir)
    
    # Load data
    data_file = os.path.join(data_path, 'purchase')
    with open(data_file, 'r') as f:
        data = f.readlines()
    
    # Parse data
    X = []
    y = []
    for line in data:
        parts = line.strip().split(',')
        features = [float(x) for x in parts[:-1]]
        label = int(parts[-1]) - 1  # Convert to 0-indexed
        X.append(features)
        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Normalize features to [0, 1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    # Split: 150K train, 47K test
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_indices = indices[:150000]
    test_indices = indices[150000:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader


def get_data_loaders(dataset_name, batch_size=256, data_dir='./data'):
    """Get data loaders for specified dataset."""
    if dataset_name == 'cifar10':
        return get_cifar10_loaders(batch_size, data_dir)
    elif dataset_name == 'cifar100':
        return get_cifar100_loaders(batch_size, data_dir)
    elif dataset_name == 'purchase100':
        return get_purchase100_loaders(batch_size, data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
