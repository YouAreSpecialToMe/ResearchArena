"""
Data loading and federated dataset splitting for FCL.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict


class ContrastiveTransform:
    """Transform for contrastive learning with two augmented views."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def get_cifar_transforms(train=True, contrastive=False):
    """Get CIFAR transforms."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    if train:
        if contrastive:
            # SimCLR augmentations
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
            return ContrastiveTransform(transform)
        else:
            # Standard training augmentations
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            return transform
    else:
        # Test transform
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def create_dirichlet_split(dataset, num_clients=10, alpha=0.5, seed=42):
    """
    Create non-IID data split using Dirichlet distribution.
    Returns a dictionary mapping client_id -> list of sample indices.
    """
    np.random.seed(seed)
    
    # Get labels
    if isinstance(dataset, datasets.CIFAR10) or isinstance(dataset, datasets.CIFAR100):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}
    
    # For each class, distribute samples to clients using Dirichlet
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        # Sample from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        
        # Assign to clients
        for i, indices in enumerate(np.split(class_indices, splits)):
            client_indices[i].extend(indices.tolist())
    
    return client_indices


def create_federated_datasets(dataset_name='cifar10', num_clients=10, alpha=0.5, data_dir='./data', seed=42):
    """
    Create federated datasets with non-IID splits.
    Returns train datasets per client and global test dataset.
    """
    # Load base dataset
    if dataset_name == 'cifar10':
        base_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
        base_test = datasets.CIFAR10(root=data_dir, train=False, download=True)
    elif dataset_name == 'cifar100':
        base_train = datasets.CIFAR100(root=data_dir, train=True, download=True)
        base_test = datasets.CIFAR100(root=data_dir, train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create non-IID split
    split_file = os.path.join(data_dir, f'{dataset_name}_dirichlet_alpha{alpha}_clients{num_clients}_seed{seed}.json')
    
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            client_indices = json.load(f)
            client_indices = {int(k): v for k, v in client_indices.items()}
    else:
        client_indices = create_dirichlet_split(base_train, num_clients, alpha, seed)
        with open(split_file, 'w') as f:
            json.dump(client_indices, f)
    
    # Create contrastive transforms for training
    train_transform = get_cifar_transforms(train=True, contrastive=True)
    test_transform = get_cifar_transforms(train=False, contrastive=False)
    
    # Create client datasets
    client_datasets = []
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        
        # Create subset with contrastive transform
        subset = Subset(base_train, indices)
        subset.dataset.transform = train_transform
        client_datasets.append(subset)
    
    # Create test dataset
    base_test.transform = test_transform
    
    return client_datasets, base_test, client_indices


def create_membership_split(client_indices, holdout_ratio=0.3, seed=42):
    """
    Split each client's data into member (training) and non-member (holdout) sets.
    Used for membership inference attacks.
    """
    np.random.seed(seed)
    
    member_indices = {}
    nonmember_indices = {}
    
    for client_id, indices in client_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        split_point = int(len(indices) * (1 - holdout_ratio))
        member_indices[client_id] = indices[:split_point].tolist()
        nonmember_indices[client_id] = indices[split_point:].tolist()
    
    return member_indices, nonmember_indices


class FederatedDataset(Dataset):
    """Wrapper for a client's dataset in federated learning."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_client_dataloader(client_dataset, batch_size=256, shuffle=True, num_workers=2):
    """Get dataloader for a client."""
    return DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def get_linear_eval_dataloaders(dataset_name='cifar10', data_dir='./data', batch_size=256, num_workers=2):
    """Get dataloaders for linear evaluation."""
    train_transform = get_cifar_transforms(train=True, contrastive=False)
    test_transform = get_cifar_transforms(train=False, contrastive=False)
    
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=test_transform)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader
