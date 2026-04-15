"""Data loading and preprocessing for CIFAR-10, CIFAR-100, and ImageNet-100."""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_cifar_transforms(dataset='cifar10', train=True):
    mean = CIFAR10_MEAN if dataset == 'cifar10' else CIFAR100_MEAN
    std = CIFAR10_STD if dataset == 'cifar10' else CIFAR100_STD

    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class NoisyLabelDataset(Dataset):
    """Wraps a dataset with noisy labels for canary injection."""
    def __init__(self, dataset, noise_rate=0.1, num_classes=10, seed=0):
        self.dataset = dataset
        self.num_classes = num_classes
        rng = np.random.RandomState(seed)
        n = len(dataset)
        self.canary_mask = rng.random(n) < noise_rate
        self.noisy_labels = []
        self.canary_indices = []
        for i in range(n):
            _, label = dataset[i]
            if self.canary_mask[i]:
                new_label = rng.randint(0, num_classes - 1)
                if new_label >= label:
                    new_label += 1
                new_label = new_label % num_classes
                self.noisy_labels.append(new_label)
                self.canary_indices.append(i)
            else:
                self.noisy_labels.append(label)
        self.canary_indices = np.array(self.canary_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.noisy_labels[idx]


def get_cifar_data(dataset='cifar10', data_dir='./data'):
    """Download and return CIFAR train and test datasets."""
    num_classes = 10 if dataset == 'cifar10' else 100
    DatasetClass = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100

    train_transform = get_cifar_transforms(dataset, train=True)
    test_transform = get_cifar_transforms(dataset, train=False)

    # Also need non-augmented train for GDS computation
    train_eval_transform = get_cifar_transforms(dataset, train=False)

    train_dataset = DatasetClass(root=data_dir, train=True, download=True, transform=train_transform)
    train_eval_dataset = DatasetClass(root=data_dir, train=True, download=True, transform=train_eval_transform)
    test_dataset = DatasetClass(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, train_eval_dataset, test_dataset, num_classes


def get_member_nonmember_split(dataset, split_seed=0):
    """Split dataset into member (50%) and non-member (50%) for MIA evaluation.
    Returns indices for class-balanced split."""
    n = len(dataset)
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(n)])

    rng = np.random.RandomState(split_seed)
    member_indices = []
    nonmember_indices = []

    for c in np.unique(labels):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)
        mid = len(class_indices) // 2
        member_indices.extend(class_indices[:mid].tolist())
        nonmember_indices.extend(class_indices[mid:].tolist())

    return np.array(member_indices), np.array(nonmember_indices)


def get_shadow_splits(dataset, num_shadows=8, base_seed=100):
    """Create random 50/50 splits for LiRA shadow models."""
    n = len(dataset)
    splits = []
    for i in range(num_shadows):
        rng = np.random.RandomState(base_seed + i)
        perm = rng.permutation(n)
        mid = n // 2
        splits.append({
            'member': perm[:mid],
            'nonmember': perm[mid:],
        })
    return splits


def make_loader(dataset, indices=None, batch_size=128, shuffle=True, num_workers=4, drop_last=False):
    """Create DataLoader, optionally using a subset of indices."""
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)
