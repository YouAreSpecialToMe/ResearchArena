"""
Data loading utilities for CIFAR-10/100 with label noise support.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class TwoCropTransform:
    """Create two crops of the same image for contrastive learning."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def add_symmetric_noise(labels, num_classes, noise_ratio, seed=42):
    """Add symmetric label noise by randomly flipping labels."""
    np.random.seed(seed)
    labels = np.array(labels)
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_ratio)
    
    # Select indices to corrupt
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    
    # For each selected sample, flip to a random different class
    new_labels = labels.copy()
    for idx in noisy_indices:
        original_label = labels[idx]
        # Choose a different class
        possible_labels = [c for c in range(num_classes) if c != original_label]
        new_labels[idx] = np.random.choice(possible_labels)
    
    return new_labels, noisy_indices


def add_asymmetric_noise_cifar10(labels, seed=42):
    """Add asymmetric noise for CIFAR-10 (class-dependent flips)."""
    np.random.seed(seed)
    labels = np.array(labels)
    # CIFAR-10: TRUCK→AUTOMOBILE, BIRD→AIRPLANE, DEER→HORSE, CAT→DOG
    transition_map = {9: 1, 2: 0, 4: 7, 3: 5}  # From: To
    
    noisy_indices = []
    new_labels = labels.copy()
    for from_class, to_class in transition_map.items():
        class_indices = np.where(labels == from_class)[0]
        # Flip 40% of samples in these classes
        n_flip = int(len(class_indices) * 0.4)
        flip_indices = np.random.choice(class_indices, n_flip, replace=False)
        new_labels[flip_indices] = to_class
        noisy_indices.extend(flip_indices)
    
    return new_labels, np.array(noisy_indices)


def add_asymmetric_noise_cifar100(labels, seed=42):
    """Add asymmetric noise for CIFAR-100 (within superclass flips)."""
    np.random.seed(seed)
    labels = np.array(labels)
    
    # CIFAR-100 has 20 superclasses, each with 5 subclasses
    # Map each class to its superclass
    superclass_map = {}
    for i in range(20):
        for j in range(5):
            superclass_map[i * 5 + j] = i
    
    new_labels = labels.copy()
    noisy_indices = []
    
    for i in range(len(labels)):
        superclass = superclass_map[labels[i]]
        # Get all classes in the same superclass
        same_superclass = [c for c in range(100) 
                          if superclass_map[c] == superclass and c != labels[i]]
        if len(same_superclass) > 0 and np.random.rand() < 0.4:
            new_labels[i] = np.random.choice(same_superclass)
            noisy_indices.append(i)
    
    return new_labels, np.array(noisy_indices)


class NoisyCIFAR(Dataset):
    """CIFAR dataset with noisy labels."""
    def __init__(self, root, train=True, transform=None, download=True,
                 dataset='cifar10', noise_type='clean', noise_ratio=0.0, seed=42):
        self.dataset_name = dataset
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.seed = seed
        
        # Load base dataset
        if dataset == 'cifar10':
            self.base_dataset = datasets.CIFAR10(root, train=train, download=download)
            num_classes = 10
        else:  # cifar100
            self.base_dataset = datasets.CIFAR100(root, train=train, download=download)
            num_classes = 100
        
        self.transform = transform
        self.train = train
        
        # Store original labels
        self.clean_labels = np.array(self.base_dataset.targets)
        self.targets = self.clean_labels.copy()
        self.noisy_indices = np.array([])
        
        # Add noise if training set and noise requested
        if train and noise_type != 'clean':
            if noise_type == 'symmetric':
                self.targets, self.noisy_indices = add_symmetric_noise(
                    self.clean_labels, num_classes, noise_ratio, seed
                )
            elif noise_type == 'asymmetric':
                if dataset == 'cifar10':
                    self.targets, self.noisy_indices = add_asymmetric_noise_cifar10(
                        self.clean_labels, seed
                    )
                else:
                    self.targets, self.noisy_indices = add_asymmetric_noise_cifar100(
                        self.clean_labels, seed
                    )
    
    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        target = self.targets[index]
        
        if self.transform is not None:
            if isinstance(self.transform, TwoCropTransform):
                img = self.transform(img)
            else:
                img = self.transform(img)
        
        return img, target, index
    
    def __len__(self):
        return len(self.base_dataset)


def get_cifar_transforms(dataset='cifar10', is_train=True, contrastive=False):
    """Get data transforms for CIFAR datasets."""
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    if is_train:
        if contrastive:
            # Strong augmentation for contrastive learning
            transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            return TwoCropTransform(transform)
        else:
            # Standard augmentation
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    else:
        # Test transform
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def get_dataloader(dataset='cifar10', train=True, batch_size=256, num_workers=4,
                   noise_type='clean', noise_ratio=0.0, seed=42, contrastive=False):
    """Get a DataLoader for CIFAR dataset."""
    transform = get_cifar_transforms(dataset, train, contrastive)
    
    ds = NoisyCIFAR(
        root='./data',
        train=train,
        transform=transform,
        download=True,
        dataset=dataset,
        noise_type=noise_type,
        noise_ratio=noise_ratio,
        seed=seed
    )
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train  # Drop last incomplete batch during training
    )
    
    return loader, ds
