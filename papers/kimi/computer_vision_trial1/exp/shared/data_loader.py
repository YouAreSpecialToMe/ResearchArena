"""
Data loading utilities for DU-VPT experiments.
Supports ImageNet-C, ImageNet-R, and ImageNet-Sketch datasets.
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
from PIL import Image


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(img_size=224, train=False):
    """Get standard ImageNet transform."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class ImageNetC(DatasetFolder):
    """ImageNet-C dataset loader."""
    
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    def __init__(self, root, corruption=None, severity=5, transform=None):
        """
        Args:
            root: Root directory of ImageNet-C
            corruption: Specific corruption type or None for all
            severity: Severity level 1-5
            transform: Transform to apply
        """
        self.root = root
        self.corruption = corruption
        self.severity = severity
        
        if transform is None:
            transform = get_transform(train=False)
        self.transform = transform
        
        # Build samples list
        self.samples = []
        self.targets = []
        
        if corruption is not None:
            # Single corruption
            corruption_dir = os.path.join(root, corruption, str(severity))
            if os.path.exists(corruption_dir):
                self._load_corruption(corruption_dir)
        else:
            # All corruptions
            for corr in self.CORRUPTIONS:
                corruption_dir = os.path.join(root, corr, str(severity))
                if os.path.exists(corruption_dir):
                    self._load_corruption(corruption_dir)
        
        self.classes = list(range(1000))  # ImageNet has 1000 classes
        self.class_to_idx = {i: i for i in range(1000)}
    
    def _load_corruption(self, corruption_dir):
        """Load images from a corruption directory."""
        for class_idx in range(1000):
            class_dir = os.path.join(corruption_dir, str(class_idx))
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.JPEG')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append(img_path)
                        self.targets.append(class_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


def load_imagenet_c(root, corruption=None, severity=5, batch_size=64, num_workers=4, seed=42):
    """Load ImageNet-C dataset."""
    dataset = ImageNetC(root, corruption=corruption, severity=severity)
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator
    )
    
    return loader, dataset


def load_imagenet_r(root, batch_size=64, num_workers=4, seed=42):
    """Load ImageNet-R dataset."""
    transform = get_transform(train=False)
    dataset = ImageFolder(root, transform=transform)
    
    generator = torch.Generator().manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator
    )
    
    return loader, dataset


def load_imagenet_sketch(root, batch_size=64, num_workers=4, seed=42):
    """Load ImageNet-Sketch dataset."""
    transform = get_transform(train=False)
    dataset = ImageFolder(root, transform=transform)
    
    generator = torch.Generator().manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator
    )
    
    return loader, dataset


def create_calibration_set(imagenet_val_dir, n_samples=1000, seed=42):
    """Create a calibration set from ImageNet validation data."""
    transform = get_transform(train=False)
    full_dataset = ImageFolder(imagenet_val_dir, transform=transform)
    
    # Random subset
    np.random.seed(seed)
    indices = np.random.choice(len(full_dataset), n_samples, replace=False)
    
    calib_dataset = Subset(full_dataset, indices)
    
    loader = DataLoader(
        calib_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return loader, calib_dataset


def get_dataset_info(dataset_name, root_dir):
    """Get information about a dataset."""
    info = {
        'name': dataset_name,
        'root': root_dir,
        'n_samples': 0,
        'n_classes': 0
    }
    
    if dataset_name == 'imagenet-c':
        info['n_classes'] = 1000
        info['corruptions'] = ImageNetC.CORRUPTIONS
        info['severities'] = [1, 2, 3, 4, 5]
    elif dataset_name == 'imagenet-r':
        info['n_classes'] = 200
    elif dataset_name == 'imagenet-sketch':
        info['n_classes'] = 1000
    
    return info
