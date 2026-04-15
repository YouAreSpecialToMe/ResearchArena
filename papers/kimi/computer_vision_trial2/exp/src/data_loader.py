"""
Data loaders for TTA experiments with corruption benchmarks.
Supports CIFAR-10-C, CIFAR-100-C, and ImageNet-C.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List
import pickle


# Corruption types for CIFAR-C
CIFAR_C_CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
    'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
    'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

# Subset for faster experimentation
CORRUPTIONS_SUBSET = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'jpeg_compression']

SEVERITY_LEVELS = [1, 2, 3, 4, 5]


class CIFAR_C_Dataset(Dataset):
    """CIFAR-10/100-C dataset loader."""
    
    def __init__(self, root: str, dataset: str = 'cifar10', corruption: str = 'gaussian_noise',
                 severity: int = 5, transform=None, download: bool = False):
        """
        Args:
            root: Root directory containing CIFAR-C data
            dataset: 'cifar10' or 'cifar100'
            corruption: Corruption type name
            severity: Severity level 1-5
            transform: Torchvision transforms
            download: Whether to download if not found
        """
        self.root = root
        self.dataset = dataset
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        # Load data
        self.data, self.labels = self._load_data()
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load corrupted images and labels."""
        if self.dataset == 'cifar10':
            data_dir = os.path.join(self.root, 'CIFAR-10-C')
            num_classes = 10
        else:
            data_dir = os.path.join(self.root, 'CIFAR-100-C')
            num_classes = 100
            
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset not found at {data_dir}. Please download from https://zenodo.org/record/3555552")
        
        # Load labels (same for all corruptions)
        labels_path = os.path.join(data_dir, 'labels.npy')
        labels = np.load(labels_path)
        
        # Load corrupted images
        corruption_file = os.path.join(data_dir, f'{self.corruption}.npy')
        if not os.path.exists(corruption_file):
            raise FileNotFoundError(f"Corruption file not found: {corruption_file}")
        
        all_images = np.load(corruption_file)
        
        # Extract images for specific severity level (each severity has 10000 images)
        start_idx = (self.severity - 1) * 10000
        end_idx = self.severity * 10000
        images = all_images[start_idx:end_idx]
        labels = labels[start_idx:end_idx]
        
        return images, labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image (CIFAR-C is stored as NHWC uint8)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


class ImageNet_C_Dataset(Dataset):
    """ImageNet-C dataset loader."""
    
    def __init__(self, root: str, corruption: str = 'gaussian_noise', severity: int = 5,
                 transform=None):
        """
        Args:
            root: Root directory containing ImageNet-C data
            corruption: Corruption type name
            severity: Severity level 1-5
            transform: Torchvision transforms
        """
        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        # Build file list
        self.samples = self._build_file_list()
        
    def _build_file_list(self) -> List[Tuple[str, int]]:
        """Build list of (image_path, label) tuples."""
        corruption_dir = os.path.join(self.root, self.corruption, str(self.severity))
        
        if not os.path.exists(corruption_dir):
            raise FileNotFoundError(f"Dataset not found at {corruption_dir}")
        
        samples = []
        # ImageNet-C is organized by class folders
        for class_idx in range(1000):  # ImageNet has 1000 classes
            class_dir = os.path.join(corruption_dir, str(class_idx))
            if os.path.exists(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, fname)
                        samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


def get_cifar_c_loader(root: str, dataset: str = 'cifar10', corruption: str = 'gaussian_noise',
                       severity: int = 5, batch_size: int = 64, num_workers: int = 4,
                       shuffle: bool = False) -> DataLoader:
    """Create CIFAR-C data loader."""
    
    # Standard CIFAR normalization
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = CIFAR_C_Dataset(root, dataset, corruption, severity, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                        num_workers=num_workers, pin_memory=True)
    
    return loader


def get_imagenet_c_loader(root: str, corruption: str = 'gaussian_noise', severity: int = 5,
                          batch_size: int = 64, num_workers: int = 4,
                          shuffle: bool = False) -> DataLoader:
    """Create ImageNet-C data loader."""
    
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = ImageNet_C_Dataset(root, corruption, severity, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    
    return loader


def get_transforms(dataset: str = 'cifar10', image_size: int = 224):
    """Get transforms for a dataset."""
    if dataset in ['cifar10', 'cifar100']:
        if dataset == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        else:
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:  # imagenet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform


# Synthetic corruption dataset for testing
class SyntheticCorruptedDataset(Dataset):
    """Generate synthetic corrupted CIFAR-like data for testing."""
    
    def __init__(self, num_samples: int = 1000, num_classes: int = 10,
                 corruption_type: str = 'noise', severity: int = 5,
                 image_size: int = 224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.corruption_type = corruption_type
        self.severity = severity
        self.image_size = image_size
        
        # Generate random clean images
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Apply corruption
        self.images = self._apply_corruption(self.images)
        
    def _apply_corruption(self, images: torch.Tensor) -> torch.Tensor:
        """Apply synthetic corruption."""
        if self.corruption_type == 'noise':
            noise_scale = self.severity * 0.1
            images = images + torch.randn_like(images) * noise_scale
        elif self.corruption_type == 'blur':
            # Simple box blur
            kernel_size = self.severity // 2 + 1
            images = torch.nn.functional.avg_pool2d(images, kernel_size, stride=1, 
                                                     padding=kernel_size//2, count_include_pad=False)
        return torch.clamp(images, -3, 3)  # Rough normalization range
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx].item()


if __name__ == '__main__':
    # Test synthetic dataset
    print("Testing synthetic dataset...")
    dataset = SyntheticCorruptedDataset(num_samples=100, corruption_type='noise', severity=3)
    loader = DataLoader(dataset, batch_size=16)
    
    for images, labels in loader:
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
    
    print("Data loader test passed!")
