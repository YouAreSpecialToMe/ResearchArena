"""
Data loading utilities for CIFAR-C, CIFAR-10.1, and ImageNet-C.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import urllib.request
import tarfile
from tqdm import tqdm


# ============== CIFAR-10/100 Clean ==============

def get_cifar_transforms(train=True):
    """Get standard CIFAR transforms"""
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform


def get_cifar100_transforms(train=True):
    """Get standard CIFAR-100 transforms"""
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    return transform


def load_cifar10(data_dir='./data', train=True, batch_size=128):
    """Load CIFAR-10 dataset"""
    transform = get_cifar_transforms(train=train)
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader, dataset


def load_cifar100(data_dir='./data', train=True, batch_size=128):
    """Load CIFAR-100 dataset"""
    transform = get_cifar100_transforms(train=train)
    dataset = datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader, dataset


# ============== CIFAR-10-C / CIFAR-100-C ==============

CIFAR10_C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

CIFAR10_C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
CIFAR100_C_URL = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"


class CIFAR_C_Dataset(Dataset):
    """Dataset for CIFAR-10-C and CIFAR-100-C"""
    
    def __init__(self, data_dir, dataset='cifar10', corruption=None, severity=5, transform=None):
        """
        Args:
            data_dir: Root directory containing CIFAR-10-C or CIFAR-100-C
            dataset: 'cifar10' or 'cifar100'
            corruption: Corruption name (e.g., 'gaussian_noise'). If None, load all.
            severity: Severity level (1-5)
            transform: Transform to apply
        """
        self.dataset = dataset
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        # Load data
        self.data, self.labels = self._load_data(data_dir)
        
    def _load_data(self, data_dir):
        if self.corruption is not None:
            # Load specific corruption
            # Try different directory naming conventions
            dataset_name_upper = self.dataset.upper()  # CIFAR10 or CIFAR100
            if dataset_name_upper == 'CIFAR10':
                possible_dirs = ['CIFAR-10-C', 'CIFAR10-C']
            elif dataset_name_upper == 'CIFAR100':
                possible_dirs = ['CIFAR-100-C', 'CIFAR100-C']
            else:
                possible_dirs = [f'{dataset_name_upper}-C']
            
            data_path = None
            labels_path = None
            for dir_name in possible_dirs:
                temp_data_path = os.path.join(data_dir, dir_name, f'{self.corruption}.npy')
                temp_labels_path = os.path.join(data_dir, dir_name, 'labels.npy')
                if os.path.exists(temp_data_path):
                    data_path = temp_data_path
                    labels_path = temp_labels_path
                    break
            
            if data_path is None:
                raise FileNotFoundError(f"Data not found for corruption {self.corruption}. Please download CIFAR-C datasets.")
            
            all_data = np.load(data_path)  # Shape: [50000, 32, 32, 3]
            all_labels = np.load(labels_path)  # Shape: [50000]
            
            # Extract specific severity (10k images per severity)
            start_idx = (self.severity - 1) * 10000
            end_idx = self.severity * 10000
            data = all_data[start_idx:end_idx]
            labels = all_labels[start_idx:end_idx]
        else:
            # Load all corruptions
            all_data = []
            all_labels = []
            for corruption in CIFAR10_C_CORRUPTIONS:
                data_path = os.path.join(data_dir, f'{self.dataset.upper()}-C', f'{corruption}.npy')
                labels_path = os.path.join(data_dir, f'{self.dataset.upper()}-C', 'labels.npy')
                
                if not os.path.exists(data_path):
                    continue
                
                corruption_data = np.load(data_path)
                corruption_labels = np.load(labels_path)
                
                start_idx = (self.severity - 1) * 10000
                end_idx = self.severity * 10000
                all_data.append(corruption_data[start_idx:end_idx])
                all_labels.append(corruption_labels[start_idx:end_idx])
            
            data = np.concatenate(all_data, axis=0)
            labels = np.concatenate(all_labels, axis=0)
        
        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def load_cifar_c(data_dir='./data', dataset='cifar10', corruption=None, severity=5, batch_size=200):
    """Load CIFAR-10-C or CIFAR-100-C dataset"""
    if dataset == 'cifar10':
        transform = get_cifar_transforms(train=False)
    else:
        transform = get_cifar100_transforms(train=False)
    
    dataset_obj = CIFAR_C_Dataset(data_dir, dataset=dataset, corruption=corruption, 
                                   severity=severity, transform=transform)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=True)
    return loader, dataset_obj


# ============== ImageNet-C (simplified) ==============

IMAGENET_C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

IMAGENET_C_SUBSET = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'pixelate', 'jpeg_compression']


class ImageNet_C_Dataset(Dataset):
    """Dataset for ImageNet-C"""
    
    def __init__(self, data_dir, corruption, severity=5, transform=None):
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        self.data_dir = os.path.join(data_dir, 'ImageNet-C', corruption, str(severity))
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"ImageNet-C data not found at {self.data_dir}")
        
        # Load all images
        self.samples = []
        self.labels = []
        for class_idx in range(1000):
            class_dir = os.path.join(self.data_dir, str(class_idx))
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    self.samples.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_imagenet_transform():
    """Get ImageNet validation transforms"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_imagenet_c(data_dir='./data', corruption='gaussian_noise', severity=5, batch_size=64):
    """Load ImageNet-C dataset"""
    transform = get_imagenet_transform()
    dataset = ImageNet_C_Dataset(data_dir, corruption, severity, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=True)
    return loader, dataset


# ============== CIFAR-10.1 ==============

CIFAR10_1_URL = "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy"
CIFAR10_1_LABELS_URL = "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy"


class CIFAR10_1_Dataset(Dataset):
    """CIFAR-10.1 dataset for natural distribution shift"""
    
    def __init__(self, data_dir, transform=None):
        self.data_path = os.path.join(data_dir, 'cifar10.1_v6_data.npy')
        self.labels_path = os.path.join(data_dir, 'cifar10.1_v6_labels.npy')
        
        self.transform = transform
        
        # Load data
        self.data = np.load(self.data_path)
        self.labels = np.load(self.labels_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def load_cifar10_1(data_dir='./data', batch_size=200):
    """Load CIFAR-10.1 dataset"""
    transform = get_cifar_transforms(train=False)
    dataset = CIFAR10_1_Dataset(data_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    return loader, dataset


# ============== Data Download Utilities ==============

def download_file(url, dest_path, desc="Downloading"):
    """Download file with progress bar"""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)


def download_cifar_c(data_dir='./data'):
    """Download and extract CIFAR-10-C and CIFAR-100-C"""
    # CIFAR-10-C
    cifar10_tar = os.path.join(data_dir, 'CIFAR-10-C.tar')
    if not os.path.exists(os.path.join(data_dir, 'CIFAR-10-C')):
        print("Downloading CIFAR-10-C...")
        # Use wget for better handling of large files
        os.system(f"wget -O {cifar10_tar} {CIFAR10_C_URL}")
        print("Extracting CIFAR-10-C...")
        with tarfile.open(cifar10_tar, 'r') as tar:
            tar.extractall(data_dir)
    
    # CIFAR-100-C
    cifar100_tar = os.path.join(data_dir, 'CIFAR-100-C.tar')
    if not os.path.exists(os.path.join(data_dir, 'CIFAR-100-C')):
        print("Downloading CIFAR-100-C...")
        os.system(f"wget -O {cifar100_tar} {CIFAR100_C_URL}")
        print("Extracting CIFAR-100-C...")
        with tarfile.open(cifar100_tar, 'r') as tar:
            tar.extractall(data_dir)


def download_cifar10_1(data_dir='./data'):
    """Download CIFAR-10.1 dataset"""
    data_path = os.path.join(data_dir, 'cifar10.1_v6_data.npy')
    labels_path = os.path.join(data_dir, 'cifar10.1_v6_labels.npy')
    
    if not os.path.exists(data_path):
        print("Downloading CIFAR-10.1 data...")
        os.system(f"wget -O {data_path} {CIFAR10_1_URL}")
    
    if not os.path.exists(labels_path):
        print("Downloading CIFAR-10.1 labels...")
        os.system(f"wget -O {labels_path} {CIFAR10_1_LABELS_URL}")


# ============== Prepare All Data ==============

def prepare_all_data(data_dir='./data'):
    """Download and prepare all datasets"""
    print("Preparing datasets...")
    
    # Download CIFAR-10 and CIFAR-100 (clean)
    print("Downloading CIFAR-10...")
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    print("Downloading CIFAR-100...")
    datasets.CIFAR100(root=data_dir, train=True, download=True)
    datasets.CIFAR100(root=data_dir, train=False, download=True)
    
    # Download CIFAR-10.1
    print("Downloading CIFAR-10.1...")
    download_cifar10_1(data_dir)
    
    # Download CIFAR-C
    print("Downloading CIFAR-C (this may take a while)...")
    download_cifar_c(data_dir)
    
    print("All datasets prepared!")
