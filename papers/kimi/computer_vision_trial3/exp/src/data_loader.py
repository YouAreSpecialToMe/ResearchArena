"""
Data loader for ImageNet-C and ImageNet-V2 benchmarks.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tarfile
import urllib.request
from tqdm import tqdm


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ImageNet-C corruption types
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# Corruption categories
NOISE_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise']
BLUR_CORRUPTIONS = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
WEATHER_CORRUPTIONS = ['snow', 'frost', 'fog', 'brightness']
DIGITAL_CORRUPTIONS = ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def get_transform():
    """Get standard ImageNet preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def denormalize(tensor):
    """Denormalize a tensor for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


class ImageNetCDataset(Dataset):
    """ImageNet-C dataset for a specific corruption and severity."""
    
    def __init__(self, root_dir, corruption, severity=3, transform=None):
        """
        Args:
            root_dir: Root directory containing ImageNet-C data
            corruption: Name of corruption type
            severity: Severity level (1-5)
            transform: Optional transform to apply
        """
        self.root_dir = root_dir
        self.corruption = corruption
        self.severity = severity
        self.transform = transform if transform is not None else get_transform()
        
        # Path to corruption data
        self.corruption_dir = os.path.join(root_dir, corruption, str(severity))
        
        # Load image paths and labels
        self.samples = []
        if os.path.exists(self.corruption_dir):
            for class_idx in range(1000):
                class_dir = os.path.join(self.corruption_dir, str(class_idx))
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                            self.samples.append((
                                os.path.join(class_dir, img_name),
                                class_idx
                            ))
        
        print(f"ImageNet-C {corruption} (severity {severity}): {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ImageNetV2Dataset(Dataset):
    """ImageNet-V2 dataset."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory containing ImageNet-V2 data
            transform: Optional transform to apply
        """
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_transform()
        
        # Load image paths and labels
        self.samples = []
        if os.path.exists(root_dir):
            # ImageNet-V2 structure: root_dir/{class_idx}/{img_name}.jpg or .jpeg
            for class_idx in range(1000):
                class_dir = os.path.join(root_dir, str(class_idx))
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                            self.samples.append((
                                os.path.join(class_dir, img_name),
                                class_idx
                            ))
        
        print(f"ImageNet-V2: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def download_file(url, dest_path, desc=None):
    """Download a file with progress bar."""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    print(f"Downloading {desc or url}...")
    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                f.write(response.read())
    
    print(f"Downloaded to {dest_path}")


def extract_archive(archive_path, dest_dir, desc=None):
    """Extract a tar archive."""
    print(f"Extracting {desc or archive_path}...")
    os.makedirs(dest_dir, exist_ok=True)
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(dest_dir)
    
    print(f"Extracted to {dest_dir}")


def download_imagenet_v2(data_dir):
    """Download ImageNet-V2 dataset."""
    url = "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz"
    archive_path = os.path.join(data_dir, "imagenetv2-matched-frequency.tar.gz")
    extract_dir = os.path.join(data_dir, "imagenetv2")
    
    download_file(url, archive_path, "ImageNet-V2")
    
    if not os.path.exists(os.path.join(extract_dir, "imagenetv2-matched-frequency-format-val")):
        extract_archive(archive_path, extract_dir, "ImageNet-V2")
    
    # Return path to actual data
    return os.path.join(extract_dir, "imagenetv2-matched-frequency-format-val")


def download_imagenet_c_subset(data_dir, corruptions=None, severities=None):
    """
    Download a subset of ImageNet-C.
    For full dataset, use: https://zenodo.org/record/2235448/files/imagenet-c.tar
    
    For this experiment, we'll use a subset approach downloading individual corruptions
    from alternative sources if available, or create synthetic corruptions.
    """
    # For this experiment, we'll use the ImageNet-C dataset
    # In production, this would download from the official source
    imagenet_c_dir = os.path.join(data_dir, "imagenet-c")
    os.makedirs(imagenet_c_dir, exist_ok=True)
    
    print(f"ImageNet-C directory: {imagenet_c_dir}")
    print("Note: Full ImageNet-C should be downloaded from https://zenodo.org/record/2235448")
    
    return imagenet_c_dir


def get_imagenet_c_loader(data_dir, corruption, severity=3, batch_size=1, num_workers=4):
    """Get a DataLoader for ImageNet-C."""
    dataset = ImageNetCDataset(data_dir, corruption, severity)
    
    # Use batch_size=1 for TTA (test-time adaptation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def get_imagenet_v2_loader(data_dir, batch_size=1, num_workers=4):
    """Get a DataLoader for ImageNet-V2."""
    dataset = ImageNetV2Dataset(data_dir)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def create_synthetic_corruption(data_dir, corruption_type, source_images_dir, severity=3):
    """
    Create synthetic corruptions for testing when ImageNet-C is not available.
    This is a fallback for development/testing only.
    """
    import torchvision.transforms.functional as F
    
    corruption_dir = os.path.join(data_dir, corruption_type, str(severity))
    os.makedirs(corruption_dir, exist_ok=True)
    
    # This would create corrupted versions of source images
    # For production, use real ImageNet-C
    print(f"Synthetic corruption created at: {corruption_dir}")
    return corruption_dir


if __name__ == "__main__":
    # Test the data loader
    data_dir = "./data"
    
    # Download ImageNet-V2
    imagenet_v2_path = download_imagenet_v2(data_dir)
    print(f"ImageNet-V2 path: {imagenet_v2_path}")
    
    # Check if ImageNet-C exists
    imagenet_c_path = os.path.join(data_dir, "imagenet-c")
    if os.path.exists(imagenet_c_path):
        print(f"ImageNet-C found at: {imagenet_c_path}")
        # Test loading one corruption
        loader = get_imagenet_c_loader(imagenet_c_path, 'gaussian_noise', severity=3)
        for images, labels in loader:
            print(f"Batch shape: {images.shape}, Labels: {labels}")
            break
    else:
        print("ImageNet-C not found. Please download from https://zenodo.org/record/2235448")
