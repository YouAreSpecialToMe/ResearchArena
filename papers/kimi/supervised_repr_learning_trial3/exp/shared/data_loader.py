"""
Data loading utilities for CAG-HNM experiments.
Supports CIFAR-100, CUB-200-2011 with attribute handling.
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def get_cifar100_transforms(train=True):
    """Get transforms for CIFAR-100."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])


def get_cub_transforms(train=True):
    """Get transforms for CUB-200."""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_cifar100_attributes(root='./data'):
    """
    Create attribute matrix for CIFAR-100 using coarse labels.
    CIFAR-100 has 100 fine classes and 20 coarse classes.
    Each fine class belongs to exactly one coarse class.
    """
    # CIFAR-100 coarse label assignment
    # Map fine label -> coarse label (0-19)
    fine_to_coarse = [
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
        3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
        0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
        16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
        2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
        18, 1, 2, 15, 6, 0, 17, 8, 14, 13
    ]
    
    num_classes = 100
    num_coarse = 20
    
    # Create binary attribute matrix
    attributes = np.zeros((num_classes, num_coarse), dtype=np.float32)
    for fine_label in range(num_classes):
        coarse_label = fine_to_coarse[fine_label]
        attributes[fine_label, coarse_label] = 1.0
    
    # Compute Jaccard similarity matrix
    similarity = compute_jaccard_similarity(attributes)
    
    save_path = os.path.join(root, 'cifar100_attributes.npz')
    os.makedirs(root, exist_ok=True)
    np.savez(save_path, attributes=attributes, similarity=similarity)
    print(f"Saved CIFAR-100 attributes to {save_path}")
    
    return attributes, similarity


def compute_jaccard_similarity(attributes):
    """
    Compute Jaccard similarity between all pairs of classes.
    attributes: (num_classes, num_attributes)
    returns: (num_classes, num_classes) similarity matrix
    """
    num_classes = attributes.shape[0]
    similarity = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    for i in range(num_classes):
        for j in range(num_classes):
            intersection = np.sum((attributes[i] > 0) & (attributes[j] > 0))
            union = np.sum((attributes[i] > 0) | (attributes[j] > 0))
            if union > 0:
                similarity[i, j] = intersection / union
            else:
                similarity[i, j] = 0.0
    
    return similarity


def load_cifar100_attributes(root='./data'):
    """Load or create CIFAR-100 attributes."""
    save_path = os.path.join(root, 'cifar100_attributes.npz')
    if os.path.exists(save_path):
        data = np.load(save_path)
        return data['attributes'], data['similarity']
    else:
        return create_cifar100_attributes(root)


class CUBDataset(Dataset):
    """CUB-200-2011 Dataset with attributes."""
    
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Load images and labels
        images_txt = os.path.join(root, 'images.txt')
        labels_txt = os.path.join(root, 'image_class_labels.txt')
        train_test_split = os.path.join(root, 'train_test_split.txt')
        
        # Check if files exist
        if not os.path.exists(images_txt):
            raise FileNotFoundError(
                f"CUB-200 dataset not found at {root}. "
                "Please download from: http://www.vision.caltech.edu/datasets/cub_200_2011/"
            )
        
        # Read image paths and labels
        self.images = []
        self.labels = []
        
        with open(images_txt, 'r') as f:
            image_paths = [line.strip().split()[1] for line in f]
        
        with open(labels_txt, 'r') as f:
            labels = [int(line.strip().split()[1]) - 1 for line in f]  # 0-indexed
        
        with open(train_test_split, 'r') as f:
            is_train = [int(line.strip().split()[1]) for line in f]
        
        # Filter by train/test split
        for i, (img_path, label, train_flag) in enumerate(zip(image_paths, labels, is_train)):
            if (train_flag == 1 and train) or (train_flag == 0 and not train):
                self.images.append(os.path.join(root, 'images', img_path))
                self.labels.append(label)
        
        # Load attributes
        self.attributes, self.similarity = self.load_cub_attributes()
    
    def load_cub_attributes(self):
        """Load or create CUB-200 attributes."""
        attr_path = os.path.join(self.root, 'cub_attributes.npz')
        
        if os.path.exists(attr_path):
            data = np.load(attr_path)
            return data['attributes'], data['similarity']
        
        # Load class attributes
        class_attr_file = os.path.join(self.root, 'attributes', 'class_attribute_labels_continuous.txt')
        
        if not os.path.exists(class_attr_file):
            print(f"Warning: CUB attributes not found at {class_attr_file}")
            # Create dummy attributes
            num_classes = 200
            attributes = np.eye(num_classes, dtype=np.float32)
        else:
            # Load continuous attributes and binarize
            attributes = []
            with open(class_attr_file, 'r') as f:
                for line in f:
                    vals = [float(x) for x in line.strip().split()]
                    # Binarize: values > 0 indicate presence
                    binary = np.array([1.0 if v > 0 else 0.0 for v in vals[1:]], dtype=np.float32)
                    attributes.append(binary)
            attributes = np.array(attributes, dtype=np.float32)
        
        similarity = compute_jaccard_similarity(attributes)
        np.savez(attr_path, attributes=attributes, similarity=similarity)
        
        return attributes, similarity
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_cifar100_dataloaders(batch_size=512, num_workers=4, root='./data'):
    """Get CIFAR-100 train and test dataloaders."""
    train_transform = get_cifar100_transforms(train=True)
    test_transform = get_cifar100_transforms(train=False)
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def get_cub_dataloaders(batch_size=256, num_workers=4, root='./data/CUB_200_2011'):
    """Get CUB-200 train and test dataloaders."""
    train_transform = get_cub_transforms(train=True)
    test_transform = get_cub_transforms(train=False)
    
    try:
        train_dataset = CUBDataset(root=root, train=True, transform=train_transform)
        test_dataset = CUBDataset(root=root, train=False, transform=test_transform)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, test_loader, train_dataset.attributes, train_dataset.similarity
    except FileNotFoundError as e:
        print(f"CUB-200 dataset not available: {e}")
        return None, None, None, None


if __name__ == '__main__':
    # Test data loading
    print("Testing CIFAR-100 data loading...")
    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
    print(f"CIFAR-100 train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    
    # Test attribute creation
    attrs, sim = load_cifar100_attributes()
    print(f"CIFAR-100 attributes shape: {attrs.shape}, similarity shape: {sim.shape}")
