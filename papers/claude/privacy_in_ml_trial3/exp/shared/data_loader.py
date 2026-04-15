"""Dataset loading with subgroup annotations."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


class SubgroupDataset(Dataset):
    """Wraps a dataset to provide (image, label, subgroup_id) tuples."""
    def __init__(self, base_dataset, subgroup_labels, transform=None):
        self.base_dataset = base_dataset
        self.subgroup_labels = subgroup_labels
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform:
            img = self.transform(img)
        subgroup = self.subgroup_labels[idx]
        return img, label, subgroup


def get_cifar10_imbalanced(seed=42):
    """CIFAR-10 with synthetic minority/majority class imbalance.

    Minority classes (2,3,5,7) subsampled to 30% (not 10% — that was too extreme).
    Subgroups: 0=majority (classes 0,1,4,6,8,9), 1=minority (classes 2,3,5,7).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_full = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_full = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    minority_classes = {2, 3, 5, 7}
    rng = np.random.RandomState(seed)

    def subsample(dataset, is_train=True):
        targets = np.array(dataset.targets)
        indices = []
        for cls in range(10):
            cls_idx = np.where(targets == cls)[0]
            if cls in minority_classes and is_train:
                # Subsample minority to 30% (was 10%, caused 0% accuracy under DP)
                n_keep = max(int(len(cls_idx) * 0.3), 50)
                cls_idx = rng.choice(cls_idx, n_keep, replace=False)
            indices.extend(cls_idx.tolist())
        rng.shuffle(indices)
        subgroups = np.array([1 if targets[i] in minority_classes else 0 for i in indices])
        return indices, subgroups

    train_idx, train_subgroups = subsample(train_full, is_train=True)
    test_idx, test_subgroups = subsample(test_full, is_train=False)

    # Split train into train/val (90/10)
    n_train = int(len(train_idx) * 0.9)
    train_indices = train_idx[:n_train]
    val_indices = train_idx[n_train:]
    train_sg = train_subgroups[:n_train]
    val_sg = train_subgroups[n_train:]

    train_ds = SubgroupDataset(Subset(train_full, train_indices), train_sg)
    val_ds = SubgroupDataset(Subset(train_full, val_indices), val_sg)
    test_ds = SubgroupDataset(Subset(test_full, test_idx), test_subgroups)

    subgroup_names = {0: "majority", 1: "minority"}
    stats = {
        "dataset": "cifar10",
        "num_classes": 10,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_idx),
        "train_subgroup_counts": dict(Counter(train_sg.tolist())),
        "subgroup_names": subgroup_names,
    }
    return train_ds, val_ds, test_ds, stats


def get_utkface(seed=42):
    """UTKFace: gender classification with ethnicity as protected attribute.

    Loaded from HuggingFace 'nu-delta/utkface'.
    Task: gender (0=Male, 1=Female).
    Subgroups by ethnicity: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others.
    """
    from datasets import load_dataset
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print("Loading UTKFace from HuggingFace...")
    hf_ds = load_dataset("nu-delta/utkface", split="train")

    ethnicity_map = {"White": 0, "Black": 1, "Asian": 2, "Indian": 3, "Other": 4}
    gender_map = {"Male": 0, "Female": 1}

    images = []
    labels = []
    subgroups = []

    for item in hf_ds:
        eth_str = item["ethnicity"]
        gen_str = item["gender"]
        if eth_str not in ethnicity_map or gen_str not in gender_map:
            continue
        try:
            img = item["image"].convert("RGB")
            img = transform(img)
            images.append(img)
            labels.append(gender_map[gen_str])
            subgroups.append(ethnicity_map[eth_str])
        except Exception:
            continue

    print(f"Loaded {len(images)} UTKFace images")

    images = torch.stack(images)
    labels = torch.tensor(labels)
    subgroups = torch.tensor(subgroups)

    # Shuffle and split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(images))
    n = len(images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    class TensorSubgroupDataset(Dataset):
        def __init__(self, imgs, labs, sgs):
            self.imgs = imgs
            self.labs = labs
            self.sgs = sgs
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            return self.imgs[idx], self.labs[idx].item(), self.sgs[idx].item()

    train_ds = TensorSubgroupDataset(images[perm[:n_train]], labels[perm[:n_train]], subgroups[perm[:n_train]])
    val_ds = TensorSubgroupDataset(images[perm[n_train:n_train+n_val]], labels[perm[n_train:n_train+n_val]], subgroups[perm[n_train:n_train+n_val]])
    test_ds = TensorSubgroupDataset(images[perm[n_train+n_val:]], labels[perm[n_train+n_val:]], subgroups[perm[n_train+n_val:]])

    subgroup_names = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
    stats = {
        "dataset": "utkface",
        "num_classes": 2,
        "train_size": n_train,
        "val_size": n_val,
        "test_size": n - n_train - n_val,
        "train_subgroup_counts": dict(Counter(subgroups[perm[:n_train]].numpy().tolist())),
        "subgroup_names": subgroup_names,
    }
    return train_ds, val_ds, test_ds, stats


def get_celeba(seed=42, max_samples=20000):
    """CelebA: smiling classification with gender as protected attribute.

    Loaded from HuggingFace 'flwrlabs/celeba'.
    Task: Smiling (binary).
    Subgroups by gender: 0=Female, 1=Male.
    """
    from datasets import load_dataset
    from PIL import Image

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print("Loading CelebA from HuggingFace (flwrlabs/celeba)...")
    hf_ds = load_dataset("flwrlabs/celeba", split="train")

    images = []
    labels = []
    subgroups = []

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(hf_ds))[:max_samples]

    for idx in indices:
        item = hf_ds[int(idx)]
        try:
            img = item["image"].convert("RGB")
            img = transform(img)
            # Smiling: 1=smiling, 0=not smiling (already binary in dataset)
            label = int(item["Smiling"])
            # Gender: Male attribute (1=Male, 0=Female as subgroup)
            gender = int(item["Male"])
            images.append(img)
            labels.append(label)
            subgroups.append(gender)
        except Exception:
            continue

        if len(images) % 5000 == 0:
            print(f"  Loaded {len(images)} CelebA images...")

    print(f"Loaded {len(images)} CelebA images")

    images = torch.stack(images)
    labels = torch.tensor(labels)
    subgroups = torch.tensor(subgroups)

    # Shuffle and split
    perm = rng.permutation(len(images))
    n = len(images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    class TensorSubgroupDataset(Dataset):
        def __init__(self, imgs, labs, sgs):
            self.imgs = imgs
            self.labs = labs
            self.sgs = sgs
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            return self.imgs[idx], self.labs[idx].item(), self.sgs[idx].item()

    train_ds = TensorSubgroupDataset(images[perm[:n_train]], labels[perm[:n_train]], subgroups[perm[:n_train]])
    val_ds = TensorSubgroupDataset(images[perm[n_train:n_train+n_val]], labels[perm[n_train:n_train+n_val]], subgroups[perm[n_train:n_train+n_val]])
    test_ds = TensorSubgroupDataset(images[perm[n_train+n_val:]], labels[perm[n_train+n_val:]], subgroups[perm[n_train+n_val:]])

    subgroup_names = {0: "Female", 1: "Male"}
    stats = {
        "dataset": "celeba",
        "num_classes": 2,
        "train_size": n_train,
        "val_size": n_val,
        "test_size": n - n_train - n_val,
        "train_subgroup_counts": dict(Counter(subgroups[perm[:n_train]].numpy().tolist())),
        "subgroup_names": subgroup_names,
    }
    return train_ds, val_ds, test_ds, stats


def get_dataset(name, seed=42):
    """Get dataset by name."""
    if name == "cifar10":
        return get_cifar10_imbalanced(seed)
    elif name == "utkface":
        return get_utkface(seed)
    elif name == "celeba":
        return get_celeba(seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def make_loader(dataset, batch_size=256, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)
