"""Data loading utilities for ImageNet validation and corruption evaluation."""
import os
import io
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from imagecorruptions import corrupt, get_corruption_names

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Corruption categories
CORRUPTION_TYPES = get_corruption_names('common')  # all 15 standard corruptions
NOISE = ['gaussian_noise', 'shot_noise', 'impulse_noise']
BLUR = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
WEATHER = ['snow', 'frost', 'fog', 'brightness']
DIGITAL = ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
CATEGORY_MAP = {}
for c in NOISE: CATEGORY_MAP[c] = 'noise'
for c in BLUR: CATEGORY_MAP[c] = 'blur'
for c in WEATHER: CATEGORY_MAP[c] = 'weather'
for c in DIGITAL: CATEGORY_MAP[c] = 'digital'

# Representative subset for ablations
REPRESENTATIVE_CORRUPTIONS = ['gaussian_noise', 'defocus_blur', 'snow', 'contrast', 'jpeg_compression']

# AlexNet reference error rates for mCE computation (from Hendrycks & Dietterich 2019)
# These are the sum of errors across 5 severity levels for AlexNet
ALEXNET_ERR = {
    'gaussian_noise': 0.886428, 'shot_noise': 0.894468, 'impulse_noise': 0.922640,
    'defocus_blur': 0.819876, 'glass_blur': 0.826268, 'motion_blur': 0.785948,
    'zoom_blur': 0.798360, 'snow': 0.866816, 'frost': 0.826572,
    'fog': 0.819324, 'brightness': 0.564592, 'contrast': 0.853204,
    'elastic_transform': 0.646056, 'pixelate': 0.717840, 'jpeg_compression': 0.606500,
}


def get_transform(input_size=224):
    """Standard ViT evaluation transform."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageNetValDataset(Dataset):
    """ImageNet validation set loaded from HuggingFace parquet files."""

    def __init__(self, data_dir, transform=None, indices=None):
        cache_dir = os.path.join(data_dir, 'hf_cache', 'datasets--Tsomaros--Imagenet-1k_validation')
        parquets = sorted(glob.glob(os.path.join(cache_dir, 'snapshots/*/data/*.parquet')))
        if not parquets:
            raise FileNotFoundError(f"No parquet files found in {cache_dir}")

        dfs = [pd.read_parquet(p) for p in parquets]
        self.df = pd.concat(dfs, ignore_index=True)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.transform = transform or get_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row['image']
        if isinstance(img_data, dict):
            img = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
        else:
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label


class CorruptedImageNetDataset(Dataset):
    """ImageNet with on-the-fly corruption applied."""

    def __init__(self, data_dir, corruption_name, severity=5, transform=None, indices=None):
        self.base_dataset = ImageNetValDataset(data_dir, transform=None, indices=indices)
        self.corruption_name = corruption_name
        self.severity = severity
        self.final_transform = transform or get_transform()
        # We need a transform that just resizes for corruption (no normalize)
        self.pre_corrupt_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ])
        self.post_corrupt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset.df.iloc[idx]
        img_data = row['image']
        if isinstance(img_data, dict):
            img = Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
        else:
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        label = int(row['label'])

        # Resize and center crop to 224x224 before corruption
        img = self.pre_corrupt_transform(img)
        # Apply corruption (requires numpy array)
        img_np = np.array(img)
        img_corrupted = corrupt(img_np, corruption_name=self.corruption_name, severity=self.severity)
        # Convert back to PIL and apply final transform
        img_corrupted = Image.fromarray(img_corrupted)
        img_tensor = self.post_corrupt_transform(img_corrupted)
        return img_tensor, label


def get_dataloader(dataset, batch_size=256, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_calibration_indices(seed, n_calibration=1000, n_total=50000):
    """Get reproducible calibration image indices."""
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, size=n_calibration, replace=False)


def compute_mce(model_errors, alexnet_errors=ALEXNET_ERR):
    """Compute mean Corruption Error (mCE).

    model_errors: dict of {corruption: [err_sev1, ..., err_sev5]}
                  or {corruption: sum_of_errors}
    """
    ce_values = []
    for corruption in CORRUPTION_TYPES:
        if corruption not in model_errors:
            continue
        model_err = model_errors[corruption]
        if isinstance(model_err, (list, np.ndarray)):
            model_err = sum(model_err)
        alexnet_err = alexnet_errors[corruption]
        ce = model_err / alexnet_err
        ce_values.append(ce)
    return np.mean(ce_values) if ce_values else float('nan')
