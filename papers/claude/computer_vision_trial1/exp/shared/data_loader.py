import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from PIL import Image
from imagecorruptions import corrupt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CORRUPTIONS = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
}

ALL_CORRUPTIONS = [c for group in CORRUPTIONS.values() for c in group]

# Some corruptions may be broken due to library version incompatibilities
BROKEN_CORRUPTIONS = set()
def _test_corruptions():
    """Test which corruptions work with current library versions."""
    import warnings
    warnings.filterwarnings('ignore')
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    broken = set()
    for corr in ALL_CORRUPTIONS:
        try:
            corrupt(test_img, corruption_name=corr, severity=3)
        except Exception:
            broken.add(corr)
    return broken

# Corruptions known to be very slow (>10x slower than others)
SLOW_CORRUPTIONS = {'zoom_blur'}

try:
    BROKEN_CORRUPTIONS = _test_corruptions()
    SKIP_CORRUPTIONS = BROKEN_CORRUPTIONS | SLOW_CORRUPTIONS
    WORKING_CORRUPTIONS = [c for c in ALL_CORRUPTIONS if c not in SKIP_CORRUPTIONS]
except Exception:
    BROKEN_CORRUPTIONS = set()
    SKIP_CORRUPTIONS = SLOW_CORRUPTIONS
    WORKING_CORRUPTIONS = [c for c in ALL_CORRUPTIONS if c not in SKIP_CORRUPTIONS]

# Cached HF dataset reference
_hf_dataset_cache = {}


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CorruptedDataset(Dataset):
    """Wraps a dataset and applies corruption on-the-fly."""
    def __init__(self, base_dataset, corruption_type, severity):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.resize_crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        img = self.resize_crop(img)
        img_np = np.array(img)
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        corrupted = corrupt(img_np, corruption_name=self.corruption_type, severity=self.severity)
        corrupted_tensor = torch.from_numpy(corrupted).permute(2, 0, 1).float() / 255.0
        corrupted_tensor = self.normalize(corrupted_tensor)
        return corrupted_tensor, label


class RawImageNetDataset(Dataset):
    """Loads ImageNet without transforms for corruption pipeline."""
    def __init__(self, root):
        self.dataset = datasets.ImageFolder(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label


def find_imagenet_val():
    """Try to find ImageNet validation set."""
    candidates = [
        os.environ.get('IMAGENET_DIR', ''),
        '/datasets/imagenet/val',
        '/data/imagenet/val',
        '/scratch/datasets/imagenet/val',
        os.path.expanduser('~/data/imagenet/val'),
        '/home/shared/imagenet/val',
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if len(subdirs) >= 100:
                return path
    return None


def _get_hf_dataset():
    """Load and cache HF ImageNet dataset."""
    if 'val' not in _hf_dataset_cache:
        from datasets import load_dataset
        _hf_dataset_cache['val'] = load_dataset(
            "evanarlian/imagenet_1k_resized_256", split="val"
        )
    return _hf_dataset_cache['val']


class HFImageNetVal(Dataset):
    """ImageNet validation set from HuggingFace (evanarlian/imagenet_1k_resized_256)."""
    def __init__(self, transform=None, raw=False):
        self.ds = _get_hf_dataset()
        self.transform = transform
        self.raw = raw

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['image']
        label = item['label']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.raw:
            return img, label
        if self.transform:
            img = self.transform(img)
        return img, label


def get_imagenet_val_loader(batch_size=128, num_workers=4, subset_size=None, seed=42):
    """Get ImageNet validation loader."""
    val_path = find_imagenet_val()
    if val_path:
        dataset = datasets.ImageFolder(val_path, transform=get_val_transform())
    else:
        dataset = HFImageNetVal(transform=get_val_transform())

    if subset_size and subset_size < len(dataset):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_corrupted_loader(corruption_type, severity, batch_size=128, num_workers=4,
                         subset_size=None, seed=42):
    """Get corrupted ImageNet loader with on-the-fly corruption."""
    val_path = find_imagenet_val()
    if val_path:
        base_dataset = RawImageNetDataset(val_path)
    else:
        base_dataset = HFImageNetVal(transform=None, raw=True)

    if subset_size and subset_size < len(base_dataset):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(base_dataset), subset_size, replace=False)
        base_dataset = Subset(base_dataset, indices)

    corrupted = CorruptedDataset(base_dataset, corruption_type, severity)
    return DataLoader(corrupted, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
