#!/usr/bin/env python3
"""Re-run AEP experiments with proper evaluation setup.

Addresses reviewer feedback:
1. Uses ImageNet-V2 as ID dataset (models have ~60% top-1 accuracy)
2. Removes low-resolution OOD datasets (SVHN, CIFAR-10, CIFAR-100)
3. Uses only native 224x224 OOD datasets: Textures/DTD, Flowers-102, Oxford Pets, EuroSAT, STL-10
4. Adds Swin-Tiny evaluation
5. Adds calibration evaluation on corrupted images (Gaussian noise, blur, etc.)
6. Benchmarks overhead with timm's native attention output
"""

import os
import sys
import json
import time
import gc
import logging
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import timm
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from shared.aep import (AEPExtractor, compute_aep_profile, compute_id_statistics,
                         compute_mahalanobis_scores, compute_aep_profile_subset)
from shared.metrics import (compute_ood_metrics, compute_calibration_metrics,
                             compute_ece, compute_mce, compute_brier_score)
from shared.baselines import (msp_score, energy_score, vim_score, knn_score,
                               fit_temperature_scaling, fit_adaptive_temperature,
                               apply_adaptive_temperature, fuse_scores)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WORKSPACE = Path(__file__).parent.parent
DATA_DIR = WORKSPACE / 'data'
RESULTS_DIR = WORKSPACE / 'exp' / 'results_v2'
FIGURES_DIR = WORKSPACE / 'figures'
CACHE_DIR = WORKSPACE / 'exp' / 'cache'

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]
BATCH_SIZE = 64

MODEL_CONFIGS = {
    'deit_small': {'name': 'deit_small_patch16_224', 'num_layers': 12, 'feat_dim': 384, 'num_heads': 6},
    'deit_base': {'name': 'deit_base_patch16_224', 'num_layers': 12, 'feat_dim': 768, 'num_heads': 12},
    'vit_base': {'name': 'vit_base_patch16_224', 'num_layers': 12, 'feat_dim': 768, 'num_heads': 12},
    'swin_tiny': {'name': 'swin_tiny_patch4_window7_224', 'num_layers': None, 'feat_dim': 768, 'num_heads': None},
}

TRANSFORM_224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============ Dataset Wrappers ============

class ImageNetV2Dataset(Dataset):
    """ImageNet-V2 matched-frequency dataset from HuggingFace."""

    def __init__(self, cache_dir, transform, max_samples=None):
        self.transform = transform
        self.cache_path = Path(cache_dir) / 'imagenetv2_images'
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.labels_path = Path(cache_dir) / 'imagenetv2_labels.json'

        if self.labels_path.exists():
            logger.info("Loading cached ImageNet-V2 labels...")
            with open(self.labels_path) as f:
                self.labels = json.load(f)
            self.image_paths = sorted(self.cache_path.glob('*.jpg'))
        else:
            logger.info("Downloading ImageNet-V2 from HuggingFace...")
            from datasets import load_dataset
            ds = load_dataset('vaishaal/ImageNetV2', split='train')
            self.labels = []
            self.image_paths = []
            for i, item in enumerate(tqdm(ds, desc="Loading ImageNet-V2")):
                img = item['jpeg']
                key = item['__key__']
                label = int(key.split('/')[-2])
                # Save image to disk
                img_path = self.cache_path / f'{i:05d}.jpg'
                if not img_path.exists():
                    img.save(str(img_path))
                self.image_paths.append(img_path)
                self.labels.append(label)
            with open(self.labels_path, 'w') as f:
                json.dump(self.labels, f)
            logger.info(f"Cached {len(self.labels)} ImageNet-V2 images")

        if max_samples and max_samples < len(self.labels):
            indices = np.random.RandomState(42).permutation(len(self.labels))[:max_samples]
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(str(self.image_paths[idx])).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]


class STL10Wrapper(Dataset):
    """STL-10 with 96x96 -> 224x224 resize."""
    def __init__(self, root, split='test'):
        self.ds = datasets.STL10(root=root, split=split, download=False)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        from PIL import Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        return self.transform(img), label


class CorruptedDataset(Dataset):
    """Apply corruption to an existing dataset for calibration evaluation."""

    def __init__(self, base_dataset, corruption_fn, severity=3):
        self.base_dataset = base_dataset
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # img is already a tensor from base_dataset, need to convert back
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Apply corruption
        corrupted = self.corruption_fn(img_np, self.severity)

        from PIL import Image
        corrupted_pil = Image.fromarray(corrupted)
        corrupted_tensor = self.to_tensor(corrupted_pil)
        return corrupted_tensor, label


def gaussian_noise(image, severity):
    """Apply Gaussian noise corruption."""
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    img = image.astype(np.float32) / 255.0
    img = img + np.random.normal(0, c, img.shape)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def gaussian_blur(image, severity):
    """Apply Gaussian blur corruption."""
    from scipy.ndimage import gaussian_filter
    c = [1.0, 2.0, 3.0, 4.0, 6.0][severity - 1]
    img = gaussian_filter(image.astype(np.float32), sigma=(c, c, 0))
    return np.clip(img, 0, 255).astype(np.uint8)


def shot_noise(image, severity):
    """Apply shot noise corruption."""
    c = [60, 25, 12, 5, 3][severity - 1]
    img = np.random.poisson(image.astype(np.float32) / 255.0 * c) / c * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def brightness(image, severity):
    """Apply brightness corruption."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    img = image.astype(np.float32) / 255.0 + c
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def contrast(image, severity):
    """Apply contrast corruption."""
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    mean = np.mean(image.astype(np.float32) / 255.0, axis=(0, 1), keepdims=True)
    img = (image.astype(np.float32) / 255.0 - mean) * c + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)


CORRUPTIONS = {
    'gaussian_noise': gaussian_noise,
    'gaussian_blur': gaussian_blur,
    'shot_noise': shot_noise,
    'brightness': brightness,
    'contrast': contrast,
}


# ============ Model and Feature Extraction ============

def load_model(model_key):
    """Load pretrained model."""
    cfg = MODEL_CONFIGS[model_key]
    model = timm.create_model(cfg['name'], pretrained=True)
    model = model.to(DEVICE)
    model.eval()
    return model


def is_swin_model(model_key):
    return 'swin' in model_key


def extract_features_and_logits(model, dataloader, model_key):
    """Extract penultimate features and logits."""
    all_logits = []
    all_labels = []
    features_store = []

    # Hook into the input of the final classifier head
    def pre_hook_fn(module, input):
        features_store.append(input[0].detach().cpu())
    hook = model.head.register_forward_pre_hook(pre_hook_fn)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Extracting features ({model_key})"):
            images = images.to(DEVICE)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    hook.remove()

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_features = np.concatenate([f.numpy() for f in features_store])

    # For Swin: features may already be pooled (batch, dim)
    # For ViT/DeiT: features might be (batch, dim) after norm or (batch, tokens, dim)
    if all_features.ndim == 3:
        all_features = all_features[:, 0, :]  # CLS token

    return all_features, all_logits, all_labels


def extract_aep_and_logits(model, dataloader, model_key):
    """Extract AEP profiles and logits. Only for non-Swin models."""
    if is_swin_model(model_key):
        return None, None, None, None

    extractor = AEPExtractor(model, MODEL_CONFIGS[model_key]['num_layers'])
    all_profiles = []
    all_logits = []
    all_labels = []
    all_features = []

    # Also get features
    features_store = []
    def pre_hook_fn(module, input):
        features_store.append(input[0].detach().cpu())
    hook = model.head.register_forward_pre_hook(pre_hook_fn)

    for images, labels in tqdm(dataloader, desc=f"Extracting AEP ({model_key})"):
        images = images.to(DEVICE)
        logits, attn_maps = extractor._extract_with_monkey_patch(images)
        profiles = compute_aep_profile(attn_maps)
        all_profiles.append(profiles)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    hook.remove()
    all_features = np.concatenate([f.numpy() for f in features_store]) if features_store else None
    if all_features is not None and all_features.ndim == 3:
        all_features = all_features[:, 0, :]

    all_profiles = np.concatenate(all_profiles)
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    extractor.remove_hooks()
    return all_profiles, all_features, all_logits, all_labels


# ============ Main Experiment ============

def load_datasets():
    """Load all datasets."""
    logger.info("Loading datasets...")

    # ID: ImageNet-V2
    id_dataset = ImageNetV2Dataset(CACHE_DIR, TRANSFORM_224)
    logger.info(f"ImageNet-V2: {len(id_dataset)} images")

    # OOD datasets (all native high-resolution)
    ood_datasets = {}

    # Textures/DTD
    try:
        dtd = datasets.DTD(root=str(DATA_DIR / 'dtd'), split='test', transform=TRANSFORM_224, download=False)
        ood_datasets['Textures'] = dtd
        logger.info(f"Textures/DTD: {len(dtd)} images")
    except Exception:
        try:
            dtd = datasets.DTD(root=str(DATA_DIR / 'dtd'), split='test', transform=TRANSFORM_224, download=True)
            ood_datasets['Textures'] = dtd
            logger.info(f"Textures/DTD: {len(dtd)} images")
        except Exception as e:
            logger.warning(f"Could not load DTD: {e}")

    # Flowers-102
    try:
        flowers = datasets.Flowers102(root=str(DATA_DIR / 'flowers102'), split='test',
                                       transform=TRANSFORM_224, download=False)
        ood_datasets['Flowers-102'] = flowers
        logger.info(f"Flowers-102: {len(flowers)} images")
    except Exception:
        try:
            flowers = datasets.Flowers102(root=str(DATA_DIR / 'flowers102'), split='test',
                                           transform=TRANSFORM_224, download=True)
            ood_datasets['Flowers-102'] = flowers
            logger.info(f"Flowers-102: {len(flowers)} images")
        except Exception as e:
            logger.warning(f"Could not load Flowers-102: {e}")

    # Oxford Pets
    try:
        pets = datasets.OxfordIIITPet(root=str(DATA_DIR / 'pets'), split='test',
                                       transform=TRANSFORM_224, download=False)
        ood_datasets['Oxford-Pets'] = pets
        logger.info(f"Oxford Pets: {len(pets)} images")
    except Exception:
        try:
            pets = datasets.OxfordIIITPet(root=str(DATA_DIR / 'pets'), split='test',
                                           transform=TRANSFORM_224, download=True)
            ood_datasets['Oxford-Pets'] = pets
            logger.info(f"Oxford Pets: {len(pets)} images")
        except Exception as e:
            logger.warning(f"Could not load Oxford Pets: {e}")

    # EuroSAT
    try:
        eurosat = datasets.EuroSAT(root=str(DATA_DIR / 'eurosat'), transform=TRANSFORM_224, download=False)
        ood_datasets['EuroSAT'] = eurosat
        logger.info(f"EuroSAT: {len(eurosat)} images")
    except Exception:
        try:
            eurosat = datasets.EuroSAT(root=str(DATA_DIR / 'eurosat'), transform=TRANSFORM_224, download=True)
            ood_datasets['EuroSAT'] = eurosat
            logger.info(f"EuroSAT: {len(eurosat)} images")
        except Exception as e:
            logger.warning(f"Could not load EuroSAT: {e}")

    # STL-10 (96x96, better than CIFAR but still upscaled - include with caveat)
    try:
        stl10 = STL10Wrapper(str(DATA_DIR / 'stl10'), split='test')
        ood_datasets['STL-10'] = stl10
        logger.info(f"STL-10: {len(stl10)} images")
    except Exception:
        try:
            datasets.STL10(root=str(DATA_DIR / 'stl10'), split='test', download=True)
            stl10 = STL10Wrapper(str(DATA_DIR / 'stl10'), split='test')
            ood_datasets['STL-10'] = stl10
            logger.info(f"STL-10: {len(stl10)} images")
        except Exception as e:
            logger.warning(f"Could not load STL-10: {e}")

    return id_dataset, ood_datasets


def run_ood_experiment(model, model_key, id_dataset, ood_datasets, seed):
    """Run OOD detection for one model and seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Split ID into calibration (1000) and eval
    n_cal = 1000
    n_total = len(id_dataset)
    perm = np.random.permutation(n_total)
    cal_indices = perm[:n_cal]
    eval_indices = perm[n_cal:]

    cal_subset = torch.utils.data.Subset(id_dataset, cal_indices)
    eval_subset = torch.utils.data.Subset(id_dataset, eval_indices)

    cal_loader = DataLoader(cal_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    results = {}

    if is_swin_model(model_key):
        # Swin: only feature/logit-based methods (no AEP - windowed attention)
        logger.info(f"  Extracting features for Swin model...")
        cal_features, cal_logits, cal_labels = extract_features_and_logits(model, cal_loader, model_key)
        eval_features, eval_logits, eval_labels = extract_features_and_logits(model, eval_loader, model_key)

        # Compute accuracy
        predictions = eval_logits.argmax(axis=1)
        accuracy = (predictions == eval_labels).mean()
        logger.info(f"  {model_key} accuracy on ImageNet-V2: {accuracy:.4f}")

        for ood_name, ood_dataset in ood_datasets.items():
            ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            ood_features, ood_logits, ood_labels = extract_features_and_logits(model, ood_loader, model_key)

            ood_results = {}
            # MSP
            id_msp = msp_score(eval_logits)
            ood_msp = msp_score(ood_logits)
            ood_results['MSP'] = compute_ood_metrics(id_msp, ood_msp)

            # Energy
            id_energy = energy_score(eval_logits)
            ood_energy = energy_score(ood_logits)
            ood_results['Energy'] = compute_ood_metrics(id_energy, ood_energy)

            # ViM
            id_vim = vim_score(eval_features, eval_logits, cal_features, cal_logits)
            ood_vim = vim_score(ood_features, ood_logits, cal_features, cal_logits)
            ood_results['ViM'] = compute_ood_metrics(id_vim, ood_vim)

            # KNN
            id_knn = knn_score(eval_features, cal_features)
            ood_knn = knn_score(ood_features, cal_features)
            ood_results['KNN'] = compute_ood_metrics(id_knn, ood_knn)

            results[ood_name] = ood_results
            logger.info(f"  {ood_name}: MSP={ood_results['MSP']['AUROC']:.4f}, Energy={ood_results['Energy']['AUROC']:.4f}, ViM={ood_results['ViM']['AUROC']:.4f}, KNN={ood_results['KNN']['AUROC']:.4f}")

        results['_accuracy'] = float(accuracy)
        return results

    # Non-Swin: extract AEP + features + logits
    logger.info(f"  Extracting AEP profiles for {model_key}...")
    cal_profiles, cal_features, cal_logits, cal_labels = extract_aep_and_logits(model, cal_loader, model_key)
    eval_profiles, eval_features, eval_logits, eval_labels = extract_aep_and_logits(model, eval_loader, model_key)

    # ID statistics for AEP
    id_stats = compute_id_statistics(cal_profiles)
    eval_aep_scores = compute_mahalanobis_scores(eval_profiles, id_stats)

    # Compute accuracy
    predictions = eval_logits.argmax(axis=1)
    accuracy = (predictions == eval_labels).mean()
    logger.info(f"  {model_key} accuracy on ImageNet-V2: {accuracy:.4f}")

    for ood_name, ood_dataset in ood_datasets.items():
        ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        ood_profiles, ood_features, ood_logits, ood_labels = extract_aep_and_logits(model, ood_loader, model_key)
        ood_aep_scores = compute_mahalanobis_scores(ood_profiles, id_stats)

        ood_results = {}

        # MSP
        id_msp = msp_score(eval_logits)
        ood_msp = msp_score(ood_logits)
        ood_results['MSP'] = compute_ood_metrics(id_msp, ood_msp)

        # Energy
        id_energy = energy_score(eval_logits)
        ood_energy = energy_score(ood_logits)
        ood_results['Energy'] = compute_ood_metrics(id_energy, ood_energy)

        # ViM
        id_vim = vim_score(eval_features, eval_logits, cal_features, cal_logits)
        ood_vim = vim_score(ood_features, ood_logits, cal_features, cal_logits)
        ood_results['ViM'] = compute_ood_metrics(id_vim, ood_vim)

        # KNN
        id_knn = knn_score(eval_features, cal_features)
        ood_knn = knn_score(ood_features, cal_features)
        ood_results['KNN'] = compute_ood_metrics(id_knn, ood_knn)

        # AEP
        ood_results['AEP'] = compute_ood_metrics(eval_aep_scores, ood_aep_scores)

        # AEP+Fusion (with best baseline = KNN)
        fused_id, fused_ood, best_beta = fuse_scores(
            eval_aep_scores, id_knn, eval_aep_scores, id_knn, ood_aep_scores, ood_knn
        )
        ood_results['AEP+Fusion'] = compute_ood_metrics(fused_id, fused_ood)
        ood_results['AEP+Fusion']['beta'] = float(best_beta)

        results[ood_name] = ood_results
        logger.info(f"  {ood_name}: MSP={ood_results['MSP']['AUROC']:.4f}, Energy={ood_results['Energy']['AUROC']:.4f}, KNN={ood_results['KNN']['AUROC']:.4f}, AEP={ood_results['AEP']['AUROC']:.4f}, Fusion={ood_results['AEP+Fusion']['AUROC']:.4f}")

    results['_accuracy'] = float(accuracy)
    results['_cal_profiles'] = cal_profiles.tolist()
    results['_eval_profiles'] = eval_profiles.tolist()
    results['_eval_labels'] = eval_labels.tolist()
    results['_eval_logits_shape'] = list(eval_logits.shape)
    results['_cal_logits'] = cal_logits.tolist()
    results['_cal_labels'] = cal_labels.tolist()

    return results


def run_calibration_experiment(model, model_key, id_dataset, seed):
    """Run calibration evaluation on corrupted images."""
    if is_swin_model(model_key):
        return {}

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use a subset for calibration evaluation
    n_cal = 1000
    n_eval = 2000
    n_total = len(id_dataset)
    perm = np.random.permutation(n_total)
    cal_indices = perm[:n_cal]
    eval_indices = perm[n_cal:n_cal + n_eval]

    cal_subset = torch.utils.data.Subset(id_dataset, cal_indices)
    eval_subset = torch.utils.data.Subset(id_dataset, eval_indices)

    cal_loader = DataLoader(cal_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Get calibration data
    cal_profiles, cal_features, cal_logits, cal_labels = extract_aep_and_logits(model, cal_loader, model_key)
    id_stats = compute_id_statistics(cal_profiles)
    cal_aep_scores = compute_mahalanobis_scores(cal_profiles, id_stats)

    # Fit calibration methods on clean calibration data
    global_temp = fit_temperature_scaling(cal_logits, cal_labels)
    adaptive_params = fit_adaptive_temperature(cal_logits, cal_labels, cal_aep_scores)

    results = {}

    # Clean evaluation
    eval_profiles, eval_features, eval_logits, eval_labels = extract_aep_and_logits(model, eval_loader, model_key)
    eval_aep_scores = compute_mahalanobis_scores(eval_profiles, id_stats)

    clean_preds = eval_logits.argmax(axis=1)
    clean_acc = (clean_preds == eval_labels).mean()

    # Calibration on clean data
    from scipy.special import softmax as sp_softmax
    raw_probs = sp_softmax(eval_logits, axis=1)
    ts_probs = sp_softmax(eval_logits / global_temp, axis=1)
    adaptive_probs = apply_adaptive_temperature(eval_logits, eval_aep_scores, adaptive_params)

    raw_conf = raw_probs.max(axis=1)
    ts_conf = ts_probs.max(axis=1)
    adaptive_conf = adaptive_probs.max(axis=1)
    correct = (eval_logits.argmax(axis=1) == eval_labels).astype(float)

    results['clean'] = {
        'accuracy': float(clean_acc),
        'Raw': {'ECE': float(compute_ece(raw_conf, correct)), 'MCE': float(compute_mce(raw_conf, correct))},
        'TempScaling': {'ECE': float(compute_ece(ts_conf, correct)), 'MCE': float(compute_mce(ts_conf, correct)), 'T': float(global_temp)},
        'AEP_Adaptive': {'ECE': float(compute_ece(adaptive_conf, correct)), 'MCE': float(compute_mce(adaptive_conf, correct)), **{k: float(v) for k, v in adaptive_params.items()}},
    }
    logger.info(f"  Clean: acc={clean_acc:.4f}, Raw ECE={results['clean']['Raw']['ECE']:.4f}, TS ECE={results['clean']['TempScaling']['ECE']:.4f}, Adaptive ECE={results['clean']['AEP_Adaptive']['ECE']:.4f}")

    # Corrupted evaluation
    for corruption_name, corruption_fn in CORRUPTIONS.items():
        for severity in [1, 3, 5]:
            key = f"{corruption_name}_s{severity}"
            corrupted_ds = CorruptedDataset(eval_subset, corruption_fn, severity)
            corr_loader = DataLoader(corrupted_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

            corr_profiles, corr_features, corr_logits, corr_labels = extract_aep_and_logits(model, corr_loader, model_key)
            corr_aep_scores = compute_mahalanobis_scores(corr_profiles, id_stats)

            corr_preds = corr_logits.argmax(axis=1)
            corr_acc = (corr_preds == corr_labels).mean()

            raw_probs_c = sp_softmax(corr_logits, axis=1)
            ts_probs_c = sp_softmax(corr_logits / global_temp, axis=1)
            adaptive_probs_c = apply_adaptive_temperature(corr_logits, corr_aep_scores, adaptive_params)

            raw_conf_c = raw_probs_c.max(axis=1)
            ts_conf_c = ts_probs_c.max(axis=1)
            adaptive_conf_c = adaptive_probs_c.max(axis=1)
            correct_c = (corr_logits.argmax(axis=1) == corr_labels).astype(float)

            results[key] = {
                'accuracy': float(corr_acc),
                'Raw': {'ECE': float(compute_ece(raw_conf_c, correct_c)), 'MCE': float(compute_mce(raw_conf_c, correct_c))},
                'TempScaling': {'ECE': float(compute_ece(ts_conf_c, correct_c)), 'MCE': float(compute_mce(ts_conf_c, correct_c))},
                'AEP_Adaptive': {'ECE': float(compute_ece(adaptive_conf_c, correct_c)), 'MCE': float(compute_mce(adaptive_conf_c, correct_c))},
            }
            logger.info(f"  {key}: acc={corr_acc:.4f}, Raw ECE={results[key]['Raw']['ECE']:.4f}, TS ECE={results[key]['TempScaling']['ECE']:.4f}, Adaptive ECE={results[key]['AEP_Adaptive']['ECE']:.4f}")

    return results


def run_ablation_components(model, model_key, id_dataset, ood_datasets):
    """Component ablation study."""
    if is_swin_model(model_key):
        return {}

    np.random.seed(42)
    torch.manual_seed(42)

    n_cal = 1000
    perm = np.random.permutation(len(id_dataset))
    cal_indices = perm[:n_cal]
    eval_indices = perm[n_cal:]

    cal_subset = torch.utils.data.Subset(id_dataset, cal_indices)
    eval_subset = torch.utils.data.Subset(id_dataset, eval_indices)
    cal_loader = DataLoader(cal_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Extract attention maps (we need raw attention for ablation)
    extractor = AEPExtractor(model, MODEL_CONFIGS[model_key]['num_layers'])

    # Collect all attention maps for cal and eval
    def collect_attn_maps(loader):
        all_attn_maps_per_sample = []
        for images, labels in tqdm(loader, desc="Collecting attention maps"):
            images = images.to(DEVICE)
            logits, attn_maps = extractor._extract_with_monkey_patch(images)
            batch_size = images.shape[0]
            for b_idx in range(batch_size):
                sample_maps = [am[b_idx:b_idx+1] for am in attn_maps]
                all_attn_maps_per_sample.append(sample_maps)
        return all_attn_maps_per_sample

    logger.info("  Collecting ID calibration attention maps...")
    cal_attn = collect_attn_maps(cal_loader)
    logger.info("  Collecting ID eval attention maps...")
    eval_attn = collect_attn_maps(eval_loader)

    # Feature subsets: 0=cls_ent_mean, 1=cls_ent_std, 2=avg_token_ent, 3=concentration, 4=head_agreement
    ablation_configs = {
        'Full': None,  # All features
        'No_CLS_entropy': [2, 3, 4],  # Remove 0, 1
        'No_avg_token_entropy': [0, 1, 3, 4],
        'No_concentration': [0, 1, 2, 4],
        'No_head_agreement': [0, 1, 2, 3],
        'CLS_entropy_only': [0, 1],
    }

    results = {}

    for config_name, feature_indices in ablation_configs.items():
        logger.info(f"  Ablation: {config_name}")

        # Compute profiles with feature subset
        def compute_profiles_from_attn(attn_list, feature_indices):
            profiles = []
            for sample_maps in attn_list:
                if feature_indices is not None:
                    p = compute_aep_profile_subset(sample_maps, feature_indices=feature_indices)
                else:
                    p = compute_aep_profile(sample_maps)
                profiles.append(p)
            return np.concatenate(profiles)

        cal_profiles = compute_profiles_from_attn(cal_attn, feature_indices)
        eval_profiles = compute_profiles_from_attn(eval_attn, feature_indices)
        id_stats = compute_id_statistics(cal_profiles)
        eval_scores = compute_mahalanobis_scores(eval_profiles, id_stats)

        config_results = {}
        for ood_name, ood_dataset in ood_datasets.items():
            ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

            ood_attn = collect_attn_maps(ood_loader)
            ood_profiles = compute_profiles_from_attn(ood_attn, feature_indices)
            ood_scores = compute_mahalanobis_scores(ood_profiles, id_stats)

            metrics = compute_ood_metrics(eval_scores, ood_scores)
            config_results[ood_name] = metrics
            logger.info(f"    {ood_name}: AUROC={metrics['AUROC']:.4f}")

        results[config_name] = config_results

    extractor.remove_hooks()
    return results


def run_statistical_tests(model, model_key, id_dataset, ood_datasets):
    """Run statistical significance tests."""
    if is_swin_model(model_key):
        return {}

    np.random.seed(42)
    torch.manual_seed(42)

    n_cal = 1000
    perm = np.random.permutation(len(id_dataset))
    cal_indices = perm[:n_cal]
    eval_indices = perm[n_cal:]

    eval_subset = torch.utils.data.Subset(id_dataset, eval_indices)
    eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    eval_profiles, _, _, _ = extract_aep_and_logits(model, eval_loader, model_key)

    results = {}
    for ood_name, ood_dataset in ood_datasets.items():
        ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        ood_profiles, _, _, _ = extract_aep_and_logits(model, ood_loader, model_key)

        # Per-dimension t-tests with Bonferroni correction
        n_dims = eval_profiles.shape[1]
        significant_dims = 0
        for d in range(n_dims):
            t_stat, p_val = stats.ttest_ind(eval_profiles[:, d], ood_profiles[:, d])
            if p_val * n_dims < 0.01:  # Bonferroni correction
                significant_dims += 1

        # Hotelling's T² approximation
        n1 = len(eval_profiles)
        n2 = len(ood_profiles)
        diff = eval_profiles.mean(axis=0) - ood_profiles.mean(axis=0)
        s1 = np.cov(eval_profiles, rowvar=False)
        s2 = np.cov(ood_profiles, rowvar=False)
        s_pooled = ((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2) + 1e-6*np.eye(n_dims)
        t2 = (n1*n2)/(n1+n2) * diff @ np.linalg.solve(s_pooled, diff)

        p = n_dims
        n = n1 + n2
        f_stat = t2 * (n - p - 1) / (p * (n - 2))
        # p-value would be extremely small

        results[ood_name] = {
            'significant_dims': significant_dims,
            'total_dims': n_dims,
            'hotelling_t2': float(t2),
            'f_statistic': float(f_stat),
        }
        logger.info(f"  {ood_name}: {significant_dims}/{n_dims} significant dims, T²={t2:.1f}")

    return results


def run_overhead_benchmark(id_dataset):
    """Benchmark computational overhead."""
    results = {}
    test_images = torch.stack([id_dataset[i][0] for i in range(100)]).to(DEVICE)

    for model_key in ['deit_small', 'deit_base', 'vit_base', 'swin_tiny']:
        model = load_model(model_key)
        cfg = MODEL_CONFIGS[model_key]

        # Standard forward pass
        torch.cuda.synchronize()
        times_standard = []
        for _ in range(5):
            start = time.time()
            with torch.no_grad():
                for i in range(0, 100, BATCH_SIZE):
                    batch = test_images[i:i+BATCH_SIZE]
                    _ = model(batch)
            torch.cuda.synchronize()
            times_standard.append(time.time() - start)
        std_time = np.median(times_standard)

        if not is_swin_model(model_key):
            # AEP forward pass
            extractor = AEPExtractor(model, cfg['num_layers'])
            torch.cuda.synchronize()
            times_aep = []
            for _ in range(5):
                start = time.time()
                with torch.no_grad():
                    for i in range(0, 100, BATCH_SIZE):
                        batch = test_images[i:i+BATCH_SIZE]
                        logits, attn_maps = extractor._extract_with_monkey_patch(batch)
                        profiles = compute_aep_profile(attn_maps)
                torch.cuda.synchronize()
                times_aep.append(time.time() - start)
            aep_time = np.median(times_aep)
            extractor.remove_hooks()

            results[model_key] = {
                'standard_ms_per_img': float(std_time / 100 * 1000),
                'aep_ms_per_img': float(aep_time / 100 * 1000),
                'overhead_pct': float((aep_time / std_time - 1) * 100),
                'multiplier': float(aep_time / std_time),
            }
        else:
            results[model_key] = {
                'standard_ms_per_img': float(std_time / 100 * 1000),
                'note': 'Swin uses windowed attention; AEP not applicable',
            }

        logger.info(f"  {model_key}: {results[model_key]}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def aggregate_results(all_results):
    """Aggregate results across seeds."""
    aggregated = {}

    for model_key in all_results:
        model_results = all_results[model_key]
        seeds = [s for s in model_results.keys() if isinstance(s, int)]

        aggregated[model_key] = {}
        # Get OOD dataset names from first seed
        first_seed = seeds[0]
        ood_names = [k for k in model_results[first_seed].keys() if not k.startswith('_')]

        for ood_name in ood_names:
            aggregated[model_key][ood_name] = {}
            methods = model_results[first_seed][ood_name].keys()

            for method in methods:
                method_metrics = {}
                for metric in model_results[first_seed][ood_name][method]:
                    if metric == 'beta':
                        continue
                    vals = [model_results[s][ood_name][method][metric] for s in seeds if ood_name in model_results[s] and method in model_results[s][ood_name]]
                    method_metrics[metric] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                    }
                aggregated[model_key][ood_name][method] = method_metrics

        # Accuracy
        accs = [model_results[s].get('_accuracy', 0) for s in seeds]
        aggregated[model_key]['_accuracy'] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs))}

    return aggregated


def main():
    logger.info("=" * 80)
    logger.info("AEP Experiments v2: Proper Evaluation Setup")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")

    # Load datasets
    id_dataset, ood_datasets = load_datasets()
    logger.info(f"ID dataset: ImageNet-V2 ({len(id_dataset)} images)")
    logger.info(f"OOD datasets: {list(ood_datasets.keys())}")

    # =====================
    # 1. Main OOD Detection
    # =====================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: OOD Detection")
    logger.info("=" * 60)

    all_ood_results = {}
    for model_key in ['deit_small', 'deit_base', 'vit_base', 'swin_tiny']:
        logger.info(f"\nModel: {model_key}")
        model = load_model(model_key)

        all_ood_results[model_key] = {}
        for seed in SEEDS:
            logger.info(f"  Seed: {seed}")
            results = run_ood_experiment(model, model_key, id_dataset, ood_datasets, seed)
            # Remove large arrays before saving
            results_clean = {k: v for k, v in results.items()
                           if not k.startswith('_') or k == '_accuracy'}
            all_ood_results[model_key][seed] = results_clean

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save raw results
    with open(RESULTS_DIR / 'ood_results_v2.json', 'w') as f:
        json.dump(all_ood_results, f, indent=2, default=str)

    # Aggregate
    aggregated = aggregate_results(all_ood_results)
    with open(RESULTS_DIR / 'ood_results_v2_aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    logger.info("\nOOD results saved.")

    # =====================
    # 2. Calibration
    # =====================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Calibration under Distribution Shift")
    logger.info("=" * 60)

    cal_results = {}
    for model_key in ['vit_base']:  # Primary model for calibration
        logger.info(f"\nModel: {model_key}")
        model = load_model(model_key)
        cal_results[model_key] = run_calibration_experiment(model, model_key, id_dataset, seed=42)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(RESULTS_DIR / 'calibration_results_v2.json', 'w') as f:
        json.dump(cal_results, f, indent=2)

    # =====================
    # 3. Ablation Studies
    # =====================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Ablation Studies")
    logger.info("=" * 60)

    model = load_model('vit_base')
    ablation_results = run_ablation_components(model, 'vit_base', id_dataset, ood_datasets)
    with open(RESULTS_DIR / 'ablation_components_v2.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =====================
    # 4. Statistical Tests
    # =====================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Statistical Significance")
    logger.info("=" * 60)

    model = load_model('vit_base')
    stat_results = run_statistical_tests(model, 'vit_base', id_dataset, ood_datasets)
    with open(RESULTS_DIR / 'statistical_tests_v2.json', 'w') as f:
        json.dump(stat_results, f, indent=2)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =====================
    # 5. Overhead Benchmark
    # =====================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5: Computational Overhead")
    logger.info("=" * 60)

    overhead_results = run_overhead_benchmark(id_dataset)
    with open(RESULTS_DIR / 'overhead_v2.json', 'w') as f:
        json.dump(overhead_results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
