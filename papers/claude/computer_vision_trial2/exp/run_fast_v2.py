#!/usr/bin/env python3
"""Fast AEP experiments v2 with proper evaluation setup.

Uses ImageNet-V2 (subset) as ID where models have real accuracy.
Addresses all reviewer concerns efficiently.
"""

import os
import sys
import json
import time
import gc
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
import timm
from tqdm import tqdm
from scipy import stats
from scipy.special import softmax as sp_softmax

sys.path.insert(0, str(Path(__file__).parent))
from shared.aep import (AEPExtractor, compute_aep_profile, compute_id_statistics,
                         compute_mahalanobis_scores, compute_aep_profile_subset)
from shared.metrics import (compute_ood_metrics, compute_ece, compute_mce)
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
BATCH_SIZE = 128  # Larger batches for speed

# Use 5K ID images: 1K cal + 4K eval (sufficient for reliable OOD metrics)
N_ID_IMAGES = 5000
N_CAL = 1000

MODEL_CONFIGS = {
    'deit_small': {'name': 'deit_small_patch16_224', 'num_layers': 12, 'feat_dim': 384},
    'deit_base': {'name': 'deit_base_patch16_224', 'num_layers': 12, 'feat_dim': 768},
    'vit_base': {'name': 'vit_base_patch16_224', 'num_layers': 12, 'feat_dim': 768},
    'swin_tiny': {'name': 'swin_tiny_patch4_window7_224', 'num_layers': None, 'feat_dim': 768},
}

TRANSFORM_224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageNetV2Dataset(Dataset):
    """ImageNet-V2 from cached files."""
    def __init__(self, cache_dir, transform, max_samples=None):
        self.transform = transform
        cache_path = Path(cache_dir) / 'imagenetv2_images'
        labels_path = Path(cache_dir) / 'imagenetv2_labels.json'

        with open(labels_path) as f:
            self.labels = json.load(f)
        self.image_paths = sorted(cache_path.glob('*.jpg'))

        if max_samples and max_samples < len(self.labels):
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(self.labels))[:max_samples]
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(str(self.image_paths[idx])).convert('RGB')
        return self.transform(img), self.labels[idx]


class STL10Wrapper(Dataset):
    def __init__(self, root, split='test'):
        self.ds = datasets.STL10(root=root, split=split, download=False)
        self.transform = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        from PIL import Image
        img, label = self.ds[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.transform(img.convert('RGB')), label


def load_model(model_key):
    cfg = MODEL_CONFIGS[model_key]
    model = timm.create_model(cfg['name'], pretrained=True)
    model = model.to(DEVICE).eval()
    return model


def is_swin(model_key):
    return 'swin' in model_key


def extract_all(model, dataloader, model_key, with_aep=True):
    """Extract features, logits, and optionally AEP profiles in one pass."""
    all_logits = []
    all_labels = []
    all_profiles = []
    features_store = []

    # Hook for penultimate features
    hook = model.head.register_forward_pre_hook(
        lambda mod, inp: features_store.append(inp[0].detach().cpu()))

    if with_aep and not is_swin(model_key):
        extractor = AEPExtractor(model, MODEL_CONFIGS[model_key]['num_layers'])
        for images, labels in tqdm(dataloader, desc=f"Extract ({model_key})", leave=False):
            images = images.to(DEVICE)
            logits, attn_maps = extractor._extract_with_monkey_patch(images)
            profiles = compute_aep_profile(attn_maps)
            all_profiles.append(profiles)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(np.array(labels) if not isinstance(labels, np.ndarray) else labels)
        extractor.remove_hooks()
    else:
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Extract ({model_key})", leave=False):
                images = images.to(DEVICE)
                logits = model(images)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(np.array(labels) if not isinstance(labels, np.ndarray) else labels)

    hook.remove()

    result = {
        'logits': np.concatenate(all_logits),
        'labels': np.concatenate(all_labels),
        'features': np.concatenate([f.numpy() for f in features_store]),
    }
    if result['features'].ndim == 3:
        result['features'] = result['features'][:, 0, :]
    if all_profiles:
        result['profiles'] = np.concatenate(all_profiles)
    return result


def compute_ood_scores(id_data, ood_data, cal_data, has_aep=True):
    """Compute all OOD scores."""
    results = {}

    # MSP
    results['MSP'] = compute_ood_metrics(
        msp_score(id_data['logits']), msp_score(ood_data['logits']))

    # Energy
    results['Energy'] = compute_ood_metrics(
        energy_score(id_data['logits']), energy_score(ood_data['logits']))

    # ViM
    results['ViM'] = compute_ood_metrics(
        vim_score(id_data['features'], id_data['logits'], cal_data['features'], cal_data['logits']),
        vim_score(ood_data['features'], ood_data['logits'], cal_data['features'], cal_data['logits']))

    # KNN
    id_knn = knn_score(id_data['features'], cal_data['features'])
    ood_knn = knn_score(ood_data['features'], cal_data['features'])
    results['KNN'] = compute_ood_metrics(id_knn, ood_knn)

    if has_aep and 'profiles' in id_data:
        id_stats = compute_id_statistics(cal_data['profiles'])
        id_aep = compute_mahalanobis_scores(id_data['profiles'], id_stats)
        ood_aep = compute_mahalanobis_scores(ood_data['profiles'], id_stats)
        results['AEP'] = compute_ood_metrics(id_aep, ood_aep)

        # Fusion with KNN
        fused_id, fused_ood, beta = fuse_scores(id_aep, id_knn, id_aep, id_knn, ood_aep, ood_knn)
        results['AEP+Fusion'] = compute_ood_metrics(fused_id, fused_ood)
        results['AEP+Fusion']['beta'] = float(beta)

    return results


def gaussian_noise(img_np, severity):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    return np.clip(img_np + np.random.normal(0, c * 255, img_np.shape), 0, 255).astype(np.uint8)

def gaussian_blur(img_np, severity):
    from scipy.ndimage import gaussian_filter
    c = [1.0, 2.0, 3.0, 4.0, 6.0][severity - 1]
    return np.clip(gaussian_filter(img_np.astype(np.float32), sigma=(c, c, 0)), 0, 255).astype(np.uint8)

def shot_noise(img_np, severity):
    c = [60, 25, 12, 5, 3][severity - 1]
    return np.clip(np.random.poisson(img_np.astype(np.float32) / 255.0 * c) / c * 255, 0, 255).astype(np.uint8)

def contrast_shift(img_np, severity):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    mean = img_np.mean(axis=(0, 1), keepdims=True)
    return np.clip((img_np - mean) * c + mean, 0, 255).astype(np.uint8)

def brightness_shift(img_np, severity):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    return np.clip(img_np.astype(np.float32) + c * 255, 0, 255).astype(np.uint8)


class CorruptedDataset(Dataset):
    """Apply corruption to pre-loaded tensor dataset."""
    def __init__(self, base_dataset, corruption_fn, severity):
        self.base = base_dataset
        self.fn = corruption_fn
        self.severity = severity
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_tensor, label = self.base[idx]
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = ((img_tensor * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        corrupted = self.fn(img, self.severity)
        tensor = self.normalize(torch.from_numpy(corrupted).permute(2, 0, 1).float() / 255.0)
        return tensor, label


def main():
    logger.info("=" * 60)
    logger.info("AEP Experiments v2 (Fast)")
    logger.info("=" * 60)

    # Load datasets
    logger.info("Loading datasets...")
    id_dataset = ImageNetV2Dataset(CACHE_DIR, TRANSFORM_224, max_samples=N_ID_IMAGES)
    logger.info(f"ImageNet-V2: {len(id_dataset)} images (subset)")

    ood_datasets = {}
    try:
        ood_datasets['Textures'] = datasets.DTD(root=str(DATA_DIR / 'dtd'), split='test',
                                                  transform=TRANSFORM_224, download=False)
    except Exception:
        ood_datasets['Textures'] = datasets.DTD(root=str(DATA_DIR / 'dtd'), split='test',
                                                  transform=TRANSFORM_224, download=True)

    try:
        ood_datasets['Flowers-102'] = datasets.Flowers102(root=str(DATA_DIR / 'flowers102'), split='test',
                                                            transform=TRANSFORM_224, download=False)
    except Exception:
        ood_datasets['Flowers-102'] = datasets.Flowers102(root=str(DATA_DIR / 'flowers102'), split='test',
                                                            transform=TRANSFORM_224, download=True)

    try:
        ood_datasets['Oxford-Pets'] = datasets.OxfordIIITPet(root=str(DATA_DIR / 'pets'), split='test',
                                                               transform=TRANSFORM_224, download=False)
    except Exception:
        ood_datasets['Oxford-Pets'] = datasets.OxfordIIITPet(root=str(DATA_DIR / 'pets'), split='test',
                                                               transform=TRANSFORM_224, download=True)

    try:
        ood_datasets['EuroSAT'] = datasets.EuroSAT(root=str(DATA_DIR / 'eurosat'),
                                                      transform=TRANSFORM_224, download=False)
    except Exception:
        ood_datasets['EuroSAT'] = datasets.EuroSAT(root=str(DATA_DIR / 'eurosat'),
                                                      transform=TRANSFORM_224, download=True)

    # Subsample large OOD datasets for speed
    for name in ood_datasets:
        ds = ood_datasets[name]
        if len(ds) > 5000:
            indices = np.random.RandomState(42).permutation(len(ds))[:5000]
            ood_datasets[name] = Subset(ds, indices)
        logger.info(f"  {name}: {len(ood_datasets[name])} images")

    # ==========================================
    # PHASE 1: OOD Detection (all models × seeds)
    # ==========================================
    logger.info("\n=== PHASE 1: OOD Detection ===")

    all_results = {}
    for model_key in ['deit_small', 'deit_base', 'vit_base', 'swin_tiny']:
        logger.info(f"\nModel: {model_key}")
        model = load_model(model_key)
        has_aep = not is_swin(model_key)
        all_results[model_key] = {}

        for seed in SEEDS:
            logger.info(f"  Seed: {seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Split ID
            perm = np.random.permutation(len(id_dataset))
            cal_subset = Subset(id_dataset, perm[:N_CAL])
            eval_subset = Subset(id_dataset, perm[N_CAL:])

            cal_loader = DataLoader(cal_subset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=4, pin_memory=True)
            eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=4, pin_memory=True)

            cal_data = extract_all(model, cal_loader, model_key, with_aep=has_aep)
            eval_data = extract_all(model, eval_loader, model_key, with_aep=has_aep)

            # Accuracy
            acc = (eval_data['logits'].argmax(axis=1) == eval_data['labels']).mean()
            logger.info(f"    Accuracy: {acc:.4f}")

            seed_results = {'_accuracy': float(acc)}

            for ood_name, ood_ds in ood_datasets.items():
                ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=4, pin_memory=True)
                ood_data = extract_all(model, ood_loader, model_key, with_aep=has_aep)
                metrics = compute_ood_scores(eval_data, ood_data, cal_data, has_aep=has_aep)
                seed_results[ood_name] = metrics

                aep_str = f", AEP={metrics['AEP']['AUROC']:.3f}" if 'AEP' in metrics else ""
                fusion_str = f", Fus={metrics['AEP+Fusion']['AUROC']:.3f}" if 'AEP+Fusion' in metrics else ""
                logger.info(f"    {ood_name}: MSP={metrics['MSP']['AUROC']:.3f}, E={metrics['Energy']['AUROC']:.3f}, "
                           f"ViM={metrics['ViM']['AUROC']:.3f}, KNN={metrics['KNN']['AUROC']:.3f}{aep_str}{fusion_str}")

            all_results[model_key][seed] = seed_results

        del model; gc.collect(); torch.cuda.empty_cache()

    # Save and aggregate
    with open(RESULTS_DIR / 'ood_results_v2.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Aggregate across seeds
    aggregated = {}
    for model_key in all_results:
        aggregated[model_key] = {}
        seeds_data = all_results[model_key]
        ood_names = [k for k in seeds_data[SEEDS[0]].keys() if not k.startswith('_')]

        for ood_name in ood_names:
            aggregated[model_key][ood_name] = {}
            for method in seeds_data[SEEDS[0]][ood_name]:
                if method == 'beta':
                    continue
                agg_metrics = {}
                for metric in ['AUROC', 'FPR95', 'AUPR_in', 'AUPR_out']:
                    vals = []
                    for s in SEEDS:
                        if ood_name in seeds_data[s] and method in seeds_data[s][ood_name]:
                            if metric in seeds_data[s][ood_name][method]:
                                vals.append(seeds_data[s][ood_name][method][metric])
                    if vals:
                        agg_metrics[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                aggregated[model_key][ood_name][method] = agg_metrics

        accs = [seeds_data[s]['_accuracy'] for s in SEEDS]
        aggregated[model_key]['_accuracy'] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs))}

    with open(RESULTS_DIR / 'ood_results_v2_aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    logger.info("\nOOD results saved.")

    # ==========================================
    # PHASE 2: Calibration (ViT-Base only, seed 42)
    # ==========================================
    logger.info("\n=== PHASE 2: Calibration ===")
    model = load_model('vit_base')
    np.random.seed(42); torch.manual_seed(42)

    perm = np.random.permutation(len(id_dataset))
    cal_subset = Subset(id_dataset, perm[:N_CAL])
    eval_subset = Subset(id_dataset, perm[N_CAL:N_CAL + 2000])

    cal_loader = DataLoader(cal_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    cal_data = extract_all(model, cal_loader, 'vit_base', with_aep=True)
    id_stats = compute_id_statistics(cal_data['profiles'])
    cal_aep_scores = compute_mahalanobis_scores(cal_data['profiles'], id_stats)

    global_temp = fit_temperature_scaling(cal_data['logits'], cal_data['labels'])
    adaptive_params = fit_adaptive_temperature(cal_data['logits'], cal_data['labels'], cal_aep_scores)

    # Evaluate on clean and corrupted
    cal_results = {}
    corruptions = {
        'gaussian_noise': gaussian_noise,
        'gaussian_blur': gaussian_blur,
        'shot_noise': shot_noise,
        'contrast': contrast_shift,
        'brightness': brightness_shift,
    }

    for condition_name, condition_loader, condition_subset in [('clean', eval_loader, eval_subset)]:
        data = extract_all(model, condition_loader, 'vit_base', with_aep=True)
        aep_scores = compute_mahalanobis_scores(data['profiles'], id_stats)
        acc = (data['logits'].argmax(axis=1) == data['labels']).mean()

        raw_probs = sp_softmax(data['logits'], axis=1)
        ts_probs = sp_softmax(data['logits'] / global_temp, axis=1)
        adap_probs = apply_adaptive_temperature(data['logits'], aep_scores, adaptive_params)
        correct = (data['logits'].argmax(axis=1) == data['labels']).astype(float)

        cal_results['clean'] = {
            'accuracy': float(acc),
            'Raw': {'ECE': float(compute_ece(raw_probs.max(1), correct))},
            'TempScaling': {'ECE': float(compute_ece(ts_probs.max(1), correct)), 'T': float(global_temp)},
            'AEP_Adaptive': {'ECE': float(compute_ece(adap_probs.max(1), correct))},
        }
        logger.info(f"  Clean: acc={acc:.4f}, Raw ECE={cal_results['clean']['Raw']['ECE']:.4f}, "
                    f"TS ECE={cal_results['clean']['TempScaling']['ECE']:.4f}, "
                    f"Adap ECE={cal_results['clean']['AEP_Adaptive']['ECE']:.4f}")

    for corr_name, corr_fn in corruptions.items():
        for severity in [1, 3, 5]:
            key = f"{corr_name}_s{severity}"
            corr_ds = CorruptedDataset(eval_subset, corr_fn, severity)
            corr_loader = DataLoader(corr_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

            data = extract_all(model, corr_loader, 'vit_base', with_aep=True)
            aep_scores = compute_mahalanobis_scores(data['profiles'], id_stats)
            acc = (data['logits'].argmax(axis=1) == data['labels']).mean()

            raw_probs = sp_softmax(data['logits'], axis=1)
            ts_probs = sp_softmax(data['logits'] / global_temp, axis=1)
            adap_probs = apply_adaptive_temperature(data['logits'], aep_scores, adaptive_params)
            correct = (data['logits'].argmax(axis=1) == data['labels']).astype(float)

            cal_results[key] = {
                'accuracy': float(acc),
                'Raw': {'ECE': float(compute_ece(raw_probs.max(1), correct))},
                'TempScaling': {'ECE': float(compute_ece(ts_probs.max(1), correct))},
                'AEP_Adaptive': {'ECE': float(compute_ece(adap_probs.max(1), correct))},
            }
            logger.info(f"  {key}: acc={acc:.4f}, Raw={cal_results[key]['Raw']['ECE']:.4f}, "
                       f"TS={cal_results[key]['TempScaling']['ECE']:.4f}, Adap={cal_results[key]['AEP_Adaptive']['ECE']:.4f}")

    with open(RESULTS_DIR / 'calibration_results_v2.json', 'w') as f:
        json.dump({'vit_base': cal_results}, f, indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()

    # ==========================================
    # PHASE 3: Ablation (ViT-Base, seed 42)
    # ==========================================
    logger.info("\n=== PHASE 3: Component Ablation ===")
    model = load_model('vit_base')
    np.random.seed(42); torch.manual_seed(42)

    perm = np.random.permutation(len(id_dataset))
    cal_subset = Subset(id_dataset, perm[:N_CAL])
    eval_subset = Subset(id_dataset, perm[N_CAL:])

    extractor = AEPExtractor(model, 12)

    def collect_attn_batch(loader):
        """Collect raw attention maps."""
        all_maps = []
        all_logits = []
        all_labels = []
        for images, labels in tqdm(loader, desc="Attn maps", leave=False):
            images = images.to(DEVICE)
            logits, attn_maps = extractor._extract_with_monkey_patch(images)
            all_maps.append(attn_maps)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(np.array(labels))
        return all_maps, np.concatenate(all_logits), np.concatenate(all_labels)

    def profiles_from_maps(all_maps, feature_indices=None, layer_indices=None):
        profiles = []
        for batch_maps in all_maps:
            if layer_indices is not None:
                batch_maps = [batch_maps[i] for i in layer_indices]
            if feature_indices is not None:
                p = compute_aep_profile_subset(batch_maps, feature_indices=feature_indices)
            else:
                p = compute_aep_profile(batch_maps)
            profiles.append(p)
        return np.concatenate(profiles)

    cal_loader = DataLoader(cal_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("  Collecting ID attention maps...")
    cal_maps, cal_logits, cal_labels = collect_attn_batch(cal_loader)
    eval_maps, eval_logits, eval_labels = collect_attn_batch(eval_loader)

    ablation_configs = {
        'Full': (None, None),
        'No_CLS_entropy': ([2, 3, 4], None),
        'No_avg_token_entropy': ([0, 1, 3, 4], None),
        'No_concentration': ([0, 1, 2, 4], None),
        'No_head_agreement': ([0, 1, 2, 3], None),
        'CLS_entropy_only': ([0, 1], None),
        'Early_1-4': (None, list(range(4))),
        'Middle_5-8': (None, list(range(4, 8))),
        'Late_9-12': (None, list(range(8, 12))),
    }

    ablation_results = {}
    for config_name, (feat_idx, layer_idx) in ablation_configs.items():
        cal_profiles = profiles_from_maps(cal_maps, feat_idx, layer_idx)
        eval_profiles = profiles_from_maps(eval_maps, feat_idx, layer_idx)
        id_stats = compute_id_statistics(cal_profiles)
        eval_scores = compute_mahalanobis_scores(eval_profiles, id_stats)

        config_results = {}
        for ood_name, ood_ds in ood_datasets.items():
            ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            ood_maps, _, _ = collect_attn_batch(ood_loader)
            ood_profiles = profiles_from_maps(ood_maps, feat_idx, layer_idx)
            ood_scores = compute_mahalanobis_scores(ood_profiles, id_stats)
            config_results[ood_name] = compute_ood_metrics(eval_scores, ood_scores)
            logger.info(f"    {config_name} / {ood_name}: AUROC={config_results[ood_name]['AUROC']:.4f}")

        ablation_results[config_name] = config_results

    with open(RESULTS_DIR / 'ablation_v2.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    extractor.remove_hooks()
    del model; gc.collect(); torch.cuda.empty_cache()

    # ==========================================
    # PHASE 4: Statistical Tests (ViT-Base)
    # ==========================================
    logger.info("\n=== PHASE 4: Statistical Tests ===")
    model = load_model('vit_base')
    np.random.seed(42); torch.manual_seed(42)

    perm = np.random.permutation(len(id_dataset))
    eval_subset = Subset(id_dataset, perm[N_CAL:])
    eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    eval_data = extract_all(model, eval_loader, 'vit_base', with_aep=True)

    stat_results = {}
    for ood_name, ood_ds in ood_datasets.items():
        ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        ood_data = extract_all(model, ood_loader, 'vit_base', with_aep=True)

        id_profiles = eval_data['profiles']
        ood_profiles = ood_data['profiles']
        n_dims = id_profiles.shape[1]

        sig_dims = 0
        for d in range(n_dims):
            _, p = stats.ttest_ind(id_profiles[:, d], ood_profiles[:, d])
            if p * n_dims < 0.01:
                sig_dims += 1

        # Hotelling's T²
        n1, n2 = len(id_profiles), len(ood_profiles)
        diff = id_profiles.mean(0) - ood_profiles.mean(0)
        s1 = np.cov(id_profiles, rowvar=False)
        s2 = np.cov(ood_profiles, rowvar=False)
        s_pool = ((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2) + 1e-6*np.eye(n_dims)
        t2 = (n1*n2)/(n1+n2) * diff @ np.linalg.solve(s_pool, diff)

        stat_results[ood_name] = {
            'significant_dims': sig_dims, 'total_dims': n_dims,
            'hotelling_t2': float(t2),
        }
        logger.info(f"  {ood_name}: {sig_dims}/{n_dims} sig dims, T²={t2:.0f}")

    with open(RESULTS_DIR / 'statistical_tests_v2.json', 'w') as f:
        json.dump(stat_results, f, indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()

    # ==========================================
    # PHASE 5: Overhead Benchmark
    # ==========================================
    logger.info("\n=== PHASE 5: Overhead ===")
    test_imgs = torch.stack([id_dataset[i][0] for i in range(100)]).to(DEVICE)

    overhead_results = {}
    for model_key in MODEL_CONFIGS:
        model = load_model(model_key)

        # Standard
        torch.cuda.synchronize()
        times = []
        for _ in range(5):
            t0 = time.time()
            with torch.no_grad():
                for i in range(0, 100, BATCH_SIZE):
                    _ = model(test_imgs[i:i+BATCH_SIZE])
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        std_time = np.median(times)

        if not is_swin(model_key):
            ext = AEPExtractor(model, MODEL_CONFIGS[model_key]['num_layers'])
            torch.cuda.synchronize()
            times = []
            for _ in range(5):
                t0 = time.time()
                for i in range(0, 100, BATCH_SIZE):
                    logits, attn = ext._extract_with_monkey_patch(test_imgs[i:i+BATCH_SIZE])
                    compute_aep_profile(attn)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            aep_time = np.median(times)
            ext.remove_hooks()

            overhead_results[model_key] = {
                'standard_ms': float(std_time / 100 * 1000),
                'aep_ms': float(aep_time / 100 * 1000),
                'multiplier': float(aep_time / std_time),
            }
        else:
            overhead_results[model_key] = {
                'standard_ms': float(std_time / 100 * 1000),
                'note': 'Swin: windowed attention, AEP not directly applicable',
            }
        logger.info(f"  {model_key}: {overhead_results[model_key]}")
        del model; gc.collect(); torch.cuda.empty_cache()

    with open(RESULTS_DIR / 'overhead_v2.json', 'w') as f:
        json.dump(overhead_results, f, indent=2)

    logger.info("\n=== ALL EXPERIMENTS COMPLETE ===")


if __name__ == '__main__':
    main()
