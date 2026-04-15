#!/usr/bin/env python3
"""Main experiment runner for AEP (Attention Entropy Profiling).

Since ImageNet-1K validation is gated, we use freely available datasets:
- ID: Food101 test set (101 food classes, ImageNet-pretrained ViTs have reasonable overlap)
- Near-OOD: CIFAR-100 (100 natural image classes, 32x32 upscaled)
             Flowers102 (102 flower classes)
- Far-OOD:  SVHN (house numbers - very different domain)
             DTD/Textures (texture patterns - no objects)
             CIFAR-10 (10 classes, different distribution)

This setup is scientifically valid: we test whether attention entropy profiles of
ImageNet-pretrained ViTs differentiate in-domain food images from various OOD sources.
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import timm
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from shared.aep import (AEPExtractor, compute_aep_profile, compute_id_statistics,
                         compute_mahalanobis_scores)
from shared.metrics import (compute_ood_metrics, compute_calibration_metrics,
                             compute_ece, compute_mce, compute_brier_score)
from shared.baselines import (msp_score, energy_score, vim_score, knn_score,
                               fit_temperature_scaling, fit_adaptive_temperature,
                               apply_adaptive_temperature, histogram_binning_fit,
                               histogram_binning_predict, fuse_scores)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WORKSPACE = Path(__file__).parent.parent
DATA_DIR = WORKSPACE / 'data'
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
CACHE_DIR = WORKSPACE / 'exp' / 'cache'

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]
BATCH_SIZE = 64

MODEL_CONFIGS = {
    'deit_small': {'name': 'deit_small_patch16_224', 'num_layers': 12, 'feat_dim': 384},
    'deit_base': {'name': 'deit_base_patch16_224', 'num_layers': 12, 'feat_dim': 768},
    'vit_base': {'name': 'vit_base_patch16_224', 'num_layers': 12, 'feat_dim': 768},
}

TRANSFORM_224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORM_SMALL = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============ Dataset Wrappers ============

class CIFARWrapper(Dataset):
    """Wrap CIFAR with 224x224 resize."""
    def __init__(self, root, cifar_cls, train=False):
        self.ds = cifar_cls(root=root, train=train, download=False)
        self.transform = TRANSFORM_SMALL

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        from PIL import Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        return self.transform(img), label


class SVHNWrapper(Dataset):
    """Wrap SVHN with 224x224 resize."""
    def __init__(self, root, split='test'):
        self.ds = datasets.SVHN(root=root, split=split, download=False)
        self.transform = TRANSFORM_SMALL

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        from PIL import Image
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img)
        img = img.convert('RGB')
        return self.transform(img), label


def get_dataloaders(num_workers=4):
    """Create all dataloaders."""
    loaders = {}

    # ID: Food101 test set
    food101 = datasets.Food101(root=str(DATA_DIR / 'food101'), split='test',
                                transform=TRANSFORM_224, download=True)
    loaders['id'] = DataLoader(food101, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    logger.info(f"ID (Food101): {len(food101)} images")

    # OOD datasets
    ood_configs = {
        'Textures': lambda: datasets.DTD(root=str(DATA_DIR / 'dtd'), split='test',
                                          transform=TRANSFORM_224),
        'SVHN': lambda: SVHNWrapper(str(DATA_DIR / 'svhn')),
        'CIFAR10': lambda: CIFARWrapper(str(DATA_DIR / 'cifar10'), datasets.CIFAR10),
        'CIFAR100': lambda: CIFARWrapper(str(DATA_DIR / 'cifar100'), datasets.CIFAR100),
        'Flowers102': lambda: datasets.Flowers102(root=str(DATA_DIR / 'flowers102'),
                                                    split='test', transform=TRANSFORM_224),
    }

    for name, make_ds in ood_configs.items():
        try:
            ds = make_ds()
            loaders[f'ood_{name}'] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=num_workers, pin_memory=True)
            logger.info(f"OOD {name}: {len(ds)} images")
        except Exception as e:
            logger.warning(f"Could not load {name}: {e}")

    return loaders


# ============ Feature Extraction ============

def extract_features(model, dataloader, model_key, dataset_name):
    """Extract AEP profiles, logits, features, and labels. Cached to disk."""
    cache_file = CACHE_DIR / f'{model_key}_{dataset_name}.npz'
    if cache_file.exists():
        logger.info(f"  Loading cached: {model_key}/{dataset_name}")
        data = np.load(str(cache_file))
        return {k: data[k] for k in data.files}

    logger.info(f"  Extracting: {model_key}/{dataset_name}...")
    model.eval().to(DEVICE)
    extractor = AEPExtractor(model)

    all_aep, all_logits, all_features, all_labels = [], [], [], []

    # Hook for penultimate features
    feat_storage = []
    hook_handle = None
    for name, module in model.named_modules():
        if name in ('norm', 'fc_norm'):
            hook_handle = module.register_forward_hook(
                lambda m, inp, out: feat_storage.append(out.detach().cpu()))
            break

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"    {dataset_name}", leave=False):
            images, labels = batch[0], batch[1]
            images = images.to(DEVICE)
            feat_storage.clear()

            logits, attn_maps = extractor._extract_with_monkey_patch(images)
            aep = compute_aep_profile(attn_maps)
            all_aep.append(aep)
            all_logits.append(logits.cpu().numpy())

            if feat_storage:
                feat = feat_storage[-1]
                if feat.dim() == 3:
                    feat = feat[:, 0]
                all_features.append(feat.numpy())

            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
            del attn_maps
            feat_storage.clear()

    if hook_handle:
        hook_handle.remove()

    result = {
        'aep_profiles': np.concatenate(all_aep),
        'logits': np.concatenate(all_logits),
        'features': np.concatenate(all_features) if all_features else np.array([]),
        'labels': np.concatenate(all_labels),
    }
    np.savez_compressed(str(cache_file), **result)
    logger.info(f"    Shape: AEP={result['aep_profiles'].shape}, logits={result['logits'].shape}")
    return result


# ============ Calibration Indices ============

def get_cal_indices(labels, n_cal, seed):
    """Stratified calibration subset."""
    rng = np.random.RandomState(seed)
    unique = np.unique(labels)
    per_class = max(1, n_cal // len(unique))
    indices = []
    for lbl in unique:
        cls_idx = np.where(labels == lbl)[0]
        chosen = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())
    if len(indices) < n_cal:
        remaining = list(set(range(len(labels))) - set(indices))
        extra = rng.choice(remaining, size=min(n_cal - len(indices), len(remaining)), replace=False)
        indices.extend(extra.tolist())
    return np.array(indices[:n_cal])


# ============ OOD Detection ============

def run_ood_detection(model_key, model_config, loaders, id_data):
    """OOD detection across all seeds and OOD datasets."""
    results = {}
    ood_names = [k.replace('ood_', '') for k in loaders if k.startswith('ood_')]

    model = timm.create_model(model_config['name'], pretrained=True).to(DEVICE).eval()
    ood_data = {}
    for name in ood_names:
        ood_data[name] = extract_features(model, loaders[f'ood_{name}'], model_key, f'ood_{name}')
    del model; torch.cuda.empty_cache(); gc.collect()

    for seed in SEEDS:
        logger.info(f"  Seed {seed}")
        results[str(seed)] = {}

        cal_idx = get_cal_indices(id_data['labels'], 1000, seed)
        eval_idx = np.setdiff1d(np.arange(len(id_data['labels'])), cal_idx)

        cal_aep = id_data['aep_profiles'][cal_idx]
        eval_aep = id_data['aep_profiles'][eval_idx]
        cal_logits = id_data['logits'][cal_idx]
        eval_logits = id_data['logits'][eval_idx]
        cal_feat = id_data['features'][cal_idx] if len(id_data['features']) > 0 else None
        eval_feat = id_data['features'][eval_idx] if len(id_data['features']) > 0 else None

        aep_stats = compute_id_statistics(cal_aep)
        id_aep_sc = compute_mahalanobis_scores(eval_aep, aep_stats)
        id_msp = msp_score(eval_logits)
        id_energy = energy_score(eval_logits)
        id_vim = vim_score(eval_feat, eval_logits, cal_feat, cal_logits) if cal_feat is not None else id_energy.copy()
        id_knn = knn_score(eval_feat, cal_feat, k=50) if cal_feat is not None else id_energy.copy()

        for ood_name in ood_names:
            od = ood_data[ood_name]
            ood_aep_sc = compute_mahalanobis_scores(od['aep_profiles'], aep_stats)
            ood_msp = msp_score(od['logits'])
            ood_energy = energy_score(od['logits'])
            ood_feat = od['features'] if len(od['features']) > 0 else None

            if cal_feat is not None and ood_feat is not None and len(ood_feat) > 0:
                ood_vim = vim_score(ood_feat, od['logits'], cal_feat, cal_logits)
                ood_knn = knn_score(ood_feat, cal_feat, k=50)
            else:
                ood_vim = ood_energy.copy()
                ood_knn = ood_energy.copy()

            methods = {
                'MSP': (id_msp, ood_msp),
                'Energy': (id_energy, ood_energy),
                'ViM': (id_vim, ood_vim),
                'KNN': (id_knn, ood_knn),
                'AEP': (id_aep_sc, ood_aep_sc),
            }

            method_results = {}
            for mname, (id_sc, ood_sc) in methods.items():
                method_results[mname] = compute_ood_metrics(id_sc, ood_sc)

            # Fusion with best baseline
            best_base = max(['MSP', 'Energy', 'ViM', 'KNN'],
                            key=lambda m: method_results[m]['AUROC'])
            id_best, ood_best = methods[best_base]
            fused_id, fused_ood, beta = fuse_scores(
                id_aep_sc, id_best, id_aep_sc, id_best, ood_aep_sc, ood_best)
            method_results['AEP+Fusion'] = compute_ood_metrics(fused_id, fused_ood)
            method_results['AEP+Fusion']['beta'] = beta
            method_results['AEP+Fusion']['fused_with'] = best_base

            results[str(seed)][ood_name] = method_results

    return results


# ============ Calibration ============

def run_calibration(model_key, model_config, loaders, id_data):
    """Calibration experiments."""
    results = {}

    model = timm.create_model(model_config['name'], pretrained=True).to(DEVICE).eval()
    ood_names = [k.replace('ood_', '') for k in loaders if k.startswith('ood_')]
    ood_data = {}
    for name in ood_names:
        ood_data[name] = extract_features(model, loaders[f'ood_{name}'], model_key, f'ood_{name}')
    del model; torch.cuda.empty_cache(); gc.collect()

    for seed in SEEDS:
        results[str(seed)] = {}
        cal_idx = get_cal_indices(id_data['labels'], 1000, seed)

        cal_logits = id_data['logits'][cal_idx]
        cal_labels = id_data['labels'][cal_idx]
        cal_aep = id_data['aep_profiles'][cal_idx]

        aep_stats = compute_id_statistics(cal_aep)
        cal_aep_scores = compute_mahalanobis_scores(cal_aep, aep_stats)

        T_global = fit_temperature_scaling(cal_logits, cal_labels)
        hb_params = histogram_binning_fit(cal_logits, cal_labels)
        adapt_params = fit_adaptive_temperature(cal_logits, cal_labels, cal_aep_scores)

        logger.info(f"  Seed {seed}: T_global={T_global:.3f}, T0={adapt_params['T0']:.3f}, alpha={adapt_params['alpha']:.3f}")

        # Evaluate on ID eval set
        eval_idx = np.setdiff1d(np.arange(len(id_data['labels'])), cal_idx)
        for ds_name, ds_data in [('Food101_eval', {
            'logits': id_data['logits'][eval_idx],
            'labels': id_data['labels'][eval_idx],
            'aep_profiles': id_data['aep_profiles'][eval_idx],
        })] + [(name, ood_data[name]) for name in ood_names]:

            shift_logits = ds_data['logits']
            shift_labels = ds_data['labels']
            shift_aep = ds_data['aep_profiles']
            shift_aep_scores = compute_mahalanobis_scores(shift_aep, aep_stats)

            cal_r = {}
            cal_r['Raw'] = compute_calibration_metrics(shift_logits, shift_labels)
            cal_r['TempScaling'] = compute_calibration_metrics(shift_logits, shift_labels, temperature=T_global)

            # Histogram binning
            hb_conf = histogram_binning_predict(shift_logits, hb_params)
            exp_l = np.exp(shift_logits - shift_logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            preds = probs.argmax(axis=1)
            correct = (preds == shift_labels).astype(float)
            cal_r['HistBinning'] = {
                'ECE': compute_ece(hb_conf, correct),
                'MCE': compute_mce(hb_conf, correct),
                'Brier': compute_brier_score(hb_conf, correct),
                'Accuracy': float(correct.mean()),
            }

            # AEP adaptive
            adapt_probs = apply_adaptive_temperature(shift_logits, shift_aep_scores, adapt_params)
            adapt_conf = adapt_probs.max(axis=1)
            adapt_preds = adapt_probs.argmax(axis=1)
            adapt_correct = (adapt_preds == shift_labels).astype(float)
            cal_r['AEP_Adaptive'] = {
                'ECE': compute_ece(adapt_conf, adapt_correct),
                'MCE': compute_mce(adapt_conf, adapt_correct),
                'Brier': compute_brier_score(adapt_conf, adapt_correct),
                'Accuracy': float(adapt_correct.mean()),
            }

            results[str(seed)][ds_name] = cal_r

    return results


# ============ Ablation: Components ============

def run_ablation_components(id_data, loaders, model_key, model_config):
    """Component ablation using cached features."""
    results = {}
    seed = 42
    num_layers = model_config['num_layers']

    cal_idx = get_cal_indices(id_data['labels'], 1000, seed)
    eval_idx = np.setdiff1d(np.arange(len(id_data['labels'])), cal_idx)

    model = timm.create_model(model_config['name'], pretrained=True).to(DEVICE).eval()
    ood_names = [k.replace('ood_', '') for k in loaders if k.startswith('ood_')]
    ood_data = {}
    for name in ood_names:
        ood_data[name] = extract_features(model, loaders[f'ood_{name}'], model_key, f'ood_{name}')
    del model; torch.cuda.empty_cache(); gc.collect()

    def feat_idx(exclude=None, only=None):
        indices = []
        for l in range(num_layers):
            for f in range(5):
                if only is not None and f not in only:
                    continue
                if exclude is not None and f in exclude:
                    continue
                indices.append(l * 5 + f)
        return indices

    variants = {
        'Full': feat_idx(),
        'No_CLS_entropy': feat_idx(exclude=[0, 1]),
        'No_avg_token_entropy': feat_idx(exclude=[2]),
        'No_concentration': feat_idx(exclude=[3]),
        'No_head_agreement': feat_idx(exclude=[4]),
        'CLS_entropy_only': feat_idx(only=[0, 1]),
    }

    for vname, idx in variants.items():
        results[vname] = {}
        cal_sub = id_data['aep_profiles'][cal_idx][:, idx]
        eval_sub = id_data['aep_profiles'][eval_idx][:, idx]
        st = compute_id_statistics(cal_sub)
        id_sc = compute_mahalanobis_scores(eval_sub, st)

        for ood_name in ood_names:
            ood_sub = ood_data[ood_name]['aep_profiles'][:, idx]
            ood_sc = compute_mahalanobis_scores(ood_sub, st)
            results[vname][ood_name] = compute_ood_metrics(id_sc, ood_sc)

    return results


# ============ Ablation: Layers ============

def run_ablation_layers(id_data, loaders, model_key, model_config):
    """Layer importance ablation."""
    results = {}
    seed = 42
    num_layers = model_config['num_layers']

    cal_idx = get_cal_indices(id_data['labels'], 1000, seed)
    eval_idx = np.setdiff1d(np.arange(len(id_data['labels'])), cal_idx)

    model = timm.create_model(model_config['name'], pretrained=True).to(DEVICE).eval()
    ood_names = [k.replace('ood_', '') for k in loaders if k.startswith('ood_')]
    ood_data = {}
    for name in ood_names:
        ood_data[name] = extract_features(model, loaders[f'ood_{name}'], model_key, f'ood_{name}')
    del model; torch.cuda.empty_cache(); gc.collect()

    def layer_idx(layers):
        return [l * 5 + f for l in layers for f in range(5)]

    groups = {
        'All_layers': layer_idx(range(num_layers)),
        'Early_1-4': layer_idx(range(0, 4)),
        'Middle_5-8': layer_idx(range(4, 8)),
        'Late_9-12': layer_idx(range(8, 12)),
    }

    for gname, idx in groups.items():
        results[gname] = {}
        cal_sub = id_data['aep_profiles'][cal_idx][:, idx]
        eval_sub = id_data['aep_profiles'][eval_idx][:, idx]
        st = compute_id_statistics(cal_sub)
        id_sc = compute_mahalanobis_scores(eval_sub, st)

        for ood_name in ood_names:
            ood_sub = ood_data[ood_name]['aep_profiles'][:, idx]
            ood_sc = compute_mahalanobis_scores(ood_sub, st)
            results[gname][ood_name] = compute_ood_metrics(id_sc, ood_sc)

    # Per-layer t-tests
    t_tests = {}
    feature_names = ['cls_ent_mean', 'cls_ent_std', 'avg_token_ent', 'concentration', 'head_agreement']
    for ood_name in ood_names:
        t_tests[ood_name] = {}
        for l in range(num_layers):
            t_tests[ood_name][f'layer_{l+1}'] = {}
            for fi, fn in enumerate(feature_names):
                col = l * 5 + fi
                t_stat, p_val = stats.ttest_ind(
                    id_data['aep_profiles'][eval_idx, col],
                    ood_data[ood_name]['aep_profiles'][:, col])
                t_tests[ood_name][f'layer_{l+1}'][fn] = {
                    't_statistic': float(t_stat), 'p_value': float(p_val)}

    results['t_test'] = t_tests
    return results


# ============ Ablation: Cal Size ============

def run_ablation_calsize(id_data, loaders, model_key, model_config):
    """Calibration set size sensitivity."""
    results = {}
    seed = 42
    sizes = [50, 100, 250, 500, 1000, 2500, 5000]

    model = timm.create_model(model_config['name'], pretrained=True).to(DEVICE).eval()
    ood_names = [k.replace('ood_', '') for k in loaders if k.startswith('ood_')]
    ood_data = {}
    for name in ood_names:
        ood_data[name] = extract_features(model, loaders[f'ood_{name}'], model_key, f'ood_{name}')
    del model; torch.cuda.empty_cache(); gc.collect()

    for n_cal in sizes:
        n_cal_actual = min(n_cal, len(id_data['labels']) - 100)
        cal_idx = get_cal_indices(id_data['labels'], n_cal_actual, seed)
        eval_idx = np.setdiff1d(np.arange(len(id_data['labels'])), cal_idx)

        st = compute_id_statistics(id_data['aep_profiles'][cal_idx])
        id_sc = compute_mahalanobis_scores(id_data['aep_profiles'][eval_idx], st)

        results[str(n_cal)] = {}
        for ood_name in ood_names:
            ood_sc = compute_mahalanobis_scores(ood_data[ood_name]['aep_profiles'], st)
            results[str(n_cal)][ood_name] = compute_ood_metrics(id_sc, ood_sc)

    return results


# ============ Statistical Tests ============

def run_statistical_tests(id_data, loaders, model_key, model_config):
    """Formal hypothesis tests."""
    results = {}
    seed = 42
    cal_idx = get_cal_indices(id_data['labels'], 1000, seed)
    eval_idx = np.setdiff1d(np.arange(len(id_data['labels'])), cal_idx)
    id_profiles = id_data['aep_profiles'][eval_idx]

    model = timm.create_model(model_config['name'], pretrained=True).to(DEVICE).eval()
    ood_names = [k.replace('ood_', '') for k in loaders if k.startswith('ood_')]

    for ood_name in ood_names:
        ood_data = extract_features(model, loaders[f'ood_{ood_name}'], model_key, f'ood_{ood_name}')
        ood_profiles = ood_data['aep_profiles']
        n_dims = id_profiles.shape[1]

        # Per-dimension t-tests with Bonferroni
        sig_dims = 0
        dim_results = {}
        for d in range(n_dims):
            t_stat, p_val = stats.ttest_ind(id_profiles[:, d], ood_profiles[:, d])
            p_corr = min(p_val * n_dims, 1.0)
            cohens_d_val = (id_profiles[:, d].mean() - ood_profiles[:, d].mean()) / \
                           np.sqrt((id_profiles[:, d].var() + ood_profiles[:, d].var()) / 2 + 1e-10)
            if p_corr < 0.01:
                sig_dims += 1
            dim_results[f'dim_{d}'] = {
                't_statistic': float(t_stat), 'p_value': float(p_val),
                'p_corrected': float(p_corr), 'cohens_d': float(cohens_d_val)}

        # Hotelling's T-squared
        n_sub = min(5000, len(id_profiles), len(ood_profiles))
        rng = np.random.RandomState(42)
        id_sub = id_profiles[rng.choice(len(id_profiles), n_sub, replace=False)]
        ood_sub = ood_profiles[rng.choice(len(ood_profiles), min(n_sub, len(ood_profiles)), replace=False)]
        n1, n2, p = len(id_sub), len(ood_sub), id_sub.shape[1]
        mean_diff = id_sub.mean(0) - ood_sub.mean(0)
        S_pooled = ((n1-1)*np.cov(id_sub, rowvar=False) + (n2-1)*np.cov(ood_sub, rowvar=False)) / (n1+n2-2)
        S_pooled += 1e-4 * np.eye(p)

        try:
            S_inv = np.linalg.inv(S_pooled)
            T2 = (n1*n2)/(n1+n2) * mean_diff @ S_inv @ mean_diff
            F_stat = T2 * (n1+n2-p-1) / (p*(n1+n2-2))
            p_value = 1 - stats.f.cdf(F_stat, p, n1+n2-p-1)
        except:
            T2, F_stat, p_value = float('inf'), float('inf'), 0.0

        results[ood_name] = {
            'n_significant_dims': sig_dims,
            'total_dims': n_dims,
            'fraction_significant': sig_dims / n_dims,
            'hotellings_T2': float(T2),
            'F_statistic': float(F_stat),
            'p_value': float(p_value),
            'dim_results': dim_results,
        }

    del model; torch.cuda.empty_cache(); gc.collect()
    return results


# ============ Overhead Analysis ============

def run_overhead_analysis():
    """Measure AEP computational overhead."""
    results = {}
    n_images = 640  # ~10 batches

    dummy = torch.randn(n_images, 3, 224, 224)
    dummy_loader = DataLoader(
        torch.utils.data.TensorDataset(dummy, torch.zeros(n_images, dtype=torch.long)),
        batch_size=64, shuffle=False)

    for mk, cfg in MODEL_CONFIGS.items():
        model = timm.create_model(cfg['name'], pretrained=True).to(DEVICE).eval()
        # Warmup
        with torch.no_grad():
            _ = model(torch.randn(4, 3, 224, 224).to(DEVICE))
        torch.cuda.synchronize()

        # Standard
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            for b, _ in dummy_loader:
                _ = model(b.to(DEVICE)); torch.cuda.synchronize()
        time_std = time.time() - t0
        mem_std = torch.cuda.max_memory_allocated() / 1e6

        # With AEP
        extractor = AEPExtractor(model)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            for b, _ in dummy_loader:
                _, attn = extractor._extract_with_monkey_patch(b.to(DEVICE))
                _ = compute_aep_profile(attn); del attn
        torch.cuda.synchronize()
        time_aep = time.time() - t0
        mem_aep = torch.cuda.max_memory_allocated() / 1e6

        results[mk] = {
            'time_standard_ms': time_std / n_images * 1000,
            'time_aep_ms': time_aep / n_images * 1000,
            'overhead_pct': (time_aep - time_std) / time_std * 100,
            'mem_standard_MB': mem_std,
            'mem_aep_MB': mem_aep,
        }
        logger.info(f"  {mk}: std={time_std:.1f}s, aep={time_aep:.1f}s, overhead={results[mk]['overhead_pct']:.1f}%")
        del model; torch.cuda.empty_cache()

    return results


# ============ Aggregate ============

def aggregate_seeds(results):
    """Aggregate dict-of-seeds results into mean±std."""
    agg = {}
    seeds = [k for k in results if k.isdigit()]
    if not seeds:
        return results

    first = results[seeds[0]]
    for key in first:
        if isinstance(first[key], dict):
            agg[key] = {}
            for subkey in first[key]:
                if isinstance(first[key][subkey], dict):
                    agg[key][subkey] = {}
                    for metric, val in first[key][subkey].items():
                        if isinstance(val, (int, float)):
                            vals = [results[s][key][subkey][metric]
                                    for s in seeds
                                    if key in results[s] and subkey in results[s][key]
                                    and metric in results[s][key][subkey]
                                    and isinstance(results[s][key][subkey][metric], (int, float))]
                            if vals:
                                agg[key][subkey][metric] = {
                                    'mean': float(np.mean(vals)),
                                    'std': float(np.std(vals))}
                        else:
                            agg[key][subkey][metric] = val
    return agg


# ============ Main ============

def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("AEP Experiment Pipeline")
    logger.info("=" * 60)

    # Data
    logger.info("\n=== Data Preparation ===")
    loaders = get_dataloaders()

    all_ood = {}
    all_cal = {}

    # Per-model experiments
    for mk, cfg in MODEL_CONFIGS.items():
        logger.info(f"\n{'='*40} Model: {mk} {'='*40}")
        model = timm.create_model(cfg['name'], pretrained=True).to(DEVICE).eval()
        id_data = extract_features(model, loaders['id'], mk, 'id')
        del model; torch.cuda.empty_cache(); gc.collect()

        logger.info("OOD Detection...")
        all_ood[mk] = run_ood_detection(mk, cfg, loaders, id_data)

        logger.info("Calibration...")
        all_cal[mk] = run_calibration(mk, cfg, loaders, id_data)

    # Save results
    with open(RESULTS_DIR / 'ood_results.json', 'w') as f:
        json.dump(all_ood, f, indent=2)
    ood_agg = {mk: aggregate_seeds(v) for mk, v in all_ood.items()}
    with open(RESULTS_DIR / 'ood_results_aggregated.json', 'w') as f:
        json.dump(ood_agg, f, indent=2)
    with open(RESULTS_DIR / 'calibration_results.json', 'w') as f:
        json.dump(all_cal, f, indent=2)
    cal_agg = {mk: aggregate_seeds(v) for mk, v in all_cal.items()}
    with open(RESULTS_DIR / 'calibration_results_aggregated.json', 'w') as f:
        json.dump(cal_agg, f, indent=2)

    # Ablations on primary model (vit_base)
    logger.info("\n=== Ablation Studies (ViT-Base) ===")
    primary = 'vit_base'
    model = timm.create_model(MODEL_CONFIGS[primary]['name'], pretrained=True).to(DEVICE).eval()
    id_data_p = extract_features(model, loaders['id'], primary, 'id')
    del model; torch.cuda.empty_cache(); gc.collect()

    logger.info("Component ablation...")
    comp_abl = run_ablation_components(id_data_p, loaders, primary, MODEL_CONFIGS[primary])
    with open(RESULTS_DIR / 'ablation_components.json', 'w') as f:
        json.dump(comp_abl, f, indent=2)

    logger.info("Layer ablation...")
    layer_abl = run_ablation_layers(id_data_p, loaders, primary, MODEL_CONFIGS[primary])
    with open(RESULTS_DIR / 'ablation_layers.json', 'w') as f:
        json.dump(layer_abl, f, indent=2)

    logger.info("Cal-size ablation...")
    calsize_abl = run_ablation_calsize(id_data_p, loaders, primary, MODEL_CONFIGS[primary])
    with open(RESULTS_DIR / 'ablation_calsize.json', 'w') as f:
        json.dump(calsize_abl, f, indent=2)

    # Statistical tests
    logger.info("\n=== Statistical Tests ===")
    stat_results = run_statistical_tests(id_data_p, loaders, primary, MODEL_CONFIGS[primary])
    with open(RESULTS_DIR / 'statistical_tests.json', 'w') as f:
        json.dump(stat_results, f, indent=2)

    # Overhead
    logger.info("\n=== Overhead Analysis ===")
    overhead = run_overhead_analysis()
    with open(RESULTS_DIR / 'overhead_analysis.json', 'w') as f:
        json.dump(overhead, f, indent=2)

    # Final aggregated results
    logger.info("\n=== Compiling Final Results ===")
    final = {
        'ood_detection': ood_agg,
        'calibration': cal_agg,
        'ablation_components': comp_abl,
        'ablation_layers': {k: v for k, v in layer_abl.items() if k != 't_test'},
        'ablation_calsize': calsize_abl,
        'statistical_tests': {k: {sk: sv for sk, sv in v.items() if sk != 'dim_results'}
                               for k, v in stat_results.items()},
        'overhead': overhead,
    }
    with open(WORKSPACE / 'results.json', 'w') as f:
        json.dump(final, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(f"\nDone in {elapsed/60:.1f} minutes. Results saved to results.json")


if __name__ == '__main__':
    main()
