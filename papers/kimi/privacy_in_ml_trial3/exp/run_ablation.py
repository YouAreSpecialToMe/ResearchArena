#!/usr/bin/env python3
"""
Run ablation studies for LGSA (individual metrics and combinations).
"""
import sys
import os
sys.path.insert(0, 'exp')

import torch
import json
import numpy as np
import time
from shared.models import get_model
from shared.data_loader import load_dataset, load_splits, get_dataloader
from shared.training import load_model, train_model
from shared.metrics import compute_accuracy
from lgsa_core.lgsa import LGSA


def run_ablation_experiment(dataset_name, model_name, seed, metric_config, device='cuda'):
    """
    Run LGSA ablation with specific metric configuration.
    
    Args:
        metric_config: tuple of (lds_weight, gas_weight, srs_weight)
    """
    config_names = {
        (1, 0, 0): 'lds_only',
        (0, 1, 0): 'gas_only',
        (0, 0, 1): 'srs_only',
        (1, 1, 0): 'lds_gas',
        (1, 0, 1): 'lds_srs',
        (0, 1, 1): 'gas_srs',
        (1, 1, 1): 'all_three'
    }
    config_name = config_names.get(metric_config, 'custom')
    
    print(f"\n{'='*60}")
    print(f"Ablation: {config_name} (seed={seed})")
    print(f"{'='*60}\n")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    train_dataset, test_dataset, num_classes, input_channels = load_dataset(dataset_name)
    splits = load_splits(f'data/{dataset_name}_splits_seed{seed}.pkl')
    forget_indices = splits['forget']
    retain_indices = splits['retain']
    
    # Load original model
    model_path = f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth'
    original_model = get_model(model_name, num_classes, input_channels)
    load_model(original_model, model_path, device)
    
    # Create retrained model (gold standard)
    retrained_model = get_model(model_name, num_classes, input_channels)
    retain_loader = get_dataloader(train_dataset, indices=retain_indices,
                                   batch_size=128, shuffle=True, num_workers=4)
    retrained_model, _ = train_model(retrained_model, retain_loader, None, 
                                      epochs=10, lr=0.1, device=device, verbose=False)
    
    # Prepare verification data
    n_forget = min(1000, len(forget_indices))
    n_retain = min(1000, len(retain_indices))
    
    forget_sample = np.random.choice(forget_indices, n_forget, replace=False)
    retain_sample = np.random.choice(retain_indices, n_retain, replace=False)
    
    forget_loader_small = get_dataloader(train_dataset, indices=forget_sample.tolist(),
                                         batch_size=n_forget, shuffle=False, num_workers=0)
    retain_loader_small = get_dataloader(train_dataset, indices=retain_sample.tolist(),
                                         batch_size=n_retain, shuffle=False, num_workers=0)
    
    forget_data, forget_targets = next(iter(forget_loader_small))
    retain_data, retain_targets = next(iter(retain_loader_small))
    
    # Run LGSA with specific weights
    lgsa = LGSA(original_model, retrained_model, device)
    weights = np.array(metric_config, dtype=np.float32)
    
    verify_start = time.time()
    results, scores, labels = lgsa.verify_unlearning(
        forget_data, forget_targets, retain_data, retain_targets, weights=weights)
    verify_time = time.time() - verify_start
    
    print(f"AUC: {results['auc']:.4f}, Config: {config_name}")
    
    ablation_result = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'config': config_name,
        'weights': metric_config,
        'auc': float(results['auc']),
        'tpr_at_1fpr': float(results['tpr_at_1fpr']),
        'verify_time': verify_time
    }
    
    os.makedirs('results/metrics', exist_ok=True)
    filename = f'results/metrics/ablation_{dataset_name}_{model_name}_{config_name}_seed{seed}.json'
    with open(filename, 'w') as f:
        json.dump(ablation_result, f, indent=2)
    
    return ablation_result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='simplecnn')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='all_three',
                        choices=['lds_only', 'gas_only', 'srs_only', 
                                'lds_gas', 'lds_srs', 'gas_srs', 'all_three'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    config_map = {
        'lds_only': (1, 0, 0),
        'gas_only': (0, 1, 0),
        'srs_only': (0, 0, 1),
        'lds_gas': (1, 1, 0),
        'lds_srs': (1, 0, 1),
        'gas_srs': (0, 1, 1),
        'all_three': (1, 1, 1)
    }
    
    run_ablation_experiment(args.dataset, args.model, args.seed, 
                            config_map[args.config], args.device)
