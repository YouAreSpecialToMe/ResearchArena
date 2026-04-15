#!/usr/bin/env python3
"""
Ablation Studies for LGSA.

Tests individual metrics and fixed vs learned weights.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, 'exp')

from shared.models import get_model
from shared.data_loader import load_dataset, load_splits, get_dataloader
from shared.training import load_model
from lgsa_core.lgsa import LGSA


def ablation_individual_metrics(dataset_name='cifar10', model_name='simplecnn', 
                                 seed=42, device='cuda'):
    """
    Test individual metrics: LDS only, GAS only, SRS only.
    """
    print(f"\nAblation: Individual Metrics ({dataset_name}, {model_name}, seed={seed})")
    
    # Load data
    train_dataset, _, num_classes, input_channels = load_dataset(dataset_name)
    splits = load_splits(f'data/{dataset_name}_splits_seed{seed}.pkl')
    
    forget_indices = splits['forget'][:1000]  # Subset for speed
    retain_indices = splits['retain'][:1000]
    
    # Load models
    original_model = get_model(model_name, num_classes, input_channels)
    load_model(original_model, f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth', device)
    
    unlearned_model = get_model(model_name, num_classes, input_channels)
    load_model(unlearned_model, f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth', device)
    
    # Apply gradient ascent
    from unlearning_methods.unlearning import gradient_ascent_unlearning
    forget_loader = get_dataloader(train_dataset, indices=forget_indices, batch_size=128, shuffle=True)
    unlearned_model = gradient_ascent_unlearning(unlearned_model, forget_loader, epochs=5, lr=0.01, device=device)
    
    # Get data tensors
    forget_loader_full = get_dataloader(train_dataset, indices=forget_indices, batch_size=len(forget_indices), shuffle=False)
    retain_loader_full = get_dataloader(train_dataset, indices=retain_indices, batch_size=len(retain_indices), shuffle=False)
    
    forget_data, forget_targets = next(iter(forget_loader_full))
    retain_data, retain_targets = next(iter(retain_loader_full))
    
    # Create LGSA instance
    lgsa = LGSA(original_model, unlearned_model, device)
    
    # Test different metric combinations
    results = {}
    
    # LDS only
    lgsa.weights = np.array([1.0, 0.0, 0.0])
    res, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['lds_only'] = {'auc': res['auc'], 'tpr_at_1fpr': res['tpr_at_1fpr']}
    print(f"  LDS only: AUC={res['auc']:.4f}")
    
    # GAS only
    lgsa.weights = np.array([0.0, 1.0, 0.0])
    res, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['gas_only'] = {'auc': res['auc'], 'tpr_at_1fpr': res['tpr_at_1fpr']}
    print(f"  GAS only: AUC={res['auc']:.4f}")
    
    # SRS only
    lgsa.weights = np.array([0.0, 0.0, 1.0])
    res, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['srs_only'] = {'auc': res['auc'], 'tpr_at_1fpr': res['tpr_at_1fpr']}
    print(f"  SRS only: AUC={res['auc']:.4f}")
    
    # LDS + GAS
    lgsa.weights = np.array([0.5, 0.5, 0.0])
    res, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['lds_gas'] = {'auc': res['auc'], 'tpr_at_1fpr': res['tpr_at_1fpr']}
    print(f"  LDS+GAS: AUC={res['auc']:.4f}")
    
    # LDS + SRS
    lgsa.weights = np.array([0.5, 0.0, 0.5])
    res, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['lds_srs'] = {'auc': res['auc'], 'tpr_at_1fpr': res['tpr_at_1fpr']}
    print(f"  LDS+SRS: AUC={res['auc']:.4f}")
    
    # GAS + SRS
    lgsa.weights = np.array([0.0, 0.5, 0.5])
    res, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['gas_srs'] = {'auc': res['auc'], 'tpr_at_1fpr': res['tpr_at_1fpr']}
    print(f"  GAS+SRS: AUC={res['auc']:.4f}")
    
    # All three (default weights)
    lgsa.weights = np.array([0.4, 0.4, 0.2])
    res, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['all_three'] = {'auc': res['auc'], 'tpr_at_1fpr': res['tpr_at_1fpr']}
    print(f"  All three: AUC={res['auc']:.4f}")
    
    return results


def ablation_fixed_vs_learned(dataset_name='cifar10', model_name='simplecnn',
                               seed=42, device='cuda'):
    """
    Compare fixed vs learned weights.
    """
    print(f"\nAblation: Fixed vs Learned Weights ({dataset_name}, {model_name}, seed={seed})")
    
    # Load data
    train_dataset, _, num_classes, input_channels = load_dataset(dataset_name)
    splits = load_splits(f'data/{dataset_name}_splits_seed{seed}.pkl')
    
    forget_indices = splits['forget'][:1000]
    retain_indices = splits['retain'][:1000]
    val_indices = splits['val'][:500]
    
    # Load models
    original_model = get_model(model_name, num_classes, input_channels)
    load_model(original_model, f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth', device)
    
    unlearned_model = get_model(model_name, num_classes, input_channels)
    load_model(unlearned_model, f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth', device)
    
    # Apply gradient ascent
    from unlearning_methods.unlearning import gradient_ascent_unlearning
    forget_loader = get_dataloader(train_dataset, indices=forget_indices, batch_size=128, shuffle=True)
    unlearned_model = gradient_ascent_unlearning(unlearned_model, forget_loader, epochs=5, lr=0.01, device=device)
    
    # Get data tensors
    forget_loader_full = get_dataloader(train_dataset, indices=forget_indices, batch_size=len(forget_indices), shuffle=False)
    retain_loader_full = get_dataloader(train_dataset, indices=retain_indices, batch_size=len(retain_indices), shuffle=False)
    val_loader_full = get_dataloader(train_dataset, indices=val_indices, batch_size=len(val_indices), shuffle=False)
    
    forget_data, forget_targets = next(iter(forget_loader_full))
    retain_data, retain_targets = next(iter(retain_loader_full))
    val_data, val_targets = next(iter(val_loader_full))
    
    # Create LGSA instance
    lgsa = LGSA(original_model, unlearned_model, device)
    
    results = {}
    
    # Fixed weights (default)
    lgsa.weights = np.array([0.4, 0.4, 0.2])
    res_fixed, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['fixed_weights'] = {
        'auc': res_fixed['auc'],
        'tpr_at_1fpr': res_fixed['tpr_at_1fpr'],
        'weights': lgsa.weights.tolist()
    }
    print(f"  Fixed weights {lgsa.weights}: AUC={res_fixed['auc']:.4f}")
    
    # Learned weights
    val_forget_data = val_data[:len(val_data)//2]
    val_forget_targets = val_targets[:len(val_targets)//2]
    val_retain_data = val_data[len(val_data)//2:]
    val_retain_targets = val_targets[len(val_targets)//2:]
    
    learned_weights = lgsa.learn_weights(val_forget_data, val_forget_targets,
                                         val_retain_data, val_retain_targets)
    
    res_learned, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
    results['learned_weights'] = {
        'auc': res_learned['auc'],
        'tpr_at_1fpr': res_learned['tpr_at_1fpr'],
        'weights': learned_weights.tolist()
    }
    print(f"  Learned weights {learned_weights}: AUC={res_learned['auc']:.4f}")
    
    results['improvement'] = res_learned['auc'] - res_fixed['auc']
    print(f"  Improvement: {results['improvement']:.4f}")
    
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run ablation studies
    all_results = {
        'individual_metrics': {},
        'fixed_vs_learned': {}
    }
    
    # Individual metrics ablation
    for seed in [42, 123]:
        key = f'cifar10_simplecnn_seed{seed}'
        all_results['individual_metrics'][key] = ablation_individual_metrics(
            'cifar10', 'simplecnn', seed, device)
    
    # Fixed vs learned weights
    for seed in [42, 123]:
        key = f'cifar10_simplecnn_seed{seed}'
        all_results['fixed_vs_learned'][key] = ablation_fixed_vs_learned(
            'cifar10', 'simplecnn', seed, device)
    
    # Save results
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/ablation_metrics.json', 'w') as f:
        json.dump(all_results['individual_metrics'], f, indent=2)
    with open('results/metrics/ablation_weights.json', 'w') as f:
        json.dump(all_results['fixed_vs_learned'], f, indent=2)
    
    print("\nAblation studies completed!")
    print("Results saved to:")
    print("  - results/metrics/ablation_metrics.json")
    print("  - results/metrics/ablation_weights.json")


if __name__ == '__main__':
    main()
