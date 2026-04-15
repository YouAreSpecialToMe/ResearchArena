#!/usr/bin/env python3
"""
Run TruVRF baseline for comparison.
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
from shared.training import load_model
from shared.metrics import compute_accuracy
from unlearning_methods.unlearning import apply_unlearning
from baseline_truvrf.truvrf import TruVRF


def run_truvrf_experiment(dataset_name, model_name, seed=42, device='cuda'):
    """Run TruVRF baseline experiment."""
    
    print(f"\n{'='*60}")
    print(f"TruVRF: {dataset_name}, {model_name}, seed={seed}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    train_dataset, test_dataset, num_classes, input_channels = load_dataset(dataset_name)
    splits = load_splits(f'data/{dataset_name}_splits_seed{seed}.pkl')
    forget_indices = splits['forget']
    retain_indices = splits['retain']
    val_indices = splits['val']
    
    # Load models
    model_path = f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth'
    original_model = get_model(model_name, num_classes, input_channels)
    load_model(original_model, model_path, device)
    
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    base_acc = compute_accuracy(original_model, test_loader, device)
    print(f"Base model test accuracy: {base_acc:.4f}")
    
    # Create unlearned model with fine-tuning
    unlearned_model = get_model(model_name, num_classes, input_channels)
    load_model(unlearned_model, model_path, device)
    
    print("\nApplying fine-tuning unlearning...")
    forget_loader = get_dataloader(train_dataset, indices=forget_indices, 
                                   batch_size=128, shuffle=True, num_workers=4)
    retain_loader = get_dataloader(train_dataset, indices=retain_indices,
                                   batch_size=128, shuffle=True, num_workers=4)
    
    unlearn_start = time.time()
    unlearned_model = apply_unlearning(unlearned_model, 'finetuning',
                                       forget_loader, retain_loader,
                                       epochs=10, lr=0.001, device=device)
    unlearn_time = time.time() - unlearn_start
    
    unlearn_acc = compute_accuracy(unlearned_model, test_loader, device)
    print(f"Unlearned model test accuracy: {unlearn_acc:.4f}")
    
    # Prepare verification data
    n_forget = min(500, len(forget_indices))
    n_retain = min(500, len(retain_indices))
    n_verify = 1000
    
    forget_sample = np.random.choice(forget_indices, n_forget, replace=False)
    retain_sample = np.random.choice(retain_indices, n_retain, replace=False)
    verify_sample = np.random.choice(val_indices, min(n_verify, len(val_indices)), replace=False)
    
    forget_loader_small = get_dataloader(train_dataset, indices=forget_sample.tolist(),
                                         batch_size=n_forget, shuffle=False, num_workers=0)
    retain_loader_small = get_dataloader(train_dataset, indices=retain_sample.tolist(),
                                         batch_size=n_retain, shuffle=False, num_workers=0)
    verify_loader_small = get_dataloader(train_dataset, indices=verify_sample.tolist(),
                                         batch_size=len(verify_sample), shuffle=False, num_workers=0)
    
    forget_data, forget_targets = next(iter(forget_loader_small))
    retain_data, retain_targets = next(iter(retain_loader_small))
    verify_data, verify_targets = next(iter(verify_loader_small))
    
    # Run TruVRF
    print("\nRunning TruVRF verification...")
    truvrf = TruVRF(original_model, unlearned_model, device)
    
    results, scores, labels = truvrf.verify_unlearning(
        forget_data, forget_targets, retain_data, retain_targets,
        verify_data, verify_targets, epochs=3, lr=0.001)
    
    print(f"\nTruVRF Results:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  TPR@1%FPR: {results['tpr_at_1fpr']:.4f}")
    print(f"  Verification time: {results['verify_time']:.2f}s")
    
    # Save results
    full_results = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'method': 'truvrf',
        'base_accuracy': base_acc,
        'unlearn_accuracy': unlearn_acc,
        'unlearn_time': unlearn_time,
        'auc': float(results['auc']),
        'tpr_at_1fpr': float(results['tpr_at_1fpr']),
        'verify_time': float(results['verify_time']),
        'forget_sens_mean': float(results['forget_sens_mean']),
        'retain_sens_mean': float(results['retain_sens_mean']),
    }
    
    os.makedirs('results/metrics', exist_ok=True)
    with open(f'results/metrics/truvrf_{dataset_name}_{model_name}_seed{seed}.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    return full_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='simplecnn')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_truvrf_experiment(args.dataset, args.model, args.seed, args.device)
