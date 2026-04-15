#!/usr/bin/env python3
"""
Run LGSA with gold standard comparison (retrain from scratch without forget set).
This creates the strongest signal for unlearning verification.
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


def run_gold_standard_experiment(dataset_name, model_name, seed=42, epochs=15, device='cuda'):
    """
    Run LGSA verification comparing original model vs retrained (without forget set).
    """
    print(f"\n{'='*60}")
    print(f"LGSA (Gold Standard): {dataset_name}, {model_name}, seed={seed}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset
    train_dataset, test_dataset, num_classes, input_channels = load_dataset(dataset_name)
    splits = load_splits(f'data/{dataset_name}_splits_seed{seed}.pkl')
    forget_indices = splits['forget']
    retain_indices = splits['retain']
    
    print(f"Forget set size: {len(forget_indices)}")
    print(f"Retain set size: {len(retain_indices)}")
    
    # Load original model (trained on full dataset)
    print("\nLoading original model...")
    model_path = f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth'
    original_model = get_model(model_name, num_classes, input_channels)
    load_model(original_model, model_path, device)
    
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    base_acc = compute_accuracy(original_model, test_loader, device)
    print(f"Original model test accuracy: {base_acc:.4f}")
    
    # Create "unlearned" model by retraining from scratch on retain set only
    # This is the gold standard for exact unlearning
    print("\nRetraining model from scratch (without forget set)...")
    retrained_model = get_model(model_name, num_classes, input_channels)
    
    retain_loader = get_dataloader(train_dataset, indices=retain_indices,
                                   batch_size=128, shuffle=True, num_workers=4)
    
    train_start = time.time()
    retrained_model, history = train_model(retrained_model, retain_loader, None, 
                                           epochs=epochs, lr=0.1, device=device, verbose=True)
    train_time = time.time() - train_start
    
    retrain_acc = compute_accuracy(retrained_model, test_loader, device)
    print(f"Retrained model test accuracy: {retrain_acc:.4f}")
    print(f"Retraining took {train_time:.2f}s")
    
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
    
    # Run LGSA
    print("\nRunning LGSA verification...")
    lgsa = LGSA(original_model, retrained_model, device)
    
    verify_start = time.time()
    results, scores, labels = lgsa.verify_unlearning(
        forget_data, forget_targets, retain_data, retain_targets)
    verify_time = time.time() - verify_start
    
    print(f"\nLGSA Results:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  TPR@1%FPR: {results['tpr_at_1fpr']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  Verification time: {verify_time:.2f}s")
    print(f"  Weights: {results['weights']}")
    
    # Save results
    full_results = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'unlearn_method': 'gold_standard_retrain',
        'base_accuracy': base_acc,
        'unlearn_accuracy': retrain_acc,
        'unlearn_time': train_time,
        'auc': float(results['auc']),
        'tpr_at_1fpr': float(results['tpr_at_1fpr']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
        'verify_time': verify_time,
        'forget_lss_mean': float(results['forget_lss_mean']),
        'retain_lss_mean': float(results['retain_lss_mean']),
        'weights': results['weights']
    }
    
    os.makedirs('results/metrics', exist_ok=True)
    with open(f'results/metrics/lgsa_{dataset_name}_{model_name}_gold_standard_seed{seed}.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    return full_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='simplecnn')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_gold_standard_experiment(args.dataset, args.model, args.seed, args.epochs, args.device)
