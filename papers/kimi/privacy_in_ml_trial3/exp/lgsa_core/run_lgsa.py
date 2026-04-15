"""
Run LGSA verification experiments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import argparse
import numpy as np
import time
from shared.models import get_model
from shared.data_loader import load_dataset, load_splits, get_dataloader
from shared.training import load_model
from shared.metrics import compute_accuracy
from unlearning_methods.unlearning import apply_unlearning
from lgsa_core.lgsa import LGSA


def run_lgsa_experiment(dataset_name, model_name, unlearn_method, seed=42, 
                        unlearn_epochs=5, unlearn_lr=0.01, use_weight_learning=False,
                        device='cuda', results_dir='results/metrics'):
    """
    Run LGSA verification experiment.
    
    Args:
        dataset_name: 'cifar10' or 'fashion-mnist'
        model_name: 'resnet18' or 'simplecnn'
        unlearn_method: 'gradient_ascent', 'finetuning', 'random_label', 'salun'
        seed: Random seed
        unlearn_epochs: Epochs for unlearning
        unlearn_lr: Learning rate for unlearning
        use_weight_learning: Whether to use weight learning
        device: Device to run on
        results_dir: Directory to save results
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"LGSA Experiment: {dataset_name}, {model_name}, {unlearn_method}, seed={seed}")
    print(f"{'='*60}\n")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    print("Loading dataset...")
    train_dataset, test_dataset, num_classes, input_channels = load_dataset(dataset_name)
    
    # Load splits
    splits_path = f'data/{dataset_name}_splits_seed{seed}.pkl'
    if not os.path.exists(splits_path):
        print(f"Error: Splits not found at {splits_path}")
        return None
    
    splits = load_splits(splits_path)
    forget_indices = splits['forget']
    retain_indices = splits['retain']
    val_indices = splits['val']
    
    print(f"Forget set size: {len(forget_indices)}")
    print(f"Retain set size: {len(retain_indices)}")
    print(f"Val set size: {len(val_indices)}")
    
    # Load base model
    print("Loading base model...")
    model_path = f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth'
    if not os.path.exists(model_path):
        print(f"Error: Base model not found at {model_path}")
        return None
    
    original_model = get_model(model_name, num_classes, input_channels)
    load_model(original_model, model_path, device)
    
    # Evaluate base model
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False)
    base_acc = compute_accuracy(original_model, test_loader, device)
    print(f"Base model test accuracy: {base_acc:.4f}")
    
    # Create unlearned model (copy of original)
    unlearned_model = get_model(model_name, num_classes, input_channels)
    load_model(unlearned_model, model_path, device)
    
    # Apply unlearning
    print(f"\nApplying {unlearn_method} unlearning...")
    forget_loader = get_dataloader(train_dataset, indices=forget_indices, 
                                   batch_size=128, shuffle=True)
    retain_loader = get_dataloader(train_dataset, indices=retain_indices,
                                   batch_size=128, shuffle=True)
    
    unlearn_start = time.time()
    unlearned_model = apply_unlearning(unlearned_model, unlearn_method,
                                       forget_loader, retain_loader,
                                       epochs=unlearn_epochs, lr=unlearn_lr, 
                                       device=device)
    unlearn_time = time.time() - unlearn_start
    print(f"Unlearning took {unlearn_time:.2f}s")
    
    # Evaluate unlearned model
    unlearn_acc = compute_accuracy(unlearned_model, test_loader, device)
    print(f"Unlearned model test accuracy: {unlearn_acc:.4f}")
    
    # Prepare data for verification
    # Sample subset for faster computation
    n_forget_samples = min(1000, len(forget_indices))
    n_retain_samples = min(1000, len(retain_indices))
    n_val_samples = min(500, len(val_indices))
    
    forget_sample = np.random.choice(forget_indices, n_forget_samples, replace=False)
    retain_sample = np.random.choice(retain_indices, n_retain_samples, replace=False)
    val_sample = np.random.choice(val_indices, n_val_samples, replace=False)
    
    # Load data tensors
    forget_loader_small = get_dataloader(train_dataset, indices=forget_sample.tolist(),
                                         batch_size=n_forget_samples, shuffle=False)
    retain_loader_small = get_dataloader(train_dataset, indices=retain_sample.tolist(),
                                         batch_size=n_retain_samples, shuffle=False)
    val_loader_small = get_dataloader(train_dataset, indices=val_sample.tolist(),
                                      batch_size=n_val_samples, shuffle=False)
    
    forget_data, forget_targets = next(iter(forget_loader_small))
    retain_data, retain_targets = next(iter(retain_loader_small))
    val_data, val_targets = next(iter(val_loader_small))
    
    # Create LGSA instance
    print("\nRunning LGSA verification...")
    lgsa = LGSA(original_model, unlearned_model, device)
    
    # Weight learning (optional)
    weights = None
    if use_weight_learning:
        # Split val into pseudo-forget and pseudo-retain
        val_forget_data = val_data[:len(val_data)//2]
        val_forget_targets = val_targets[:len(val_targets)//2]
        val_retain_data = val_data[len(val_data)//2:]
        val_retain_targets = val_targets[len(val_targets)//2:]
        
        weights = lgsa.learn_weights(val_forget_data, val_forget_targets,
                                     val_retain_data, val_retain_targets)
    
    # Verify unlearning
    results, scores, labels = lgsa.verify_unlearning(
        forget_data, forget_targets, retain_data, retain_targets, weights)
    
    print(f"\nLGSA Results:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  TPR@1%FPR: {results['tpr_at_1fpr']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  Verification time: {results['verify_time']:.2f}s")
    print(f"  Weights: {results['weights']}")
    
    # Compile full results
    full_results = {
        'dataset': dataset_name,
        'model': model_name,
        'unlearn_method': unlearn_method,
        'seed': seed,
        'use_weight_learning': use_weight_learning,
        'unlearn_epochs': unlearn_epochs,
        'unlearn_lr': unlearn_lr,
        'base_accuracy': base_acc,
        'unlearn_accuracy': unlearn_acc,
        'unlearn_time': unlearn_time,
        **results,
        'forget_lss_mean': float(results['forget_lss_mean']),
        'forget_lss_std': float(results['forget_lss_std']),
        'retain_lss_mean': float(results['retain_lss_mean']),
        'retain_lss_std': float(results['retain_lss_std'])
    }
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, 
                               f'lgsa_{dataset_name}_{model_name}_{unlearn_method}_seed{seed}.json')
    with open(result_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved results to {result_path}")
    
    return full_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--unlearn_method', type=str, default='gradient_ascent')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unlearn_epochs', type=int, default=5)
    parser.add_argument('--unlearn_lr', type=float, default=0.01)
    parser.add_argument('--weight_learning', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    run_lgsa_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
        unlearn_method=args.unlearn_method,
        seed=args.seed,
        unlearn_epochs=args.unlearn_epochs,
        unlearn_lr=args.unlearn_lr,
        use_weight_learning=args.weight_learning,
        device=args.device
    )
