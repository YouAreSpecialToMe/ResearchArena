#!/usr/bin/env python3
"""
Run LGSA with aggressive gradient ascent to create clear unlearning signal.
"""
import sys
import os
sys.path.insert(0, 'exp')

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import time
from shared.models import get_model
from shared.data_loader import load_dataset, load_splits, get_dataloader
from shared.training import load_model
from shared.metrics import compute_accuracy
from lgsa_core.lgsa import LGSA


def aggressive_gradient_ascent(model, forget_loader, epochs=3, lr=0.001, device='cuda'):
    """Aggressive gradient ascent with gradient clipping."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        
        for data, target in forget_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = -criterion(output, target)  # Negative for ascent
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            count += data.size(0)
        
        avg_loss = total_loss / count
        print(f"GA Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    return model


def run_experiment(dataset_name, model_name, seed=42, device='cuda'):
    """Run LGSA verification with aggressive unlearning."""
    
    print(f"\n{'='*60}")
    print(f"LGSA (Aggressive GA): {dataset_name}, {model_name}, seed={seed}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    train_dataset, test_dataset, num_classes, input_channels = load_dataset(dataset_name)
    splits = load_splits(f'data/{dataset_name}_splits_seed{seed}.pkl')
    forget_indices = splits['forget']
    retain_indices = splits['retain']
    
    # Load base model
    model_path = f'results/models/{dataset_name}_{model_name}_seed{seed}_base.pth'
    original_model = get_model(model_name, num_classes, input_channels)
    load_model(original_model, model_path, device)
    
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    base_acc = compute_accuracy(original_model, test_loader, device)
    print(f"Base model test accuracy: {base_acc:.4f}")
    
    # Create unlearned model
    unlearned_model = get_model(model_name, num_classes, input_channels)
    load_model(unlearned_model, model_path, device)
    
    # Apply aggressive gradient ascent
    print("\nApplying aggressive gradient ascent...")
    forget_loader = get_dataloader(train_dataset, indices=forget_indices, 
                                   batch_size=128, shuffle=True, num_workers=4)
    
    unlearn_start = time.time()
    unlearned_model = aggressive_gradient_ascent(unlearned_model, forget_loader, 
                                                  epochs=3, lr=0.001, device=device)
    unlearn_time = time.time() - unlearn_start
    
    unlearn_acc = compute_accuracy(unlearned_model, test_loader, device)
    print(f"Unlearned model test accuracy: {unlearn_acc:.4f}")
    
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
    lgsa = LGSA(original_model, unlearned_model, device)
    
    verify_start = time.time()
    results, scores, labels = lgsa.verify_unlearning(
        forget_data, forget_targets, retain_data, retain_targets)
    verify_time = time.time() - verify_start
    
    print(f"\nLGSA Results:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  TPR@1%FPR: {results['tpr_at_1fpr']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Verification time: {verify_time:.2f}s")
    
    # Save results
    full_results = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'unlearn_method': 'aggressive_ga',
        'base_accuracy': base_acc,
        'unlearn_accuracy': unlearn_acc,
        'unlearn_time': unlearn_time,
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
    with open(f'results/metrics/lgsa_{dataset_name}_{model_name}_aggressive_ga_seed{seed}.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    return full_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    run_experiment(args.dataset, args.model, args.seed, args.device)
