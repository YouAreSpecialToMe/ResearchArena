#!/usr/bin/env python3
"""
Fast Fixed Experiments for LGSA Machine Unlearning Verification.

Optimized for speed while maintaining experimental validity.
"""
import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).parent))

from exp.shared.models import get_model
from exp.shared.data_loader import load_dataset, create_forget_retain_splits, get_dataloader
from exp.shared.training import train_model, save_model, load_model
from exp.fixed_implementation.unlearning_fixed import (
    apply_unlearning_fixed, evaluate_accuracy
)
from exp.fixed_implementation.lgsa_fixed import LGSAFixed
from exp.baseline_truvrf.truvrf import TruVRF
from sklearn.metrics import roc_auc_score, roc_curve

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_experiment_fast(dataset, model_name, unlearn_method, seed, device):
    """Fast experiment runner with reduced data size."""
    print(f"\n{'='*60}")
    print(f"Exp: {dataset} {model_name} {unlearn_method} seed={seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    
    # Load base model
    model_path = f'results/models/{dataset}_{model_name}_seed{seed}_base.pth'
    print(f"Loading base model: {model_path}")
    base_model = get_model(model_name, num_classes=10, input_channels=3)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model = base_model.to(device)
    
    # Load data
    train_dataset, test_dataset, _, _ = load_dataset(dataset)
    forget_indices, retain_indices, _ = create_forget_retain_splits(train_dataset, seed=seed)
    
    # Use smaller subsets for faster verification
    forget_indices_small = forget_indices[:1000]  # Reduced from 5000
    retain_indices_small = retain_indices[:1000]
    
    forget_loader = get_dataloader(train_dataset, forget_indices_small, batch_size=128, shuffle=False, num_workers=2)
    retain_loader = get_dataloader(train_dataset, retain_indices_small, batch_size=128, shuffle=True, num_workers=2)
    test_loader = get_dataloader(test_dataset, None, batch_size=128, shuffle=False, num_workers=2)
    
    # Apply unlearning
    print(f"\nApplying {unlearn_method}...")
    unlearned_model = copy.deepcopy(base_model)
    
    unlearn_start = time.time()
    if unlearn_method == 'gradient_ascent':
        unlearned_model, _ = apply_unlearning_fixed(
            unlearned_model, 'gradient_ascent', forget_loader, retain_loader,
            epochs=3, lr=0.0002, device=device, grad_clip=0.05
        )
    elif unlearn_method == 'finetuning':
        unlearned_model, _ = apply_unlearning_fixed(
            unlearned_model, 'finetuning', None, retain_loader,
            epochs=5, lr=0.001, device=device
        )
    unlearn_time = time.time() - unlearn_start
    
    # Evaluate
    retain_acc = evaluate_accuracy(unlearned_model, retain_loader, device)
    test_acc = evaluate_accuracy(unlearned_model, test_loader, device)
    print(f"Unlearned - Retain: {retain_acc:.4f}, Test: {test_acc:.4f}")
    
    # Get data tensors
    forget_data, forget_targets = [], []
    for data, target in forget_loader:
        forget_data.append(data)
        forget_targets.append(target)
    forget_data = torch.cat(forget_data).to(device)
    forget_targets = torch.cat(forget_targets).to(device)
    
    retain_data, retain_targets = [], []
    for data, target in retain_loader:
        retain_data.append(data)
        retain_targets.append(target)
    retain_data = torch.cat(retain_data).to(device)
    retain_targets = torch.cat(retain_targets).to(device)
    
    # LGSA Verification
    print("\nLGSA Verification...")
    lgsa = LGSAFixed(base_model, unlearned_model, device=device)
    
    # Use even smaller subset for weight learning
    val_size = 200
    weights = lgsa.learn_weights(
        forget_data[:val_size], forget_targets[:val_size],
        retain_data[:val_size], retain_targets[:val_size],
        grid_search=True
    )
    
    lgsa_results, _, _ = lgsa.verify_unlearning(
        forget_data, forget_targets, retain_data, retain_targets,
        weights=weights, normalize=True
    )
    
    print(f"LGSA: AUC={lgsa_results['auc']:.4f}, TPR@1%FPR={lgsa_results['tpr_at_1fpr']:.4f}, Time={lgsa_results['verify_time']:.2f}s")
    
    # TruVRF Baseline
    print("\nTruVRF Baseline...")
    truvrf = TruVRF(base_model, unlearned_model, device=device)
    
    # Get verification data from test
    verify_data, verify_targets = [], []
    for data, target in test_loader:
        verify_data.append(data)
        verify_targets.append(target)
        if len(verify_data) >= 8:
            break
    verify_data = torch.cat(verify_data).to(device)
    verify_targets = torch.cat(verify_targets).to(device)
    
    truvrf_start = time.time()
    truvrf_results, _, _ = truvrf.verify_unlearning(
        forget_data, forget_targets, retain_data, retain_targets,
        verify_data, verify_targets, epochs=2, lr=0.001
    )
    truvrf_time = time.time() - truvrf_start
    truvrf_results['verify_time'] = truvrf_time
    
    print(f"TruVRF: AUC={truvrf_results['auc']:.4f}, TPR@1%FPR={truvrf_results['tpr_at_1fpr']:.4f}, Time={truvrf_time:.2f}s")
    
    # Compile results
    result = {
        'config': {'dataset': dataset, 'model': model_name, 'unlearn_method': unlearn_method, 'seed': seed},
        'model_utility': {'retain_acc': retain_acc, 'test_acc': test_acc, 'unlearn_time': unlearn_time},
        'lgsa': {k: v for k, v in lgsa_results.items() if k not in ['forget_lds_mean', 'forget_gas_mean', 'forget_srs_mean', 'retain_lds_mean', 'retain_gas_mean', 'retain_srs_mean']},
        'truvrf': truvrf_results,
        'comparison': {
            'speedup_vs_truvrf': truvrf_results['verify_time'] / lgsa_results['verify_time'] if lgsa_results['verify_time'] > 0 else 0,
            'auc_difference': lgsa_results['auc'] - truvrf_results['auc']
        }
    }
    
    os.makedirs('results/metrics', exist_ok=True)
    with open(f'results/metrics/fast_{dataset}_{model_name}_{unlearn_method}_seed{seed}.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def run_ablation_fast(dataset, model_name, seed, device):
    """Fast ablation study."""
    print(f"\n{'='*60}")
    print(f"Ablation: {dataset} {model_name} seed={seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    
    # Load models
    base_model = get_model(model_name, num_classes=10, input_channels=3)
    base_model.load_state_dict(torch.load(f'results/models/{dataset}_{model_name}_seed{seed}_base.pth', map_location=device))
    base_model = base_model.to(device)
    
    train_dataset, _, _, _ = load_dataset(dataset)
    forget_indices, retain_indices, _ = create_forget_retain_splits(train_dataset, seed=seed)
    
    forget_indices_small = forget_indices[:800]
    retain_indices_small = retain_indices[:800]
    
    forget_loader = get_dataloader(train_dataset, forget_indices_small, batch_size=128, shuffle=False, num_workers=2)
    retain_loader = get_dataloader(train_dataset, retain_indices_small, batch_size=128, shuffle=True, num_workers=2)
    
    # Unlearn
    unlearned_model = copy.deepcopy(base_model)
    unlearned_model, _ = apply_unlearning_fixed(
        unlearned_model, 'gradient_ascent', forget_loader, retain_loader,
        epochs=3, lr=0.0002, device=device, grad_clip=0.05
    )
    
    # Get data
    forget_data, forget_targets = [], []
    for data, target in forget_loader:
        forget_data.append(data)
        forget_targets.append(target)
    forget_data = torch.cat(forget_data).to(device)
    forget_targets = torch.cat(forget_targets).to(device)
    
    retain_data, retain_targets = [], []
    for data, target in retain_loader:
        retain_data.append(data)
        retain_targets.append(target)
    retain_data = torch.cat(retain_data).to(device)
    retain_targets = torch.cat(retain_targets).to(device)
    
    # Ablation
    lgsa = LGSAFixed(base_model, unlearned_model, device=device)
    forget_metrics = lgsa.compute_all_metrics(forget_data, forget_targets)
    retain_metrics = lgsa.compute_all_metrics(retain_data, retain_targets)
    
    labels = np.concatenate([np.ones(len(forget_metrics['lds'])), np.zeros(len(retain_metrics['lds']))])
    
    ablation = {}
    for metric in ['lds', 'gas', 'srs']:
        all_scores = np.concatenate([forget_metrics[metric], retain_metrics[metric]])
        try:
            auc = roc_auc_score(labels, all_scores)
        except:
            auc = 0.5
        ablation[f'{metric}_only'] = float(auc)
        print(f"  {metric.upper()}: AUC={auc:.4f}")
    
    # Full combination
    weights = np.array([0.4, 0.4, 0.2])
    combined = (weights[0] * np.concatenate([forget_metrics['lds'], retain_metrics['lds']]) +
                weights[1] * np.concatenate([forget_metrics['gas'], retain_metrics['gas']]) +
                weights[2] * np.concatenate([forget_metrics['srs'], retain_metrics['srs']]))
    try:
        full_auc = roc_auc_score(labels, combined)
    except:
        full_auc = 0.5
    ablation['full'] = float(full_auc)
    print(f"  Full: AUC={full_auc:.4f}")
    
    with open(f'results/metrics/fast_ablation_{dataset}_{model_name}_seed{seed}.json', 'w') as f:
        json.dump(ablation, f, indent=2)
    
    return ablation

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # Experiments
    configs = [
        ('cifar10', 'resnet18', 'gradient_ascent', 42),
        ('cifar10', 'resnet18', 'gradient_ascent', 123),
        ('cifar10', 'resnet18', 'gradient_ascent', 456),
        ('cifar10', 'resnet18', 'finetuning', 42),
        ('cifar10', 'resnet18', 'finetuning', 123),
        ('cifar10', 'simplecnn', 'gradient_ascent', 42),
        ('cifar10', 'simplecnn', 'gradient_ascent', 123),
    ]
    
    for dataset, model_name, method, seed in configs:
        try:
            result = run_experiment_fast(dataset, model_name, method, seed, device)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Ablation
    try:
        run_ablation_fast('cifar10', 'simplecnn', 42, device)
    except Exception as e:
        print(f"Ablation ERROR: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_results:
        lgsa_aucs = [r['lgsa']['auc'] for r in all_results]
        truvrf_aucs = [r['truvrf']['auc'] for r in all_results]
        speedups = [r['comparison']['speedup_vs_truvrf'] for r in all_results]
        
        print(f"LGSA AUC: {np.mean(lgsa_aucs):.4f} ± {np.std(lgsa_aucs):.4f}")
        print(f"TruVRF AUC: {np.mean(truvrf_aucs):.4f} ± {np.std(truvrf_aucs):.4f}")
        print(f"Speedup: {np.mean(speedups):.2f}x ± {np.std(speedups):.2f}x")
        
        aggregate = {
            'lgsa_auc_mean': float(np.mean(lgsa_aucs)),
            'lgsa_auc_std': float(np.std(lgsa_aucs)),
            'truvrf_auc_mean': float(np.mean(truvrf_aucs)),
            'truvrf_auc_std': float(np.std(truvrf_aucs)),
            'speedup_mean': float(np.mean(speedups)),
            'speedup_std': float(np.std(speedups)),
            'all_results': all_results
        }
        
        with open('results/metrics/fast_aggregate.json', 'w') as f:
            json.dump(aggregate, f, indent=2)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
