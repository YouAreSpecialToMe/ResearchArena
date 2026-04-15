#!/usr/bin/env python3
"""
Main LGSA Experiments Runner - Consolidated.

This script runs all experiments in a streamlined manner.
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, 'exp')

from shared.models import get_model, count_parameters
from shared.data_loader import load_dataset, create_forget_retain_splits, load_splits, save_splits, get_dataloader
from shared.training import train_model, load_model, save_model, train_shadow_models
from shared.metrics import compute_accuracy
from unlearning_methods.unlearning import apply_unlearning
from lgsa_core.lgsa import LGSA


def train_base_models(device='cuda'):
    """Train all base models."""
    print("\n" + "="*60)
    print("TRAINING BASE MODELS")
    print("="*60)
    
    configs = [
        ('cifar10', 'resnet18', 42, 30),
        ('cifar10', 'resnet18', 123, 30),
        ('cifar10', 'resnet18', 456, 30),
        ('cifar10', 'simplecnn', 42, 20),
        ('cifar10', 'simplecnn', 123, 20),
        ('cifar10', 'simplecnn', 456, 20),
        ('fashion-mnist', 'resnet18', 42, 20),
    ]
    
    for dataset, model_name, seed, epochs in configs:
        print(f"\nTraining {dataset} {model_name} seed={seed}...")
        
        # Load dataset
        train_dataset, test_dataset, num_classes, input_channels = load_dataset(dataset)
        
        # Create and save splits
        splits_path = f'data/{dataset}_splits_seed{seed}.pkl'
        if not os.path.exists(splits_path):
            forget_indices, retain_indices, val_indices = create_forget_retain_splits(
                train_dataset, forget_ratio=0.1, seed=seed)
            save_splits({
                'forget': forget_indices,
                'retain': retain_indices,
                'val': val_indices,
                'seed': seed
            }, splits_path)
        else:
            splits = load_splits(splits_path)
            forget_indices = splits['forget']
            retain_indices = splits['retain']
        
        # Train model
        full_indices = forget_indices + retain_indices
        train_loader = get_dataloader(train_dataset, indices=full_indices, batch_size=128, shuffle=True)
        test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False)
        
        model = get_model(model_name, num_classes, input_channels)
        model, history = train_model(model, train_loader, epochs=epochs, lr=0.1, device=device, verbose=False)
        
        # Save model
        save_model(model, f'results/models/{dataset}_{model_name}_seed{seed}_base.pth')
        
        # Evaluate
        test_acc = compute_accuracy(model, test_loader, device)
        print(f"  Test accuracy: {test_acc:.4f}")


def run_lgsa_experiments(device='cuda'):
    """Run LGSA verification experiments."""
    print("\n" + "="*60)
    print("RUNNING LGSA EXPERIMENTS")
    print("="*60)
    
    configs = [
        ('cifar10', 'resnet18', 'gradient_ascent', [42, 123, 456], 5, 0.01),
        ('cifar10', 'resnet18', 'finetuning', [42, 123, 456], 10, 0.001),
        ('cifar10', 'simplecnn', 'gradient_ascent', [42, 123, 456], 5, 0.01),
        ('fashion-mnist', 'resnet18', 'gradient_ascent', [42], 5, 0.01),
    ]
    
    all_results = []
    
    for dataset, model_name, method, seeds, unlearn_epochs, unlearn_lr in configs:
        for seed in seeds:
            print(f"\nLGSA: {dataset} {model_name} {method} seed={seed}")
            
            # Load dataset
            train_dataset, _, num_classes, input_channels = load_dataset(dataset)
            splits = load_splits(f'data/{dataset}_splits_seed{seed}.pkl')
            forget_indices = splits['forget'][:1000]
            retain_indices = splits['retain'][:1000]
            
            # Load models
            original_model = get_model(model_name, num_classes, input_channels)
            load_model(original_model, f'results/models/{dataset}_{model_name}_seed{seed}_base.pth', device)
            
            unlearned_model = get_model(model_name, num_classes, input_channels)
            load_model(unlearned_model, f'results/models/{dataset}_{model_name}_seed{seed}_base.pth', device)
            
            # Apply unlearning
            forget_loader = get_dataloader(train_dataset, indices=forget_indices, batch_size=128, shuffle=True)
            retain_loader = get_dataloader(train_dataset, indices=retain_indices, batch_size=128, shuffle=True)
            
            unlearn_start = time.time()
            unlearned_model = apply_unlearning(unlearned_model, method, forget_loader, retain_loader,
                                              epochs=unlearn_epochs, lr=unlearn_lr, device=device)
            unlearn_time = time.time() - unlearn_start
            
            # Get data for verification
            forget_loader_full = get_dataloader(train_dataset, indices=forget_indices[:500], 
                                               batch_size=500, shuffle=False)
            retain_loader_full = get_dataloader(train_dataset, indices=retain_indices[:500],
                                               batch_size=500, shuffle=False)
            
            forget_data, forget_targets = next(iter(forget_loader_full))
            retain_data, retain_targets = next(iter(retain_loader_full))
            
            # Run LGSA
            lgsa = LGSA(original_model, unlearned_model, device)
            results, scores, labels = lgsa.verify_unlearning(forget_data, forget_targets, 
                                                              retain_data, retain_targets)
            
            result_entry = {
                'method': 'LGSA',
                'dataset': dataset,
                'model': model_name,
                'unlearn_method': method,
                'seed': seed,
                'unlearn_time': unlearn_time,
                'auc': results['auc'],
                'tpr_at_1fpr': results['tpr_at_1fpr'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'verify_time': results['verify_time'],
                'weights': results['weights']
            }
            
            all_results.append(result_entry)
            
            print(f"  AUC: {results['auc']:.4f}")
            print(f"  Verify time: {results['verify_time']:.2f}s")
    
    # Save results
    with open('results/metrics/lgsa_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def run_ablation_studies(device='cuda'):
    """Run ablation studies."""
    print("\n" + "="*60)
    print("RUNNING ABLATION STUDIES")
    print("="*60)
    
    dataset = 'cifar10'
    model_name = 'simplecnn'
    
    ablation_results = {
        'individual_metrics': [],
        'fixed_vs_learned': []
    }
    
    for seed in [42, 123]:
        print(f"\nAblation seed={seed}")
        
        # Load data
        train_dataset, _, num_classes, input_channels = load_dataset(dataset)
        splits = load_splits(f'data/{dataset}_splits_seed{seed}.pkl')
        forget_indices = splits['forget'][:500]
        retain_indices = splits['retain'][:500]
        val_indices = splits['val'][:300]
        
        # Load models
        original_model = get_model(model_name, num_classes, input_channels)
        load_model(original_model, f'results/models/{dataset}_{model_name}_seed{seed}_base.pth', device)
        
        unlearned_model = get_model(model_name, num_classes, input_channels)
        load_model(unlearned_model, f'results/models/{dataset}_{model_name}_seed{seed}_base.pth', device)
        
        # Apply unlearning
        forget_loader = get_dataloader(train_dataset, indices=forget_indices, batch_size=128, shuffle=True)
        unlearned_model = apply_unlearning(unlearned_model, 'gradient_ascent', forget_loader, None,
                                          epochs=5, lr=0.01, device=device)
        
        # Get data
        forget_loader_full = get_dataloader(train_dataset, indices=forget_indices, batch_size=len(forget_indices), shuffle=False)
        retain_loader_full = get_dataloader(train_dataset, indices=retain_indices, batch_size=len(retain_indices), shuffle=False)
        val_loader_full = get_dataloader(train_dataset, indices=val_indices, batch_size=len(val_indices), shuffle=False)
        
        forget_data, forget_targets = next(iter(forget_loader_full))
        retain_data, retain_targets = next(iter(retain_loader_full))
        val_data, val_targets = next(iter(val_loader_full))
        
        lgsa = LGSA(original_model, unlearned_model, device)
        
        # Test individual metrics
        weights_configs = {
            'lds_only': [1.0, 0.0, 0.0],
            'gas_only': [0.0, 1.0, 0.0],
            'srs_only': [0.0, 0.0, 1.0],
            'lds_gas': [0.5, 0.5, 0.0],
            'lds_srs': [0.5, 0.0, 0.5],
            'gas_srs': [0.0, 0.5, 0.5],
            'all_three': [0.4, 0.4, 0.2]
        }
        
        for name, weights in weights_configs.items():
            lgsa.weights = np.array(weights)
            results, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
            ablation_results['individual_metrics'].append({
                'seed': seed,
                'config': name,
                'weights': weights,
                'auc': results['auc'],
                'tpr_at_1fpr': results['tpr_at_1fpr']
            })
            print(f"  {name}: AUC={results['auc']:.4f}")
        
        # Fixed vs learned weights
        lgsa.weights = np.array([0.4, 0.4, 0.2])
        res_fixed, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
        
        # Learn weights
        val_forget_data = val_data[:len(val_data)//2]
        val_forget_targets = val_targets[:len(val_targets)//2]
        val_retain_data = val_data[len(val_data)//2:]
        val_retain_targets = val_targets[len(val_targets)//2:]
        
        learned_weights = lgsa.learn_weights(val_forget_data, val_forget_targets,
                                            val_retain_data, val_retain_targets)
        res_learned, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, retain_data, retain_targets)
        
        ablation_results['fixed_vs_learned'].append({
            'seed': seed,
            'fixed_auc': res_fixed['auc'],
            'learned_auc': res_learned['auc'],
            'improvement': res_learned['auc'] - res_fixed['auc'],
            'fixed_weights': [0.4, 0.4, 0.2],
            'learned_weights': learned_weights.tolist()
        })
        print(f"  Fixed: {res_fixed['auc']:.4f}, Learned: {res_learned['auc']:.4f}, Improvement: {res_learned['auc']-res_fixed['auc']:.4f}")
    
    # Save results
    with open('results/metrics/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    return ablation_results


def run_complexity_analysis(device='cuda'):
    """Run complexity analysis."""
    print("\n" + "="*60)
    print("RUNNING COMPLEXITY ANALYSIS")
    print("="*60)
    
    dataset = 'cifar10'
    seed = 42
    
    complexity_results = []
    
    for model_name in ['simplecnn', 'resnet18']:
        print(f"\nModel: {model_name}")
        
        # Load data
        train_dataset, _, num_classes, input_channels = load_dataset(dataset)
        splits = load_splits(f'data/{dataset}_splits_seed{seed}.pkl')
        forget_indices = splits['forget'][:500]
        
        # Load models
        original_model = get_model(model_name, num_classes, input_channels)
        load_model(original_model, f'results/models/{dataset}_{model_name}_seed{seed}_base.pth', device)
        
        unlearned_model = get_model(model_name, num_classes, input_channels)
        load_model(unlearned_model, f'results/models/{dataset}_{model_name}_seed{seed}_base.pth', device)
        
        # Get data
        forget_loader_full = get_dataloader(train_dataset, indices=forget_indices, batch_size=len(forget_indices), shuffle=False)
        forget_data, forget_targets = next(iter(forget_loader_full))
        
        # Time LGSA
        lgsa = LGSA(original_model, unlearned_model, device)
        
        times = []
        for _ in range(3):
            start = time.time()
            _ = lgsa.compute_all_metrics(forget_data[:100], forget_targets[:100])
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        n_params = count_parameters(original_model)
        
        complexity_results.append({
            'model': model_name,
            'n_params': n_params,
            'time_100_samples': avg_time,
            'time_per_sample': avg_time / 100
        })
        
        print(f"  Parameters: {n_params:,}")
        print(f"  Time for 100 samples: {avg_time:.3f}s")
    
    # Save results
    with open('results/metrics/complexity_analysis.json', 'w') as f:
        json.dump(complexity_results, f, indent=2)
    
    return complexity_results


def compile_final_results():
    """Compile all results into final summary."""
    print("\n" + "="*60)
    print("COMPILING FINAL RESULTS")
    print("="*60)
    
    results = {}
    
    # Load LGSA results
    if os.path.exists('results/metrics/lgsa_results.json'):
        with open('results/metrics/lgsa_results.json', 'r') as f:
            lgsa_results = json.load(f)
        results['lgsa'] = lgsa_results
        
        # Compute aggregated statistics
        for dataset in ['cifar10', 'fashion-mnist']:
            for model in ['resnet18', 'simplecnn']:
                for method in ['gradient_ascent', 'finetuning']:
                    entries = [e for e in lgsa_results 
                              if e['dataset'] == dataset and e['model'] == model 
                              and e['unlearn_method'] == method]
                    if entries:
                        aucs = [e['auc'] for e in entries]
                        times = [e['verify_time'] for e in entries]
                        results[f'{dataset}_{model}_{method}'] = {
                            'auc_mean': float(np.mean(aucs)),
                            'auc_std': float(np.std(aucs)),
                            'verify_time_mean': float(np.mean(times)),
                            'n_seeds': len(entries)
                        }
    
    # Load ablation results
    if os.path.exists('results/metrics/ablation_results.json'):
        with open('results/metrics/ablation_results.json', 'r') as f:
            ablation_results = json.load(f)
        results['ablation'] = ablation_results
    
    # Load complexity results
    if os.path.exists('results/metrics/complexity_analysis.json'):
        with open('results/metrics/complexity_analysis.json', 'r') as f:
            complexity_results = json.load(f)
        results['complexity'] = complexity_results
    
    # Save final results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults compiled and saved to results.json")
    
    return results


def main():
    """Run all experiments."""
    start_time = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    for d in ['results/models', 'results/metrics', 'results/figures', 'results/tables', 'data']:
        os.makedirs(d, exist_ok=True)
    
    # Run experiments
    train_base_models(device)
    run_lgsa_experiments(device)
    run_ablation_studies(device)
    run_complexity_analysis(device)
    
    # Compile results
    results = compile_final_results()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED IN {total_time/3600:.2f} HOURS")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
