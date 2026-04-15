"""
Main experiment runner for AdaToken.
Runs all baselines and AdaToken on corruption benchmarks.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_configs import load_model, get_num_classes
from src.data_loader import (
    SyntheticCorruptedDataset, 
    get_cifar_c_loader, 
    CORRUPTIONS_SUBSET,
    CIFAR_C_CORRUPTIONS
)
from src.baselines.source import run_source_baseline
from src.baselines.tent import run_tent_baseline
from src.baselines.eata import run_eata_baseline
from src.adatoken import run_adatoken


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_synthetic_loader(dataset: str, corruption: str, severity: int, 
                            batch_size: int = 64, num_samples: int = 5000):
    """Create a synthetic corrupted dataset loader."""
    num_classes = get_num_classes(dataset)
    image_size = 32 if 'cifar' in dataset else 224
    
    synthetic_dataset = SyntheticCorruptedDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        corruption_type=corruption,
        severity=severity,
        image_size=image_size
    )
    
    loader = DataLoader(synthetic_dataset, batch_size=batch_size, 
                        shuffle=False, num_workers=2)
    return loader


def run_method(method_name: str, model: nn.Module, loader, 
               num_classes: int, device: str, **kwargs) -> Dict:
    """Run a TTA method and return metrics."""
    print(f"\n{'='*60}")
    print(f"Running {method_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if method_name == 'source':
        metrics = run_source_baseline(model, loader, device)
    elif method_name == 'tent':
        lr = kwargs.get('lr', 1e-3)
        metrics = run_tent_baseline(model, loader, lr=lr, device=device)
    elif method_name == 'eata':
        lr = kwargs.get('lr', 1e-3)
        metrics = run_eata_baseline(model, loader, lr=lr, device=device)
    elif method_name == 'adatoken':
        lr = kwargs.get('lr', 1e-3)
        head_layer = kwargs.get('head_layer', 8)
        alpha = kwargs.get('alpha', 0.5)
        metrics = run_adatoken(model, loader, num_classes=num_classes,
                               head_layer=head_layer, alpha=alpha, 
                               lr=lr, device=device)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    total_time = time.time() - start_time
    metrics['total_runtime_seconds'] = total_time
    
    print(f"\n{method_name} Results:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Time per sample: {metrics['time_per_sample_ms']:.2f} ms")
    
    return metrics


def run_single_experiment(dataset: str, corruption: str, severity: int,
                          model_name: str, method: str, seed: int,
                          use_real_data: bool = False) -> Dict:
    """Run a single experiment configuration."""
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    num_classes = get_num_classes(dataset)
    model = load_model(model_name, num_classes=num_classes, pretrained=False)
    
    # Create data loader
    if use_real_data and os.path.exists(f'data/{dataset}-c'):
        try:
            loader = get_cifar_c_loader(f'data', dataset=dataset,
                                        corruption=corruption, severity=severity,
                                        batch_size=64, num_workers=2)
        except:
            loader = create_synthetic_loader(dataset, corruption, severity)
    else:
        loader = create_synthetic_loader(dataset, corruption, severity)
    
    # Run method
    metrics = run_method(method, model, loader, num_classes, device)
    
    # Add experiment metadata
    metrics['dataset'] = dataset
    metrics['corruption'] = corruption
    metrics['severity'] = severity
    metrics['model'] = model_name
    metrics['method'] = method
    metrics['seed'] = seed
    
    return metrics


def run_full_experiment_suite(dataset: str = 'cifar10',
                              model_name: str = 'deit_small_patch16_224',
                              seeds: List[int] = [42, 123, 2024],
                              corruptions: List[str] = None,
                              severities: List[int] = [3, 5]):
    """Run full experiment suite with multiple seeds, corruptions, and methods."""
    
    if corruptions is None:
        corruptions = ['gaussian_noise', 'defocus_blur', 'jpeg_compression']
    
    methods = ['source', 'tent', 'eata', 'adatoken']
    
    all_results = []
    
    for seed in seeds:
        print(f"\n{'#'*80}")
        print(f"# SEED {seed}")
        print(f"{'#'*80}")
        
        for corruption in corruptions:
            for severity in severities:
                print(f"\n{'#'*60}")
                print(f"# {corruption} - Severity {severity}")
                print(f"{'#'*60}")
                
                for method in methods:
                    result = run_single_experiment(
                        dataset=dataset,
                        corruption=corruption,
                        severity=severity,
                        model_name=model_name,
                        method=method,
                        seed=seed
                    )
                    all_results.append(result)
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    return all_results, aggregated


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
    aggregated = {}
    
    # Group by method, corruption, severity
    groups = {}
    for r in results:
        key = (r['method'], r['corruption'], r['severity'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    # Compute statistics
    for key, group in groups.items():
        method, corruption, severity = key
        
        accs = [r['accuracy'] for r in group]
        times = [r['time_per_sample_ms'] for r in group]
        
        if key not in aggregated:
            aggregated[key] = {}
        
        aggregated[key] = {
            'method': method,
            'corruption': corruption,
            'severity': severity,
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'n_seeds': len(accs)
        }
        
        # Add method-specific metrics
        if 'avg_selection_ratio' in group[0]:
            ratios = [r['avg_selection_ratio'] for r in group]
            aggregated[key]['selection_ratio_mean'] = np.mean(ratios)
            aggregated[key]['selection_ratio_std'] = np.std(ratios)
        
        if 'filter_ratio' in group[0]:
            filters = [r['filter_ratio'] for r in group]
            aggregated[key]['filter_ratio_mean'] = np.mean(filters)
            aggregated[key]['filter_ratio_std'] = np.std(filters)
    
    return aggregated


def save_results(all_results: List[Dict], aggregated: Dict, output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save aggregated results
    # Convert tuple keys to strings
    agg_serializable = {}
    for key, value in aggregated.items():
        key_str = f"{key[0]}_{key[1]}_sev{key[2]}"
        agg_serializable[key_str] = value
    
    with open(os.path.join(output_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(agg_serializable, f, indent=2)
    
    # Create summary table
    summary_lines = []
    summary_lines.append("Method | Corruption | Severity | Acc (mean±std) | Time (ms)")
    summary_lines.append("-" * 70)
    
    for key, value in aggregated.items():
        line = f"{value['method']:10s} | {value['corruption']:15s} | {value['severity']} | "
        line += f"{value['accuracy_mean']:.2f}±{value['accuracy_std']:.2f} | "
        line += f"{value['time_mean']:.2f}"
        summary_lines.append(line)
    
    summary_text = "\n".join(summary_lines)
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(summary_text)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TTA experiments')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, default='deit_small_patch16_224')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024])
    parser.add_argument('--corruptions', type=str, nargs='+', 
                        default=['gaussian_noise', 'defocus_blur', 'jpeg_compression'])
    parser.add_argument('--severities', type=int, nargs='+', default=[3, 5])
    
    args = parser.parse_args()
    
    print(f"Running experiments:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Corruptions: {args.corruptions}")
    print(f"  Severities: {args.severities}")
    
    all_results, aggregated = run_full_experiment_suite(
        dataset=args.dataset,
        model_name=args.model,
        seeds=args.seeds,
        corruptions=args.corruptions,
        severities=args.severities
    )
    
    save_results(all_results, aggregated, args.output_dir)
    
    print("\nExperiment suite complete!")
