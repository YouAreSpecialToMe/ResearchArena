"""
Complete ablation study with all metric combinations.

Tests:
1. Individual metrics: LDS only, GAS only, SRS only
2. Pairwise combinations: LDS+GAS, LDS+SRS, GAS+SRS  
3. Full combination: LDS+GAS+SRS
"""
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.models import SimpleCNN
from shared.data_loader import get_cifar10_loaders, create_forget_retain_split
from lgsa_core.lgsa import LGSA


def run_ablation_config(original_model, unlearned_model, forget_data, forget_targets,
                        retain_data, retain_targets, weights, config_name, seed, device='cuda'):
    """Run LGSA with specific weight configuration."""
    
    lgsa = LGSA(original_model, unlearned_model, device=device)
    lgsa.weights = np.array(weights)
    
    start_time = time.time()
    results, _, _ = lgsa.verify_unlearning(
        forget_data, forget_targets,
        retain_data, retain_targets,
        weights=weights
    )
    verify_time = time.time() - start_time
    
    result = {
        'dataset': 'cifar10',
        'model': 'simplecnn',
        'seed': seed,
        'config': config_name,
        'weights': weights,
        'auc': results['auc'],
        'tpr_at_1fpr': results['tpr_at_1fpr'],
        'verify_time': verify_time,
        'forget_lss_mean': results['forget_lss_mean'],
        'retain_lss_mean': results['retain_lss_mean']
    }
    
    print(f"  {config_name:20s}: AUC={results['auc']:.4f}, Time={verify_time:.2f}s")
    
    return result


def run_complete_ablation(seed=42, device='cuda'):
    """Run complete ablation study with all metric combinations."""
    
    print(f"\n{'='*60}")
    print(f"Complete Ablation Study - All Metric Combinations")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load models
    base_model_path = f'../../results/models/cifar10_simplecnn_seed{seed}_base.pth'
    unlearned_model_path = f'../../results/models/cifar10_simplecnn_seed{seed}_unlearned.pth'
    
    # Try gold standard if unlearned not available
    if not os.path.exists(unlearned_model_path):
        unlearned_model_path = f'../../results/models/cifar10_simplecnn_seed{seed}_gold_standard.pth'
    
    if not os.path.exists(base_model_path) or not os.path.exists(unlearned_model_path):
        print(f"Models not found. Please train base models first.")
        return []
    
    original_model = SimpleCNN(num_classes=10)
    original_model.load_state_dict(torch.load(base_model_path, map_location=device))
    original_model = original_model.to(device)
    
    unlearned_model = SimpleCNN(num_classes=10)
    unlearned_model.load_state_dict(torch.load(unlearned_model_path, map_location=device))
    unlearned_model = unlearned_model.to(device)
    
    # Load data splits
    import pickle
    split_path = f'../../data/cifar10_splits_seed{seed}.pkl'
    with open(split_path, 'rb') as f:
        splits = pickle.load(f)
    
    forget_data = torch.tensor(splits['forget_data']).float()[:1000].to(device)
    forget_targets = torch.tensor(splits['forget_targets']).long()[:1000].to(device)
    retain_data = torch.tensor(splits['retain_data']).float()[:1000].to(device)
    retain_targets = torch.tensor(splits['retain_targets']).long()[:1000].to(device)
    
    results = []
    
    print("\n1. Individual Metrics:")
    print("-" * 40)
    
    # LDS only
    result = run_ablation_config(original_model, unlearned_model, 
                                 forget_data, forget_targets,
                                 retain_data, retain_targets,
                                 [1.0, 0.0, 0.0], 'lds_only', seed, device)
    results.append(result)
    
    # GAS only
    result = run_ablation_config(original_model, unlearned_model,
                                 forget_data, forget_targets,
                                 retain_data, retain_targets,
                                 [0.0, 1.0, 0.0], 'gas_only', seed, device)
    results.append(result)
    
    # SRS only
    result = run_ablation_config(original_model, unlearned_model,
                                 forget_data, forget_targets,
                                 retain_data, retain_targets,
                                 [0.0, 0.0, 1.0], 'srs_only', seed, device)
    results.append(result)
    
    print("\n2. Pairwise Combinations:")
    print("-" * 40)
    
    # LDS + GAS
    result = run_ablation_config(original_model, unlearned_model,
                                 forget_data, forget_targets,
                                 retain_data, retain_targets,
                                 [0.5, 0.5, 0.0], 'lds_gas', seed, device)
    results.append(result)
    
    # LDS + SRS
    result = run_ablation_config(original_model, unlearned_model,
                                 forget_data, forget_targets,
                                 retain_data, retain_targets,
                                 [0.5, 0.0, 0.5], 'lds_srs', seed, device)
    results.append(result)
    
    # GAS + SRS
    result = run_ablation_config(original_model, unlearned_model,
                                 forget_data, forget_targets,
                                 retain_data, retain_targets,
                                 [0.0, 0.5, 0.5], 'gas_srs', seed, device)
    results.append(result)
    
    print("\n3. Full Combination (Default Weights):")
    print("-" * 40)
    
    # LDS + GAS + SRS (default weights)
    result = run_ablation_config(original_model, unlearned_model,
                                 forget_data, forget_targets,
                                 retain_data, retain_targets,
                                 [0.4, 0.4, 0.2], 'lds_gas_srs', seed, device)
    results.append(result)
    
    # LDS + GAS + SRS (equal weights)
    result = run_ablation_config(original_model, unlearned_model,
                                 forget_data, forget_targets,
                                 retain_data, retain_targets,
                                 [0.333, 0.333, 0.334], 'lds_gas_srs_equal', seed, device)
    results.append(result)
    
    # Save results
    results_dir = '../../results/metrics'
    os.makedirs(results_dir, exist_ok=True)
    
    for result in results:
        config_name = result['config']
        result_path = os.path.join(results_dir, f'ablation_cifar10_simplecnn_{config_name}_seed{seed}.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Ablation study complete!")
    print(f"Results saved to {results_dir}/")
    print(f"{'='*60}\n")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run complete ablation study')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    run_complete_ablation(seed=args.seed, device=args.device)


if __name__ == '__main__':
    main()
