"""Train baseline interpretability methods (Random, PCA) on synthetic data."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
from sklearn.decomposition import PCA
import os
import json

from exp.shared.utils import set_seed, save_json

def train_random_baseline(seed: int, dict_size: int, hidden_train: np.ndarray, 
                          hidden_val: np.ndarray) -> dict:
    """Generate random projection baseline."""
    set_seed(seed)
    
    input_dim = hidden_train.shape[1]
    overcomplete = dict_size // input_dim
    
    print(f"\n  Generating Random {overcomplete}x (dict_size={dict_size}) with seed={seed}...")
    
    # Generate random projection matrix
    projection = np.random.randn(input_dim, dict_size).astype(np.float32)
    projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
    
    # Apply projection
    features_train = hidden_train @ projection
    features_val = hidden_val @ projection
    
    # Reconstruction (pseudo-inverse)
    recon_val = features_val @ np.linalg.pinv(projection)
    recon_error = np.mean((recon_val - hidden_val) ** 2)
    
    # Compute sparsity
    l0_sparsity = np.mean(features_val < 1e-6)
    l1_sparsity = np.mean(np.sum(np.abs(features_val), axis=1))
    
    result = {
        'seed': seed,
        'dict_size': dict_size,
        'overcomplete': overcomplete,
        'recon_error': float(recon_error),
        'l0_sparsity': float(l0_sparsity),
        'l1_sparsity': float(l1_sparsity),
    }
    
    # Save
    save_path = f'models/baseline_random_{overcomplete}x_seed{seed}.pt'
    torch.save({
        'projection': projection,
        'features_train': features_train,
        'features_val': features_val,
        'result': result
    }, save_path)
    
    print(f"    Recon error: {recon_error:.6f}")
    
    return result

def train_pca_baseline(seed: int, n_components: int, hidden_train: np.ndarray, 
                       hidden_val: np.ndarray) -> dict:
    """Train PCA baseline."""
    set_seed(seed)
    
    input_dim = hidden_train.shape[1]
    overcomplete = n_components // input_dim
    
    print(f"\n  Training PCA {overcomplete}x (n_components={n_components}) with seed={seed}...")
    
    # Fit PCA
    pca = PCA(n_components=n_components, random_state=seed)
    features_train = pca.fit_transform(hidden_train)
    features_val = pca.transform(hidden_val)
    
    # Reconstruction
    recon_val = pca.inverse_transform(features_val)
    recon_error = np.mean((recon_val - hidden_val) ** 2)
    
    # Explained variance
    explained_var = np.sum(pca.explained_variance_ratio_)
    
    # Compute sparsity
    l0_sparsity = np.mean(features_val < 1e-6)
    l1_sparsity = np.mean(np.sum(np.abs(features_val), axis=1))
    
    result = {
        'seed': seed,
        'n_components': n_components,
        'overcomplete': overcomplete,
        'recon_error': float(recon_error),
        'explained_variance': float(explained_var),
        'l0_sparsity': float(l0_sparsity),
        'l1_sparsity': float(l1_sparsity),
    }
    
    # Save
    save_path = f'models/baseline_pca_{overcomplete}x_seed{seed}.pt'
    torch.save({
        'pca': pca,
        'features_train': features_train,
        'features_val': features_val,
        'result': result
    }, save_path)
    
    print(f"    Recon error: {recon_error:.6f}, Explained var: {explained_var:.4f}")
    
    return result

def main():
    print("="*60)
    print("Training Baseline Methods on Synthetic Data")
    print("="*60)
    
    # Configuration
    SEEDS = [42, 123, 456]
    DICT_SIZES = [64]  # Only 1x for synthetic (hidden_dim=64)
    
    # Load data
    print("\nLoading synthetic data...")
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    hidden_train = data['hidden_train']
    hidden_val = data['hidden_val']
    
    print(f"  Train activations: {hidden_train.shape}")
    print(f"  Val activations: {hidden_val.shape}")
    
    # Train Random baselines
    print("\n" + "="*60)
    print("Training Random Projection Baselines")
    print("="*60)
    
    random_results = {}
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 64
        random_results[f'{overcomplete}x'] = []
        
        for seed in SEEDS:
            result = train_random_baseline(seed, dict_size, hidden_train, hidden_val)
            random_results[f'{overcomplete}x'].append(result)
    
    # Train PCA baselines
    print("\n" + "="*60)
    print("Training PCA Baselines")
    print("="*60)
    
    pca_results = {}
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 64
        pca_results[f'{overcomplete}x'] = []
        
        for seed in SEEDS:
            result = train_pca_baseline(seed, dict_size, hidden_train, hidden_val)
            pca_results[f'{overcomplete}x'].append(result)
    
    # Compute summaries
    def compute_summary(results_dict):
        summary = {}
        for key, results in results_dict.items():
            summary[key] = {
                'recon_error_mean': np.mean([r['recon_error'] for r in results]),
                'recon_error_std': np.std([r['recon_error'] for r in results]),
                'l0_sparsity_mean': np.mean([r['l0_sparsity'] for r in results]),
            }
        return summary
    
    random_summary = compute_summary(random_results)
    pca_summary = compute_summary(pca_results)
    
    # Add explained variance for PCA
    for key in pca_results:
        pca_summary[key]['explained_var_mean'] = np.mean([r['explained_variance'] for r in pca_results[key]])
    
    # Save results
    save_json({
        'random': {'all_results': random_results, 'summary': random_summary},
        'pca': {'all_results': pca_results, 'summary': pca_summary}
    }, 'exp/synthetic/baselines/results.json')
    
    print("\n" + "="*60)
    print("Baseline Training Summary:")
    print("="*60)
    print("\nRandom Baselines:")
    for key, stats in random_summary.items():
        print(f"  {key}: Recon error = {stats['recon_error_mean']:.6f} ± {stats['recon_error_std']:.6f}")
    
    print("\nPCA Baselines:")
    for key, stats in pca_summary.items():
        print(f"  {key}: Recon error = {stats['recon_error_mean']:.6f} ± {stats['recon_error_std']:.6f}, "
              f"Explained var = {stats['explained_var_mean']:.4f}")
    
    print("\nResults saved to exp/synthetic/baselines/results.json")

if __name__ == '__main__':
    main()
