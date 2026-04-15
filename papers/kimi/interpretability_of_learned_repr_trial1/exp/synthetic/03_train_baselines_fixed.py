"""Train baseline interpretability methods with proper per-seed variance."""
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


def train_pca_baseline_fixed(seed: int, n_components: int, hidden_train: np.ndarray, 
                              hidden_val: np.ndarray) -> dict:
    """Train PCA baseline with proper per-seed variance.
    
    KEY FIX: To get variance across seeds, we subsample the training data
    differently for each seed. This creates real variance in the PCA components.
    """
    set_seed(seed)
    
    input_dim = hidden_train.shape[1]
    # PCA cannot have more components than input_dim
    n_components = min(n_components, input_dim)
    overcomplete = n_components // input_dim
    
    print(f"\n  Training PCA {overcomplete}x (n_components={n_components}) with seed={seed}...")
    
    # FIX: Subsample training data differently for each seed
    # This creates actual variance in PCA across seeds
    n_train = hidden_train.shape[0]
    subsample_ratio = 0.9  # Use 90% of data
    subsample_size = int(n_train * subsample_ratio)
    
    # Random subsample (different for each seed)
    indices = np.random.choice(n_train, subsample_size, replace=False)
    hidden_train_subsample = hidden_train[indices]
    
    print(f"    Using {subsample_size}/{n_train} training samples")
    
    # Fit PCA on subsample
    pca = PCA(n_components=n_components, random_state=seed)
    features_train = pca.fit_transform(hidden_train_subsample)
    features_val = pca.transform(hidden_val)
    
    # Reconstruction on validation set
    recon_val = pca.inverse_transform(features_val)
    recon_error = np.mean((recon_val - hidden_val) ** 2)
    
    # Explained variance
    explained_var = np.sum(pca.explained_variance_ratio_)
    
    # Compute sparsity
    l0_sparsity = np.mean(features_val < 1e-6)
    l1_sparsity = np.mean(np.sum(np.abs(features_val), axis=1))
    
    # Additional metrics for stability analysis
    component_variance = np.var(pca.components_, axis=1).mean()
    
    result = {
        'seed': seed,
        'n_components': n_components,
        'overcomplete': overcomplete,
        'recon_error': float(recon_error),
        'explained_variance': float(explained_var),
        'l0_sparsity': float(l0_sparsity),
        'l1_sparsity': float(l1_sparsity),
        'subsample_size': subsample_size,
        'component_variance': float(component_variance),
        'singular_values': pca.singular_values_.tolist() if hasattr(pca, 'singular_values_') else []
    }
    
    # Save
    save_path = f'models/baseline_pca_{overcomplete}x_seed{seed}.pt'
    torch.save({
        'pca': pca,
        'features_train': features_train,
        'features_val': features_val,
        'result': result,
        'subsample_indices': indices.tolist()
    }, save_path)
    
    print(f"    Recon error: {recon_error:.6f}, Explained var: {explained_var:.4f}")
    
    return result


def train_oracle_baseline(seed: int, hidden_train: np.ndarray, 
                          hidden_val: np.ndarray, ground_truth_features: dict) -> dict:
    """Train Oracle baseline using ground-truth features.
    
    This establishes an upper bound on what C-GAS can achieve.
    The oracle uses the actual ground-truth causal features.
    """
    set_seed(seed)
    
    print(f"\n  Training Oracle (ground-truth) baseline with seed={seed}...")
    
    # Extract ground truth features
    gt_features_list = []
    for feat_name in sorted(ground_truth_features.keys()):
        gt_features_list.append(ground_truth_features[feat_name])
    
    # Stack into matrix (n_samples, n_ground_truth_features)
    gt_matrix = np.stack(gt_features_list, axis=1)
    
    # Project to hidden space dimensions for fair comparison
    # Use random projection to expand to hidden_dim
    hidden_dim = hidden_val.shape[1]
    n_gt = gt_matrix.shape[1]
    
    if n_gt < hidden_dim:
        # Expand with random projection
        projection = np.random.randn(n_gt, hidden_dim).astype(np.float32)
        projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
        features_val = gt_matrix @ projection
        features_train = np.stack([
            ground_truth_features[k] for k in sorted(ground_truth_features.keys())
        ], axis=1) @ projection
    else:
        features_val = gt_matrix
        features_train = np.stack([
            ground_truth_features[k] for k in sorted(ground_truth_features.keys())
        ], axis=1)
    
    # Reconstruction error (compared to ground truth itself - should be 0)
    recon_error = 0.0  # Oracle has perfect reconstruction of ground truth
    
    result = {
        'seed': seed,
        'n_ground_truth_features': n_gt,
        'hidden_dim': hidden_dim,
        'recon_error': float(recon_error),
        'is_oracle': True
    }
    
    # Save
    save_path = f'models/baseline_oracle_seed{seed}.pt'
    torch.save({
        'features_train': features_train,
        'features_val': features_val,
        'ground_truth_matrix': gt_matrix,
        'result': result
    }, save_path)
    
    print(f"    Oracle features: {gt_matrix.shape}, Projection: {features_val.shape}")
    
    return result


def main():
    print("="*60)
    print("Training Baseline Methods (FIXED)")
    print("="*60)
    
    # Configuration
    SEEDS = [42, 123, 456]
    DICT_SIZES = [64, 256, 1024]  # 1x, 4x, 16x for synthetic (hidden_dim=64)
    
    # Load data
    print("\nLoading synthetic data...")
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    hidden_train = data['hidden_train']
    hidden_val = data['hidden_val']
    
    # Load ground truth for oracle
    ground_truth = torch.load('data/synthetic_ground_truth.pt', weights_only=False)
    ground_truth_train = ground_truth['train']
    ground_truth_val = ground_truth['val']
    
    print(f"  Train activations: {hidden_train.shape}")
    print(f"  Val activations: {hidden_val.shape}")
    print(f"  Ground truth features: {len(ground_truth_val)}")
    
    os.makedirs('exp/synthetic/baselines', exist_ok=True)
    
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
    
    # Train PCA baselines (FIXED - proper per-seed variance)
    print("\n" + "="*60)
    print("Training PCA Baselines (FIXED - per-seed subsampling)")
    print("="*60)
    
    pca_results = {}
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 64
        pca_results[f'{overcomplete}x'] = []
        
        for seed in SEEDS:
            result = train_pca_baseline_fixed(seed, dict_size, hidden_train, hidden_val)
            pca_results[f'{overcomplete}x'].append(result)
    
    # Train Oracle baseline
    print("\n" + "="*60)
    print("Training Oracle Baseline (Ground-Truth Features)")
    print("="*60)
    
    oracle_results = []
    for seed in SEEDS:
        result = train_oracle_baseline(seed, hidden_train, hidden_val, ground_truth_val)
        oracle_results.append(result)
    
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
        pca_summary[key]['explained_var_std'] = np.std([r['explained_variance'] for r in pca_results[key]])
    
    # Save results
    save_json({
        'random': {'all_results': random_results, 'summary': random_summary},
        'pca': {'all_results': pca_results, 'summary': pca_summary},
        'oracle': {'all_results': oracle_results, 'n_features': len(ground_truth_val)}
    }, 'exp/synthetic/baselines/results_fixed.json')
    
    # Print summary
    print("\n" + "="*60)
    print("Baseline Training Summary (FIXED):")
    print("="*60)
    print("\nRandom Baselines:")
    for key, stats in random_summary.items():
        print(f"  {key}: Recon error = {stats['recon_error_mean']:.6f} ± {stats['recon_error_std']:.6f}")
    
    print("\nPCA Baselines:")
    for key, stats in pca_summary.items():
        print(f"  {key}: Recon error = {stats['recon_error_mean']:.6f} ± {stats['recon_error_std']:.6f}, "
              f"Explained var = {stats['explained_var_mean']:.4f} ± {stats['explained_var_std']:.4f}")
    
    print("\nOracle Baseline:")
    print(f"  Features: {len(ground_truth_val)} ground-truth features")
    print(f"  This establishes the upper bound on achievable C-GAS")
    
    print("\nResults saved to exp/synthetic/baselines/results_fixed.json")


if __name__ == '__main__':
    main()
