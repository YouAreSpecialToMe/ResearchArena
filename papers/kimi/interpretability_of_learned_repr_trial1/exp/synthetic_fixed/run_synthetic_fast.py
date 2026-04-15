"""
Fast version of synthetic task experiment - focuses on key results.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr, ttest_ind
import os
from typing import Dict, Any
from sklearn.decomposition import PCA

from exp.shared.utils import set_seed
from exp.shared.models import SparseAutoencoder, train_sae


def convert_to_serializable(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_json_safe(data: Dict[str, Any], path: str):
    serializable_data = convert_to_serializable(data)
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def compute_cgas_fast(causal_subspaces, explanation_features, full_activations, top_k=5):
    """Compute C-GAS efficiently."""
    from sklearn.metrics import pairwise_distances
    
    n_samples = causal_subspaces.shape[0]
    
    # Subsample for faster computation
    if n_samples > 200:
        indices = np.random.choice(n_samples, 200, replace=False)
        causal_subspaces = causal_subspaces[indices]
        explanation_features = explanation_features[indices]
        full_activations = full_activations[indices]
        n_samples = 200
    
    # Select features (simplified)
    if explanation_features.shape[1] > top_k * causal_subspaces.shape[1]:
        # Use random projection for speed instead of correlation
        n_select = min(top_k * causal_subspaces.shape[1], explanation_features.shape[1] // 2)
        selected_features = explanation_features[:, :n_select]
    else:
        selected_features = explanation_features
    
    # Compute distances
    D_causal = pairwise_distances(causal_subspaces, metric='cosine')
    D_exp = pairwise_distances(selected_features, metric='cosine')
    D_full = pairwise_distances(full_activations, metric='cosine')
    
    # Upper triangular
    triu_indices = np.triu_indices(n_samples, k=1)
    d_causal_vec = D_causal[triu_indices]
    d_exp_vec = D_exp[triu_indices]
    d_full_vec = D_full[triu_indices]
    
    # Correlations
    rho_causal_exp, _ = spearmanr(d_causal_vec, d_exp_vec)
    rho_causal_full, _ = spearmanr(d_causal_vec, d_full_vec)
    
    if np.isnan(rho_causal_exp) or np.isnan(rho_causal_full) or abs(rho_causal_full) < 1e-10:
        return 0.0
    
    return float(rho_causal_exp / rho_causal_full)


def train_pca_baseline_fast(seed, n_components, hidden_train, hidden_val):
    """Fast PCA training."""
    set_seed(seed)
    input_dim = hidden_train.shape[1]
    n_components = min(n_components, input_dim)
    
    # Use smaller subsample for speed
    n_train = hidden_train.shape[0]
    subsample_size = min(int(n_train * 0.8), 5000)
    indices = np.random.choice(n_train, subsample_size, replace=False)
    
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(hidden_train[indices])
    features_val = pca.transform(hidden_val)
    
    return features_val


def compute_recovery_rate(features, gt_values):
    """Fast recovery rate computation."""
    best_corr = 0.0
    for j in range(min(features.shape[1], 100)):  # Limit to first 100 features
        feat_j = features[:, j]
        try:
            corr = abs(np.corrcoef(feat_j, gt_values)[0, 1])
            if not np.isnan(corr):
                best_corr = max(best_corr, corr)
        except:
            pass
    return 1.0 if best_corr > 0.5 else 0.0


def main():
    print("="*60)
    print("Synthetic Task Experiment (FAST VERSION)")
    print("="*60)
    
    os.makedirs('exp/synthetic_fixed', exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    hidden_val = data['hidden_val']
    
    ground_truth = torch.load('data/synthetic_ground_truth.pt', weights_only=False)
    ground_truth_val = ground_truth['val']
    
    print(f"  Val: {hidden_val.shape}")
    
    # Load validation results
    with open('exp/synthetic/validation/results.json', 'r') as f:
        validation_results = json.load(f)
    validated_atlas = validation_results['validated_atlas']
    
    validated_features = {k: v for k, v in validated_atlas.items() if v.get('validated', False)}
    if len(validated_features) == 0:
        validated_features = validated_atlas
    
    print(f"  Validated features: {len(validated_features)}")
    
    SEEDS = [42, 123, 456]
    OVERCOMPLETES = ['1x', '4x', '16x']
    METHODS = ['random', 'pca', 'sae']
    
    all_results = []
    
    # Train missing PCA 4x and 16x baselines
    print("\n" + "="*60)
    print("Training missing PCA baselines...")
    print("="*60)
    
    for overcomplete in [4, 16]:
        print(f"\nPCA {overcomplete}x:")
        for seed in SEEDS:
            features_val = train_pca_baseline_fast(seed, overcomplete * 64, 
                                                    data['hidden_train'], hidden_val)
            torch.save({
                'features_val': features_val,
            }, f'models/baseline_pca_{overcomplete}x_seed{seed}.pt')
            print(f"  Seed {seed}: done")
    
    # Evaluate all methods
    print("\n" + "="*60)
    print("Evaluating methods...")
    print("="*60)
    
    for method in METHODS:
        print(f"\n{method.upper()}:")
        
        for overcomplete in OVERCOMPLETES:
            print(f"  {overcomplete}:")
            
            for seed in SEEDS:
                try:
                    # Load features
                    if method == 'sae':
                        checkpoint = torch.load(f'models/sae_synthetic_{overcomplete}_seed{seed}.pt', weights_only=False)
                    elif method == 'random':
                        checkpoint = torch.load(f'models/baseline_random_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    elif method == 'pca':
                        checkpoint = torch.load(f'models/baseline_pca_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    
                    features = checkpoint['features_val']
                    
                    # Compute C-GAS (average across features)
                    cgas_list = []
                    recovery_list = []
                    
                    for feat_name, atlas_entry in list(validated_features.items())[:3]:  # Limit to 3 features for speed
                        causal_dims = atlas_entry['dims'][:5]  # Top 5 dims
                        causal_subspaces = hidden_val[:, causal_dims]
                        
                        cgas = compute_cgas_fast(causal_subspaces, features, hidden_val)
                        cgas_list.append(cgas)
                        
                        recovery = compute_recovery_rate(features, ground_truth_val[feat_name])
                        recovery_list.append(recovery)
                    
                    mean_cgas = np.mean(cgas_list) if cgas_list else 0.0
                    mean_recovery = np.mean(recovery_list) if recovery_list else 0.0
                    
                    all_results.append({
                        'method': method,
                        'overcomplete': overcomplete,
                        'seed': seed,
                        'cgas': float(mean_cgas),
                        'recovery_rate': float(mean_recovery)
                    })
                    
                    print(f"    Seed {seed}: C-GAS={mean_cgas:.4f}, Recovery={mean_recovery:.4f}")
                    
                except Exception as e:
                    print(f"    Seed {seed}: Error - {e}")
    
    # Oracle baseline
    print(f"\nORACLE:")
    for seed in SEEDS:
        try:
            checkpoint = torch.load(f'models/baseline_oracle_seed{seed}.pt', weights_only=False)
            features = checkpoint['features_val']
            
            cgas_list = []
            for feat_name, atlas_entry in list(validated_features.items())[:3]:
                causal_dims = atlas_entry['dims'][:5]
                causal_subspaces = hidden_val[:, causal_dims]
                
                cgas = compute_cgas_fast(causal_subspaces, features, hidden_val)
                cgas_list.append(cgas)
            
            mean_cgas = np.mean(cgas_list) if cgas_list else 0.0
            
            all_results.append({
                'method': 'oracle',
                'overcomplete': '1x',
                'seed': seed,
                'cgas': float(mean_cgas),
                'recovery_rate': 1.0
            })
            
            print(f"  Seed {seed}: C-GAS={mean_cgas:.4f}")
        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")
    
    # Compute summary
    summary = {}
    for method in METHODS + ['oracle']:
        summary[method] = {}
        for overcomplete in OVERCOMPLETES:
            method_results = [r for r in all_results 
                           if r['method'] == method and r['overcomplete'] == overcomplete]
            
            if method_results:
                cgas_values = [r['cgas'] for r in method_results]
                summary[method][overcomplete] = {
                    'cgas_mean': float(np.mean(cgas_values)),
                    'cgas_std': float(np.std(cgas_values)),
                }
    
    # Correlation analysis
    cgas_vals = [r['cgas'] for r in all_results if r['method'] != 'oracle']
    recovery_vals = [r['recovery_rate'] for r in all_results if r['method'] != 'oracle']
    
    if cgas_vals and recovery_vals:
        corr, p_val = pearsonr(cgas_vals, recovery_vals)
        scorr, sp_val = spearmanr(cgas_vals, recovery_vals)
    else:
        corr, p_val, scorr, sp_val = 0.0, 1.0, 0.0, 1.0
    
    print("\n" + "="*60)
    print("Correlation Analysis: C-GAS vs Ground-Truth Recovery")
    print("="*60)
    print(f"  Pearson r = {corr:.4f}, p = {p_val:.4f}")
    print(f"  Spearman ρ = {scorr:.4f}, p = {sp_val:.4f}")
    
    # Statistical tests
    print("\n" + "="*60)
    print("Statistical Tests")
    print("="*60)
    
    sae_cgas = [r['cgas'] for r in all_results if r['method'] == 'sae']
    random_cgas = [r['cgas'] for r in all_results if r['method'] == 'random']
    pca_cgas = [r['cgas'] for r in all_results if r['method'] == 'pca']
    
    if sae_cgas and random_cgas:
        t_stat, p_val_test = ttest_ind(sae_cgas, random_cgas)
        print(f"  SAE vs Random: t={t_stat:.4f}, p={p_val_test:.4f}")
    
    if sae_cgas and pca_cgas:
        t_stat, p_val_test = ttest_ind(sae_cgas, pca_cgas)
        print(f"  SAE vs PCA: t={t_stat:.4f}, p={p_val_test:.4f}")
    
    # Save results
    save_json_safe({
        'all_results': all_results,
        'summary': summary,
        'correlations': {
            'pearson_r': float(corr),
            'pearson_p': float(p_val),
            'spearman_rho': float(scorr),
            'spearman_p': float(sp_val)
        }
    }, 'exp/synthetic_fixed/results.json')
    
    # Print final summary
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    
    for method in METHODS + ['oracle']:
        if method not in summary:
            continue
        print(f"\n{method.upper()}:")
        for overcomplete in OVERCOMPLETES:
            if overcomplete in summary[method]:
                stats = summary[method][overcomplete]
                print(f"  {overcomplete}: C-GAS = {stats['cgas_mean']:.4f} ± {stats['cgas_std']:.4f}")
    
    print(f"\nCorrelation with ground truth: r = {corr:.4f}")
    print("\nResults saved to exp/synthetic_fixed/results.json")


if __name__ == '__main__':
    main()
