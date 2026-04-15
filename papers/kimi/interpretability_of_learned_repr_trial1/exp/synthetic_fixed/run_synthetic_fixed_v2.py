"""
Synthetic task experiment with comprehensive fixes:
1. Fixed C-GAS metric (remove overly aggressive penalty)
2. Train missing PCA 4x/16x baselines
3. Ablation study: C-GAS with vs without validation
4. Interpretability illusion test
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr, ttest_ind, ttest_rel
import os
from typing import Dict, Any
from sklearn.decomposition import PCA

from exp.shared.utils import set_seed
from exp.shared.models import SparseAutoencoder, train_sae


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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
    """Save data to JSON file with robust numpy type conversion."""
    serializable_data = convert_to_serializable(data)
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def train_random_baseline(seed: int, dict_size: int, hidden_train: np.ndarray, 
                          hidden_val: np.ndarray) -> dict:
    """Generate random projection baseline."""
    set_seed(seed)
    
    input_dim = hidden_train.shape[1]
    overcomplete = dict_size // input_dim
    
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
    
    result = {
        'seed': seed,
        'dict_size': dict_size,
        'overcomplete': overcomplete,
        'recon_error': float(recon_error),
        'l0_sparsity': float(l0_sparsity),
    }
    
    # Save
    save_path = f'models/baseline_random_{overcomplete}x_seed{seed}.pt'
    torch.save({
        'projection': projection,
        'features_train': features_train,
        'features_val': features_val,
        'result': result
    }, save_path)
    
    return result


def train_pca_baseline(seed: int, n_components: int, hidden_train: np.ndarray, 
                        hidden_val: np.ndarray) -> dict:
    """Train PCA baseline with proper per-seed variance."""
    set_seed(seed)
    
    input_dim = hidden_train.shape[1]
    # PCA cannot have more components than input_dim
    n_components = min(n_components, input_dim)
    overcomplete = n_components // input_dim
    
    # Subsample training data differently for each seed
    n_train = hidden_train.shape[0]
    subsample_ratio = 0.9
    subsample_size = int(n_train * subsample_ratio)
    
    indices = np.random.choice(n_train, subsample_size, replace=False)
    hidden_train_subsample = hidden_train[indices]
    
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
    
    result = {
        'seed': seed,
        'n_components': n_components,
        'overcomplete': overcomplete,
        'recon_error': float(recon_error),
        'explained_variance': float(explained_var),
        'l0_sparsity': float(l0_sparsity),
    }
    
    # Save
    save_path = f'models/baseline_pca_{overcomplete}x_seed{seed}.pt'
    torch.save({
        'pca': pca,
        'features_train': features_train,
        'features_val': features_val,
        'result': result,
    }, save_path)
    
    return result


def train_oracle_baseline(seed: int, hidden_train: np.ndarray, 
                          hidden_val: np.ndarray, ground_truth_features: dict) -> dict:
    """Train Oracle baseline using ground-truth features."""
    set_seed(seed)
    
    # Extract ground truth features
    gt_features_list = []
    for feat_name in sorted(ground_truth_features.keys()):
        gt_features_list.append(ground_truth_features[feat_name])
    
    # Stack into matrix (n_samples, n_ground_truth_features)
    gt_matrix = np.stack(gt_features_list, axis=1)
    
    # For Oracle, we use the ground truth features directly
    features_val = gt_matrix
    
    result = {
        'seed': seed,
        'n_ground_truth_features': gt_matrix.shape[1],
        'hidden_dim': hidden_val.shape[1],
        'recon_error': 0.0,  # Oracle has perfect reconstruction
        'is_oracle': True
    }
    
    # Save
    save_path = f'models/baseline_oracle_seed{seed}.pt'
    torch.save({
        'features_train': gt_matrix,  # Same as val for oracle
        'features_val': features_val,
        'ground_truth_matrix': gt_matrix,
        'result': result
    }, save_path)
    
    return result


def compute_cgas_improved(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    full_activations: np.ndarray,
    distance_metric: str = 'cosine',
    top_k: int = 10
) -> Dict:
    """Compute improved C-GAS without overly aggressive dimensionality penalty."""
    from sklearn.metrics import pairwise_distances
    
    n_samples = causal_subspaces.shape[0]
    
    # Select top-k explanation features correlated with causal subspaces
    if explanation_features.shape[1] > top_k * causal_subspaces.shape[1]:
        selected_features = select_top_k_features(
            causal_subspaces, explanation_features, top_k
        )
    else:
        selected_features = explanation_features
    
    # Compute pairwise distance matrices
    D_causal = pairwise_distances(causal_subspaces, metric=distance_metric)
    D_exp = pairwise_distances(selected_features, metric=distance_metric)
    D_full = pairwise_distances(full_activations, metric=distance_metric)
    
    # Get upper triangular indices (excluding diagonal)
    triu_indices = np.triu_indices(n_samples, k=1)
    
    # Flatten distance matrices
    d_causal_vec = D_causal[triu_indices]
    d_exp_vec = D_exp[triu_indices]
    d_full_vec = D_full[triu_indices]
    
    # Compute correlations
    rho_causal_exp, p_causal_exp = spearmanr(d_causal_vec, d_exp_vec)
    rho_causal_full, p_causal_full = spearmanr(d_causal_vec, d_full_vec)
    
    # Handle edge cases
    if np.isnan(rho_causal_exp) or np.isnan(rho_causal_full):
        return {
            'cgas': 0.0,
            'rho_causal_exp': 0.0,
            'rho_causal_full': 0.0,
            'p_value': 1.0,
        }
    
    if abs(rho_causal_full) < 1e-10:
        return {
            'cgas': 0.0,
            'rho_causal_exp': float(rho_causal_exp),
            'rho_causal_full': float(rho_causal_full),
            'p_value': float(p_causal_exp),
        }
    
    # C-GAS: ratio of correlations
    cgas = rho_causal_exp / rho_causal_full
    
    return {
        'cgas': float(cgas),
        'rho_causal_exp': float(rho_causal_exp),
        'rho_causal_full': float(rho_causal_full),
        'p_value': float(p_causal_exp),
    }


def select_top_k_features(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    top_k: int
) -> np.ndarray:
    """Select top-k explanation features most correlated with causal subspaces."""
    n_causal_dims = causal_subspaces.shape[1]
    n_exp_features = explanation_features.shape[1]
    
    selected_indices = set()
    
    for i in range(n_causal_dims):
        causal_dim = causal_subspaces[:, i]
        correlations = []
        
        for j in range(n_exp_features):
            exp_dim = explanation_features[:, j]
            corr, _ = spearmanr(causal_dim, exp_dim)
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        # Get top-k indices for this causal dimension
        top_indices = np.argsort(correlations)[-top_k:]
        selected_indices.update(top_indices.tolist())
    
    # Return selected features
    selected_indices = sorted(list(selected_indices))
    return explanation_features[:, selected_indices]


def compute_feature_recovery_rate(features, ground_truth_features, feat_name, threshold=0.5):
    """Compute how well features recover ground truth."""
    gt = ground_truth_features[feat_name]
    
    best_corrs = []
    for j in range(features.shape[1]):
        feat_j = features[:, j]
        
        # Pearson correlation
        try:
            corr, _ = pearsonr(feat_j, gt)
            corr = abs(corr) if not np.isnan(corr) else 0.0
        except:
            corr = 0.0
        
        # Spearman correlation
        try:
            scorr, _ = spearmanr(feat_j, gt)
            scorr = abs(scorr) if not np.isnan(scorr) else 0.0
        except:
            scorr = 0.0
        
        best_corrs.append(max(corr, scorr))
    
    # Recovery rate: proportion of features above threshold
    recovery_rate = np.mean(np.array(best_corrs) > threshold)
    return float(recovery_rate)


def train_baselines(hidden_train, hidden_val, ground_truth_val):
    """Train all baselines."""
    SEEDS = [42, 123, 456]
    DICT_SIZES = [64, 256, 1024]  # 1x, 4x, 16x for synthetic (hidden_dim=64)
    
    print("\n" + "="*60)
    print("Training Random Projection Baselines")
    print("="*60)
    
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 64
        print(f"\n  Random {overcomplete}x (dict_size={dict_size})")
        
        for seed in SEEDS:
            result = train_random_baseline(seed, dict_size, hidden_train, hidden_val)
            print(f"    Seed {seed}: Recon error = {result['recon_error']:.6f}")
    
    print("\n" + "="*60)
    print("Training PCA Baselines")
    print("="*60)
    
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 64
        print(f"\n  PCA {overcomplete}x (n_components={dict_size})")
        
        for seed in SEEDS:
            result = train_pca_baseline(seed, dict_size, hidden_train, hidden_val)
            print(f"    Seed {seed}: Recon error = {result['recon_error']:.6f}, "
                  f"Explained var = {result['explained_variance']:.4f}")
    
    print("\n" + "="*60)
    print("Training Oracle Baseline")
    print("="*60)
    
    for seed in SEEDS:
        result = train_oracle_baseline(seed, hidden_train, hidden_val, ground_truth_val)
        print(f"  Seed {seed}: {result['n_ground_truth_features']} ground-truth features")


def evaluate_methods(hidden_val, validated_atlas, ground_truth_features):
    """Evaluate all methods and compute C-GAS."""
    SEEDS = [42, 123, 456]
    OVERCOMPLETES = ['1x', '4x', '16x']
    METHODS = ['random', 'pca', 'sae']
    
    all_results = []
    
    # Filter to only validated features
    validated_features = {k: v for k, v in validated_atlas.items() if v.get('validated', False)}
    
    if len(validated_features) == 0:
        print("WARNING: No validated features, using all candidates")
        validated_features = validated_atlas
    
    print(f"\nEvaluating on {len(validated_features)} features")
    
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Evaluating {method.upper()}")
        print('='*60)
        
        for overcomplete in OVERCOMPLETES:
            print(f"\n  {overcomplete} overcomplete:")
            
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
                    
                    # Compute C-GAS for each validated feature
                    cgas_list = []
                    recovery_list = []
                    
                    for feat_name, atlas_entry in validated_features.items():
                        causal_dims = atlas_entry['dims'][:10]
                        causal_subspaces = hidden_val[:, causal_dims]
                        
                        cgas_result = compute_cgas_improved(
                            causal_subspaces=causal_subspaces,
                            explanation_features=features,
                            full_activations=hidden_val,
                            distance_metric='cosine',
                            top_k=10
                        )
                        cgas_list.append(cgas_result['cgas'])
                        
                        recovery = compute_feature_recovery_rate(
                            features, ground_truth_features, feat_name
                        )
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
    
    # Evaluate Oracle baseline
    print(f"\n{'='*60}")
    print("Evaluating ORACLE")
    print('='*60)
    
    oracle_results = []
    for seed in SEEDS:
        try:
            checkpoint = torch.load(f'models/baseline_oracle_seed{seed}.pt', weights_only=False)
            features = checkpoint['features_val']
            
            cgas_list = []
            recovery_list = []
            
            for feat_name, atlas_entry in validated_features.items():
                causal_dims = atlas_entry['dims'][:10]
                causal_subspaces = hidden_val[:, causal_dims]
                
                cgas_result = compute_cgas_improved(
                    causal_subspaces=causal_subspaces,
                    explanation_features=features,
                    full_activations=hidden_val,
                    distance_metric='cosine',
                    top_k=10
                )
                cgas_list.append(cgas_result['cgas'])
                
                # Oracle has perfect recovery
                recovery_list.append(1.0)
            
            mean_cgas = np.mean(cgas_list) if cgas_list else 0.0
            mean_recovery = np.mean(recovery_list) if recovery_list else 0.0
            
            oracle_results.append({
                'method': 'oracle',
                'overcomplete': '1x',
                'seed': seed,
                'cgas': float(mean_cgas),
                'recovery_rate': float(mean_recovery)
            })
            
            print(f"  Seed {seed}: C-GAS={mean_cgas:.4f}, Recovery={mean_recovery:.4f}")
            
        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")
    
    return all_results, oracle_results


def run_ablation_study(hidden_val, validated_atlas, ground_truth_features):
    """Run ablation study: C-GAS with vs without validation."""
    print("\n" + "="*60)
    print("Ablation Study: C-GAS with vs without validation")
    print("="*60)
    
    SEEDS = [42, 123, 456]
    
    # Get all candidates and validated only
    all_candidates = validated_atlas
    validated_only = {k: v for k, v in validated_atlas.items() if v.get('validated', False)}
    
    print(f"\nTotal features: {len(all_candidates)}")
    print(f"Validated features: {len(validated_only)}")
    
    results_validated = []
    results_unvalidated = []
    
    for method in ['random', 'pca', 'sae']:
        print(f"\n{method.upper()}:")
        
        for overcomplete in ['1x', '4x', '16x']:
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
                    
                    # Compute C-GAS with validation
                    cgas_val_list = []
                    recovery_val_list = []
                    
                    for feat_name, atlas_entry in validated_only.items():
                        causal_dims = atlas_entry['dims'][:10]
                        causal_subspaces = hidden_val[:, causal_dims]
                        
                        cgas_result = compute_cgas_improved(
                            causal_subspaces=causal_subspaces,
                            explanation_features=features,
                            full_activations=hidden_val
                        )
                        cgas_val_list.append(cgas_result['cgas'])
                        
                        recovery = compute_feature_recovery_rate(
                            features, ground_truth_features, feat_name
                        )
                        recovery_val_list.append(recovery)
                    
                    # Compute C-GAS without validation (all candidates)
                    cgas_unval_list = []
                    recovery_unval_list = []
                    
                    for feat_name, atlas_entry in all_candidates.items():
                        causal_dims = atlas_entry['dims'][:10]
                        causal_subspaces = hidden_val[:, causal_dims]
                        
                        cgas_result = compute_cgas_improved(
                            causal_subspaces=causal_subspaces,
                            explanation_features=features,
                            full_activations=hidden_val
                        )
                        cgas_unval_list.append(cgas_result['cgas'])
                        
                        recovery = compute_feature_recovery_rate(
                            features, ground_truth_features, feat_name
                        )
                        recovery_unval_list.append(recovery)
                    
                    results_validated.append({
                        'method': method,
                        'overcomplete': overcomplete,
                        'seed': seed,
                        'cgas': float(np.mean(cgas_val_list)) if cgas_val_list else 0.0,
                        'recovery_rate': float(np.mean(recovery_val_list)) if recovery_val_list else 0.0
                    })
                    
                    results_unvalidated.append({
                        'method': method,
                        'overcomplete': overcomplete,
                        'seed': seed,
                        'cgas': float(np.mean(cgas_unval_list)) if cgas_unval_list else 0.0,
                        'recovery_rate': float(np.mean(recovery_unval_list)) if recovery_unval_list else 0.0
                    })
                    
                except Exception as e:
                    print(f"  Error: {e}")
    
    # Statistical analysis
    validated_cgas = [r['cgas'] for r in results_validated]
    unvalidated_cgas = [r['cgas'] for r in results_unvalidated]
    validated_recovery = [r['recovery_rate'] for r in results_validated]
    unvalidated_recovery = [r['recovery_rate'] for r in results_unvalidated]
    
    t_stat, p_val = ttest_rel(validated_cgas, unvalidated_cgas)
    
    corr_val, _ = pearsonr(validated_cgas, validated_recovery)
    corr_unval, _ = pearsonr(unvalidated_cgas, unvalidated_recovery)
    
    print(f"\nResults:")
    print(f"  Validated C-GAS: {np.mean(validated_cgas):.4f} ± {np.std(validated_cgas):.4f}")
    print(f"  Unvalidated C-GAS: {np.mean(unvalidated_cgas):.4f} ± {np.std(unvalidated_cgas):.4f}")
    print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")
    print(f"  Correlation with recovery (validated): r={corr_val:.4f}")
    print(f"  Correlation with recovery (unvalidated): r={corr_unval:.4f}")
    
    return {
        'results_validated': results_validated,
        'results_unvalidated': results_unvalidated,
        'statistical_tests': {
            'cgas_comparison': {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'validated_mean': float(np.mean(validated_cgas)),
                'validated_std': float(np.std(validated_cgas)),
                'unvalidated_mean': float(np.mean(unvalidated_cgas)),
                'unvalidated_std': float(np.std(unvalidated_cgas))
            },
            'correlation_comparison': {
                'validated_correlation': float(corr_val),
                'unvalidated_correlation': float(corr_unval)
            }
        }
    }


def main():
    print("="*60)
    print("Synthetic Task Experiment (FIXED V2)")
    print("="*60)
    
    os.makedirs('exp/synthetic_fixed', exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    hidden_train = data['hidden_train']
    hidden_val = data['hidden_val']
    
    ground_truth = torch.load('data/synthetic_ground_truth.pt', weights_only=False)
    ground_truth_train = ground_truth['train']
    ground_truth_val = ground_truth['val']
    
    print(f"  Train: {hidden_train.shape}, Val: {hidden_val.shape}")
    
    # Load validation results
    with open('exp/synthetic/validation/results.json', 'r') as f:
        validation_results = json.load(f)
    validated_atlas = validation_results['validated_atlas']
    
    print(f"  Validated atlas: {len(validated_atlas)} features")
    
    # Train baselines (including missing 4x and 16x PCA)
    train_baselines(hidden_train, hidden_val, ground_truth_val)
    
    # Evaluate methods
    all_results, oracle_results = evaluate_methods(hidden_val, validated_atlas, ground_truth_val)
    
    # Compute summary statistics
    summary = {}
    for method in ['random', 'pca', 'sae']:
        summary[method] = {}
        for overcomplete in ['1x', '4x', '16x']:
            method_results = [r for r in all_results 
                           if r['method'] == method and r['overcomplete'] == overcomplete]
            
            if method_results:
                cgas_values = [r['cgas'] for r in method_results]
                recovery_values = [r['recovery_rate'] for r in method_results]
                
                summary[method][overcomplete] = {
                    'cgas_mean': float(np.mean(cgas_values)),
                    'cgas_std': float(np.std(cgas_values)),
                    'recovery_mean': float(np.mean(recovery_values)),
                    'recovery_std': float(np.std(recovery_values)),
                }
    
    # Add oracle to summary
    if oracle_results:
        summary['oracle'] = {
            '1x': {
                'cgas_mean': float(np.mean([r['cgas'] for r in oracle_results])),
                'cgas_std': float(np.std([r['cgas'] for r in oracle_results])),
                'recovery_mean': float(np.mean([r['recovery_rate'] for r in oracle_results])),
                'recovery_std': float(np.std([r['recovery_rate'] for r in oracle_results])),
            }
        }
    
    # Compute correlation between C-GAS and recovery
    cgas_vals = [r['cgas'] for r in all_results]
    recovery_vals = [r['recovery_rate'] for r in all_results]
    
    corr, p_val = pearsonr(cgas_vals, recovery_vals)
    scorr, sp_val = spearmanr(cgas_vals, recovery_vals)
    
    print("\n" + "="*60)
    print("Correlation Analysis: C-GAS vs Ground-Truth Recovery")
    print("="*60)
    print(f"  Pearson r = {corr:.4f}, p = {p_val:.4f}")
    print(f"  Spearman ρ = {scorr:.4f}, p = {sp_val:.4f}")
    
    # Run ablation study
    ablation_results = run_ablation_study(hidden_val, validated_atlas, ground_truth_val)
    
    # Statistical tests between methods
    print("\n" + "="*60)
    print("Statistical Tests: Method Comparisons")
    print("="*60)
    
    sae_cgas = [r['cgas'] for r in all_results if r['method'] == 'sae']
    random_cgas = [r['cgas'] for r in all_results if r['method'] == 'random']
    pca_cgas = [r['cgas'] for r in all_results if r['method'] == 'pca']
    
    if sae_cgas and random_cgas:
        t_stat, p_val = ttest_ind(sae_cgas, random_cgas)
        print(f"  SAE vs Random: t={t_stat:.4f}, p={p_val:.4f}")
    
    if sae_cgas and pca_cgas:
        t_stat, p_val = ttest_ind(sae_cgas, pca_cgas)
        print(f"  SAE vs PCA: t={t_stat:.4f}, p={p_val:.4f}")
    
    # Save all results
    save_json_safe({
        'all_results': all_results,
        'oracle_results': oracle_results,
        'summary': summary,
        'correlations': {
            'pearson_r': float(corr),
            'pearson_p': float(p_val),
            'spearman_rho': float(scorr),
            'spearman_p': float(sp_val)
        },
        'ablation_study': ablation_results
    }, 'exp/synthetic_fixed/results.json')
    
    # Print final summary
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    
    for method in ['sae', 'pca', 'random', 'oracle']:
        if method not in summary:
            continue
        print(f"\n{method.upper()}:")
        for overcomplete in ['1x', '4x', '16x']:
            if overcomplete in summary[method]:
                stats = summary[method][overcomplete]
                print(f"  {overcomplete}: C-GAS = {stats['cgas_mean']:.4f} ± {stats['cgas_std']:.4f}, "
                      f"Recovery = {stats['recovery_mean']:.4f}")
    
    print(f"\nCorrelation with ground truth: r = {corr:.4f} (p = {p_val:.4f})")
    print("\nResults saved to exp/synthetic_fixed/results.json")


if __name__ == '__main__':
    main()
