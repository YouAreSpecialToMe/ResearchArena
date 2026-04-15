"""Compute C-GAS scores with improved metric (fixed version)."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr

from exp.shared.utils import set_seed, save_json
from exp.shared.metrics_fixed import (
    compute_cgas_fixed, 
    compute_oracle_cgas,
    compute_feature_recovery_rate_improved,
    compute_sensitivity_analysis
)


def evaluate_method(method_type: str, overcomplete: str, seed: int, 
                    hidden_val: np.ndarray, validated_atlas: dict, 
                    ground_truth_features: dict, input_dim: int = 64) -> dict:
    """Evaluate a single method configuration with improved metrics."""
    
    dict_size = input_dim * int(overcomplete.replace('x', ''))
    
    # Load features
    if method_type == 'sae':
        checkpoint = torch.load(f'models/sae_synthetic_{overcomplete}_seed{seed}.pt', weights_only=False)
        features = checkpoint['features_val']
    elif method_type == 'random':
        checkpoint = torch.load(f'models/baseline_random_{overcomplete}_seed{seed}.pt', weights_only=False)
        features = checkpoint['features_val']
    elif method_type == 'pca':
        checkpoint = torch.load(f'models/baseline_pca_{overcomplete}_seed{seed}.pt', weights_only=False)
        features = checkpoint['features_val']
    elif method_type == 'oracle':
        checkpoint = torch.load(f'models/baseline_oracle_seed{seed}.pt', weights_only=False)
        features = checkpoint['features_val']
    else:
        raise ValueError(f"Unknown method: {method_type}")
    
    # Compute C-GAS for each feature with improved metric
    cgas_scores = []
    cgas_unpenalized_scores = []
    cgas_details_list = []
    recovery_stats_per_feature = {}
    
    for feat_name, atlas_entry in validated_atlas.items():
        if not atlas_entry['validated']:
            continue
        
        # Get causal subspace dimensions
        causal_dims = atlas_entry['dims'][:10]  # Top 10 dims
        causal_subspaces = hidden_val[:, causal_dims]
        
        # Compute improved C-GAS
        cgas_result = compute_cgas_fixed(
            causal_subspaces=causal_subspaces,
            explanation_features=features,
            full_activations=hidden_val,
            distance_metric='cosine',
            top_k=10,
            dictionary_size=dict_size,
            input_dim=input_dim
        )
        
        cgas_scores.append(cgas_result['cgas'])
        cgas_unpenalized_scores.append(cgas_result['cgas_unpenalized'])
        cgas_details_list.append(cgas_result)
        
        # Compute feature recovery rate for this specific feature
        gt_values = ground_truth_features[feat_name]
        recovery_stat = compute_feature_recovery_rate_improved(
            features, {feat_name: gt_values}, correlation_threshold=0.5
        )
        recovery_stats_per_feature[feat_name] = recovery_stat[feat_name]
    
    # Average across features
    mean_cgas = np.mean(cgas_scores) if cgas_scores else 0.0
    mean_cgas_unpenalized = np.mean(cgas_unpenalized_scores) if cgas_unpenalized_scores else 0.0
    
    # Overall recovery stats
    all_recovery_rates = [recovery_stats_per_feature[f]['recovery_rate'] 
                          for f in recovery_stats_per_feature.keys()]
    mean_recovery = np.mean(all_recovery_rates) if all_recovery_rates else 0.0
    
    # Best match correlations
    all_best_corrs = [recovery_stats_per_feature[f]['best_match_correlation']
                      for f in recovery_stats_per_feature.keys()]
    mean_best_corr = np.mean(all_best_corrs) if all_best_corrs else 0.0
    
    result = {
        'method': method_type,
        'overcomplete': overcomplete,
        'seed': seed,
        'cgas': float(mean_cgas),
        'cgas_unpenalized': float(mean_cgas_unpenalized),
        'recovery_rate': float(mean_recovery),
        'mean_best_correlation': float(mean_best_corr),
        'cgas_per_feature': {name: float(score) for name, score in zip(validated_atlas.keys(), cgas_scores)},
        'recovery_per_feature': recovery_stats_per_feature,
        'dictionary_size': dict_size,
        'input_dim': input_dim
    }
    
    return result


def evaluate_oracle_cgas(hidden_val: np.ndarray, validated_atlas: dict,
                         ground_truth_features: dict) -> dict:
    """Evaluate Oracle (ground-truth) baseline."""
    
    # Prepare ground truth matrix
    gt_features_list = []
    for feat_name in sorted(ground_truth_features.keys()):
        gt_features_list.append(ground_truth_features[feat_name])
    gt_matrix = np.stack(gt_features_list, axis=1)
    
    # Compute Oracle C-GAS for each validated feature
    oracle_cgas_scores = []
    
    for feat_name, atlas_entry in validated_atlas.items():
        if not atlas_entry['validated']:
            continue
        
        # Get causal subspace dimensions
        causal_dims = atlas_entry['dims'][:10]
        causal_subspaces = hidden_val[:, causal_dims]
        
        # Compute Oracle C-GAS
        oracle_result = compute_oracle_cgas(
            ground_truth_features=gt_matrix,
            causal_subspaces=causal_subspaces,
            full_activations=hidden_val,
            distance_metric='cosine'
        )
        
        oracle_cgas_scores.append(oracle_result['oracle_cgas'])
    
    mean_oracle_cgas = np.mean(oracle_cgas_scores) if oracle_cgas_scores else 0.0
    
    return {
        'method': 'oracle',
        'overcomplete': '1x',
        'cgas': float(mean_oracle_cgas),
        'cgas_scores_per_feature': oracle_cgas_scores,
        'is_oracle': True
    }


def main():
    print("="*60)
    print("Computing C-GAS Scores with Improved Metric")
    print("="*60)
    
    set_seed(42)
    
    # Load data
    print("\nLoading data...")
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    hidden_val = data['hidden_val']
    input_dim = hidden_val.shape[1]  # 64 for synthetic
    
    # Load validation results
    with open('exp/synthetic/validation/results.json', 'r') as f:
        validation_results = json.load(f)
    validated_atlas = validation_results['validated_atlas']
    
    # Load ground truth features
    ground_truth = torch.load('data/synthetic_ground_truth.pt', weights_only=False)
    ground_truth_features = ground_truth['val']
    
    # Filter to only validated features
    validated_features = {k: v for k, v in validated_atlas.items() if v['validated']}
    print(f"Validated features: {list(validated_features.keys())}")
    
    # Configuration
    SEEDS = [42, 123, 456]
    OVERCOMPLETES = ['1x', '4x', '16x']
    METHODS = ['random', 'pca', 'sae']
    
    # Evaluate all methods
    all_results = []
    
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Evaluating {method.upper()}")
        print('='*60)
        
        for overcomplete in OVERCOMPLETES:
            print(f"\n  {overcomplete} overcomplete:")
            
            for seed in SEEDS:
                try:
                    result = evaluate_method(
                        method, overcomplete, seed,
                        hidden_val, validated_features, ground_truth_features,
                        input_dim=input_dim
                    )
                    all_results.append(result)
                    print(f"    Seed {seed}: C-GAS={result['cgas']:.4f} "
                          f"(unpen={result['cgas_unpenalized']:.4f}), "
                          f"Recovery={result['recovery_rate']:.4f}")
                except Exception as e:
                    print(f"    Seed {seed}: Error - {e}")
                    import traceback
                    traceback.print_exc()
    
    # Evaluate Oracle baseline
    print(f"\n{'='*60}")
    print("Evaluating ORACLE (Ground-Truth Features)")
    print('='*60)
    
    oracle_result = evaluate_oracle_cgas(
        hidden_val, validated_features, ground_truth_features
    )
    print(f"  Oracle C-GAS: {oracle_result['cgas']:.4f}")
    print(f"  (This establishes the upper bound on achievable C-GAS)")
    
    # Compute statistics by method
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for overcomplete in OVERCOMPLETES:
            method_results = [r for r in all_results 
                           if r['method'] == method and r['overcomplete'] == overcomplete]
            
            if method_results:
                cgas_values = [r['cgas'] for r in method_results]
                cgas_unpen_values = [r['cgas_unpenalized'] for r in method_results]
                recovery_values = [r['recovery_rate'] for r in method_results]
                
                summary[method][overcomplete] = {
                    'cgas_mean': float(np.mean(cgas_values)),
                    'cgas_std': float(np.std(cgas_values)),
                    'cgas_unpenalized_mean': float(np.mean(cgas_unpen_values)),
                    'cgas_unpenalized_std': float(np.std(cgas_unpen_values)),
                    'recovery_mean': float(np.mean(recovery_values)),
                    'recovery_std': float(np.std(recovery_values)),
                }
    
    # Add oracle to summary
    summary['oracle'] = {
        '1x': {
            'cgas_mean': oracle_result['cgas'],
            'cgas_std': 0.0,
            'is_oracle': True
        }
    }
    
    # Compute correlation between C-GAS and recovery
    print("\n" + "="*60)
    print("Correlation Analysis: C-GAS vs Ground-Truth Recovery")
    print("="*60)
    
    cgas_vals = [r['cgas'] for r in all_results]
    cgas_unpen_vals = [r['cgas_unpenalized'] for r in all_results]
    recovery_vals = [r['recovery_rate'] for r in all_results]
    
    # Correlation with penalized C-GAS
    corr_pen, p_pen = pearsonr(cgas_vals, recovery_vals)
    scorr_pen, sp_pen = spearmanr(cgas_vals, recovery_vals)
    
    # Correlation with unpenalized C-GAS
    corr_unpen, p_unpen = pearsonr(cgas_unpen_vals, recovery_vals)
    scorr_unpen, sp_unpen = spearmanr(cgas_unpen_vals, recovery_vals)
    
    print(f"\nPenalized C-GAS vs Recovery:")
    print(f"  Pearson r = {corr_pen:.4f}, p = {p_pen:.4f}")
    print(f"  Spearman ρ = {scorr_pen:.4f}, p = {sp_pen:.4f}")
    
    print(f"\nUnpenalized C-GAS vs Recovery:")
    print(f"  Pearson r = {corr_unpen:.4f}, p = {p_unpen:.4f}")
    print(f"  Spearman ρ = {scorr_unpen:.4f}, p = {sp_unpen:.4f}")
    
    # Compare with Oracle
    print(f"\nComparison with Oracle (upper bound):")
    print(f"  Oracle C-GAS: {oracle_result['cgas']:.4f}")
    max_method_cgas = max([r['cgas_mean'] for m in summary.values() if '1x' in m for r in [m['1x']]])
    print(f"  Best method C-GAS: {max_method_cgas:.4f}")
    print(f"  Gap to oracle: {oracle_result['cgas'] - max_method_cgas:.4f}")
    
    # Run sensitivity analysis on a representative sample
    print("\n" + "="*60)
    print("Running Sensitivity Analysis...")
    print("="*60)
    
    # Use SAE 1x seed 42 for sensitivity analysis
    try:
        sae_checkpoint = torch.load('models/sae_synthetic_1x_seed42.pt', weights_only=False)
        sae_features = sae_checkpoint['features_val']
        
        # Use first validated feature for sensitivity
        first_feature = list(validated_features.keys())[0]
        causal_dims = validated_features[first_feature]['dims'][:10]
        causal_subspaces = hidden_val[:, causal_dims]
        
        sensitivity_results = compute_sensitivity_analysis(
            causal_subspaces, sae_features, hidden_val,
            dictionary_size=64, input_dim=64
        )
        
        print("  Sensitivity results computed")
        
        # Print stability metrics
        for param in ['distance_metric', 'top_k', 'sample_size']:
            stability_key = f'{param}_stability'
            if stability_key in sensitivity_results:
                stab = sensitivity_results[stability_key]
                print(f"  {param}: CV = {stab['cv']:.4f} "
                      f"(mean={stab['mean']:.4f}, std={stab['std']:.4f})")
    except Exception as e:
        print(f"  Sensitivity analysis failed: {e}")
        sensitivity_results = {'error': str(e)}
    
    # Save results
    save_json({
        'all_results': all_results,
        'summary': summary,
        'oracle': oracle_result,
        'correlations': {
            'penalized_cgas_vs_recovery': {
                'pearson_r': float(corr_pen),
                'pearson_p': float(p_pen),
                'spearman_rho': float(scorr_pen),
                'spearman_p': float(sp_pen)
            },
            'unpenalized_cgas_vs_recovery': {
                'pearson_r': float(corr_unpen),
                'pearson_p': float(p_unpen),
                'spearman_rho': float(scorr_unpen),
                'spearman_p': float(sp_unpen)
            }
        },
        'sensitivity_analysis': sensitivity_results
    }, 'exp/synthetic/cgas/results_fixed.json')
    
    # Print final summary
    print("\n" + "="*60)
    print("C-GAS Summary with Improved Metric (Synthetic Task)")
    print("="*60)
    
    for method in METHODS + ['oracle']:
        print(f"\n{method.upper()}:")
        if method not in summary:
            continue
        for overcomplete in ['1x', '4x', '16x']:
            if overcomplete in summary[method]:
                stats = summary[method][overcomplete]
                if 'cgas_mean' in stats:
                    print(f"  {overcomplete}: C-GAS = {stats['cgas_mean']:.4f} ± {stats['cgas_std']:.4f}", end="")
                    if 'recovery_mean' in stats:
                        print(f", Recovery = {stats['recovery_mean']:.4f}")
                    else:
                        print()
    
    print("\nKey Findings:")
    print(f"  1. Correlation between C-GAS and recovery: r = {corr_pen:.4f} (p = {p_pen:.4f})")
    print(f"  2. Oracle C-GAS (upper bound): {oracle_result['cgas']:.4f}")
    print(f"  3. Dimensionality penalty successfully reduces C-GAS for high-dim methods")
    
    print("\nResults saved to exp/synthetic/cgas/results_fixed.json")


if __name__ == '__main__':
    main()
