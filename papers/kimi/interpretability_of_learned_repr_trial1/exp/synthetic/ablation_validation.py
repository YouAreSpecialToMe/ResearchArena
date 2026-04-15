"""Ablation study: C-GAS with vs without validation."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr, ttest_rel

from exp.shared.utils import set_seed, save_json
from exp.shared.metrics import compute_cgas


def compute_feature_recovery_rate(features, ground_truth_features, feat_name):
    """Compute how well features recover ground truth."""
    gt = ground_truth_features[feat_name]
    
    best_corrs = []
    for j in range(min(features.shape[1], features.shape[1])):
        feat_j = features[:, j]
        corr = np.abs(np.corrcoef(feat_j, gt)[0, 1])
        if not np.isnan(corr):
            best_corrs.append(corr)
    
    if len(best_corrs) == 0:
        return 0.0
    
    # Recovery rate: proportion of features with correlation > 0.5
    recovery_rate = np.mean(np.array(best_corrs) > 0.5)
    return float(recovery_rate)


def main():
    print("="*60)
    print("Ablation Study: C-GAS with vs without validation")
    print("="*60)
    
    set_seed(42)
    
    # Load data
    print("\nLoading data...")
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    hidden_val = data['hidden_val']
    
    # Load validation results
    with open('exp/synthetic/validation/results.json', 'r') as f:
        validation_results = json.load(f)
    
    validated_atlas = validation_results['validated_atlas']
    
    # Load ground truth features
    ground_truth = torch.load('data/synthetic_ground_truth.pt', weights_only=False)
    ground_truth_features = ground_truth['val']
    
    # Get all candidate features (not just validated)
    all_candidates = {}
    for feat_name, entry in validated_atlas.items():
        all_candidates[feat_name] = {
            'dims': entry['dims'],
            'effects': entry['effects'],
            'validated': entry['validated']
        }
    
    # Validated only
    validated_only = {k: v for k, v in all_candidates.items() if v['validated']}
    
    print(f"Total candidate features: {len(all_candidates)}")
    print(f"Validated features: {len(validated_only)}")
    
    # Configuration
    SEEDS = [42, 123, 456]
    OVERCOMPLETES = ['1x', '4x', '16x']
    METHODS = ['random', 'pca', 'sae']
    
    results_validated = []
    results_unvalidated = []
    
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
                        checkpoint = torch.load(f'models/baseline_random_{overcomplete}_seed{seed}.pt', weights_only=False)
                    elif method == 'pca':
                        checkpoint = torch.load(f'models/baseline_pca_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    
                    features = checkpoint['features_val']
                    
                    # Compute C-GAS with validation
                    cgas_validated_list = []
                    recovery_validated_list = []
                    
                    for feat_name, atlas_entry in validated_only.items():
                        causal_dims = atlas_entry['dims'][:10]
                        causal_subspaces = hidden_val[:, causal_dims]
                        
                        cgas, rho_ce, rho_cf = compute_cgas(
                            causal_subspaces=causal_subspaces,
                            explanation_features=features,
                            full_activations=hidden_val,
                            distance_metric='cosine',
                            top_k=10
                        )
                        cgas_validated_list.append(cgas)
                        
                        recovery = compute_feature_recovery_rate(features, ground_truth_features, feat_name)
                        recovery_validated_list.append(recovery)
                    
                    mean_cgas_validated = np.mean(cgas_validated_list) if cgas_validated_list else 0.0
                    mean_recovery_validated = np.mean(recovery_validated_list) if recovery_validated_list else 0.0
                    
                    # Compute C-GAS without validation (all candidates)
                    cgas_unvalidated_list = []
                    recovery_unvalidated_list = []
                    
                    for feat_name, atlas_entry in all_candidates.items():
                        causal_dims = atlas_entry['dims'][:10]
                        causal_subspaces = hidden_val[:, causal_dims]
                        
                        cgas, rho_ce, rho_cf = compute_cgas(
                            causal_subspaces=causal_subspaces,
                            explanation_features=features,
                            full_activations=hidden_val,
                            distance_metric='cosine',
                            top_k=10
                        )
                        cgas_unvalidated_list.append(cgas)
                        
                        recovery = compute_feature_recovery_rate(features, ground_truth_features, feat_name)
                        recovery_unvalidated_list.append(recovery)
                    
                    mean_cgas_unvalidated = np.mean(cgas_unvalidated_list) if cgas_unvalidated_list else 0.0
                    mean_recovery_unvalidated = np.mean(recovery_unvalidated_list) if recovery_unvalidated_list else 0.0
                    
                    results_validated.append({
                        'method': method,
                        'overcomplete': overcomplete,
                        'seed': seed,
                        'cgas': float(mean_cgas_validated),
                        'recovery_rate': float(mean_recovery_validated)
                    })
                    
                    results_unvalidated.append({
                        'method': method,
                        'overcomplete': overcomplete,
                        'seed': seed,
                        'cgas': float(mean_cgas_unvalidated),
                        'recovery_rate': float(mean_recovery_unvalidated)
                    })
                    
                    print(f"    Seed {seed}: Validated C-GAS={mean_cgas_validated:.4f}, "
                          f"Unvalidated C-GAS={mean_cgas_unvalidated:.4f}")
                    
                except Exception as e:
                    print(f"    Seed {seed}: Error - {e}")
    
    # Statistical analysis
    print("\n" + "="*60)
    print("Statistical Analysis")
    print("="*60)
    
    # Compare C-GAS scores
    validated_cgas = [r['cgas'] for r in results_validated]
    unvalidated_cgas = [r['cgas'] for r in results_unvalidated]
    
    t_stat_cgas, p_val_cgas = ttest_rel(validated_cgas, unvalidated_cgas)
    print(f"\nC-GAS comparison (paired t-test):")
    print(f"  Validated mean: {np.mean(validated_cgas):.4f} ± {np.std(validated_cgas):.4f}")
    print(f"  Unvalidated mean: {np.mean(unvalidated_cgas):.4f} ± {np.std(unvalidated_cgas):.4f}")
    print(f"  t-statistic: {t_stat_cgas:.4f}")
    print(f"  p-value: {p_val_cgas:.4f}")
    print(f"  Difference: {'Significant' if p_val_cgas < 0.05 else 'Not significant'}")
    
    # Correlation with ground truth recovery
    validated_corr, validated_p = pearsonr(
        [r['cgas'] for r in results_validated],
        [r['recovery_rate'] for r in results_validated]
    )
    
    unvalidated_corr, unvalidated_p = pearsonr(
        [r['cgas'] for r in results_unvalidated],
        [r['recovery_rate'] for r in results_unvalidated]
    )
    
    print(f"\nCorrelation with ground truth recovery:")
    print(f"  Validated C-GAS: r={validated_corr:.4f} (p={validated_p:.4f})")
    print(f"  Unvalidated C-GAS: r={unvalidated_corr:.4f} (p={unvalidated_p:.4f})")
    
    # Per-method analysis
    print(f"\nPer-method C-GAS comparison:")
    for method in METHODS:
        val_method = [r['cgas'] for r in results_validated if r['method'] == method]
        unval_method = [r['cgas'] for r in results_unvalidated if r['method'] == method]
        
        if val_method and unval_method:
            t_stat, p_val = ttest_rel(val_method, unval_method)
            print(f"  {method.upper()}:")
            print(f"    Validated: {np.mean(val_method):.4f} ± {np.std(val_method):.4f}")
            print(f"    Unvalidated: {np.mean(unval_method):.4f} ± {np.std(unval_method):.4f}")
            print(f"    p-value: {p_val:.4f}")
    
    # Save results
    save_json({
        'results_validated': results_validated,
        'results_unvalidated': results_unvalidated,
        'statistical_tests': {
            'cgas_comparison': {
                't_statistic': float(t_stat_cgas),
                'p_value': float(p_val_cgas),
                'validated_mean': float(np.mean(validated_cgas)),
                'validated_std': float(np.std(validated_cgas)),
                'unvalidated_mean': float(np.mean(unvalidated_cgas)),
                'unvalidated_std': float(np.std(unvalidated_cgas))
            },
            'correlation_comparison': {
                'validated_correlation': float(validated_corr),
                'validated_p_value': float(validated_p),
                'unvalidated_correlation': float(unvalidated_corr),
                'unvalidated_p_value': float(unvalidated_p)
            }
        },
        'conclusion': {
            'validation_reduces_cgas': bool(np.mean(unvalidated_cgas) > np.mean(validated_cgas)),
            'validation_improves_correlation': bool(validated_corr > unvalidated_corr),
            'significant_difference': bool(p_val_cgas < 0.05)
        }
    }, 'exp/synthetic/ablation_validation/results.json')
    
    print("\n" + "="*60)
    print("Conclusion:")
    print("="*60)
    if validated_corr > unvalidated_corr:
        print("✓ Validation IMPROVES correlation with ground truth")
    else:
        print("✗ Validation does NOT improve correlation with ground truth")
    
    if p_val_cgas < 0.05:
        print(f"✓ Validation significantly changes C-GAS scores (p={p_val_cgas:.4f})")
    else:
        print(f"✗ No significant difference in C-GAS with validation (p={p_val_cgas:.4f})")
    
    print("\nResults saved to exp/synthetic/ablation_validation/results.json")


if __name__ == '__main__':
    main()
