"""Compute C-GAS scores for all methods on synthetic task."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
import json

from exp.shared.utils import set_seed, save_json
from exp.shared.metrics import compute_cgas, compute_reconstruction_error

def compute_feature_recovery_rate(features, ground_truth_features, feat_name):
    """Compute how well features recover ground truth."""
    gt = ground_truth_features[feat_name]
    
    best_corrs = []
    for j in range(min(features.shape[1], 100)):  # Check up to 100 features
        feat_j = features[:, j]
        corr = np.abs(np.corrcoef(feat_j, gt)[0, 1])
        if not np.isnan(corr):
            best_corrs.append(corr)
    
    if len(best_corrs) == 0:
        return 0.0
    
    # Recovery rate: proportion of features with correlation > 0.5
    recovery_rate = np.mean(np.array(best_corrs) > 0.5)
    return float(recovery_rate)

def evaluate_method(method_type: str, overcomplete: str, seed: int, 
                    hidden_val: np.ndarray, validated_atlas: dict, 
                    ground_truth_features: dict) -> dict:
    """Evaluate a single method configuration."""
    
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
    else:
        raise ValueError(f"Unknown method: {method_type}")
    
    # Compute C-GAS for each feature
    cgas_scores = []
    recovery_rates = []
    
    for feat_name, atlas_entry in validated_atlas.items():
        if not atlas_entry['validated']:
            continue
        
        # Get causal subspace dimensions
        causal_dims = atlas_entry['dims'][:10]  # Top 10 dims
        causal_subspaces = hidden_val[:, causal_dims]
        
        # Compute C-GAS
        cgas, rho_ce, rho_cf = compute_cgas(
            causal_subspaces=causal_subspaces,
            explanation_features=features,
            full_activations=hidden_val,
            distance_metric='cosine',
            top_k=10
        )
        
        cgas_scores.append(cgas)
        
        # Compute feature recovery rate
        recovery = compute_feature_recovery_rate(features, ground_truth_features, feat_name)
        recovery_rates.append(recovery)
    
    # Average across features
    mean_cgas = np.mean(cgas_scores) if cgas_scores else 0.0
    mean_recovery = np.mean(recovery_rates) if recovery_rates else 0.0
    
    result = {
        'method': method_type,
        'overcomplete': overcomplete,
        'seed': seed,
        'cgas': float(mean_cgas),
        'recovery_rate': float(mean_recovery),
        'cgas_per_feature': {name: float(score) for name, score in zip(validated_atlas.keys(), cgas_scores)}
    }
    
    return result

def main():
    print("="*60)
    print("Computing C-GAS Scores (Synthetic Task)")
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
                        hidden_val, validated_features, ground_truth_features
                    )
                    all_results.append(result)
                    print(f"    Seed {seed}: C-GAS={result['cgas']:.4f}, Recovery={result['recovery_rate']:.4f}")
                except Exception as e:
                    print(f"    Seed {seed}: Error - {e}")
    
    # Compute statistics
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for overcomplete in OVERCOMPLETES:
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
    
    # Save results
    save_json({
        'all_results': all_results,
        'summary': summary
    }, 'exp/synthetic/cgas/results.json')
    
    # Print summary
    print("\n" + "="*60)
    print("C-GAS Summary (Synthetic Task)")
    print("="*60)
    
    for method in METHODS:
        print(f"\n{method.upper()}:")
        for overcomplete in OVERCOMPLETES:
            if overcomplete in summary[method]:
                stats = summary[method][overcomplete]
                print(f"  {overcomplete}: C-GAS = {stats['cgas_mean']:.4f} ± {stats['cgas_std']:.4f}, "
                      f"Recovery = {stats['recovery_mean']:.4f}")
    
    # Compute correlation between C-GAS and recovery
    cgas_vals = [r['cgas'] for r in all_results]
    recovery_vals = [r['recovery_rate'] for r in all_results]
    correlation = np.corrcoef(cgas_vals, recovery_vals)[0, 1]
    print(f"\nCorrelation between C-GAS and recovery rate: {correlation:.4f}")
    
    print("\nResults saved to exp/synthetic/cgas/results.json")

if __name__ == '__main__':
    main()
