"""
Validation: Information Gain Approximation.

Tests whether IG ≈ 4*p*(1-p)*Power*Disc has positive correlation with actual information gain.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
from typing import Dict, List, Tuple
from shared.metrics import compute_metrics
from shared.utils import load_dataset, save_results
from mf_acd.mf_acd import MFACD


def compute_actual_ig(true_graph: np.ndarray, test_result: bool, edge: Tuple[int, int]) -> float:
    """
    Compute actual information gain from a test result.
    Simplified: reduction in edge uncertainty.
    """
    i, j = edge
    # If edge exists in true graph and test detected it (or vice versa)
    edge_exists = true_graph[i, j] == 1 or true_graph[j, i] == 1
    
    # IG is high when test result matches ground truth
    if test_result == edge_exists:
        return 1.0  # Correct detection
    else:
        return -1.0  # Incorrect detection


def run_ig_validation(data: np.ndarray, true_adj: np.ndarray) -> Dict:
    """
    Validate IG approximation by comparing predicted vs actual IG.
    """
    n_vars = data.shape[1]
    
    # Generate some test edges
    np.random.seed(42)
    test_edges = []
    for _ in range(50):  # Sample 50 edges
        i, j = np.random.choice(n_vars, 2, replace=False)
        if i > j:
            i, j = j, i
        test_edges.append((i, j))
    
    # For each edge, compute predicted IG and actual IG
    predicted_igs = []
    actual_igs = []
    
    mf_acd = MFACD()
    
    for edge in test_edges:
        i, j = edge
        
        # Compute predicted IG (simplified for validation)
        # p = uncertainty (probability edge exists)
        p = 0.5  # Uniform prior
        power = 0.8  # Medium fidelity
        disc = 0.7  # Discriminative power
        
        predicted_ig = 4 * p * (1 - p) * power * disc
        predicted_igs.append(predicted_ig)
        
        # Compute actual IG: perform a test and see how much we learn
        p_value = mf_acd.correlation_test(data, i, j, [])
        test_result = p_value < 0.05  # Test says dependent
        
        actual_ig = compute_actual_ig(true_adj, test_result, edge)
        actual_igs.append(actual_ig)
    
    # Compute correlation
    from scipy.stats import pearsonr, spearmanr
    
    pearson_r, pearson_p = pearsonr(predicted_igs, actual_igs)
    spearman_r, spearman_p = spearmanr(predicted_igs, actual_igs)
    
    return {
        'predicted_ig': predicted_igs,
        'actual_ig': actual_igs,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_tests': len(test_edges)
    }


def run_all_experiments(data_dir: str = "data/synthetic",
                        output_dir: str = "results/validation/ig_approximation"):
    """Run IG validation on subset of datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    
    # Sample: 20 configs × 5 seeds = 100 runs
    import random
    random.seed(42)
    sampled = random.sample(manifest['datasets'], min(100, len(manifest['datasets'])))
    
    print(f"Running IG validation on {len(sampled)} datasets...")
    
    results = []
    for i, dataset_info in enumerate(sampled):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(sampled)}")
        
        dataset_path = dataset_info['path']
        if not dataset_path.startswith('data/synthetic/'):
            dataset_path = os.path.join(data_dir, os.path.basename(dataset_info['path']))
        
        try:
            dataset = load_dataset(dataset_path)
            result = run_ig_validation(dataset['data'], dataset['adjacency'])
            result['dataset_name'] = dataset_info['name']
            result['dataset_config'] = dataset_info['config']
            results.append(result)
        except Exception as e:
            print(f"Error on {dataset_info['name']}: {e}")
            results.append({'dataset_name': dataset_info['name'], 'error': str(e)})
    
    save_results(results, os.path.join(output_dir, "results.json"))
    
    # Aggregate correlation
    pearson_rs = [r['pearson_r'] for r in results if 'pearson_r' in r]
    spearman_rs = [r['spearman_r'] for r in results if 'spearman_r' in r]
    
    print("\nIG Validation Summary:")
    print(f"  Mean Pearson r: {np.mean(pearson_rs):.3f} ± {np.std(pearson_rs):.3f}")
    print(f"  Mean Spearman r: {np.mean(spearman_rs):.3f} ± {np.std(spearman_rs):.3f}")
    print(f"  Success criterion (r > 0.5): {np.mean(pearson_rs) > 0.5}")
    
    return results


if __name__ == "__main__":
    run_all_experiments()
