"""
Run Sachs dataset experiments for SPICED and baselines.
"""
import sys
import os
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

import numpy as np
from spiced.spiced_main import spiced
from notears.run_fixed import notears_linear_fixed
from shared.metrics import compute_all_metrics


def run_sachs_experiments():
    """Run all methods on Sachs dataset."""
    print("="*60)
    print("Sachs Dataset Experiments")
    print("="*60)
    
    # Load Sachs data
    sachs_file = os.path.join(PROJECT_ROOT, "data/processed/real_world/sachs.npz")
    sachs_data = np.load(sachs_file)
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    var_names = sachs_data['var_names']
    
    print(f"Data shape: {data.shape}")
    print(f"True edges: {int(true_adj.sum())}")
    print(f"Variables: {var_names}")
    
    results = {
        'spiced': [],
        'notears': []
    }
    
    # Run SPICED with 5 seeds
    print("\n--- Running SPICED ---")
    for seed in range(1, 6):
        start = time.time()
        pred_adj, timing, _ = spiced(
            data,
            k_neighbors=5,
            alpha=0.05,
            lambda1=0.1,
            lambda3=0.01,
            max_iter=100,
            w_threshold=0.3,
            seed=seed
        )
        runtime = time.time() - start
        
        metrics = compute_all_metrics(true_adj, pred_adj)
        
        result = {
            'seed': seed,
            'runtime': runtime,
            'runtime_phase1': timing.get('phase1_skeleton', 0),
            'runtime_phase2': timing.get('phase2_constraints', 0),
            'runtime_phase3': timing.get('phase3_optimization', 0),
            **metrics
        }
        results['spiced'].append(result)
        
        print(f"  Seed {seed}: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}, "
              f"Time={runtime:.2f}s")
    
    # Run NOTEARS with 5 seeds
    print("\n--- Running NOTEARS ---")
    for seed in range(1, 6):
        start = time.time()
        pred_adj, _ = notears_linear_fixed(
            data,
            lambda1=0.1,
            max_iter=100,
            w_threshold=0.3,
            seed=seed
        )
        runtime = time.time() - start
        
        metrics = compute_all_metrics(true_adj, pred_adj)
        
        result = {
            'seed': seed,
            'runtime': runtime,
            **metrics
        }
        results['notears'].append(result)
        
        print(f"  Seed {seed}: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}, "
              f"Time={runtime:.2f}s")
    
    # Compute summary statistics
    print("\n--- Summary Statistics ---")
    for method_name, method_results in results.items():
        shds = [r['shd'] for r in method_results]
        tprs = [r['tpr'] for r in method_results]
        fdrs = [r['fdr'] for r in method_results]
        times = [r['runtime'] for r in method_results]
        
        print(f"\n{method_name.upper()}:")
        print(f"  SHD: {np.mean(shds):.2f} ± {np.std(shds):.2f}")
        print(f"  TPR: {np.mean(tprs):.3f} ± {np.std(tprs):.3f}")
        print(f"  FDR: {np.mean(fdrs):.3f} ± {np.std(fdrs):.3f}")
        print(f"  Runtime: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
    
    # Save results
    os.makedirs(os.path.join(PROJECT_ROOT, "results/real_world"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results/real_world/sachs_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\nSachs results saved.")
    return results


if __name__ == "__main__":
    run_sachs_experiments()
