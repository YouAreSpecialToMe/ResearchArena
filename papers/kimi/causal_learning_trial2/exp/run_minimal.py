"""
Minimal experiment for quick validation.
Runs on a very small subset to demonstrate the concept.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import time
import glob
import warnings
warnings.filterwarnings('ignore')

from shared.metrics import compute_all_metrics
from notears.run import notears_linear
from golem.run import golem_ev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spiced'))
from spiced_main import spiced


def run_minimal_experiment():
    """Run minimal experiment on very small subset."""
    
    # Get minimal subset
    dataset_files = glob.glob("data/processed/datasets/*.npz")
    
    # Take only first 50 files for quick testing
    # Filter for n_nodes=10 and N=50,100
    filtered_files = []
    for f in dataset_files:
        data_dict = np.load(f)
        n_samples = int(data_dict['n_samples'])
        adj = data_dict['adj']
        n_nodes = adj.shape[0]
        graph_id = int(data_dict['graph_id'])
        
        if n_nodes == 10 and n_samples in [50, 100] and graph_id <= 10:
            filtered_files.append(f)
    
    # Limit to 50 files
    filtered_files = filtered_files[:50]
    
    print(f"Running on {len(filtered_files)} datasets...")
    
    results = {
        'notears': [],
        'golem': [],
        'spiced': []
    }
    
    for i, dataset_file in enumerate(filtered_files):
        print(f"Processing {i+1}/{len(filtered_files)}: {os.path.basename(dataset_file)}")
        
        data_dict = np.load(dataset_file)
        data = data_dict['data']
        true_adj = data_dict['adj']
        
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        n_nodes = true_adj.shape[0]
        
        base_info = {
            'graph_id': graph_id,
            'mechanism': mechanism,
            'n_samples': n_samples,
            'seed': seed,
            'n_nodes': n_nodes
        }
        
        # Run NOTEARS (fast)
        try:
            start = time.time()
            pred_adj = notears_linear(data, lambda1=0.1, max_iter=30, 
                                     w_threshold=0.3, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['notears'].append({**base_info, 'runtime': runtime, **metrics})
            print(f"  NOTEARS: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}")
        except Exception as e:
            print(f"  NOTEARS error: {e}")
        
        # Run GOLEM (fast)
        try:
            start = time.time()
            pred_adj = golem_ev(data, lambda1=0.02, lambda2=5.0, 
                               num_iter=300, learning_rate=0.001,
                               w_threshold=0.3, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['golem'].append({**base_info, 'runtime': runtime, **metrics})
            print(f"  GOLEM: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}")
        except Exception as e:
            print(f"  GOLEM error: {e}")
        
        # Run SPICED (fast)
        try:
            pred_adj, timing, _ = spiced(
                data,
                k_neighbors=5,
                alpha=0.05,
                lambda1=0.1,
                lambda3=0.01,
                max_iter=30,
                w_threshold=0.3,
                use_it_init=True,
                use_structural_constraints=True,
                seed=seed
            )
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['spiced'].append({
                **base_info, 
                'runtime': timing['total'],
                **metrics
            })
            print(f"  SPICED: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}")
        except Exception as e:
            print(f"  SPICED error: {e}")
    
    # Save results
    os.makedirs("results/synthetic", exist_ok=True)
    
    for method, data in results.items():
        with open(f"results/synthetic/{method}_summary.json", 'w') as f:
            json.dump(data, f, indent=2, default=float)
        print(f"Saved {method}: {len(data)} results")
    
    return results


def run_sachs_experiment():
    """Run all methods on Sachs dataset."""
    
    sachs_data = np.load("data/processed/real_world/sachs.npz")
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    
    results = {
        'notears': [],
        'golem': [],
        'spiced': []
    }
    
    for seed in range(1, 4):
        print(f"\nSachs seed {seed}:")
        
        # NOTEARS
        try:
            start = time.time()
            pred_adj = notears_linear(data, lambda1=0.1, max_iter=50, 
                                     w_threshold=0.3, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['notears'].append({'dataset': 'sachs', 'seed': seed, 'runtime': runtime, **metrics})
            print(f"  NOTEARS: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
        except Exception as e:
            print(f"  NOTEARS error: {e}")
        
        # GOLEM
        try:
            start = time.time()
            pred_adj = golem_ev(data, lambda1=0.02, lambda2=5.0, 
                               num_iter=500, learning_rate=0.001,
                               w_threshold=0.3, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['golem'].append({'dataset': 'sachs', 'seed': seed, 'runtime': runtime, **metrics})
            print(f"  GOLEM: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
        except Exception as e:
            print(f"  GOLEM error: {e}")
        
        # SPICED
        try:
            pred_adj, timing, _ = spiced(
                data,
                k_neighbors=5,
                alpha=0.05,
                lambda1=0.1,
                lambda3=0.01,
                max_iter=50,
                w_threshold=0.3,
                use_it_init=True,
                use_structural_constraints=True,
                seed=seed
            )
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['spiced'].append({
                'dataset': 'sachs', 
                'seed': seed, 
                'runtime': timing['total'],
                **metrics
            })
            print(f"  SPICED: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
        except Exception as e:
            print(f"  SPICED error: {e}")
    
    # Save results
    os.makedirs("results/real_world", exist_ok=True)
    for method, data in results.items():
        with open(f"results/real_world/{method}_sachs.json", 'w') as f:
            json.dump(data, f, indent=2, default=float)
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Running minimal experiment")
    print("=" * 60)
    
    results = run_minimal_experiment()
    
    print("\n" + "=" * 60)
    print("Running on Sachs dataset")
    print("=" * 60)
    
    sachs_results = run_sachs_experiment()
    
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
