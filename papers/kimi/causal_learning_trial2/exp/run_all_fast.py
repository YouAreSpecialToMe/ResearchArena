"""
Fast version of experiments on reduced dataset.
Focuses on key configurations to demonstrate sample efficiency.
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

# Import methods
from notears.run import notears_linear
from golem.run import golem_ev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spiced'))
from spiced_main import spiced


def run_all_methods_fast():
    """Run all methods on a reduced dataset for faster completion."""
    
    # Get subset of datasets: focus on key configurations
    dataset_files = glob.glob("data/processed/datasets/*.npz")
    
    # Filter: n_nodes=10,20 and specific sample sizes
    filtered_files = []
    for f in dataset_files:
        data_dict = np.load(f)
        n_samples = int(data_dict['n_samples'])
        adj = data_dict['adj']
        n_nodes = adj.shape[0]
        graph_id = int(data_dict['graph_id'])
        
        # Focus on smaller graphs and key sample sizes
        # Use graph_id to subsample (only use first 5 graphs per config)
        if n_nodes <= 20 and n_samples in [50, 100, 200, 500] and (graph_id % 10) <= 4:
            filtered_files.append(f)
    
    print(f"Running on {len(filtered_files)} datasets (filtered subset)...")
    
    results = {
        'notears': [],
        'golem': [],
        'spiced': []
    }
    
    total = len(filtered_files)
    
    for i, dataset_file in enumerate(filtered_files):
        if i % 10 == 0:
            print(f"Progress: {i}/{total} ({100*i/total:.1f}%)")
        
        # Load dataset
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
        
        # Run NOTEARS
        try:
            start = time.time()
            pred_adj = notears_linear(data, lambda1=0.1, max_iter=50, 
                                     w_threshold=0.3, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['notears'].append({**base_info, 'runtime': runtime, **metrics})
        except Exception as e:
            print(f"NOTEARS error on {dataset_file}: {e}")
            results['notears'].append({**base_info, 'runtime': None, 'error': str(e)})
        
        # Run GOLEM
        try:
            start = time.time()
            pred_adj = golem_ev(data, lambda1=0.02, lambda2=5.0, 
                               num_iter=500, learning_rate=0.001,
                               w_threshold=0.3, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['golem'].append({**base_info, 'runtime': runtime, **metrics})
        except Exception as e:
            print(f"GOLEM error on {dataset_file}: {e}")
            results['golem'].append({**base_info, 'runtime': None, 'error': str(e)})
        
        # Run SPICED
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
                **base_info, 
                'runtime': timing['total'],
                'runtime_phase1': timing['phase1_skeleton'],
                'runtime_phase2': timing['phase2_constraints'],
                'runtime_phase3': timing['phase3_optimization'],
                **metrics
            })
        except Exception as e:
            print(f"SPICED error on {dataset_file}: {e}")
            results['spiced'].append({**base_info, 'runtime': None, 'error': str(e)})
    
    # Save all results
    os.makedirs("results/synthetic", exist_ok=True)
    
    for method, data in results.items():
        np.savez(f"results/synthetic/{method}_results.npz", results=data)
        with open(f"results/synthetic/{method}_summary.json", 'w') as f:
            json.dump(data, f, indent=2, default=float)
        print(f"Saved {method}: {len(data)} results")
    
    return results


def run_real_world_fast():
    """Run all methods on Sachs dataset."""
    
    # Load Sachs dataset
    sachs_data = np.load("data/processed/real_world/sachs.npz")
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    
    results = {
        'notears': [],
        'golem': [],
        'spiced': []
    }
    
    for seed in range(1, 4):
        # NOTEARS
        try:
            start = time.time()
            pred_adj = notears_linear(data, lambda1=0.1, max_iter=50, 
                                     w_threshold=0.3, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['notears'].append({'dataset': 'sachs', 'seed': seed, 'runtime': runtime, **metrics})
            print(f"NOTEARS Sachs seed {seed}: SHD={metrics['shd']}")
        except Exception as e:
            print(f"NOTEARS Sachs error: {e}")
        
        # GOLEM
        try:
            start = time.time()
            pred_adj = golem_ev(data, lambda1=0.02, lambda2=5.0, 
                               num_iter=500, learning_rate=0.001,
                               w_threshold=0.3, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['golem'].append({'dataset': 'sachs', 'seed': seed, 'runtime': runtime, **metrics})
            print(f"GOLEM Sachs seed {seed}: SHD={metrics['shd']}")
        except Exception as e:
            print(f"GOLEM Sachs error: {e}")
        
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
            print(f"SPICED Sachs seed {seed}: SHD={metrics['shd']}")
        except Exception as e:
            print(f"SPICED Sachs error: {e}")
    
    # Save results
    os.makedirs("results/real_world", exist_ok=True)
    for method, data in results.items():
        with open(f"results/real_world/{method}_sachs.json", 'w') as f:
            json.dump(data, f, indent=2, default=float)
    
    return results


if __name__ == "__main__":
    print("Running fast experiments on subset of datasets...")
    print("=" * 60)
    
    run_all_methods_fast()
    
    print("\n" + "=" * 60)
    print("Running on Sachs dataset...")
    print("=" * 60)
    
    run_real_world_fast()
    
    print("\nAll experiments complete!")
