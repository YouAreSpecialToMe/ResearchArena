"""
Ablation study: Effect of Structural Constraints
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp/spiced'))

import numpy as np
import json
import glob
from spiced_main import spiced
from shared.metrics import compute_all_metrics


def run_ablation_constraints():
    """Compare with/without structural constraints."""
    results = []
    
    dataset_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    # Filter for n_nodes <= 30 and N <= 500
    filtered_files = []
    for f in dataset_files:
        data_dict = np.load(f)
        n_samples = int(data_dict['n_samples'])
        adj = data_dict['adj']
        n_nodes = adj.shape[0]
        
        if n_nodes <= 30 and n_samples <= 500:
            filtered_files.append(f)
    
    print(f"Running structural constraints ablation on {len(filtered_files)} datasets...")
    
    for i, dataset_file in enumerate(filtered_files):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(filtered_files)}")
        
        data_dict = np.load(dataset_file)
        data = data_dict['data']
        true_adj = data_dict['adj']
        
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        n_nodes = true_adj.shape[0]
        
        # Run with structural constraints
        try:
            pred_adj_with, timing_with, _ = spiced(
                data,
                k_neighbors=5,
                alpha=0.05,
                lambda1=0.1,
                lambda3=0.01,
                max_iter=100,
                w_threshold=0.3,
                use_it_init=True,
                use_structural_constraints=True,
                seed=seed
            )
            
            metrics_with = compute_all_metrics(true_adj, pred_adj_with)
            
            result_with = {
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'use_constraints': True,
                'runtime': timing_with['total'],
                **metrics_with
            }
            results.append(result_with)
            
        except Exception as e:
            print(f"Error (with constraints) on {dataset_file}: {e}")
        
        # Run without structural constraints
        try:
            pred_adj_without, timing_without, _ = spiced(
                data,
                k_neighbors=5,
                alpha=0.05,
                lambda1=0.1,
                lambda3=0.0,
                max_iter=100,
                w_threshold=0.3,
                use_it_init=True,
                use_structural_constraints=False,
                seed=seed
            )
            
            metrics_without = compute_all_metrics(true_adj, pred_adj_without)
            
            result_without = {
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'use_constraints': False,
                'runtime': timing_without['total'],
                **metrics_without
            }
            results.append(result_without)
            
        except Exception as e:
            print(f"Error (without constraints) on {dataset_file}: {e}")
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results/ablations"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results/ablations/structural_constraints.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"Structural constraints ablation complete.")
    return results


if __name__ == "__main__":
    print("Running ablation study: Effect of structural constraints...")
    run_ablation_constraints()
