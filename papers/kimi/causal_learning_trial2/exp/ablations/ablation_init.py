"""
Ablation study: Information-Theoretic Initialization vs Random Initialization
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


def run_ablation_initialization():
    """Compare IT initialization vs random initialization."""
    results = []
    
    dataset_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    # Filter for n_nodes=10,20 and N=50,100,200 for efficiency
    filtered_files = []
    for f in dataset_files:
        data_dict = np.load(f)
        n_samples = int(data_dict['n_samples'])
        adj = data_dict['adj']
        n_nodes = adj.shape[0]
        
        if n_nodes <= 20 and n_samples <= 200:
            filtered_files.append(f)
    
    print(f"Running initialization ablation on {len(filtered_files)} datasets...")
    
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
        
        # Run with IT initialization
        try:
            pred_adj_it, timing_it, _ = spiced(
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
            
            metrics_it = compute_all_metrics(true_adj, pred_adj_it)
            
            result_it = {
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'init_method': 'IT',
                'runtime': timing_it['total'],
                **metrics_it
            }
            results.append(result_it)
            
        except Exception as e:
            print(f"Error (IT init) on {dataset_file}: {e}")
        
        # Run with random initialization
        try:
            pred_adj_rand, timing_rand, _ = spiced(
                data,
                k_neighbors=5,
                alpha=0.05,
                lambda1=0.1,
                lambda3=0.01,
                max_iter=100,
                w_threshold=0.3,
                use_it_init=False,
                use_structural_constraints=True,
                seed=seed
            )
            
            metrics_rand = compute_all_metrics(true_adj, pred_adj_rand)
            
            result_rand = {
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'init_method': 'random',
                'runtime': timing_rand['total'],
                **metrics_rand
            }
            results.append(result_rand)
            
        except Exception as e:
            print(f"Error (random init) on {dataset_file}: {e}")
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results/ablations"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results/ablations/initialization.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"Initialization ablation complete. Processed {len(filtered_files)} datasets x 2 methods.")
    return results


if __name__ == "__main__":
    print("Running ablation study: IT initialization vs random initialization...")
    run_ablation_initialization()
