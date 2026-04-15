"""
PC algorithm baseline for causal discovery.
Uses causal-learn library.
"""
import sys
import os

# Get project root (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

import numpy as np
import json
import time
import glob
from causallearn.search.ConstraintBased.PC import pc
from shared.metrics import compute_all_metrics


def run_pc_on_data(data, alpha=0.05, indep_test='fisherz'):
    """Run PC algorithm on data."""
    cg = pc(data, alpha=alpha, indep_test=indep_test, stable=True, show_progress=False)
    pred_adj = cg.G.graph
    
    # Convert to binary adjacency (0/1)
    binary_adj = np.zeros_like(pred_adj)
    binary_adj[pred_adj == 1] = 1  # Directed edge
    binary_adj[pred_adj == -1] = 1  # Undirected edge -> include both directions
    
    return binary_adj


def run_pc_baseline():
    """Run PC on all synthetic datasets."""
    results = []
    
    dataset_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    print(f"Running PC on {len(dataset_files)} datasets...")
    
    for i, dataset_file in enumerate(dataset_files):
        if i % 500 == 0:
            print(f"Progress: {i}/{len(dataset_files)}")
        
        data_dict = np.load(dataset_file)
        data = data_dict['data']
        true_adj = data_dict['adj']
        
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        n_nodes = true_adj.shape[0]
        
        start_time = time.time()
        try:
            pred_adj = run_pc_on_data(data, alpha=0.05, indep_test='fisherz')
            runtime = time.time() - start_time
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            result = {
                'graph_id': int(graph_id),
                'mechanism': mechanism,
                'n_samples': int(n_samples),
                'seed': int(seed),
                'n_nodes': int(n_nodes),
                'runtime': float(runtime),
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error on {dataset_file}: {e}")
            result = {
                'graph_id': int(graph_id),
                'mechanism': mechanism,
                'n_samples': int(n_samples),
                'seed': int(seed),
                'n_nodes': int(n_nodes),
                'runtime': None,
                'error': str(e)
            }
            results.append(result)
    
    results_dir = os.path.join(PROJECT_ROOT, "results/synthetic/pc")
    os.makedirs(results_dir, exist_ok=True)
    np.savez(os.path.join(results_dir, "pc_results.npz"), results=results)
    
    with open(os.path.join(PROJECT_ROOT, "results/synthetic/pc_summary.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"PC baseline complete. Processed {len(results)} datasets.")
    return results


def run_pc_sachs():
    """Run PC on Sachs dataset."""
    sachs_file = os.path.join(PROJECT_ROOT, "data/processed/real_world/sachs.npz")
    sachs_data = np.load(sachs_file)
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    
    results = []
    for seed in range(1, 6):  # 5 seeds
        np.random.seed(seed)
        start_time = time.time()
        
        try:
            pred_adj = run_pc_on_data(data, alpha=0.05, indep_test='fisherz')
            runtime = time.time() - start_time
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            result = {
                'dataset': 'sachs',
                'seed': seed,
                'runtime': runtime,
                **metrics
            }
            results.append(result)
            print(f"Sachs seed {seed}: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
            
        except Exception as e:
            print(f"Error on Sachs seed {seed}: {e}")
    
    results_dir = os.path.join(PROJECT_ROOT, "results/real_world")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "pc_sachs.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


if __name__ == "__main__":
    print("Running PC baseline on synthetic data...")
    run_pc_baseline()
    
    print("\nRunning PC on Sachs dataset...")
    run_pc_sachs()
    print("\nPC experiments complete!")
