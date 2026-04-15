"""
Run n=50 scalability experiments.
Addresses the self-review feedback about missing n=50 experiments.
"""
import sys
import os
import json
import time
import glob
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

from spiced.spiced_main import spiced
from notears.run_fixed import notears_linear_fixed
from shared.metrics import compute_all_metrics


def run_n50_experiments():
    """Run experiments on n=50 graphs."""
    print("="*60)
    print("n=50 Scalability Experiments")
    print("="*60)
    
    # Get all n=50 dataset files
    dataset_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/graph_01*.npz"))
    
    # Filter for a subset to save time
    # We'll run on graphs 101-120, all mechanisms, N=500, seeds 1-3
    selected_files = []
    for f in dataset_files:
        data_dict = np.load(f)
        graph_id = int(data_dict['graph_id'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        
        if 101 <= graph_id <= 120 and n_samples == 500 and seed <= 3:
            selected_files.append(f)
    
    print(f"Selected {len(selected_files)} n=50 datasets for testing")
    
    results = {
        'spiced_n50': [],
        'notears_n50': []
    }
    
    # Run SPICED
    print("\n[1/2] Running SPICED on n=50 graphs...")
    for i, dataset_file in enumerate(selected_files):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(selected_files)}")
        
        data_dict = np.load(dataset_file)
        data = data_dict['data']
        true_adj = data_dict['adj']
        
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        n_nodes = true_adj.shape[0]
        
        try:
            start = time.time()
            pred_adj, timing, _ = spiced(
                data, k_neighbors=5, alpha=0.05, lambda1=0.1, lambda3=0.01,
                max_iter=100, w_threshold=0.3, seed=seed
            )
            runtime = time.time() - start
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            results['spiced_n50'].append({
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'runtime': runtime,
                'runtime_phase1': timing.get('phase1_skeleton', 0),
                'runtime_phase2': timing.get('phase2_constraints', 0),
                'runtime_phase3': timing.get('phase3_optimization', 0),
                **metrics,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  Error on graph {graph_id}: {e}")
            results['spiced_n50'].append({
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'runtime': None,
                'error': str(e),
                'status': 'failed'
            })
    
    # Run NOTEARS
    print("\n[2/2] Running NOTEARS on n=50 graphs...")
    for i, dataset_file in enumerate(selected_files):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(selected_files)}")
        
        data_dict = np.load(dataset_file)
        data = data_dict['data']
        true_adj = data_dict['adj']
        
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        n_nodes = true_adj.shape[0]
        
        try:
            start = time.time()
            pred_adj, _ = notears_linear_fixed(
                data, lambda1=0.1, max_iter=100, w_threshold=0.3, seed=seed
            )
            runtime = time.time() - start
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            results['notears_n50'].append({
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'runtime': runtime,
                **metrics,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  Error on graph {graph_id}: {e}")
            results['notears_n50'].append({
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'runtime': None,
                'error': str(e),
                'status': 'failed'
            })
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    for method_name, method_results in results.items():
        result_dir = os.path.join(PROJECT_ROOT, 'results', 'synthetic', method_name)
        os.makedirs(result_dir, exist_ok=True)
        
        with open(os.path.join(result_dir, 'results.json'), 'w') as f:
            json.dump(method_results, f, indent=2, default=float)
        
        # Print summary
        successful = [r for r in method_results if r.get('status') == 'success']
        if successful:
            runtimes = [r['runtime'] for r in successful if r.get('runtime')]
            shds = [r['shd'] for r in successful]
            
            print(f"\n{method_name}:")
            print(f"  Completed: {len(successful)}/{len(method_results)}")
            if runtimes:
                print(f"  Runtime: {np.mean(runtimes):.2f}s ± {np.std(runtimes):.2f}s")
                print(f"  Median runtime: {np.median(runtimes):.2f}s")
            if shds:
                print(f"  SHD: {np.mean(shds):.2f} ± {np.std(shds):.2f}")
            
            # Check success criterion
            if 'spiced' in method_name and runtimes:
                if np.median(runtimes) < 300:
                    print(f"  Scalability: PASS (< 5 min)")
                else:
                    print(f"  Scalability: FAIL (>= 5 min)")
    
    return results


if __name__ == "__main__":
    start = time.time()
    results = run_n50_experiments()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
