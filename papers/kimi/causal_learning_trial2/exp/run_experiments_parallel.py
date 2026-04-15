"""
Parallel experiment runner for SPICED experiments.
Runs experiments in parallel to maximize CPU utilization.
"""
import sys
import os
import json
import time
import glob
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

from spiced.spiced_main import spiced
from notears.run_fixed import notears_linear_fixed
from shared.metrics import compute_all_metrics


def run_single_experiment(args):
    """Run a single experiment (for parallel execution)."""
    dataset_file, method_name, method_func, kwargs = args
    
    try:
        data_dict = np.load(dataset_file)
        data = data_dict['data']
        true_adj = data_dict['adj']
        
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        n_nodes = true_adj.shape[0]
        
        start_time = time.time()
        
        if method_name == 'spiced':
            pred_adj, timing, _ = method_func(data, **kwargs, seed=seed)
            runtime = time.time() - start_time
            result = {
                'method': method_name,
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'runtime': runtime,
                'runtime_phase1': timing.get('phase1_skeleton', 0),
                'runtime_phase2': timing.get('phase2_constraints', 0),
                'runtime_phase3': timing.get('phase3_optimization', 0),
                **compute_all_metrics(true_adj, pred_adj),
                'status': 'success'
            }
        else:
            pred_adj, _ = method_func(data, **kwargs, seed=seed)
            runtime = time.time() - start_time
            result = {
                'method': method_name,
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'runtime': runtime,
                **compute_all_metrics(true_adj, pred_adj),
                'status': 'success'
            }
        
        return result
    except Exception as e:
        return {
            'method': method_name,
            'graph_id': int(data_dict['graph_id']),
            'mechanism': str(data_dict['mechanism']),
            'n_samples': int(data_dict['n_samples']),
            'seed': int(data_dict['seed']),
            'n_nodes': true_adj.shape[0],
            'runtime': None,
            'error': str(e),
            'status': 'failed'
        }


def run_method_parallel(method_name, method_func, exp_name, filter_func=None, **kwargs):
    """Run a method on all matching datasets in parallel."""
    print(f"\n{'='*60}")
    print(f"Running {method_name} on {exp_name}")
    print(f"{'='*60}")
    
    # Get all dataset files
    all_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    # Filter files
    dataset_files = []
    for f in all_files:
        if filter_func:
            data_dict = np.load(f)
            if filter_func(data_dict):
                dataset_files.append(f)
        else:
            dataset_files.append(f)
    
    print(f"Processing {len(dataset_files)} datasets...")
    
    # Prepare arguments for parallel execution
    args_list = [(f, method_name, method_func, kwargs) for f in dataset_files]
    
    # Run in parallel
    n_workers = min(2, cpu_count())  # Use up to 2 cores
    with Pool(n_workers) as pool:
        results = pool.map(run_single_experiment, args_list)
    
    # Save results
    result_dir = os.path.join(PROJECT_ROOT, 'results', 'synthetic', exp_name)
    os.makedirs(result_dir, exist_ok=True)
    
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Print summary
    successful = [r for r in results if r.get('status') == 'success']
    print(f"Completed: {len(successful)}/{len(results)} successful")
    
    return results


def main():
    """Run all experiments."""
    print("="*60)
    print("SPICED Experiments - Parallel Execution")
    print(f"CPU cores available: {cpu_count()}")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Main comparison: SPICED vs NOTEARS on n=10,20,30
    print("\n[1/5] Running main comparison experiments...")
    
    # SPICED on n=10,20,30
    run_method_parallel(
        'spiced', spiced, 'spiced_main',
        filter_func=lambda d: int(d['graph_id']) <= 60,
        k_neighbors=5, alpha=0.05, lambda1=0.1, lambda3=0.01,
        max_iter=100, w_threshold=0.3
    )
    
    # NOTEARS (fixed) on n=10,20,30
    run_method_parallel(
        'notears_fixed', notears_linear_fixed, 'notears_fixed',
        filter_func=lambda d: int(d['graph_id']) <= 60,
        lambda1=0.1, max_iter=100, w_threshold=0.3
    )
    
    # 2. Scalability: n=50 experiments
    print("\n[2/5] Running n=50 scalability experiments...")
    
    run_method_parallel(
        'spiced', spiced, 'spiced_n50',
        filter_func=lambda d: int(d['graph_id']) >= 101,
        k_neighbors=5, alpha=0.05, lambda1=0.1, lambda3=0.01,
        max_iter=100, w_threshold=0.3
    )
    
    run_method_parallel(
        'notears_fixed', notears_linear_fixed, 'notears_n50',
        filter_func=lambda d: int(d['graph_id']) >= 101,
        lambda1=0.1, max_iter=100, w_threshold=0.3
    )
    
    # 3. Ablation: SPICED without structural constraints
    print("\n[3/5] Running ablation experiments...")
    
    run_method_parallel(
        'spiced_no_constraints', spiced, 'spiced_no_constraints',
        filter_func=lambda d: int(d['graph_id']) <= 24,  # Smaller subset
        k_neighbors=5, alpha=0.05, lambda1=0.1, lambda3=0.0,
        max_iter=100, w_threshold=0.3, use_structural_constraints=False
    )
    
    # 4. Ablation: SPICED without IT initialization
    run_method_parallel(
        'spiced_random_init', spiced, 'spiced_random_init',
        filter_func=lambda d: int(d['graph_id']) <= 24,
        k_neighbors=5, alpha=0.05, lambda1=0.1, lambda3=0.01,
        max_iter=100, w_threshold=0.3, use_it_init=False
    )
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All experiments completed in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
