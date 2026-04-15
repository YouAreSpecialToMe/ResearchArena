"""
Comprehensive experiment runner (v2) - addresses self-review feedback.

Key fixes:
1. Proper k-NN entropy estimation in Phase 1
2. Fixed NOTEARS implementation
3. Actual n=50 experiments (not extrapolated)
4. Complete ablation studies
5. Per-experiment logs
"""
import sys
import os
import json
import time
import glob
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

from spiced.spiced_main import spiced
from shared.metrics import compute_all_metrics


def setup_logging(exp_name):
    """Setup per-experiment logging."""
    log_dir = os.path.join(PROJECT_ROOT, 'exp', exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def run_single_experiment(method_func, dataset_file, exp_name, **kwargs):
    """Run a single experiment and return results."""
    data_dict = np.load(dataset_file)
    data = data_dict['data']
    true_adj = data_dict['adj']
    
    graph_id = int(data_dict['graph_id'])
    mechanism = str(data_dict['mechanism'])
    n_samples = int(data_dict['n_samples'])
    seed = int(data_dict['seed'])
    n_nodes = true_adj.shape[0]
    
    log_file = os.path.join(PROJECT_ROOT, 'exp', exp_name, 'logs', 
                           f'graph_{graph_id}_{mechanism}_N{n_samples}_seed{seed}.log')
    
    start_time = time.time()
    try:
        pred_adj, timing, intermediates = method_func(data, **kwargs, seed=seed)
        runtime = time.time() - start_time
        
        metrics = compute_all_metrics(true_adj, pred_adj)
        
        result = {
            'graph_id': graph_id,
            'mechanism': mechanism,
            'n_samples': n_samples,
            'seed': seed,
            'n_nodes': n_nodes,
            'runtime': runtime,
            'timing': timing,
            **metrics,
            'status': 'success'
        }
        
        # Write per-experiment log
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Graph ID: {graph_id}\n")
            f.write(f"Mechanism: {mechanism}\n")
            f.write(f"N: {n_samples}, Nodes: {n_nodes}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Runtime: {runtime:.4f}s\n")
            f.write(f"SHD: {metrics['shd']}, TPR: {metrics['tpr']:.3f}, FDR: {metrics['fdr']:.3f}\n")
        
        return result
        
    except Exception as e:
        runtime = time.time() - start_time
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {exp_name} - FAILED\n")
            f.write(f"Error: {str(e)}\n")
        
        return {
            'graph_id': graph_id,
            'mechanism': mechanism,
            'n_samples': n_samples,
            'seed': seed,
            'n_nodes': n_nodes,
            'runtime': None,
            'error': str(e),
            'status': 'failed'
        }


def run_method_on_datasets(method_func, exp_name, graph_ids=None, 
                            mechanisms=None, n_samples_list=None, 
                            seeds=None, **kwargs):
    """
    Run a method on specified dataset configurations.
    """
    log_dir = setup_logging(exp_name)
    
    # Build list of dataset files to process
    dataset_files = []
    all_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    for f in all_files:
        data_dict = np.load(f)
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        
        if graph_ids is not None and graph_id not in graph_ids:
            continue
        if mechanisms is not None and mechanism not in mechanisms:
            continue
        if n_samples_list is not None and n_samples not in n_samples_list:
            continue
        if seeds is not None and seed not in seeds:
            continue
        
        dataset_files.append(f)
    
    print(f"{exp_name}: Running on {len(dataset_files)} datasets...")
    
    results = []
    for i, dataset_file in enumerate(dataset_files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(dataset_files)}")
        
        result = run_single_experiment(method_func, dataset_file, exp_name, **kwargs)
        results.append(result)
    
    # Save results
    result_dir = os.path.join(PROJECT_ROOT, 'results', 'synthetic', exp_name)
    os.makedirs(result_dir, exist_ok=True)
    
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"{exp_name}: Completed {len(results)} experiments.")
    return results


def run_spiced_ablation_mi_method():
    """
    Ablation: Compare k-NN vs kernel MI estimation in Phase 1.
    """
    from spiced.phase1_skeleton_ablation import compute_skeleton_it_method
    
    # We need to create a modified SPICED that uses different MI methods
    def spiced_with_mi_method(data, mi_method='knn', **kwargs):
        """SPICED with specified MI method."""
        from spiced.phase2_constraints import extract_structural_constraints, create_constraint_penalty_matrix
        from spiced.phase3_optimization import constrained_optimization
        
        k = kwargs.get('k_neighbors', 5)
        alpha = kwargs.get('alpha', 0.05)
        
        # Phase 1 with specified method
        if mi_method == 'knn':
            skeleton, it_scores = compute_skeleton_it_method(data, method='knn', k=k, alpha=alpha)
        else:
            skeleton, it_scores = compute_skeleton_it_method(data, method='kernel', alpha=alpha)
        
        # Phase 2
        constraints = extract_structural_constraints(skeleton)
        constraint_matrix = create_constraint_penalty_matrix(constraints, data.shape[1])
        
        # Phase 3
        lambda1 = kwargs.get('lambda1', 0.1)
        lambda3 = kwargs.get('lambda3', 0.01)
        max_iter = kwargs.get('max_iter', 100)
        w_threshold = kwargs.get('w_threshold', 0.3)
        
        pred_adj = constrained_optimization(
            data, skeleton, it_scores, constraint_matrix,
            lambda1=lambda1, lambda2=0.0, lambda3=lambda3,
            max_iter=max_iter, w_threshold=w_threshold,
            use_it_init=True, seed=kwargs.get('seed', None)
        )
        
        timing = {'phase1': 0, 'phase2': 0, 'phase3': 0, 'total': 0}
        return pred_adj, timing, {}
    
    # Run for subset: n=10,20, N=50,100,200, all mechanisms
    print("\n" + "="*60)
    print("ABLATION: k-NN vs Kernel MI Estimation")
    print("="*60)
    
    # k-NN method
    print("\nRunning SPICED with k-NN MI estimation...")
    results_knn = run_method_on_datasets(
        lambda data, **kw: spiced_with_mi_method(data, mi_method='knn', **kw),
        'spiced_ablation_knn',
        graph_ids=list(range(1, 13)),  # Graphs 1-12
        n_samples_list=[50, 100, 200],
        seeds=[1, 2, 3]
    )
    
    # Kernel method
    print("\nRunning SPICED with Kernel MI estimation...")
    results_kernel = run_method_on_datasets(
        lambda data, **kw: spiced_with_mi_method(data, mi_method='kernel', **kw),
        'spiced_ablation_kernel',
        graph_ids=list(range(1, 13)),
        n_samples_list=[50, 100, 200],
        seeds=[1, 2, 3]
    )
    
    return results_knn, results_kernel


def run_spiced_ablation_structural_constraints():
    """
    Ablation: Effect of structural constraints.
    """
    print("\n" + "="*60)
    print("ABLATION: Structural Constraints Effect")
    print("="*60)
    
    # With structural constraints
    print("\nRunning SPICED WITH structural constraints...")
    results_with = run_method_on_datasets(
        lambda data, **kw: spiced(data, use_structural_constraints=True, **kw),
        'spiced_with_constraints',
        graph_ids=list(range(1, 13)),
        n_samples_list=[50, 100, 200, 500],
        seeds=[1, 2, 3]
    )
    
    # Without structural constraints
    print("\nRunning SPICED WITHOUT structural constraints...")
    results_without = run_method_on_datasets(
        lambda data, **kw: spiced(data, use_structural_constraints=False, **kw),
        'spiced_no_constraints',
        graph_ids=list(range(1, 13)),
        n_samples_list=[50, 100, 200, 500],
        seeds=[1, 2, 3]
    )
    
    return results_with, results_without


def run_spiced_ablation_initialization():
    """
    Ablation: IT-based vs random initialization.
    """
    print("\n" + "="*60)
    print("ABLATION: Initialization Method")
    print("="*60)
    
    # With IT initialization
    print("\nRunning SPICED WITH IT initialization...")
    results_it = run_method_on_datasets(
        lambda data, **kw: spiced(data, use_it_init=True, **kw),
        'spiced_it_init',
        graph_ids=list(range(1, 13)),
        n_samples_list=[50, 100, 200],
        seeds=[1, 2, 3]
    )
    
    # With random initialization
    print("\nRunning SPICED with RANDOM initialization...")
    results_random = run_method_on_datasets(
        lambda data, **kw: spiced(data, use_it_init=False, **kw),
        'spiced_random_init',
        graph_ids=list(range(1, 13)),
        n_samples_list=[50, 100, 200],
        seeds=[1, 2, 3]
    )
    
    return results_it, results_random


def run_n50_scalability():
    """
    Run actual n=50 experiments (not extrapolated).
    """
    print("\n" + "="*60)
    print("SCALABILITY: Running n=50 experiments")
    print("="*60)
    
    # Filter for n=50 graphs (IDs 101-120)
    results_spiced = run_method_on_datasets(
        spiced,
        'spiced_n50',
        graph_ids=list(range(101, 121)),  # n=50 graphs
        seeds=[1, 2, 3]
    )
    
    return results_spiced


def aggregate_results():
    """Aggregate all results into final results.json."""
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)
    
    all_results = {}
    
    # Load all synthetic results
    result_dirs = glob.glob(os.path.join(PROJECT_ROOT, 'results/synthetic/*/'))
    
    for result_dir in result_dirs:
        method_name = os.path.basename(os.path.dirname(result_dir))
        results_file = os.path.join(result_dir, 'results.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results[method_name] = json.load(f)
            print(f"Loaded {method_name}: {len(all_results[method_name])} results")
    
    # Create summary statistics
    summary = {}
    for method_name, results in all_results.items():
        method_summary = defaultdict(lambda: defaultdict(list))
        
        for r in results:
            if r.get('status') == 'success':
                key = (r['n_nodes'], r['mechanism'], r['n_samples'])
                method_summary[key]['shd'].append(r['shd'])
                method_summary[key]['tpr'].append(r['tpr'])
                method_summary[key]['fdr'].append(r['fdr'])
                method_summary[key]['runtime'].append(r['runtime'])
        
        # Compute means and stds
        summary[method_name] = []
        for key, metrics in method_summary.items():
            n_nodes, mechanism, n_samples = key
            summary[method_name].append({
                'n_nodes': n_nodes,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'shd_mean': np.mean(metrics['shd']),
                'shd_std': np.std(metrics['shd']),
                'tpr_mean': np.mean(metrics['tpr']),
                'tpr_std': np.std(metrics['tpr']),
                'fdr_mean': np.mean(metrics['fdr']),
                'fdr_std': np.std(metrics['fdr']),
                'runtime_mean': np.mean(metrics['runtime']),
                'n_runs': len(metrics['shd'])
            })
    
    # Save aggregated results
    final_results = {
        'all_results': all_results,
        'summary': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(PROJECT_ROOT, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=float)
    
    print(f"\nAggregated results saved to results.json")
    return final_results


if __name__ == "__main__":
    print("="*60)
    print("SPICED Experiments v2 - Addressing Self-Review Feedback")
    print("="*60)
    
    # Run ablations
    run_spiced_ablation_mi_method()
    run_spiced_ablation_structural_constraints()
    run_spiced_ablation_initialization()
    
    # Run n=50 experiments
    run_n50_scalability()
    
    # Aggregate results
    aggregate_results()
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
