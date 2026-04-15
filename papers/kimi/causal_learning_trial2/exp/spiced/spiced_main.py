"""
SPICED: Sample-Efficient Prior-Informed Causal Estimation via Directed Information
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

import numpy as np
import time
import json
import glob

from phase1_skeleton_fixed import compute_skeleton_it, compute_directed_information_scores
from phase2_constraints import extract_structural_constraints, create_constraint_penalty_matrix
from phase3_optimization_fixed import constrained_optimization
from shared.metrics import compute_all_metrics


def spiced(data, k_neighbors=5, alpha=0.05, lambda1=0.1, lambda2=0.0, lambda3=0.01,
           max_iter=100, w_threshold=0.3, use_it_init=True, 
           use_structural_constraints=True, seed=None):
    """SPICED algorithm for causal discovery."""
    timing = {}
    
    # Phase 1: Information-Theoretic Skeleton Discovery
    start = time.time()
    skeleton = compute_skeleton_it(data, k=k_neighbors, alpha=alpha)
    it_scores = compute_directed_information_scores(data, skeleton, k=k_neighbors)
    timing['phase1_skeleton'] = time.time() - start
    
    # Phase 2: Structural Constraint Extraction
    start = time.time()
    if use_structural_constraints:
        constraints = extract_structural_constraints(skeleton)
        constraint_matrix = create_constraint_penalty_matrix(constraints, data.shape[1])
    else:
        constraints = None
        constraint_matrix = np.zeros((data.shape[1], data.shape[1]))
        lambda3 = 0.0
    timing['phase2_constraints'] = time.time() - start
    
    # Phase 3: Constrained Optimization
    start = time.time()
    pred_adj = constrained_optimization(
        data, skeleton, it_scores, constraint_matrix,
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
        max_iter=max_iter, w_threshold=w_threshold,
        use_it_init=use_it_init, seed=seed
    )
    timing['phase3_optimization'] = time.time() - start
    
    timing['total'] = timing['phase1_skeleton'] + timing['phase2_constraints'] + timing['phase3_optimization']
    
    intermediates = {
        'skeleton': skeleton,
        'it_scores': it_scores,
        'constraints': constraints,
        'constraint_matrix': constraint_matrix
    }
    
    return pred_adj, timing, intermediates


def run_spiced_baseline():
    """Run SPICED on all synthetic datasets."""
    results = []
    
    dataset_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    print(f"Running SPICED on {len(dataset_files)} datasets...")
    print("This may take a while...")
    
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
        
        max_iter = 100 if n_nodes <= 20 else 50
        
        try:
            pred_adj, timing, _ = spiced(
                data,
                k_neighbors=5,
                alpha=0.05,
                lambda1=0.1,
                lambda3=0.01,
                max_iter=max_iter,
                w_threshold=0.3,
                use_it_init=True,
                use_structural_constraints=True,
                seed=seed
            )
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            result = {
                'graph_id': int(graph_id),
                'mechanism': mechanism,
                'n_samples': int(n_samples),
                'seed': int(seed),
                'n_nodes': int(n_nodes),
                'runtime': timing['total'],
                'runtime_phase1': timing['phase1_skeleton'],
                'runtime_phase2': timing['phase2_constraints'],
                'runtime_phase3': timing['phase3_optimization'],
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error on graph {graph_id}: {e}")
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
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results/synthetic/spiced"), exist_ok=True)
    np.savez(os.path.join(PROJECT_ROOT, "results/synthetic/spiced_results.npz"), results=results)
    
    with open(os.path.join(PROJECT_ROOT, "results/synthetic/spiced_summary.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"SPICED complete. Processed {len(results)} datasets.")
    return results


def run_spiced_sachs():
    """Run SPICED on Sachs dataset."""
    sachs_file = os.path.join(PROJECT_ROOT, "data/processed/real_world/sachs.npz")
    sachs_data = np.load(sachs_file)
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    
    results = []
    for seed in range(1, 6):
        try:
            pred_adj, timing, _ = spiced(
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
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            result = {
                'dataset': 'sachs',
                'seed': seed,
                'runtime': timing['total'],
                'runtime_phase1': timing['phase1_skeleton'],
                'runtime_phase2': timing['phase2_constraints'],
                'runtime_phase3': timing['phase3_optimization'],
                **metrics
            }
            results.append(result)
            print(f"Sachs seed {seed}: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
            
        except Exception as e:
            print(f"Error on Sachs seed {seed}: {e}")
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results/real_world"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results/real_world/spiced_sachs.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


if __name__ == "__main__":
    print("Running SPICED on synthetic data...")
    run_spiced_baseline()
    print("\nRunning SPICED on Sachs dataset...")
    run_spiced_sachs()
    print("\nSPICED experiments complete!")
