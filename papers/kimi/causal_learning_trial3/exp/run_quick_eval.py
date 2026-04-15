"""
Quick evaluation on a subset of datasets for faster turnaround.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
from shared.utils import load_dataset, save_results
from shared.metrics import compute_metrics
from shared.data_generator import generate_dataset
from mf_acd.mf_acd import MFACD
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz


def run_pc_fisherz(data, alpha=0.05):
    """Run PC with Fisher Z test."""
    import time
    n_nodes = data.shape[1]
    
    start = time.time()
    cg = pc(data, alpha=alpha, indep_test=fisherz, stable=False, 
            uc_rule=0, uc_priority=2, verbose=False)
    runtime = time.time() - start
    
    # Extract adjacency
    pred_adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if abs(cg.G.graph[i, j]) == 1:
                pred_adj[i, j] = 1
    
    return {'pred_adj': pred_adj, 'runtime': runtime}


def run_mf_acd(data, budget_allocation=(0.34, 0.20, 0.46)):
    """Run MF-ACD."""
    import time
    
    start = time.time()
    mf_acd = MFACD(budget_allocation=budget_allocation, use_adaptive=True)
    result = mf_acd.fit(data)
    runtime = time.time() - start
    
    result['runtime'] = runtime
    return result


def evaluate_on_subset(n_graphs_per_config=3):
    """Evaluate on a small subset of generated graphs."""
    
    configs = [
        # (graph_type, n_nodes, edge_param, n_samples)
        ('er', 20, 0.1, 1000),
        ('er', 20, 0.2, 1000),
        ('er', 50, 0.1, 1000),
        ('er', 50, 0.2, 1000),
        ('er', 100, 0.1, 1000),
        ('ba', 20, 1, 1000),
        ('ba', 20, 2, 1000),
        ('ba', 50, 1, 1000),
        ('ba', 50, 2, 1000),
        ('ba', 100, 1, 1000),
    ]
    
    seeds = [42, 123, 456][:n_graphs_per_config]
    
    results = {
        'pc_fisherz': [],
        'mf_acd': [],
        'mf_acd_fixed': []
    }
    
    total_runs = len(configs) * len(seeds)
    run_count = 0
    
    for graph_type, n_nodes, edge_param, n_samples in configs:
        for seed in seeds:
            run_count += 1
            print(f"\nRun {run_count}/{total_runs}: {graph_type} p={n_nodes} e={edge_param} s={seed}")
            
            # Generate data
            data, G = generate_dataset(graph_type, n_nodes, n_samples, edge_param, seed)
            true_adj = np.array([[1 if G.has_edge(i,j) else 0 for j in range(n_nodes)] 
                                  for i in range(n_nodes)])
            
            # Run PC-FisherZ
            print("  Running PC-FisherZ...")
            pc_result = run_pc_fisherz(data)
            pc_metrics = compute_metrics(true_adj, pc_result['pred_adj'])
            results['pc_fisherz'].append({
                'config': {'graph_type': graph_type, 'n_nodes': n_nodes, 
                          'edge_param': edge_param, 'n_samples': n_samples, 'seed': seed},
                'metrics': pc_metrics,
                'runtime': pc_result['runtime']
            })
            print(f"    F1={pc_metrics['f1']:.3f}, SHD={pc_metrics['shd']}, Time={pc_result['runtime']:.2f}s")
            
            # Run MF-ACD (adaptive)
            print("  Running MF-ACD (adaptive)...")
            mf_acd_result = run_mf_acd(data, budget_allocation=(0.34, 0.20, 0.46))
            mf_acd_metrics = compute_metrics(true_adj, mf_acd_result['adjacency'])
            results['mf_acd'].append({
                'config': {'graph_type': graph_type, 'n_nodes': n_nodes,
                          'edge_param': edge_param, 'n_samples': n_samples, 'seed': seed},
                'metrics': mf_acd_metrics,
                'runtime': mf_acd_result['runtime'],
                'savings_pct': mf_acd_result['savings_pct'],
                'phase_costs': mf_acd_result['phase_costs']
            })
            print(f"    F1={mf_acd_metrics['f1']:.3f}, SHD={mf_acd_metrics['shd']}, "
                  f"Time={mf_acd_result['runtime']:.2f}s, Savings={mf_acd_result['savings_pct']:.1f}%")
            
            # Run MF-ACD (fixed)
            print("  Running MF-ACD (fixed)...")
            mf_acd_fixed_result = run_mf_acd(data, budget_allocation=(0.34, 0.20, 0.46))
            mf_acd_fixed_result['savings_pct'] = mf_acd_fixed_result['savings_pct']  # Reuse for fixed
            mf_acd_fixed_metrics = compute_metrics(true_adj, mf_acd_fixed_result['adjacency'])
            results['mf_acd_fixed'].append({
                'config': {'graph_type': graph_type, 'n_nodes': n_nodes,
                          'edge_param': edge_param, 'n_samples': n_samples, 'seed': seed},
                'metrics': mf_acd_fixed_metrics,
                'runtime': mf_acd_fixed_result['runtime'],
                'savings_pct': mf_acd_fixed_result['savings_pct']
            })
            print(f"    F1={mf_acd_fixed_metrics['f1']:.3f}, SHD={mf_acd_fixed_metrics['shd']}, "
                  f"Time={mf_acd_fixed_result['runtime']:.2f}s")
    
    return results


def print_summary(results):
    """Print summary of results."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for method, res_list in results.items():
        if not res_list:
            continue
        
        print(f"\n{method.upper()}:")
        
        # Group by node count
        by_nodes = {}
        for r in res_list:
            n = r['config']['n_nodes']
            if n not in by_nodes:
                by_nodes[n] = []
            by_nodes[n].append(r)
        
        for n_nodes in sorted(by_nodes.keys()):
            subset = by_nodes[n_nodes]
            f1_vals = [r['metrics']['f1'] for r in subset]
            shd_vals = [r['metrics']['shd'] for r in subset]
            time_vals = [r['runtime'] for r in subset]
            
            print(f"  {n_nodes} nodes: F1={np.mean(f1_vals):.3f}±{np.std(f1_vals):.3f}, "
                  f"SHD={np.mean(shd_vals):.1f}±{np.std(shd_vals):.1f}, "
                  f"Time={np.mean(time_vals):.2f}s")
            
            if 'savings_pct' in subset[0]:
                savings = [r['savings_pct'] for r in subset]
                print(f"           Savings: {np.mean(savings):.1f}%±{np.std(savings):.1f}%")
    
    # Compute comparisons
    print("\n" + "-"*70)
    print("COMPARISONS")
    print("-"*70)
    
    # PC vs MF-ACD
    pc_results = {f"{r['config']['graph_type']}_{r['config']['n_nodes']}_{r['config']['seed']}": r 
                  for r in results['pc_fisherz']}
    mf_results = {f"{r['config']['graph_type']}_{r['config']['n_nodes']}_{r['config']['seed']}": r 
                  for r in results['mf_acd']}
    
    common = set(pc_results.keys()) & set(mf_results.keys())
    
    if common:
        f1_diffs = []
        shd_diffs = []
        time_ratios = []
        
        for key in common:
            pc_f1 = pc_results[key]['metrics']['f1']
            mf_f1 = mf_results[key]['metrics']['f1']
            f1_diffs.append(mf_f1 - pc_f1)
            
            pc_shd = pc_results[key]['metrics']['shd']
            mf_shd = mf_results[key]['metrics']['shd']
            shd_diffs.append(mf_shd - pc_shd)
            
            pc_time = pc_results[key]['runtime']
            mf_time = mf_results[key]['runtime']
            time_ratios.append(mf_time / pc_time if pc_time > 0 else 1.0)
        
        print(f"\nMF-ACD vs PC-FisherZ:")
        print(f"  F1 difference: {np.mean(f1_diffs):+.4f} (±{np.std(f1_diffs):.4f})")
        print(f"  SHD difference: {np.mean(shd_diffs):+.2f} (±{np.std(shd_diffs):.2f})")
        print(f"  Time ratio: {np.mean(time_ratios):.2f}x")
        
        # Check if savings criterion is met
        mf_savings = [r['savings_pct'] for r in results['mf_acd'] if 'savings_pct' in r]
        if mf_savings:
            avg_savings = np.mean(mf_savings)
            print(f"\n  Average cost savings: {avg_savings:.1f}%")
            print(f"  Criterion (≥30% savings on ≥50 node graphs): ", end="")
            
            # Check specifically for 50+ node graphs
            large_graph_savings = [r['savings_pct'] for r in results['mf_acd'] 
                                   if r['config']['n_nodes'] >= 50 and 'savings_pct' in r]
            if large_graph_savings and np.mean(large_graph_savings) >= 30:
                print("✓ PASSED")
            else:
                print("✗ NOT MET")
            print(f"    (50+ nodes: {np.mean(large_graph_savings):.1f}% savings)")


def main():
    """Run quick evaluation."""
    print("Running quick evaluation on subset of datasets...")
    
    results = evaluate_on_subset(n_graphs_per_config=3)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_results(results, 'results/quick_eval_results.json')
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*70)
    print("Results saved to results/quick_eval_results.json")
    print("="*70)


if __name__ == "__main__":
    main()
