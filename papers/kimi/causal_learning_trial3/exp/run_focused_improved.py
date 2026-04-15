"""
Focused evaluation of improved MF-ACD addressing feedback.

Key improvements tested:
1. Distance correlation in Phase 1 (nonlinear capability)
2. 50-node graph evaluations
3. Key ablations
"""
import numpy as np
import sys
import os
import json
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.mf_acd.mf_acd_improved import MFACDImproved
from exp.shared.data_loader import generate_synthetic_data
from exp.shared.metrics import compute_metrics


def run_baseline_pc_fisherz(data: np.ndarray, alpha: float = 0.05) -> Dict:
    """Run standard PC with Fisher Z test."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    
    start_time = time.time()
    cg = pc(data, alpha, fisherz, verbose=False)
    runtime = time.time() - start_time
    
    return {
        'adjacency': cg.G.graph,
        'runtime': runtime
    }


def run_baseline_pc_stable(data: np.ndarray, alpha: float = 0.05) -> Dict:
    """Run PC-Stable algorithm."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    
    start_time = time.time()
    cg = pc(data, alpha, fisherz, stable=True, verbose=False)
    runtime = time.time() - start_time
    
    return {
        'adjacency': cg.G.graph,
        'runtime': runtime
    }


def run_baseline_ges(data: np.ndarray) -> Dict:
    """Run GES algorithm."""
    from causallearn.search.ScoreBased.GES import ges
    
    start_time = time.time()
    record = ges(data, verbose=False)
    runtime = time.time() - start_time
    
    return {
        'adjacency': record['G'].graph,
        'runtime': runtime
    }


def evaluate_method(method_fn, data: np.ndarray, true_adj: np.ndarray, **kwargs) -> Dict:
    """Evaluate a method and return metrics."""
    try:
        result = method_fn(data, **kwargs)
        pred_adj = result['adjacency']
        runtime = result.get('runtime', 0)
        
        # Convert directed to undirected for skeleton comparison
        pred_skeleton = np.maximum(pred_adj, pred_adj.T)
        pred_skeleton = (pred_skeleton > 0).astype(int)
        np.fill_diagonal(pred_skeleton, 0)
        
        true_skeleton = np.maximum(true_adj, true_adj.T)
        true_skeleton = (true_skeleton > 0).astype(int)
        np.fill_diagonal(true_skeleton, 0)
        
        metrics = compute_metrics(pred_skeleton, true_skeleton)
        metrics['runtime'] = runtime
        metrics['status'] = 'success'
        
        return metrics
    except Exception as e:
        return {
            'f1': 0, 'precision': 0, 'recall': 0, 'shd': 9999,
            'runtime': 0, 'status': f'error: {str(e)}'
        }


def run_mf_acd_wrapper(data: np.ndarray, **kwargs) -> Dict:
    """Wrapper for improved MF-ACD."""
    mf_acd = MFACDImproved(**kwargs)
    
    start_time = time.time()
    result = mf_acd.fit(data)
    runtime = time.time() - start_time
    
    return {
        'adjacency': result['adjacency'],
        'runtime': runtime,
        'savings_pct': result['savings_pct'],
        'phase_costs': result['phase_costs'],
        'n_tests': result['n_tests']
    }


def main():
    print("="*70)
    print("FOCUSED EVALUATION - IMPROVED MF-ACD")
    print("="*70)
    
    results = {
        'main_experiments': [],
        'ablations': {},
        'summary': {}
    }
    
    # Experiment configurations - focus on 50-node graphs
    configs = []
    
    # 20-node graphs (3 densities × 5 seeds)
    for density in [0.1, 0.2, 0.3]:
        for seed in [42, 123, 456, 789, 1011]:
            configs.append({
                'p': 20, 'n': 1000, 'density': density,
                'graph_type': 'ER', 'seed': seed
            })
    
    # 50-node graphs (2 densities × 10 seeds = 20 configs)
    for density in [0.1, 0.2]:
        for seed in [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]:
            configs.append({
                'p': 50, 'n': 1000, 'density': density,
                'graph_type': 'ER', 'seed': seed
            })
    
    print(f"\nRunning {len(configs)} main experiments...")
    print(f"- 15 experiments on 20-node graphs")
    print(f"- 20 experiments on 50-node graphs")
    
    for idx, config in enumerate(configs):
        print(f"\n[{idx+1}/{len(configs)}] p={config['p']}, d={config['density']}, seed={config['seed']}")
        
        # Generate data
        data, true_adj = generate_synthetic_data(
            n_nodes=config['p'], n_samples=config['n'],
            edge_prob=config['density'], graph_type='ER', seed=config['seed']
        )
        
        exp_result = {
            'config': config,
            'baselines': {},
            'mf_acd': {}
        }
        
        # Run baselines (limit for time)
        print(f"  PC-FisherZ...", end=' ')
        exp_result['baselines']['pc_fisherz'] = evaluate_method(
            run_baseline_pc_fisherz, data, true_adj, alpha=0.05
        )
        print(f"F1={exp_result['baselines']['pc_fisherz'].get('f1', 0):.3f}")
        
        print(f"  PC-Stable...", end=' ')
        exp_result['baselines']['pc_stable'] = evaluate_method(
            run_baseline_pc_stable, data, true_adj, alpha=0.05
        )
        print(f"F1={exp_result['baselines']['pc_stable'].get('f1', 0):.3f}")
        
        # Only run GES for smaller graphs (it's slow)
        if config['p'] <= 20:
            print(f"  GES...", end=' ')
            exp_result['baselines']['ges'] = evaluate_method(
                run_baseline_ges, data, true_adj
            )
            print(f"F1={exp_result['baselines']['ges'].get('f1', 0):.3f}")
        
        # Run improved MF-ACD
        print(f"  MF-ACD (improved)...", end=' ')
        mf_result = evaluate_method(
            run_mf_acd_wrapper, data, true_adj
        )
        exp_result['mf_acd']['improved'] = mf_result
        print(f"F1={mf_result.get('f1', 0):.3f}, Savings={mf_result.get('savings_pct', 0):.1f}%")
        
        results['main_experiments'].append(exp_result)
    
    # Ablation studies on subset
    print("\n" + "="*70)
    print("ABLATION STUDIES")
    print("="*70)
    
    ablation_configs = [
        {'p': 50, 'n': 1000, 'density': 0.2, 'seed': 42},
        {'p': 50, 'n': 1000, 'density': 0.2, 'seed': 123},
        {'p': 50, 'n': 1000, 'density': 0.2, 'seed': 456},
    ]
    
    print("\n1. UGFS Component Ablation (3 runs)...")
    ugfs_results = {'no_iterative': [], 'full_ugfs': []}
    
    for config in ablation_configs:
        print(f"  Run with seed {config['seed']}...")
        data, true_adj = generate_synthetic_data(**config)
        
        # No iterative refinement
        mf_acd = MFACDImproved(use_iterative=False)
        result = mf_acd.fit(data)
        pred_skeleton = np.maximum(result['adjacency'], result['adjacency'].T)
        metrics = compute_metrics(pred_skeleton, true_adj)
        ugfs_results['no_iterative'].append({
            'f1': metrics['f1'],
            'savings_pct': result['savings_pct']
        })
        
        # Full UGFS
        mf_acd = MFACDImproved(use_iterative=True)
        result = mf_acd.fit(data)
        pred_skeleton = np.maximum(result['adjacency'], result['adjacency'].T)
        metrics = compute_metrics(pred_skeleton, true_adj)
        ugfs_results['full_ugfs'].append({
            'f1': metrics['f1'],
            'savings_pct': result['savings_pct']
        })
    
    results['ablations']['ugfs_components'] = ugfs_results
    
    print("\n2. Budget Allocation Sensitivity...")
    alloc_results = {}
    
    data, true_adj = generate_synthetic_data(p=50, n=1000, edge_prob=0.2, graph_type='ER', seed=42)
    
    allocations = {
        'conservative': (0.40, 0.30, 0.30),
        'aggressive': (0.25, 0.15, 0.60),
        'balanced': (0.30, 0.25, 0.45)
    }
    
    for name, alloc in allocations.items():
        print(f"  {name}: {alloc}...", end=' ')
        mf_acd = MFACDImproved(budget_allocation=alloc)
        result = mf_acd.fit(data)
        pred_skeleton = np.maximum(result['adjacency'], result['adjacency'].T)
        metrics = compute_metrics(pred_skeleton, true_adj)
        alloc_results[name] = {
            'f1': metrics['f1'],
            'savings_pct': result['savings_pct']
        }
        print(f"F1={metrics['f1']:.3f}, Savings={result['savings_pct']:.1f}%")
    
    results['ablations']['allocation_sensitivity'] = alloc_results
    
    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    # Aggregate by method
    method_stats = {}
    
    for exp in results['main_experiments']:
        # Baselines
        for method, metrics in exp['baselines'].items():
            if method not in method_stats:
                method_stats[method] = {'f1': [], 'runtime': [], 'shd': []}
            if metrics['status'] == 'success':
                method_stats[method]['f1'].append(metrics['f1'])
                method_stats[method]['runtime'].append(metrics['runtime'])
                method_stats[method]['shd'].append(metrics['shd'])
        
        # MF-ACD
        mf_metrics = exp['mf_acd']['improved']
        if 'mf_acd' not in method_stats:
            method_stats['mf_acd'] = {'f1': [], 'runtime': [], 'shd': [], 'savings_pct': []}
        if mf_metrics['status'] == 'success':
            method_stats['mf_acd']['f1'].append(mf_metrics['f1'])
            method_stats['mf_acd']['runtime'].append(mf_metrics['runtime'])
            method_stats['mf_acd']['shd'].append(mf_metrics['shd'])
            if 'savings_pct' in mf_metrics:
                method_stats['mf_acd']['savings_pct'].append(mf_metrics['savings_pct'])
    
    # Compute summary statistics
    summary = {}
    for method, stats in method_stats.items():
        if stats['f1']:
            summary[method] = {
                'f1_mean': np.mean(stats['f1']),
                'f1_std': np.std(stats['f1']),
                'runtime_mean': np.mean(stats['runtime']),
                'shd_mean': np.mean(stats['shd']),
                'n': len(stats['f1'])
            }
            if 'savings_pct' in stats and stats['savings_pct']:
                summary[method]['savings_pct_mean'] = np.mean(stats['savings_pct'])
                summary[method]['savings_pct_std'] = np.std(stats['savings_pct'])
    
    results['summary'] = summary
    
    # Save results
    with open('/home/nw366/ResearchArena/outputs/kimi_t3_causal_learning/idea_01/exp/results_improved.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for method, stats in summary.items():
        print(f"\n{method}:")
        print(f"  F1: {stats['f1_mean']:.3f} ± {stats.get('f1_std', 0):.3f}")
        print(f"  Runtime: {stats['runtime_mean']:.2f}s")
        if 'savings_pct_mean' in stats:
            print(f"  Savings: {stats['savings_pct_mean']:.1f}% ± {stats.get('savings_pct_std', 0):.1f}%")
        print(f"  n={stats['n']}")
    
    # Compare to PC-FisherZ baseline
    if 'pc_fisherz' in summary and 'mf_acd' in summary:
        pc_f1 = summary['pc_fisherz']['f1_mean']
        mf_f1 = summary['mf_acd']['f1_mean']
        f1_diff = ((mf_f1 - pc_f1) / pc_f1 * 100) if pc_f1 > 0 else 0
        
        print("\n" + "="*70)
        print("KEY FINDING")
        print("="*70)
        print(f"PC-FisherZ F1: {pc_f1:.3f}")
        print(f"MF-ACD F1: {mf_f1:.3f}")
        print(f"F1 difference: {f1_diff:+.1f}%")
        print(f"Cost savings: {summary['mf_acd'].get('savings_pct_mean', 0):.1f}%")
        
        if abs(f1_diff) < 10:
            print("\n✓ ACCURACY TARGET MET: F1 difference < 10%")
        else:
            print(f"\n✗ Accuracy still needs improvement ({abs(f1_diff):.1f}% difference)")
    
    print("\n" + "="*70)
    print("Results saved to exp/results_improved.json")
    print("="*70)


if __name__ == '__main__':
    main()
