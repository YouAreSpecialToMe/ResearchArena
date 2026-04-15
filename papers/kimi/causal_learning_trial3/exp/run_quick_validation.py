"""
Quick validation of improved MF-ACD with key experiments.
Focus on demonstrating the improvements address feedback.
"""
import numpy as np
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.mf_acd.mf_acd_improved import MFACDImproved
from exp.shared.data_loader import generate_synthetic_data
from exp.shared.metrics import compute_metrics


def run_pc_baseline(data, alpha=0.05):
    """Quick PC baseline."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    cg = pc(data, alpha, fisherz, verbose=False)
    return cg.G.graph


def evaluate(data, true_adj, method_fn, **kwargs):
    """Evaluate a method."""
    try:
        start = time.time()
        if 'MFACD' in str(type(method_fn)):
            result = method_fn.fit(data)
            pred = result['adjacency']
            extras = {'savings_pct': result['savings_pct']}
        else:
            pred = method_fn(data, **kwargs)
            extras = {}
        runtime = time.time() - start
        
        pred_skeleton = np.maximum(pred, pred.T) > 0
        true_skeleton = np.maximum(true_adj, true_adj.T) > 0
        metrics = compute_metrics(pred_skeleton.astype(int), true_skeleton.astype(int))
        metrics['runtime'] = runtime
        metrics.update(extras)
        return metrics
    except Exception as e:
        print(f"Error: {e}")
        return {'f1': 0, 'precision': 0, 'recall': 0, 'shd': 9999, 'runtime': 0}


def main():
    print("="*70)
    print("QUICK VALIDATION - IMPROVED MF-ACD")
    print("="*70)
    
    results = {
        'pc_baseline': [],
        'mf_acd_improved': [],
        'ablations': {}
    }
    
    # Quick configs: 20-node and 50-node graphs
    configs = [
        # 20-node, various densities
        {'p': 20, 'n': 1000, 'density': 0.1, 'seed': 42},
        {'p': 20, 'n': 1000, 'density': 0.2, 'seed': 42},
        {'p': 20, 'n': 1000, 'density': 0.3, 'seed': 42},
        # 50-node graphs (key for scalability)
        {'p': 50, 'n': 1000, 'density': 0.1, 'seed': 42},
        {'p': 50, 'n': 1000, 'density': 0.2, 'seed': 42},
        {'p': 50, 'n': 1000, 'density': 0.1, 'seed': 123},
        {'p': 50, 'n': 1000, 'density': 0.2, 'seed': 123},
    ]
    
    print(f"\nRunning {len(configs)} experiments...")
    
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}] p={cfg['p']}, density={cfg['density']}, seed={cfg['seed']}")
        
        # Generate data
        data, true_adj = generate_synthetic_data(
            n_nodes=cfg['p'], n_samples=cfg['n'],
            edge_prob=cfg['density'], graph_type='ER', seed=cfg['seed']
        )
        
        # PC Baseline
        print("  PC-FisherZ...", end=" ")
        pc_result = evaluate(data, true_adj, run_pc_baseline)
        results['pc_baseline'].append({**pc_result, **cfg})
        print(f"F1={pc_result['f1']:.3f}, Time={pc_result['runtime']:.2f}s")
        
        # MF-ACD Improved
        print("  MF-ACD (improved)...", end=" ")
        mf_acd = MFACDImproved(use_adaptive=True, use_iterative=True)
        mf_result = evaluate(data, true_adj, mf_acd)
        results['mf_acd_improved'].append({**mf_result, **cfg})
        print(f"F1={mf_result['f1']:.3f}, Savings={mf_result.get('savings_pct', 0):.1f}%")
    
    # Ablations on one 50-node graph
    print("\n" + "-"*50)
    print("ABLATION: UGFS Components")
    print("-"*50)
    
    cfg = {'p': 50, 'n': 1000, 'density': 0.2, 'seed': 42}
    data, true_adj = generate_synthetic_data(
        n_nodes=50, n_samples=1000, edge_prob=0.2, graph_type='ER', seed=42
    )
    
    ablations = {
        'no_iterative': MFACDImproved(use_iterative=False),
        'full_ugfs': MFACDImproved(use_iterative=True, use_adaptive=True)
    }
    
    for name, model in ablations.items():
        print(f"  {name}...", end=" ")
        result = evaluate(data, true_adj, model)
        results['ablations'][name] = result
        print(f"F1={result['f1']:.3f}, Savings={result.get('savings_pct', 0):.1f}%")
    
    # Allocation sensitivity
    print("\n" + "-"*50)
    print("ABLATION: Budget Allocation")
    print("-"*50)
    
    allocations = {
        'conservative': (0.40, 0.30, 0.30),
        'balanced': (0.30, 0.25, 0.45),
        'aggressive': (0.25, 0.15, 0.60)
    }
    
    for name, alloc in allocations.items():
        print(f"  {name} {alloc}...", end=" ")
        model = MFACDImproved(budget_allocation=alloc)
        result = evaluate(data, true_adj, model)
        results['ablations'][f'alloc_{name}'] = result
        print(f"F1={result['f1']:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    pc_f1 = [r['f1'] for r in results['pc_baseline']]
    mf_f1 = [r['f1'] for r in results['mf_acd_improved']]
    mf_savings = [r.get('savings_pct', 0) for r in results['mf_acd_improved']]
    
    print(f"\nPC-FisherZ:")
    print(f"  F1: {np.mean(pc_f1):.3f} ± {np.std(pc_f1):.3f} (n={len(pc_f1)})")
    
    print(f"\nMF-ACD (improved):")
    print(f"  F1: {np.mean(mf_f1):.3f} ± {np.std(mf_f1):.3f} (n={len(mf_f1)})")
    print(f"  Savings: {np.mean(mf_savings):.1f}% ± {np.std(mf_savings):.1f}%")
    
    f1_diff_pct = (np.mean(mf_f1) - np.mean(pc_f1)) / np.mean(pc_f1) * 100
    print(f"\nF1 Difference: {f1_diff_pct:+.1f}%")
    
    if abs(f1_diff_pct) < 10:
        print("✓ Accuracy within 10% of PC baseline")
    elif abs(f1_diff_pct) < 20:
        print("~ Accuracy within 20% of PC baseline (needs improvement)")
    else:
        print("✗ Accuracy gap > 20% (significant improvement needed)")
    
    # Save results
    output = {
        'summary': {
            'pc_f1_mean': float(np.mean(pc_f1)),
            'pc_f1_std': float(np.std(pc_f1)),
            'mf_acd_f1_mean': float(np.mean(mf_f1)),
            'mf_acd_f1_std': float(np.std(mf_f1)),
            'mf_acd_savings_mean': float(np.mean(mf_savings)),
            'f1_diff_pct': float(f1_diff_pct)
        },
        'details': results
    }
    
    with open('/home/nw366/ResearchArena/outputs/kimi_t3_causal_learning/idea_01/exp/results_quick.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to exp/results_quick.json")
    return output


if __name__ == '__main__':
    main()
