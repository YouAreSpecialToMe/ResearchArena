"""
Minimal experiment run to demonstrate improvements.
Focus on 20-node graphs for speed.
"""
import numpy as np
import sys
import json
import time
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_causal_learning/idea_01')

from exp.mf_acd.mf_acd_improved import MFACDImproved
from exp.shared.data_loader import generate_synthetic_data
from exp.shared.metrics import compute_metrics
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

results = {'pc': [], 'mf_acd': [], 'configs': []}

configs = [
    (20, 1000, 0.1, 42), (20, 1000, 0.1, 123),
    (20, 1000, 0.2, 42), (20, 1000, 0.2, 123),
    (20, 1000, 0.3, 42), (20, 1000, 0.3, 123),
    (50, 1000, 0.1, 42), (50, 1000, 0.2, 42),
]

for p, n, d, seed in configs:
    print(f"Running p={p}, d={d}, seed={seed}")
    data, true_adj = generate_synthetic_data(p, n, d, 'ER', seed)
    
    # PC
    t0 = time.time()
    cg = pc(data, 0.05, fisherz, verbose=False)
    pc_time = time.time() - t0
    pc_adj = cg.G.graph
    pc_skeleton = (np.maximum(pc_adj, pc_adj.T) > 0).astype(int)
    true_skeleton = (np.maximum(true_adj, true_adj.T) > 0).astype(int)
    pc_metrics = compute_metrics(pc_skeleton, true_skeleton)
    
    # MF-ACD
    t0 = time.time()
    mf = MFACDImproved()
    mf_result = mf.fit(data)
    mf_time = time.time() - t0
    mf_adj = mf_result['adjacency']
    mf_skeleton = (np.maximum(mf_adj, mf_adj.T) > 0).astype(int)
    mf_metrics = compute_metrics(mf_skeleton, true_skeleton)
    
    results['pc'].append({
        'f1': pc_metrics['f1'], 'shd': pc_metrics['shd'], 
        'time': pc_time, 'p': p, 'd': d, 'seed': seed
    })
    results['mf_acd'].append({
        'f1': mf_metrics['f1'], 'shd': mf_metrics['shd'],
        'time': mf_time, 'savings': mf_result['savings_pct'],
        'p': p, 'd': d, 'seed': seed
    })
    
    print(f"  PC: F1={pc_metrics['f1']:.3f}, Time={pc_time:.2f}s")
    print(f"  MF: F1={mf_metrics['f1']:.3f}, Time={mf_time:.2f}s, Savings={mf_result['savings_pct']:.1f}%")

# Summary
pc_f1 = [r['f1'] for r in results['pc']]
mf_f1 = [r['f1'] for r in results['mf_acd']]
pc_time = [r['time'] for r in results['pc']]
mf_time = [r['time'] for r in results['mf_acd']]
mf_savings = [r['savings'] for r in results['mf_acd']]

summary = {
    'pc_f1_mean': float(np.mean(pc_f1)),
    'pc_f1_std': float(np.std(pc_f1)),
    'mf_acd_f1_mean': float(np.mean(mf_f1)),
    'mf_acd_f1_std': float(np.std(mf_f1)),
    'pc_time_mean': float(np.mean(pc_time)),
    'mf_acd_time_mean': float(np.mean(mf_time)),
    'savings_mean': float(np.mean(mf_savings)),
    'f1_diff_pct': float((np.mean(mf_f1) - np.mean(pc_f1)) / np.mean(pc_f1) * 100),
    'n_experiments': len(configs)
}

results['summary'] = summary

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"PC-FisherZ: F1={summary['pc_f1_mean']:.3f} ± {summary['pc_f1_std']:.3f}")
print(f"MF-ACD:     F1={summary['mf_acd_f1_mean']:.3f} ± {summary['mf_acd_f1_std']:.3f}")
print(f"F1 Diff:    {summary['f1_diff_pct']:+.1f}%")
print(f"Savings:    {summary['savings_mean']:.1f}%")

with open('/home/nw366/ResearchArena/outputs/kimi_t3_causal_learning/idea_01/exp/results_minimal.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved!")
