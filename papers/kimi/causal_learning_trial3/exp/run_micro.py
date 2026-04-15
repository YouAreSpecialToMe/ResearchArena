"""
Micro experiment - just MF-ACD on small graphs.
"""
import numpy as np
import json
import time
import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_causal_learning/idea_01')

from exp.mf_acd.mf_acd_improved import MFACDImproved
from exp.shared.data_loader import generate_synthetic_data
from exp.shared.metrics import compute_metrics

results = []

# Just 4 quick runs
configs = [
    (20, 1000, 0.1, 42),
    (20, 1000, 0.2, 42),
    (20, 1000, 0.1, 123),
    (20, 1000, 0.2, 123),
]

for p, n, d, seed in configs:
    print(f"Running p={p}, d={d}, seed={seed}...", flush=True)
    data, true_adj = generate_synthetic_data(p, n, d, 'ER', seed)
    
    t0 = time.time()
    mf = MFACDImproved()
    mf_result = mf.fit(data)
    mf_time = time.time() - t0
    
    mf_adj = mf_result['adjacency']
    mf_skeleton = (np.maximum(mf_adj, mf_adj.T) > 0).astype(int)
    true_skeleton = (np.maximum(true_adj, true_adj.T) > 0).astype(int)
    metrics = compute_metrics(mf_skeleton, true_skeleton)
    
    results.append({
        'f1': float(metrics['f1']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'shd': int(metrics['shd']),
        'runtime': float(mf_time),
        'savings_pct': float(mf_result['savings_pct']),
        'p': p, 'density': d, 'seed': seed
    })
    print(f"  F1={metrics['f1']:.3f}, Savings={mf_result['savings_pct']:.1f}%", flush=True)

# Save
with open('/home/nw366/ResearchArena/outputs/kimi_t3_causal_learning/idea_01/exp/results_micro.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nDone!")
