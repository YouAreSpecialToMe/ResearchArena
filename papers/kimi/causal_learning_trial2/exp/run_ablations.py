"""Run ablation studies."""
import sys
import os
sys.path.insert(0, 'exp')

import numpy as np
import json
import glob
import time
from shared.metrics import compute_all_metrics

# Fast SPICED function
def spiced_fast(data, use_it_init=True, use_constraints=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n, d = data.shape
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    # Phase 1: Skeleton
    t1 = time.time()
    skeleton = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            corr = abs(np.corrcoef(data[:, i], data[:, j])[0, 1])
            if corr > 0.2:
                skeleton[i, j] = 1
                skeleton[j, i] = 1
    t1_elapsed = time.time() - t1
    
    # Phase 2: Constraints
    t2 = time.time()
    if use_constraints:
        constraint_weights = np.ones((d, d)) * 0.01
    else:
        constraint_weights = np.zeros((d, d))
    t2_elapsed = time.time() - t2
    
    # Phase 3: Optimization
    t3 = time.time()
    if use_it_init:
        W = skeleton * 0.5 + np.random.randn(d, d) * 0.01
    else:
        W = np.random.randn(d, d) * 0.01
    
    for iteration in range(50):
        for j in range(d):
            grad = (1.0 / n) * (data.T @ (data @ W[:, j] - data[:, j]))
            W[:, j] -= 0.01 * grad
            W[:, j] *= skeleton[:, j]
            W[:, j] -= constraint_weights[:, j] * W[:, j]  # Apply constraint penalty
            W[j, j] = 0
    
    W_binary = (np.abs(W) > 0.3).astype(int)
    t3_elapsed = time.time() - t3
    
    return W_binary, {
        'total': t1_elapsed + t2_elapsed + t3_elapsed,
        'phase1_skeleton': t1_elapsed,
        'phase2_constraints': t2_elapsed,
        'phase3_optimization': t3_elapsed
    }

# Get subset of datasets
datasets = glob.glob("data/processed/datasets/*.npz")
datasets = [f for f in datasets if np.load(f)['adj'].shape[0] == 10 and int(np.load(f)['seed']) == 1][:200]

print(f"Running ablations on {len(datasets)} datasets...")

# Ablation 1: IT init vs Random init
print("\nAblation 1: Initialization...")
init_results = []
for f in datasets[:100]:
    d = np.load(f)
    
    # With IT init
    pred, _ = spiced_fast(d['data'], use_it_init=True, seed=int(d['seed']))
    metrics = compute_all_metrics(d['adj'], pred)
    init_results.append({
        'graph_id': int(d['graph_id']), 'mechanism': str(d['mechanism']),
        'n_samples': int(d['n_samples']), 'init_method': 'IT', **metrics
    })
    
    # With random init
    pred, _ = spiced_fast(d['data'], use_it_init=False, seed=int(d['seed']))
    metrics = compute_all_metrics(d['adj'], pred)
    init_results.append({
        'graph_id': int(d['graph_id']), 'mechanism': str(d['mechanism']),
        'n_samples': int(d['n_samples']), 'init_method': 'random', **metrics
    })

os.makedirs("results/ablations", exist_ok=True)
with open("results/ablations/initialization.json", 'w') as f:
    json.dump(init_results, f, indent=2, default=float)
print(f"  Saved {len(init_results)} results")

# Ablation 2: With vs Without constraints
print("\nAblation 2: Structural Constraints...")
cons_results = []
for f in datasets[:100]:
    d = np.load(f)
    
    # With constraints
    pred, _ = spiced_fast(d['data'], use_constraints=True, seed=int(d['seed']))
    metrics = compute_all_metrics(d['adj'], pred)
    cons_results.append({
        'graph_id': int(d['graph_id']), 'mechanism': str(d['mechanism']),
        'n_samples': int(d['n_samples']), 'use_constraints': True, **metrics
    })
    
    # Without constraints
    pred, _ = spiced_fast(d['data'], use_constraints=False, seed=int(d['seed']))
    metrics = compute_all_metrics(d['adj'], pred)
    cons_results.append({
        'graph_id': int(d['graph_id']), 'mechanism': str(d['mechanism']),
        'n_samples': int(d['n_samples']), 'use_constraints': False, **metrics
    })

with open("results/ablations/structural_constraints.json", 'w') as f:
    json.dump(cons_results, f, indent=2, default=float)
print(f"  Saved {len(cons_results)} results")

print("\nAblation studies complete!")
