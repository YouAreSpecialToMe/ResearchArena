"""Run SPICED on n=10 datasets."""
import sys
import os
sys.path.insert(0, 'exp')
sys.path.insert(0, 'exp/spiced')

import numpy as np
import json
import glob
import time
from spiced_main import spiced
from shared.metrics import compute_all_metrics

datasets = sorted(glob.glob("data/processed/datasets/*.npz"),
                 key=lambda f: (np.load(f)['adj'].shape[0], int(np.load(f)['seed'])))
datasets = datasets[:2400]

print(f"Running SPICED on {len(datasets)} datasets...")
results = []
for i, f in enumerate(datasets):
    if i % 200 == 0:
        print(f"  {i}/{len(datasets)}")
    d = np.load(f)
    try:
        pred, timing, _ = spiced(d['data'], k_neighbors=5, alpha=0.05, lambda1=0.1, 
                                 lambda3=0.01, max_iter=100, w_threshold=0.3,
                                 use_it_init=True, use_structural_constraints=True, seed=int(d['seed']))
        metrics = compute_all_metrics(d['adj'], pred)
        results.append({'graph_id': int(d['graph_id']), 'mechanism': str(d['mechanism']),
                       'n_samples': int(d['n_samples']), 'seed': int(d['seed']),
                       'n_nodes': d['adj'].shape[0], 'runtime': timing['total'],
                       'runtime_phase1': timing['phase1_skeleton'],
                       'runtime_phase2': timing['phase2_constraints'],
                       'runtime_phase3': timing['phase3_optimization'], **metrics})
    except Exception as e:
        pass

os.makedirs("results/synthetic", exist_ok=True)
with open("results/synthetic/spiced_summary.json", 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"SPICED: {len(results)} results")
