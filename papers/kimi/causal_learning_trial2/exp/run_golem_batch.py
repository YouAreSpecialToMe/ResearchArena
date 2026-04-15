"""Run GOLEM on n=10 datasets."""
import sys
import os
sys.path.insert(0, 'exp')

import numpy as np
import json
import glob
import time
from shared.metrics import compute_all_metrics

def golem_ev(X, lambda1=0.02, lambda2=5.0, num_iter=1000, w_threshold=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n, d = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    W = np.random.randn(d, d) * 0.01
    m, v = np.zeros_like(W), np.zeros_like(W)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, num_iter + 1):
        residual = X @ W - X
        grad_ls = (1.0 / n) * (X.T @ residual)
        grad_l1 = lambda1 * np.sign(W)
        M = W * W
        try:
            eigvals = np.linalg.eigvalsh(M)
            h = np.sum(np.exp(eigvals)) - d
            eigvecs = np.linalg.eigh(M)[1]
            exp_M = eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
            grad_dag = lambda2 * 2 * W * exp_M.T * h
        except:
            grad_dag = np.zeros_like(W)
        grad = grad_ls + grad_l1 + grad_dag
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W -= 0.001 * m_hat / (np.sqrt(v_hat) + eps)
        W = np.clip(W, -10, 10)
    return (np.abs(W) > w_threshold).astype(int)

datasets = sorted(glob.glob("data/processed/datasets/*.npz"),
                 key=lambda f: (np.load(f)['adj'].shape[0], int(np.load(f)['seed'])))
datasets = datasets[:2400]

print(f"Running GOLEM on {len(datasets)} datasets...")
results = []
for i, f in enumerate(datasets):
    if i % 200 == 0:
        print(f"  {i}/{len(datasets)}")
    d = np.load(f)
    start = time.time()
    try:
        pred = golem_ev(d['data'], seed=int(d['seed']))
        runtime = time.time() - start
        metrics = compute_all_metrics(d['adj'], pred)
        results.append({'graph_id': int(d['graph_id']), 'mechanism': str(d['mechanism']),
                       'n_samples': int(d['n_samples']), 'seed': int(d['seed']),
                       'n_nodes': d['adj'].shape[0], 'runtime': runtime, **metrics})
    except:
        pass

os.makedirs("results/synthetic", exist_ok=True)
with open("results/synthetic/golem_summary.json", 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"GOLEM: {len(results)} results")
