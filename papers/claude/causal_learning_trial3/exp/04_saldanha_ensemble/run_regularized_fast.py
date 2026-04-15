"""Fast Saldanha ensemble with L2 regularization - subset of settings."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
from scipy.optimize import minimize
from src.data_generator import generate_sem_data
from src.portfolio import run_portfolio
from src.metrics import shd, edge_f1

seeds = [42, 123]
# Use a representative subset: 2 func × 2 noise × 2 densities = 8 settings × 2 seeds = 16 runs
configs = []
for func in ['linear', 'nonlinear']:
    for noise in ['gaussian', 'laplace']:
        for density in [1.5, 2.5]:
            configs.append({'func': func, 'noise': noise, 'density': density})

print(f"Running {len(configs)} configs × {len(seeds)} seeds = {len(configs)*len(seeds)} runs...")
all_data = []
for i, cfg in enumerate(configs):
    for seed in seeds:
        rng = np.random.RandomState(seed)
        data, true_adj, _ = generate_sem_data(
            num_nodes=8, edge_density=cfg['density'],
            functional_form=cfg['func'], noise_type=cfg['noise'],
            faithfulness_mode='faithful', confounder_fraction=0.0,
            sample_size=500, rng=rng
        )
        algo_outputs, _ = run_portfolio(data, timeout=60)
        all_data.append({
            'algo_outputs': algo_outputs,
            'true_adj': true_adj,
        })
    print(f"  Config {i+1}/{len(configs)} done")

algo_names = list(all_data[0]['algo_outputs'].keys())
n_algos = len(algo_names)

from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)

reg_lambdas = [0.01, 0.1, 0.5, 1.0, 2.0]
best_overall = None

for reg_lambda in reg_lambdas:
    fold_shds = []
    fold_weights_list = []

    indices = np.arange(len(all_data))
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        train_data = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]

        def loss(log_weights):
            w = np.exp(log_weights) / np.sum(np.exp(log_weights))
            total = 0.0
            count = 0
            for item in train_data:
                graphs = item['algo_outputs']
                true_adj = item['true_adj']
                n = true_adj.shape[0]
                for i in range(n):
                    for j in range(n):
                        pred = sum(w[k] * float(graphs[algo_names[k]][i, j] > 0)
                                  for k in range(n_algos))
                        pred = np.clip(pred, 1e-6, 1 - 1e-6)
                        t = true_adj[i, j]
                        total -= t * np.log(pred) + (1 - t) * np.log(1 - pred)
                        count += 1
            reg = reg_lambda * np.sum(log_weights ** 2)
            return total / max(count, 1) + reg

        result = minimize(loss, np.zeros(n_algos), method='L-BFGS-B',
                         options={'maxiter': 100})
        weights = np.exp(result.x) / np.sum(np.exp(result.x))
        fold_weights_list.append(weights)

        fold_shd_vals = []
        for item in test_data:
            graphs = item['algo_outputs']
            true_adj = item['true_adj']
            n = true_adj.shape[0]
            pred = np.zeros((n, n))
            for k, name in enumerate(algo_names):
                pred += weights[k] * (graphs[name] > 0).astype(float)
            pred_adj = (pred >= 0.5).astype(float)
            fold_shd_vals.append(shd(pred_adj, true_adj))
        fold_shds.extend(fold_shd_vals)

    mean_shd = float(np.mean(fold_shds))
    std_shd = float(np.std(fold_shds))
    avg_weights = np.mean(fold_weights_list, axis=0)

    print(f"  reg_lambda={reg_lambda:.2f}: SHD={mean_shd:.2f}+/-{std_shd:.2f}, "
          f"max_w={max(avg_weights):.3f}, weights={dict(zip(algo_names, [f'{w:.3f}' for w in avg_weights]))}")

    if best_overall is None or mean_shd < best_overall['shd_mean']:
        best_overall = {
            'reg_lambda': reg_lambda,
            'shd_mean': mean_shd,
            'shd_std': std_shd,
            'weights': {name: float(w) for name, w in zip(algo_names, avg_weights)},
        }

output = {
    'experiment': 'saldanha_ensemble_regularized_fast',
    'best': best_overall,
    'collapse_check': {
        'max_weight': max(best_overall['weights'].values()),
        'min_weight': min(best_overall['weights'].values()),
        'collapsed': max(best_overall['weights'].values()) > 0.9,
    }
}

with open(os.path.join(os.path.dirname(__file__), 'results_regularized.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nBest: lambda={best_overall['reg_lambda']}, SHD={best_overall['shd_mean']:.2f}")
print(f"Collapsed: {output['collapse_check']['collapsed']}")
