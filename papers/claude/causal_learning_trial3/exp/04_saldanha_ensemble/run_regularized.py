"""Run Saldanha ensemble with L2 regularization to prevent weight collapse."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
from scipy.optimize import minimize
from src.data_generator import generate_sem_data, generate_experiment_grid
from src.portfolio import run_portfolio
from src.metrics import shd, edge_f1
from src.baselines import majority_vote

seeds = [42, 123, 456]
settings = generate_experiment_grid()

# Generate all datasets and run portfolio
print("Running portfolio on all settings...")
all_data = []
for s in settings:
    for seed in seeds:
        rng = np.random.RandomState(seed)
        data, true_adj, _ = generate_sem_data(
            s['num_nodes'], s['edge_density'], s['functional_form'],
            s['noise_type'], s['faithfulness_mode'], s['confounder_fraction'],
            s['sample_size'], rng
        )
        algo_outputs, _ = run_portfolio(data, timeout=120)
        all_data.append({
            'setting_id': s['setting_id'],
            'seed': seed,
            'algo_outputs': algo_outputs,
            'true_adj': true_adj,
        })
    if (s['setting_id'] + 1) % 20 == 0:
        print(f"  Settings {s['setting_id']+1}/{len(settings)} done")

algo_names = list(all_data[0]['algo_outputs'].keys())
n_algos = len(algo_names)

# 5-fold cross-validation with L2 regularization
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Try different regularization strengths
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
            # L2 regularization toward uniform (log_weights=0)
            reg = reg_lambda * np.sum(log_weights ** 2)
            return total / max(count, 1) + reg

        result = minimize(loss, np.zeros(n_algos), method='L-BFGS-B',
                         options={'maxiter': 100})
        weights = np.exp(result.x) / np.sum(np.exp(result.x))
        fold_weights_list.append(weights)

        # Evaluate on test fold
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

    print(f"  reg_lambda={reg_lambda:.2f}: SHD={mean_shd:.2f}±{std_shd:.2f}, "
          f"weights={dict(zip(algo_names, [f'{w:.3f}' for w in avg_weights]))}")

    if best_overall is None or mean_shd < best_overall['shd_mean']:
        best_overall = {
            'reg_lambda': reg_lambda,
            'shd_mean': mean_shd,
            'shd_std': std_shd,
            'weights': {name: float(w) for name, w in zip(algo_names, avg_weights)},
            'all_shds': fold_shds,
        }

# Also compute F1 for the best regularization
# Rerun with best lambda on all data (train on all, report)
best_reg = best_overall['reg_lambda']
def loss_final(log_weights):
    w = np.exp(log_weights) / np.sum(np.exp(log_weights))
    total = 0.0
    count = 0
    for item in all_data:
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
    reg = best_reg * np.sum(log_weights ** 2)
    return total / max(count, 1) + reg

result_final = minimize(loss_final, np.zeros(n_algos), method='L-BFGS-B',
                       options={'maxiter': 100})
final_weights = np.exp(result_final.x) / np.sum(np.exp(result_final.x))

# Compute full metrics with final weights
all_shds = []
all_f1s = []
for item in all_data:
    graphs = item['algo_outputs']
    true_adj = item['true_adj']
    n = true_adj.shape[0]
    pred = np.zeros((n, n))
    for k, name in enumerate(algo_names):
        pred += final_weights[k] * (graphs[name] > 0).astype(float)
    pred_adj = (pred >= 0.5).astype(float)
    all_shds.append(shd(pred_adj, true_adj))
    all_f1s.append(edge_f1(pred_adj, true_adj)['f1'])

output = {
    'experiment': 'saldanha_ensemble_regularized',
    'metrics': {
        'shd_mean': float(np.mean(all_shds)),
        'shd_std': float(np.std(all_shds)),
        'f1_mean': float(np.mean(all_f1s)),
        'n': len(all_data),
    },
    'config': {
        'best_reg_lambda': best_reg,
        'cv_folds': 5,
        'final_weights': {name: float(w) for name, w in zip(algo_names, final_weights)},
        'reg_lambdas_tested': reg_lambdas,
    },
    'cv_results': {
        'shd_mean': best_overall['shd_mean'],
        'shd_std': best_overall['shd_std'],
    },
    'collapse_check': {
        'max_weight': float(max(final_weights)),
        'min_weight': float(min(final_weights)),
        'collapsed': bool(max(final_weights) > 0.9),
    }
}

with open(os.path.join(os.path.dirname(__file__), 'results_regularized.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nBest regularized Saldanha: SHD={np.mean(all_shds):.2f}±{np.std(all_shds):.2f}, "
      f"F1={np.mean(all_f1s):.3f}")
print(f"Final weights: {dict(zip(algo_names, [f'{w:.3f}' for w in final_weights]))}")
print(f"Collapsed: {max(final_weights) > 0.9}")
