"""Run sample size ablation: ADECD and baselines on n={200,500,1000,2000}."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
from src.data_generator import generate_sem_data
from src.portfolio import run_portfolio
from src.assumption_profiler import profile_edges
from src.reconciler import reconcile, get_candidate_edges
from src.baselines import majority_vote
from src.metrics import shd, edge_f1

# Pretrained beta from exp/06
BETA_PRETRAINED = np.array([1.0707959500535706, 3.1266909740418325, -2.7828049176207035, 5.0])

sample_sizes = [200, 500, 1000, 2000]
seeds = [42, 123, 456]
results = {n: {'adecd': [], 'majority': [], 'notears': [], 'ges': []} for n in sample_sizes}

for func in ['linear', 'nonlinear', 'mixed']:
    for noise in ['gaussian', 'laplace']:
        for n_samples in sample_sizes:
            for seed in seeds:
                rng = np.random.RandomState(seed)
                data, true_adj, _ = generate_sem_data(
                    num_nodes=10, edge_density=2.0,
                    functional_form=func, noise_type=noise,
                    faithfulness_mode='faithful', confounder_fraction=0.0,
                    sample_size=n_samples, rng=rng
                )

                # Run portfolio
                algo_outputs, _ = run_portfolio(data, timeout=60)

                # ADECD transfer
                candidate_edges = get_candidate_edges(algo_outputs)
                profiles = profile_edges(data, candidate_edges, algo_outputs)
                adecd_adj, _, _ = reconcile(algo_outputs, profiles, BETA_PRETRAINED, threshold=0.5)

                # Majority vote
                maj_adj = majority_vote(algo_outputs)

                # Individual: NOTEARS, GES
                notears_adj = algo_outputs.get('NOTEARS', np.zeros_like(true_adj))
                ges_adj = algo_outputs.get('GES', np.zeros_like(true_adj))

                results[n_samples]['adecd'].append(shd(adecd_adj, true_adj))
                results[n_samples]['majority'].append(shd(maj_adj, true_adj))
                results[n_samples]['notears'].append(shd(notears_adj, true_adj))
                results[n_samples]['ges'].append(shd(ges_adj, true_adj))

            print(f"  Done: func={func}, noise={noise}, n={n_samples}")

# Aggregate
summary = {}
for n in sample_sizes:
    summary[str(n)] = {}
    for method in ['adecd', 'majority', 'notears', 'ges']:
        vals = results[n][method]
        summary[str(n)][method] = {
            'shd_mean': float(np.mean(vals)),
            'shd_std': float(np.std(vals)),
            'n_runs': len(vals),
        }

output = {
    'experiment': 'ablation_sample_size_complete',
    'metrics': summary,
    'config': {
        'sample_sizes': sample_sizes,
        'methods': ['adecd', 'majority', 'notears', 'ges'],
        'n_settings_per_size': 6,
        'n_seeds': 3,
    }
}

with open(os.path.join(os.path.dirname(__file__), 'results_complete.json'), 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults:")
for n in sample_sizes:
    line = f"n={n:5d}:"
    for method in ['adecd', 'majority', 'notears', 'ges']:
        m = summary[str(n)][method]['shd_mean']
        s = summary[str(n)][method]['shd_std']
        line += f"  {method}={m:.1f}±{s:.1f}"
    print(line)
