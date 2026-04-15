"""Experiment 1 rerun with 1000 trials for tighter coverage estimates."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
import time
from src.adaquantcs import AdaQuantCS
from src.baselines import FullMemoryQuantileCS, SketchBootstrapCI
from src.utils import (
    DISTRIBUTIONS, QUANTILE_LEVELS, CHECKPOINTS_100K,
    generate_stream, true_quantile, get_initial_range,
    check_anytime_coverage, check_pointwise_coverage,
    save_results, NumpyEncoder,
)

N = 100_000
N_TRIALS = 1000
ALPHA = 0.05
K = 50


def run_single_trial(dist_name, p, seed):
    """Run all three methods on one stream."""
    stream = generate_stream(dist_name, N, seed)
    tq = true_quantile(dist_name, p)
    init_range = get_initial_range(dist_name)

    methods = {
        'adaquantcs': AdaQuantCS(p=p, k=K, alpha=ALPHA, initial_range=init_range),
        'fullmemory': FullMemoryQuantileCS(p=p, alpha=ALPHA),
        'bootstrap': SketchBootstrapCI(p=p, alpha=ALPHA, reservoir_size=1000, n_bootstrap=500),
    }

    results = {name: {'ci_lower': [], 'ci_upper': [], 'ci_width': [], 'memory': []}
               for name in methods}

    checkpoint_set = set(CHECKPOINTS_100K)

    for i, x in enumerate(stream):
        t = i + 1
        for name, method in methods.items():
            method.update(x)

        if t in checkpoint_set:
            for name, method in methods.items():
                lo, hi = method.get_ci()
                results[name]['ci_lower'].append(lo)
                results[name]['ci_upper'].append(hi)
                results[name]['ci_width'].append(hi - lo)
                results[name]['memory'].append(method.memory_usage())

    for name in methods:
        results[name]['anytime_covered'] = check_anytime_coverage(
            results[name]['ci_lower'], results[name]['ci_upper'], tq
        )
        results[name]['pointwise_covered'] = check_pointwise_coverage(
            results[name]['ci_lower'], results[name]['ci_upper'], tq
        )

    return results


def main():
    print("=" * 60)
    print("Experiment 1: Coverage Validation (1000 trials)")
    print("=" * 60)

    all_results = {}
    start_time = time.time()

    for dist_name in DISTRIBUTIONS:
        for p in QUANTILE_LEVELS:
            key = f"{dist_name}_p{p}"
            print(f"\nRunning {key} ({N_TRIALS} trials)...")
            tq = true_quantile(dist_name, p)
            print(f"  True quantile: {tq:.4f}")

            trial_results = []
            for trial in range(N_TRIALS):
                seed = trial
                res = run_single_trial(dist_name, p, seed)
                trial_results.append(res)

                if (trial + 1) % 200 == 0:
                    aq_cov = np.mean([r['adaquantcs']['anytime_covered'] for r in trial_results])
                    print(f"  Trial {trial+1}/{N_TRIALS}, AQ coverage: {aq_cov:.4f}")

            # Aggregate
            agg = {}
            for method_name in ['adaquantcs', 'fullmemory', 'bootstrap']:
                anytime_covs = [r[method_name]['anytime_covered'] for r in trial_results]
                pointwise_covs = np.array([r[method_name]['pointwise_covered'] for r in trial_results])
                ci_widths = np.array([r[method_name]['ci_width'] for r in trial_results])
                memories = np.array([r[method_name]['memory'] for r in trial_results])

                agg[method_name] = {
                    'anytime_coverage': float(np.mean(anytime_covs)),
                    'anytime_coverage_std': float(np.std(anytime_covs)),
                    'pointwise_coverage_per_checkpoint': np.mean(pointwise_covs, axis=0).tolist(),
                    'ci_width_mean': np.mean(ci_widths, axis=0).tolist(),
                    'ci_width_std': np.std(ci_widths, axis=0).tolist(),
                    'memory_mean': np.mean(memories, axis=0).tolist(),
                    'n_trials': N_TRIALS,
                }

                print(f"  {method_name}: cov={agg[method_name]['anytime_coverage']:.4f}, "
                      f"width={np.mean(ci_widths[:, -1]):.4f}±{np.std(ci_widths[:, -1]):.4f}")

            all_results[key] = {
                'distribution': dist_name,
                'quantile_level': p,
                'true_quantile': float(tq),
                'n': N,
                'checkpoints': CHECKPOINTS_100K,
                'methods': agg,
            }

    elapsed = time.time() - start_time
    all_results['runtime_seconds'] = elapsed
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    outpath = os.path.join(os.path.dirname(__file__), 'results_1000.json')
    save_results(all_results, outpath)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
