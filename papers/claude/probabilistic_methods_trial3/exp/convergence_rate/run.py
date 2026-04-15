"""Convergence rate analysis: verify LIL rate sqrt(log log t / t)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from src.adaquantcs import AdaQuantCS
from src.baselines import FullMemoryQuantileCS
from src.utils import (
    generate_stream, true_quantile, get_initial_range,
    save_results,
)

# Use 1M instead of 10M to keep runtime reasonable
N = 1_000_000
ALPHA = 0.05
P = 0.5
DIST = 'gaussian'
SEEDS = [42, 123, 456]
CHECKPOINTS = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]


def main():
    print("=" * 60)
    print("Convergence Rate Analysis")
    print("=" * 60)

    tq = true_quantile(DIST, P)
    init_range = get_initial_range(DIST)
    start_time = time.time()

    aq_results = {'ci_width': []}
    fm_results = {'ci_width': []}

    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        stream = generate_stream(DIST, N, seed)

        aq = AdaQuantCS(p=P, k=50, alpha=ALPHA, initial_range=init_range)
        fm = FullMemoryQuantileCS(p=P, alpha=ALPHA)

        checkpoint_set = set(CHECKPOINTS)
        aq_widths = []
        fm_widths = []

        for i, x in enumerate(stream):
            aq.update(x)
            fm.update(x)
            t = i + 1

            if t in checkpoint_set:
                lo_aq, hi_aq = aq.get_ci()
                lo_fm, hi_fm = fm.get_ci()
                aq_widths.append(hi_aq - lo_aq)
                fm_widths.append(hi_fm - lo_fm)
                print(f"  t={t:>8d}: AQ_width={aq_widths[-1]:.5f}, FM_width={fm_widths[-1]:.5f}")

        aq_results['ci_width'].append(aq_widths)
        fm_results['ci_width'].append(fm_widths)

    aq_results['ci_width_mean'] = np.mean(aq_results['ci_width'], axis=0).tolist()
    aq_results['ci_width_std'] = np.std(aq_results['ci_width'], axis=0).tolist()
    fm_results['ci_width_mean'] = np.mean(fm_results['ci_width'], axis=0).tolist()
    fm_results['ci_width_std'] = np.std(fm_results['ci_width'], axis=0).tolist()

    # Estimate convergence rate via log-log regression
    log_t = np.log(np.array(CHECKPOINTS, dtype=float))
    log_aq_width = np.log(np.array(aq_results['ci_width_mean']))
    log_fm_width = np.log(np.array(fm_results['ci_width_mean']))

    # Linear regression: log(width) = a + b * log(t)
    aq_slope, aq_intercept = np.polyfit(log_t, log_aq_width, 1)
    fm_slope, fm_intercept = np.polyfit(log_t, log_fm_width, 1)

    print(f"\nConvergence exponents:")
    print(f"  AdaQuantCS: slope = {aq_slope:.4f} (theory: -0.5)")
    print(f"  FullMemory: slope = {fm_slope:.4f} (theory: -0.5)")

    # Compute normalized width: width * sqrt(t / log(log(t)))
    t_arr = np.array(CHECKPOINTS, dtype=float)
    lil_factor = np.sqrt(t_arr / np.log(np.log(t_arr)))
    aq_normalized = np.array(aq_results['ci_width_mean']) * lil_factor
    fm_normalized = np.array(fm_results['ci_width_mean']) * lil_factor

    # Width ratio
    width_ratio = (np.array(aq_results['ci_width_mean']) / np.array(fm_results['ci_width_mean'])).tolist()
    print(f"\nWidth ratio (AQ/FM) at checkpoints: {[f'{r:.2f}' for r in width_ratio]}")

    all_results = {
        'checkpoints': CHECKPOINTS,
        'adaquantcs': aq_results,
        'fullmemory': fm_results,
        'convergence_exponent_aq': float(aq_slope),
        'convergence_exponent_fm': float(fm_slope),
        'aq_normalized_width': aq_normalized.tolist(),
        'fm_normalized_width': fm_normalized.tolist(),
        'width_ratio': width_ratio,
        'runtime_seconds': time.time() - start_time,
    }

    outpath = os.path.join(os.path.dirname(__file__), 'results.json')
    save_results(all_results, outpath)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
