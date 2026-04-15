"""Experiment 2: Memory-accuracy tradeoff.

Varies grid size k and compares CI width and memory with full-memory baseline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from src.adaquantcs import AdaQuantCS
from src.baselines import FullMemoryQuantileCS
from src.utils import (
    generate_stream, true_quantile, get_initial_range,
    save_results, SEEDS, CHECKPOINTS_1M,
)

N = 1_000_000
ALPHA = 0.05
P = 0.5
DIST = 'gaussian'
K_VALUES = [5, 10, 20, 50, 100, 200]


def main():
    print("=" * 60)
    print("Experiment 2: Memory-Accuracy Tradeoff")
    print("=" * 60)

    tq = true_quantile(DIST, P)
    init_range = get_initial_range(DIST)
    start_time = time.time()

    all_results = {
        'distribution': DIST,
        'quantile_level': P,
        'true_quantile': float(tq),
        'n': N,
        'checkpoints': CHECKPOINTS_1M,
        'k_values': K_VALUES,
        'seeds': SEEDS,
    }

    # Run full-memory baseline
    print("\nRunning full-memory baseline...")
    fm_results = {'ci_width': [], 'memory': []}
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        stream = generate_stream(DIST, N, seed)
        fm = FullMemoryQuantileCS(p=P, alpha=ALPHA)

        checkpoint_set = set(CHECKPOINTS_1M)
        widths = []
        mems = []
        for i, x in enumerate(stream):
            fm.update(x)
            t = i + 1
            if t in checkpoint_set:
                lo, hi = fm.get_ci()
                widths.append(hi - lo)
                mems.append(fm.memory_usage())
        fm_results['ci_width'].append(widths)
        fm_results['memory'].append(mems)

    fm_results['ci_width_mean'] = np.mean(fm_results['ci_width'], axis=0).tolist()
    fm_results['ci_width_std'] = np.std(fm_results['ci_width'], axis=0).tolist()
    fm_results['memory_mean'] = np.mean(fm_results['memory'], axis=0).tolist()
    all_results['fullmemory'] = fm_results

    # Run AdaQuantCS for each k
    aq_results = {}
    for k in K_VALUES:
        print(f"\nRunning AdaQuantCS with k={k}...")
        k_res = {'ci_width': [], 'memory': [], 'covered': []}

        for seed in SEEDS:
            stream = generate_stream(DIST, N, seed)
            aq = AdaQuantCS(p=P, k=k, alpha=ALPHA, initial_range=init_range)

            checkpoint_set = set(CHECKPOINTS_1M)
            widths = []
            mems = []
            for i, x in enumerate(stream):
                aq.update(x)
                t = i + 1
                if t in checkpoint_set:
                    lo, hi = aq.get_ci()
                    widths.append(hi - lo)
                    mems.append(aq.memory_usage())

            k_res['ci_width'].append(widths)
            k_res['memory'].append(mems)
            final_lo, final_hi = aq.get_ci()
            k_res['covered'].append(final_lo <= tq <= final_hi)

        k_res['ci_width_mean'] = np.mean(k_res['ci_width'], axis=0).tolist()
        k_res['ci_width_std'] = np.std(k_res['ci_width'], axis=0).tolist()
        k_res['memory_mean'] = np.mean(k_res['memory'], axis=0).tolist()

        # Width ratio vs full memory
        width_ratio = (np.array(k_res['ci_width_mean']) / np.array(fm_results['ci_width_mean'])).tolist()
        k_res['width_ratio_vs_fullmem'] = width_ratio

        # Memory ratio
        mem_ratio = (np.array(fm_results['memory_mean']) / np.array(k_res['memory_mean'])).tolist()
        k_res['memory_ratio_vs_fullmem'] = mem_ratio

        print(f"  Final width ratio: {width_ratio[-1]:.2f}, Memory ratio: {mem_ratio[-1]:.0f}")
        aq_results[f'k={k}'] = k_res

    all_results['adaquantcs'] = aq_results

    elapsed = time.time() - start_time
    all_results['runtime_seconds'] = elapsed
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    outpath = os.path.join(os.path.dirname(__file__), 'results.json')
    save_results(all_results, outpath)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
