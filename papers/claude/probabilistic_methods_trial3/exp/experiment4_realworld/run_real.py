"""Experiment 4 rerun: Real-world data (actual Yahoo Finance + synthetic latency)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from src.adaquantcs import AdaQuantCS
from src.baselines import FullMemoryQuantileCS, SketchBootstrapCI
from src.utils import (
    generate_latency_stream, latency_true_quantile,
    save_results, SEEDS,
)


def run_financial_real():
    """Run on actual S&P 500 daily log returns from Yahoo Finance."""
    print("\n  Financial returns experiment (REAL Yahoo Finance data)...")

    returns = np.load(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sp500_returns.npy'))
    n = len(returns)
    print(f"  Loaded {n} daily log returns (S&P 500, 2000-2025)")

    # We don't know the true quantile for real data, but we can use the
    # full-sample empirical quantile as a reference
    p = 0.05  # VaR at 5th percentile
    empirical_q = np.percentile(returns, 100 * p)
    print(f"  Full-sample 5th percentile: {empirical_q:.6f}")

    checkpoints = [100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    checkpoints = [c for c in checkpoints if c <= n]
    checkpoint_set = set(checkpoints)

    results_by_seed = []

    # Run with different reservoir seeds but same data order
    for seed_idx, seed in enumerate(SEEDS):
        print(f"    Seed {seed}...")

        aq = AdaQuantCS(p=p, k=50, alpha=0.05, initial_range=(-0.15, 0.15))
        fm = FullMemoryQuantileCS(p=p, alpha=0.05)
        bs = SketchBootstrapCI(p=p, alpha=0.05, reservoir_size=500)

        aq_cis = []
        fm_cis = []
        bs_cis = []

        # Optionally shuffle the data for different seeds (to test robustness)
        # But seed 0 uses original order
        if seed_idx == 0:
            data = returns
        else:
            rng = np.random.default_rng(seed)
            data = rng.permutation(returns)

        for i, x in enumerate(data):
            aq.update(x)
            fm.update(x)
            bs.update(x)
            t = i + 1

            if t in checkpoint_set:
                aq_cis.append(aq.get_ci())
                fm_cis.append(fm.get_ci())
                bs_cis.append(bs.get_ci())

        results_by_seed.append({
            'checkpoints': checkpoints,
            'n_observations': n,
            'empirical_quantile': float(empirical_q),
            'aq_ci_lower': [c[0] for c in aq_cis],
            'aq_ci_upper': [c[1] for c in aq_cis],
            'fm_ci_lower': [c[0] for c in fm_cis],
            'fm_ci_upper': [c[1] for c in fm_cis],
            'bs_ci_lower': [c[0] for c in bs_cis],
            'bs_ci_upper': [c[1] for c in bs_cis],
            'aq_memory': aq.memory_usage(),
            'fm_memory': fm.memory_usage(),
            'bs_memory': bs.memory_usage(),
            'aq_width_final': aq_cis[-1][1] - aq_cis[-1][0],
            'fm_width_final': fm_cis[-1][1] - fm_cis[-1][0],
            'bs_width_final': bs_cis[-1][1] - bs_cis[-1][0],
        })

    return results_by_seed


def run_latency_experiment():
    """Run on synthetic network latency data, tracking p99."""
    print("\n  Network latency experiment (synthetic log-normal mixture)...")

    n = 500_000
    p = 0.99
    true_q = latency_true_quantile(p)
    print(f"    True p99 latency: {true_q:.2f}")

    checkpoints = [1000, 5000, 10000, 50000, 100000, 250000, 500000]
    checkpoint_set = set(checkpoints)

    results_by_seed = []

    for seed in SEEDS:
        print(f"    Seed {seed}...")
        stream = generate_latency_stream(n, seed)

        aq = AdaQuantCS(p=p, k=50, alpha=0.05, initial_range=(0.0, 500.0))
        fm = FullMemoryQuantileCS(p=p, alpha=0.05)
        bs = SketchBootstrapCI(p=p, alpha=0.05, reservoir_size=1000)

        aq_cis = []
        fm_cis = []
        bs_cis = []

        for i, x in enumerate(stream):
            aq.update(x)
            fm.update(x)
            bs.update(x)
            t = i + 1

            if t in checkpoint_set:
                aq_cis.append(aq.get_ci())
                fm_cis.append(fm.get_ci())
                bs_cis.append(bs.get_ci())

        results_by_seed.append({
            'checkpoints': checkpoints,
            'true_quantile': float(true_q),
            'aq_ci_lower': [c[0] for c in aq_cis],
            'aq_ci_upper': [c[1] for c in aq_cis],
            'fm_ci_lower': [c[0] for c in fm_cis],
            'fm_ci_upper': [c[1] for c in fm_cis],
            'bs_ci_lower': [c[0] for c in bs_cis],
            'bs_ci_upper': [c[1] for c in bs_cis],
            'aq_memory': aq.memory_usage(),
            'fm_memory': fm.memory_usage(),
            'bs_memory': bs.memory_usage(),
            'aq_covered': all(lo <= true_q <= hi for lo, hi in aq_cis),
            'fm_covered': all(lo <= true_q <= hi for lo, hi in fm_cis),
            'aq_width_final': aq_cis[-1][1] - aq_cis[-1][0],
            'fm_width_final': fm_cis[-1][1] - fm_cis[-1][0],
            'bs_width_final': bs_cis[-1][1] - bs_cis[-1][0],
        })

    return results_by_seed


def main():
    print("=" * 60)
    print("Experiment 4: Real-World Data (with actual Yahoo Finance data)")
    print("=" * 60)

    start_time = time.time()

    financial_results = run_financial_real()
    latency_results = run_latency_experiment()

    all_results = {
        'financial': financial_results,
        'latency': latency_results,
        'runtime_seconds': time.time() - start_time,
    }

    # Summary stats
    print("\n  Financial summary (real S&P 500 data):")
    for method in ['aq', 'fm', 'bs']:
        widths = [r[f'{method}_width_final'] for r in financial_results]
        mems = [r[f'{method}_memory'] for r in financial_results]
        print(f"    {method}: width={np.mean(widths):.5f}+/-{np.std(widths):.5f}, mem={np.mean(mems):.0f}")

    print("\n  Latency summary:")
    for method in ['aq', 'fm', 'bs']:
        widths = [r[f'{method}_width_final'] for r in latency_results]
        mems = [r[f'{method}_memory'] for r in latency_results]
        print(f"    {method}: width={np.mean(widths):.2f}+/-{np.std(widths):.2f}, mem={np.mean(mems):.0f}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    outpath = os.path.join(os.path.dirname(__file__), 'results_real.json')
    save_results(all_results, outpath)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
