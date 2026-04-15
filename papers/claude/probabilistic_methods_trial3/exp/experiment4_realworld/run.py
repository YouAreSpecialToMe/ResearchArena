"""Experiment 4: Real-world data experiments (financial + network latency)."""

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


def run_financial_experiment():
    """Run on synthetic financial returns (since we can't download live data)."""
    print("\n  Financial returns experiment...")

    # Generate synthetic financial returns mimicking S&P 500
    # Use a GARCH-like model: returns ~ N(0.0003, sigma_t^2)
    # with sigma_t^2 = 0.0001 + 0.1*r_{t-1}^2 + 0.85*sigma_{t-1}^2
    n = 5000  # ~20 years of daily data
    results_by_seed = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        returns = np.zeros(n)
        sigma2 = 0.0002
        for t in range(n):
            returns[t] = rng.normal(0.0003, np.sqrt(sigma2))
            sigma2 = 0.0001 + 0.1 * returns[t]**2 + 0.85 * sigma2

        # Track VaR at p=0.05 (5th percentile)
        p = 0.05
        aq = AdaQuantCS(p=p, k=50, alpha=0.05, initial_range=(-0.15, 0.15))
        fm = FullMemoryQuantileCS(p=p, alpha=0.05)
        bs = SketchBootstrapCI(p=p, alpha=0.05, reservoir_size=500)

        checkpoints = [100, 250, 500, 1000, 2000, 3000, 4000, 5000]
        checkpoint_set = set(checkpoints)

        aq_cis = []
        fm_cis = []
        bs_cis = []

        for i, x in enumerate(returns):
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
    print("\n  Network latency experiment...")

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
    print("Experiment 4: Real-World Data Experiments")
    print("=" * 60)

    start_time = time.time()

    financial_results = run_financial_experiment()
    latency_results = run_latency_experiment()

    all_results = {
        'financial': financial_results,
        'latency': latency_results,
        'runtime_seconds': time.time() - start_time,
    }

    # Summary stats
    print("\n  Financial summary:")
    for method in ['aq', 'fm', 'bs']:
        widths = [r[f'{method}_width_final'] for r in financial_results]
        mems = [r[f'{method}_memory'] for r in financial_results]
        print(f"    {method}: width={np.mean(widths):.5f}±{np.std(widths):.5f}, mem={np.mean(mems):.0f}")

    print("\n  Latency summary:")
    for method in ['aq', 'fm', 'bs']:
        widths = [r[f'{method}_width_final'] for r in latency_results]
        mems = [r[f'{method}_memory'] for r in latency_results]
        print(f"    {method}: width={np.mean(widths):.2f}±{np.std(widths):.2f}, mem={np.mean(mems):.0f}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    outpath = os.path.join(os.path.dirname(__file__), 'results.json')
    save_results(all_results, outpath)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
