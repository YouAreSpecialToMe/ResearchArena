"""Experiment 3: Comparison with sketch+bootstrap focusing on peeking penalty."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from src.adaquantcs import AdaQuantCS
from src.baselines import SketchBootstrapCI
from src.utils import (
    DISTRIBUTIONS, QUANTILE_LEVELS, SEEDS,
    generate_stream, true_quantile, get_initial_range,
    save_results,
)

N = 100_000
ALPHA = 0.05
K = 50
N_TRIALS = 30


def run_peeking_experiment(dist_name, p, n_peeks, n_trials=N_TRIALS):
    """Measure anytime coverage as a function of number of peeks."""
    tq = true_quantile(dist_name, p)
    init_range = get_initial_range(dist_name)

    peek_times = set(np.linspace(100, N, n_peeks, dtype=int).tolist())

    aq_coverage_count = 0
    bs_coverage_count = 0

    for trial in range(n_trials):
        stream = generate_stream(dist_name, N, trial)

        aq = AdaQuantCS(p=p, k=K, alpha=ALPHA, initial_range=init_range)
        # Use fewer bootstrap samples for speed
        bs = SketchBootstrapCI(p=p, alpha=ALPHA, reservoir_size=1000, n_bootstrap=200)

        aq_always_covered = True
        bs_always_covered = True

        for i, x in enumerate(stream):
            aq.update(x)
            bs.update(x)
            t = i + 1

            if t in peek_times:
                lo_aq, hi_aq = aq.get_ci()
                if not (lo_aq <= tq <= hi_aq):
                    aq_always_covered = False

                lo_bs, hi_bs = bs.get_ci()
                if not (lo_bs <= tq <= hi_bs):
                    bs_always_covered = False

        aq_coverage_count += aq_always_covered
        bs_coverage_count += bs_always_covered

    return aq_coverage_count / n_trials, bs_coverage_count / n_trials


def main():
    print("=" * 60)
    print("Experiment 3: Comparison & Peeking Penalty")
    print("=" * 60)

    start_time = time.time()
    all_results = {}

    # Part 1: Standard comparison on all distributions
    print("\nPart 1: Standard comparison across distributions...")
    for dist_name in DISTRIBUTIONS:
        for p in QUANTILE_LEVELS:
            key = f"{dist_name}_p{p}"
            print(f"  Running {key}...")
            init_range = get_initial_range(dist_name)

            method_results = {
                'adaquantcs': {'ci_width': [], 'time': []},
                'bootstrap': {'ci_width': [], 'time': []},
            }

            for seed in SEEDS:
                stream = generate_stream(dist_name, N, seed)
                aq = AdaQuantCS(p=p, k=K, alpha=ALPHA, initial_range=init_range)
                bs = SketchBootstrapCI(p=p, alpha=ALPHA)

                t0 = time.time()
                for x in stream:
                    aq.update(x)
                    bs.update(x)
                elapsed = time.time() - t0

                lo_aq, hi_aq = aq.get_ci()
                lo_bs, hi_bs = bs.get_ci()

                method_results['adaquantcs']['ci_width'].append(hi_aq - lo_aq)
                method_results['adaquantcs']['time'].append(elapsed)
                method_results['bootstrap']['ci_width'].append(hi_bs - lo_bs)
                method_results['bootstrap']['time'].append(elapsed)

            for m in method_results:
                method_results[m]['ci_width_mean'] = float(np.mean(method_results[m]['ci_width']))
                method_results[m]['ci_width_std'] = float(np.std(method_results[m]['ci_width']))
                method_results[m]['time_mean'] = float(np.mean(method_results[m]['time']))

            all_results[key] = method_results

    # Part 2: Peeking penalty analysis
    print("\nPart 2: Peeking penalty analysis...")
    peek_counts = [1, 5, 10, 50, 100, 500]
    peeking_results = {}

    for dist_name, p in [('gaussian', 0.5), ('student_t', 0.95)]:
        key = f"{dist_name}_p{p}"
        print(f"  Peeking analysis for {key}...")
        aq_covs = []
        bs_covs = []

        for n_peeks in peek_counts:
            print(f"    n_peeks={n_peeks}...", end=" ", flush=True)
            aq_cov, bs_cov = run_peeking_experiment(dist_name, p, n_peeks)
            aq_covs.append(aq_cov)
            bs_covs.append(bs_cov)
            print(f"AQ={aq_cov:.3f}, BS={bs_cov:.3f}")

        peeking_results[key] = {
            'peek_counts': peek_counts,
            'adaquantcs_coverage': aq_covs,
            'bootstrap_coverage': bs_covs,
        }

    all_results['peeking_analysis'] = peeking_results

    elapsed = time.time() - start_time
    all_results['runtime_seconds'] = elapsed
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    outpath = os.path.join(os.path.dirname(__file__), 'results.json')
    save_results(all_results, outpath)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
