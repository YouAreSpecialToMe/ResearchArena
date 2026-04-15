"""Experiment 5: Multiple quantile tracking."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from src.adaquantcs import AdaQuantCS
from src.utils import (
    ALL_QUANTILE_LEVELS, SEEDS,
    generate_stream, true_quantile, get_initial_range,
    check_anytime_coverage, save_results,
)

N = 100_000
ALPHA = 0.05
K_INDEPENDENT = 50
K_SHARED = 200
CHECKPOINTS = [100, 500, 1000, 5000, 10000, 50000, 100000]


def run_independent_grids(dist_name, seed):
    """Run independent AdaQuantCS instances for each quantile level."""
    stream = generate_stream(dist_name, N, seed)
    init_range = get_initial_range(dist_name)

    trackers = {}
    for p in ALL_QUANTILE_LEVELS:
        trackers[p] = AdaQuantCS(p=p, k=K_INDEPENDENT, alpha=ALPHA, initial_range=init_range)

    checkpoint_set = set(CHECKPOINTS)
    results = {p: {'ci_lower': [], 'ci_upper': [], 'ci_width': []} for p in ALL_QUANTILE_LEVELS}

    for i, x in enumerate(stream):
        for p in ALL_QUANTILE_LEVELS:
            trackers[p].update(x)
        t = i + 1
        if t in checkpoint_set:
            for p in ALL_QUANTILE_LEVELS:
                lo, hi = trackers[p].get_ci()
                results[p]['ci_lower'].append(lo)
                results[p]['ci_upper'].append(hi)
                results[p]['ci_width'].append(hi - lo)

    total_memory = sum(trackers[p].memory_usage() for p in ALL_QUANTILE_LEVELS)
    return results, total_memory


def run_shared_grid(dist_name, seed):
    """Run a single shared grid covering all quantile levels."""
    stream = generate_stream(dist_name, N, seed)
    init_range = get_initial_range(dist_name)

    # Use a wider range and more thresholds
    lo_range, hi_range = init_range
    margin = (hi_range - lo_range) * 0.3
    shared_range = (lo_range - margin, hi_range + margin)

    # Create one tracker per quantile but with shared grid concept
    # In practice, each tracker still needs its own counters, but they can share thresholds
    trackers = {}
    for p in ALL_QUANTILE_LEVELS:
        trackers[p] = AdaQuantCS(p=p, k=K_SHARED, alpha=ALPHA,
                                  initial_range=shared_range, grid_type='fixed')

    checkpoint_set = set(CHECKPOINTS)
    results = {p: {'ci_lower': [], 'ci_upper': [], 'ci_width': []} for p in ALL_QUANTILE_LEVELS}

    for i, x in enumerate(stream):
        for p in ALL_QUANTILE_LEVELS:
            trackers[p].update(x)
        t = i + 1
        if t in checkpoint_set:
            for p in ALL_QUANTILE_LEVELS:
                lo, hi = trackers[p].get_ci()
                results[p]['ci_lower'].append(lo)
                results[p]['ci_upper'].append(hi)
                results[p]['ci_width'].append(hi - lo)

    total_memory = sum(trackers[p].memory_usage() for p in ALL_QUANTILE_LEVELS)
    return results, total_memory


def main():
    print("=" * 60)
    print("Experiment 5: Multiple Quantile Tracking")
    print("=" * 60)

    start_time = time.time()
    all_results = {}

    for dist_name in ['gaussian', 'student_t']:
        print(f"\nDistribution: {dist_name}")
        dist_results = {'independent': [], 'shared': []}

        for seed in SEEDS:
            print(f"  Seed {seed}...")

            # Independent grids
            ind_res, ind_mem = run_independent_grids(dist_name, seed)
            dist_results['independent'].append({
                'results': {str(p): ind_res[p] for p in ALL_QUANTILE_LEVELS},
                'total_memory': ind_mem,
            })

            # Shared grid
            shr_res, shr_mem = run_shared_grid(dist_name, seed)
            dist_results['shared'].append({
                'results': {str(p): shr_res[p] for p in ALL_QUANTILE_LEVELS},
                'total_memory': shr_mem,
            })

        # Compute coverage and joint coverage
        for approach in ['independent', 'shared']:
            for seed_idx, seed_data in enumerate(dist_results[approach]):
                coverages = {}
                for p in ALL_QUANTILE_LEVELS:
                    tq = true_quantile(dist_name, p)
                    p_key = str(p)
                    covered = check_anytime_coverage(
                        seed_data['results'][p_key]['ci_lower'],
                        seed_data['results'][p_key]['ci_upper'],
                        tq,
                    )
                    coverages[p_key] = covered
                seed_data['per_quantile_coverage'] = coverages
                seed_data['joint_coverage'] = all(coverages.values())

            joint_cov = np.mean([s['joint_coverage'] for s in dist_results[approach]])
            print(f"  {approach}: joint_coverage={joint_cov:.3f}, "
                  f"mem={np.mean([s['total_memory'] for s in dist_results[approach]]):.0f}")

        # Compute Bonferroni-corrected version
        for seed_idx in range(len(SEEDS)):
            seed = SEEDS[seed_idx]
            stream = generate_stream(dist_name, N, seed)
            init_range = get_initial_range(dist_name)
            alpha_bonf = ALPHA / len(ALL_QUANTILE_LEVELS)

            trackers = {p: AdaQuantCS(p=p, k=K_INDEPENDENT, alpha=alpha_bonf,
                                       initial_range=init_range)
                        for p in ALL_QUANTILE_LEVELS}
            for x in stream:
                for p in ALL_QUANTILE_LEVELS:
                    trackers[p].update(x)

            bonf_covered = {}
            for p in ALL_QUANTILE_LEVELS:
                tq = true_quantile(dist_name, p)
                lo, hi = trackers[p].get_ci()
                bonf_covered[str(p)] = lo <= tq <= hi

            if 'bonferroni' not in dist_results:
                dist_results['bonferroni'] = []
            dist_results['bonferroni'].append({
                'per_quantile_coverage': bonf_covered,
                'joint_coverage': all(bonf_covered.values()),
            })

        if 'bonferroni' in dist_results:
            bonf_joint = np.mean([s['joint_coverage'] for s in dist_results['bonferroni']])
            print(f"  bonferroni: joint_coverage={bonf_joint:.3f}")

        all_results[dist_name] = dist_results

    all_results['quantile_levels'] = ALL_QUANTILE_LEVELS
    all_results['checkpoints'] = CHECKPOINTS
    elapsed = time.time() - start_time
    all_results['runtime_seconds'] = elapsed
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    outpath = os.path.join(os.path.dirname(__file__), 'results.json')
    save_results(all_results, outpath)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
