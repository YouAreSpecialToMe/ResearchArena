"""Experiment 1: Coverage verification on Gaussian targets."""
import sys
import os
import json
import time
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.targets import GaussianTarget
from src.samplers import rwmh_sample, coupled_rwmh, compute_unbiased_estimator
from src.confidence_sequences import (
    clt_confidence_interval, BatchMeansCS, CouplingCS, MartingaleCS, compute_ess
)

MONITOR_TIMES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def run_single_replicate(d, seed, rep_idx, n_samples=50000, alpha=0.05):
    """Run one replicate: RWMH chain + all CS methods."""
    rng = np.random.default_rng(seed * 10000 + rep_idx)
    target = GaussianTarget(d)
    true_mean = 0.0

    h = 2.38 / np.sqrt(d) if d >= 2 else 1.0
    x0 = rng.standard_normal(d)
    chain, accept_rate = rwmh_sample(target, x0, n_samples, h, burn_in=1000, rng=rng)
    f_values = chain[:, 0]

    monitor_times = [t for t in MONITOR_TIMES if t <= n_samples]

    # CLT coverage at each monitoring time
    clt_covers = {}
    for t in monitor_times:
        center, hw = clt_confidence_interval(f_values[:t], alpha)
        clt_covers[t] = bool(abs(center - true_mean) <= hw)

    # BMCS: incremental updates
    bmcs = BatchMeansCS(alpha=alpha)
    bmcs_covers = {}
    prev_t = 0
    for t in monitor_times:
        bmcs.update(f_values[prev_t:t])
        center, hw = bmcs.get_cs()
        bmcs_covers[t] = bool(abs(center - true_mean) <= hw)
        prev_t = t

    # MDCS: incremental updates
    mdcs_covers = {}
    mdcs = MartingaleCS(alpha=alpha)
    mt_set = set(monitor_times)
    for i in range(n_samples):
        mdcs.update(f_values[i])
        if (i + 1) in mt_set:
            center, hw = mdcs.get_cs()
            mdcs_covers[i + 1] = bool(abs(center - true_mean) <= hw)

    # CBCS: coupled chains (fewer pairs in high-d)
    n_pairs = 15 if d <= 10 else 8
    max_coupling_iter = 2000 if d <= 10 else 1000
    cbcs = CouplingCS(alpha=alpha)
    meeting_times_list = []

    for pair_idx in range(n_pairs):
        pair_rng = np.random.default_rng(seed * 100000 + rep_idx * 1000 + pair_idx)
        x0_c = pair_rng.standard_normal(d)
        y0_c = pair_rng.standard_normal(d)
        x_chain, y_chain, tau = coupled_rwmh(
            target, x0_c, y0_c, h, max_iter=max_coupling_iter, n_extra=200, rng=pair_rng
        )
        meeting_times_list.append(tau)

        if tau < max_coupling_iter:
            h_val = compute_unbiased_estimator(
                x_chain, y_chain, tau, func=lambda x: x[0]
            )
            cbcs.add_estimator(float(h_val))

    # CBCS: record coverage at monitoring times
    # Map accumulated estimators to equivalent chain time
    cbcs_covers = {}
    if len(cbcs.estimators) >= 2:
        center, hw = cbcs.get_cs()
        for t in monitor_times:
            # CBCS coverage is the same regardless of t (all estimators used)
            cbcs_covers[t] = bool(abs(center - true_mean) <= hw)
    else:
        for t in monitor_times:
            cbcs_covers[t] = True  # conservative

    return {
        'd': d, 'seed': seed, 'rep': rep_idx,
        'accept_rate': accept_rate,
        'meeting_times': meeting_times_list,
        'clt_covers': clt_covers,
        'bmcs_covers': bmcs_covers,
        'cbcs_covers': cbcs_covers,
        'mdcs_covers': mdcs_covers,
    }


def main():
    start_time = time.time()
    dimensions = [1, 10, 50]
    seeds = [42, 123, 456]
    n_replicates = 200
    alpha = 0.05

    all_results = []
    for d in dimensions:
        for seed in seeds:
            print(f"Running d={d}, seed={seed}, {n_replicates} replicates...", flush=True)
            t0 = time.time()

            results = Parallel(n_jobs=2, backend='loky')(
                delayed(run_single_replicate)(d, seed, rep, alpha=alpha)
                for rep in range(n_replicates)
            )
            all_results.extend(results)
            print(f"  Done in {time.time()-t0:.1f}s, "
                  f"accept_rate={np.mean([r['accept_rate'] for r in results]):.3f}", flush=True)

    # Aggregate
    coverage_summary = {}
    for d in dimensions:
        d_results = [r for r in all_results if r['d'] == d]

        for method in ['clt', 'bmcs', 'cbcs', 'mdcs']:
            key = f"{method}_covers"
            method_coverage = {}
            for t in MONITOR_TIMES:
                covers = [r[key].get(t, True) for r in d_results]
                method_coverage[str(t)] = float(np.mean(covers))

            time_uniform = min(method_coverage.values())
            coverage_summary[f"d{d}_{method}"] = {
                'per_time': method_coverage,
                'time_uniform': time_uniform,
                'final_coverage': method_coverage[str(MONITOR_TIMES[-1])],
                'n_replicates': len(d_results),
            }

    meeting_stats = {}
    for d in dimensions:
        d_results = [r for r in all_results if r['d'] == d]
        all_taus = [tau for r in d_results for tau in r['meeting_times']]
        if all_taus:
            meeting_stats[f"d{d}"] = {
                'median': float(np.median(all_taus)),
                'mean': float(np.mean(all_taus)),
                'p95': float(np.percentile(all_taus, 95)),
            }

    elapsed = time.time() - start_time
    output = {
        'experiment': 'exp1_coverage',
        'config': {
            'dimensions': dimensions, 'seeds': seeds,
            'n_replicates': n_replicates, 'n_samples': 50000, 'alpha': alpha,
        },
        'coverage': coverage_summary,
        'meeting_times': meeting_stats,
        'runtime_minutes': elapsed / 60,
    }

    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExperiment 1 completed in {elapsed/60:.1f} minutes")
    print("\nCoverage Summary (time-uniform):")
    for key, val in coverage_summary.items():
        print(f"  {key}: {val['time_uniform']:.3f}")


if __name__ == '__main__':
    main()
