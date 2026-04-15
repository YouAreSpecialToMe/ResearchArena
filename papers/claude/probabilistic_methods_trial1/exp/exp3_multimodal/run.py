"""Experiment 3: Robustness under poor mixing (multimodal targets)."""
import sys
import os
import json
import time
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.targets import MixtureGaussianTarget
from src.samplers import rwmh_sample
from src.confidence_sequences import (
    clt_confidence_interval, BatchMeansCS, MartingaleCS, compute_ess, compute_iat
)


def count_mode_switches(chain_coord, threshold=0.0):
    """Count transitions across the mode boundary."""
    signs = np.sign(chain_coord)
    return int(np.sum(signs[1:] != signs[:-1]))


def run_single_replicate(d, separation, seed, rep_idx, n_samples=50000, alpha=0.05):
    rng = np.random.default_rng(seed * 10000 + rep_idx)
    target = MixtureGaussianTarget(d, separation)
    true_mean = 0.0

    h = 2.38 / np.sqrt(d)
    x0 = rng.standard_normal(d)
    chain, accept_rate = rwmh_sample(target, x0, n_samples, h, burn_in=2000, rng=rng)
    f_values = chain[:, 0]

    monitor_times = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    monitor_times = [t for t in monitor_times if t <= n_samples]

    # CLT
    clt_covers = {}
    for t in monitor_times:
        c, hw = clt_confidence_interval(f_values[:t], alpha)
        clt_covers[t] = bool(abs(c - true_mean) <= hw)

    # BMCS
    bmcs = BatchMeansCS(alpha=alpha)
    bmcs_covers = {}
    prev_t = 0
    for t in monitor_times:
        bmcs.update(f_values[prev_t:t])
        c, hw = bmcs.get_cs()
        bmcs_covers[t] = bool(abs(c - true_mean) <= hw)
        prev_t = t

    # MDCS
    mdcs = MartingaleCS(alpha=alpha)
    mdcs_covers = {}
    mt_set = set(monitor_times)
    for i in range(n_samples):
        mdcs.update(f_values[i])
        if (i + 1) in mt_set:
            c, hw = mdcs.get_cs()
            mdcs_covers[i + 1] = bool(abs(c - true_mean) <= hw)

    n_switches = count_mode_switches(f_values)
    iat = float(compute_iat(f_values))

    return {
        'd': d, 'separation': separation, 'seed': seed, 'rep': rep_idx,
        'accept_rate': accept_rate,
        'n_switches': n_switches,
        'iat': iat,
        'clt_covers': clt_covers,
        'bmcs_covers': bmcs_covers,
        'mdcs_covers': mdcs_covers,
    }


def main():
    start_time = time.time()
    d = 5
    separations = [1.0, 2.0, 3.0, 4.0]
    seeds = [42, 123, 456]
    n_replicates = 100

    all_results = []
    monitor_times = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    for sep in separations:
        for seed in seeds:
            print(f"Running sep={sep}, seed={seed}, {n_replicates} replicates...", flush=True)
            t0 = time.time()
            results = Parallel(n_jobs=2, backend='loky')(
                delayed(run_single_replicate)(d, sep, seed, rep)
                for rep in range(n_replicates)
            )
            all_results.extend(results)
            avg_switches = np.mean([r['n_switches'] for r in results])
            print(f"  Done in {time.time()-t0:.1f}s, avg_switches={avg_switches:.1f}", flush=True)

    # Aggregate
    coverage_summary = {}
    mixing_summary = {}

    for sep in separations:
        sep_results = [r for r in all_results if r['separation'] == sep]
        n_total = len(sep_results)

        for method in ['clt', 'bmcs', 'mdcs']:
            key = f"{method}_covers"
            method_coverage = {}
            for t in monitor_times:
                covers = [r[key].get(t, True) for r in sep_results]
                method_coverage[str(t)] = float(np.mean(covers))

            time_uniform = min(method_coverage.values())
            coverage_summary[f"sep{sep}_{method}"] = {
                'per_time': method_coverage,
                'time_uniform': time_uniform,
                'n_replicates': n_total,
            }

        mixing_summary[f"sep{sep}"] = {
            'mean_switches': float(np.mean([r['n_switches'] for r in sep_results])),
            'mean_iat': float(np.mean([r['iat'] for r in sep_results])),
            'median_iat': float(np.median([r['iat'] for r in sep_results])),
        }

    elapsed = time.time() - start_time
    output = {
        'experiment': 'exp3_multimodal',
        'config': {
            'd': d, 'separations': separations, 'seeds': seeds,
            'n_replicates': n_replicates, 'n_samples': 100000, 'alpha': 0.05,
        },
        'coverage': coverage_summary,
        'mixing': mixing_summary,
        'runtime_minutes': elapsed / 60,
    }

    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExperiment 3 completed in {elapsed/60:.1f} minutes")
    print("\nCoverage (time-uniform) by separation:")
    for sep in separations:
        print(f"\n  sep={sep} (switches={mixing_summary[f'sep{sep}']['mean_switches']:.0f}, "
              f"IAT={mixing_summary[f'sep{sep}']['mean_iat']:.0f}):")
        for method in ['clt', 'bmcs', 'mdcs']:
            print(f"    {method}: {coverage_summary[f'sep{sep}_{method}']['time_uniform']:.3f}")


if __name__ == '__main__':
    main()
