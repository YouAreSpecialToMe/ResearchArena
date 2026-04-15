"""Experiment 4: Sequential stopping efficiency."""
import sys
import os
import json
import time
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.targets import GaussianTarget
from src.samplers import rwmh_sample
from src.confidence_sequences import (
    clt_confidence_interval, fwsr_stopping_time, BatchMeansCS, compute_ess
)


def run_stopping_replicate(d, seed, rep_idx, epsilon, n_max=100000, alpha=0.05):
    rng = np.random.default_rng(seed * 10000 + rep_idx)
    target = GaussianTarget(d)
    true_mean = 0.0

    h = 2.38 / np.sqrt(d) if d >= 2 else 1.0
    x0 = rng.standard_normal(d)
    chain, accept_rate = rwmh_sample(target, x0, n_max, h, burn_in=1000, rng=rng)
    f_values = chain[:, 0]

    # ESS for oracle
    ess = compute_ess(f_values)
    sigma2_asy = np.var(f_values) * len(f_values) / ess
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    oracle_stop = min(int(np.ceil(sigma2_asy * z**2 / epsilon**2)), n_max)

    # FWSR stopping
    fwsr_stop = fwsr_stopping_time(f_values, epsilon, alpha)

    # BMCS stopping
    bmcs = BatchMeansCS(alpha=alpha)
    bmcs_stop = n_max
    check_interval = 500
    for i in range(0, n_max, check_interval):
        end = min(i + check_interval, n_max)
        bmcs.update(f_values[i:end])
        c, hw = bmcs.get_cs()
        if hw < epsilon:
            bmcs_stop = end
            break

    # Coverage at stopping
    def check_coverage(stop_t, use_bmcs=False):
        if stop_t >= n_max:
            return None
        if use_bmcs:
            b = BatchMeansCS(alpha=alpha)
            b.update(f_values[:stop_t])
            c, hw = b.get_cs()
        else:
            c, hw = clt_confidence_interval(f_values[:stop_t], alpha)
        return bool(abs(c - true_mean) <= hw)

    return {
        'd': d, 'seed': seed, 'rep': rep_idx, 'epsilon': epsilon,
        'oracle_stop': oracle_stop,
        'fwsr_stop': fwsr_stop,
        'bmcs_stop': bmcs_stop,
        'fwsr_ratio': fwsr_stop / max(1, oracle_stop),
        'bmcs_ratio': bmcs_stop / max(1, oracle_stop),
        'fwsr_coverage': check_coverage(fwsr_stop, False),
        'bmcs_coverage': check_coverage(bmcs_stop, True),
    }


def main():
    start_time = time.time()
    seeds = [42, 123, 456]
    epsilons = [0.1, 0.05]
    n_replicates = 80

    all_results = []
    for d in [1, 10, 50]:
        for eps in epsilons:
            for seed in seeds:
                print(f"Gaussian d={d}, eps={eps}, seed={seed}...", flush=True)
                t0 = time.time()
                results = Parallel(n_jobs=2, backend='loky')(
                    delayed(run_stopping_replicate)(d, seed, rep, eps)
                    for rep in range(n_replicates)
                )
                all_results.extend(results)
                print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Aggregate
    summary = {}
    for d in [1, 10, 50]:
        for eps in epsilons:
            key = f"gaussian_d{d}_eps{eps}"
            t_results = [r for r in all_results if r['d'] == d and r['epsilon'] == eps]
            if not t_results:
                continue

            fwsr_stops = [r['fwsr_stop'] for r in t_results]
            bmcs_stops = [r['bmcs_stop'] for r in t_results]
            fwsr_covs = [r['fwsr_coverage'] for r in t_results if r['fwsr_coverage'] is not None]
            bmcs_covs = [r['bmcs_coverage'] for r in t_results if r['bmcs_coverage'] is not None]

            summary[key] = {
                'n': len(t_results),
                'fwsr_mean_stop': float(np.mean(fwsr_stops)),
                'fwsr_std_stop': float(np.std(fwsr_stops)),
                'bmcs_mean_stop': float(np.mean(bmcs_stops)),
                'bmcs_std_stop': float(np.std(bmcs_stops)),
                'fwsr_mean_ratio': float(np.mean([r['fwsr_ratio'] for r in t_results])),
                'bmcs_mean_ratio': float(np.mean([r['bmcs_ratio'] for r in t_results])),
                'fwsr_coverage': float(np.mean(fwsr_covs)) if fwsr_covs else None,
                'bmcs_coverage': float(np.mean(bmcs_covs)) if bmcs_covs else None,
                'savings_bmcs': float(
                    (np.mean(fwsr_stops) - np.mean(bmcs_stops)) / np.mean(fwsr_stops) * 100
                ),
            }

    elapsed = time.time() - start_time
    output = {
        'experiment': 'exp4_stopping',
        'config': {'seeds': seeds, 'epsilons': epsilons, 'n_replicates': n_replicates},
        'summary': summary,
        'runtime_minutes': elapsed / 60,
    }

    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExperiment 4 completed in {elapsed/60:.1f} minutes")
    for key, val in summary.items():
        print(f"\n{key}:")
        print(f"  FWSR: stop={val['fwsr_mean_stop']:.0f}+/-{val['fwsr_std_stop']:.0f}, "
              f"ratio={val['fwsr_mean_ratio']:.2f}, cov={val['fwsr_coverage']}")
        print(f"  BMCS: stop={val['bmcs_mean_stop']:.0f}+/-{val['bmcs_std_stop']:.0f}, "
              f"ratio={val['bmcs_mean_ratio']:.2f}, cov={val['bmcs_coverage']}, "
              f"savings={val['savings_bmcs']:.1f}%")


if __name__ == '__main__':
    main()
