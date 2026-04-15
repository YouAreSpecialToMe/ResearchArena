"""Experiment 5: Scalability analysis."""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.targets import GaussianTarget
from src.samplers import rwmh_sample, coupled_rwmh, compute_unbiased_estimator
from src.confidence_sequences import (
    clt_confidence_interval, BatchMeansCS, CouplingCS, MartingaleCS
)


def run_dimension(d, seed, n_samples=50000, alpha=0.05, n_reps=50):
    rng_base = np.random.default_rng(seed)
    target = GaussianTarget(d)
    h = 2.38 / np.sqrt(d) if d >= 2 else 1.0

    chain_times = []
    bmcs_times = []
    mdcs_times = []
    clt_times = []
    bmcs_widths = []
    mdcs_widths = []
    clt_widths = []
    cbcs_meeting_times = []

    for rep in range(n_reps):
        rep_rng = np.random.default_rng(seed * 10000 + rep)
        x0 = rep_rng.standard_normal(d)

        # Chain timing
        t0 = time.time()
        chain, _ = rwmh_sample(target, x0, n_samples, h, burn_in=500, rng=rep_rng)
        chain_time = time.time() - t0
        chain_times.append(chain_time)

        f_values = chain[:, 0]

        # CLT
        t0 = time.time()
        _, hw = clt_confidence_interval(f_values, alpha)
        clt_times.append(time.time() - t0)
        clt_widths.append(float(hw))

        # BMCS
        t0 = time.time()
        bmcs = BatchMeansCS(alpha=alpha)
        bmcs.update(f_values)
        _, hw = bmcs.get_cs()
        bmcs_times.append(time.time() - t0)
        bmcs_widths.append(float(hw))

        # MDCS
        t0 = time.time()
        mdcs = MartingaleCS(alpha=alpha)
        for v in f_values[:5000]:  # Only first 5000 to keep timing manageable
            mdcs.update(v)
        _, hw = mdcs.get_cs()
        mdcs_time = time.time() - t0
        mdcs_times.append(mdcs_time * (n_samples / 5000))  # Extrapolate
        mdcs_widths.append(float(hw))

        # CBCS meeting time (just 3 pairs per rep)
        if rep < 10:
            for pair_idx in range(3):
                pair_rng = np.random.default_rng(seed * 100000 + rep * 100 + pair_idx)
                x0_c = pair_rng.standard_normal(d)
                y0_c = pair_rng.standard_normal(d)
                _, _, tau = coupled_rwmh(
                    target, x0_c, y0_c, h, max_iter=5000, n_extra=0, rng=pair_rng
                )
                cbcs_meeting_times.append(tau)

    return {
        'd': d,
        'seed': seed,
        'n_reps': n_reps,
        'chain_time_per_iter': float(np.mean(chain_times) / n_samples),
        'clt_overhead': float(np.mean(clt_times)),
        'bmcs_overhead': float(np.mean(bmcs_times)),
        'mdcs_overhead': float(np.mean(mdcs_times)),
        'clt_width': {'mean': float(np.mean(clt_widths)), 'std': float(np.std(clt_widths))},
        'bmcs_width': {'mean': float(np.mean(bmcs_widths)), 'std': float(np.std(bmcs_widths))},
        'mdcs_width': {'mean': float(np.mean(mdcs_widths)), 'std': float(np.std(mdcs_widths))},
        'cbcs_meeting_time': {
            'median': float(np.median(cbcs_meeting_times)) if cbcs_meeting_times else None,
            'mean': float(np.mean(cbcs_meeting_times)) if cbcs_meeting_times else None,
        },
        'overhead_ratio_bmcs': float(np.mean(bmcs_times) / np.mean(chain_times)),
        'overhead_ratio_mdcs': float(np.mean(mdcs_times) / np.mean(chain_times)),
    }


def main():
    start_time = time.time()
    dimensions = [2, 5, 10, 20, 50, 100, 200]
    seeds = [42, 123, 456]

    all_results = []
    for d in dimensions:
        for seed in seeds:
            print(f"Running d={d}, seed={seed}...", flush=True)
            t0 = time.time()
            r = run_dimension(d, seed)
            all_results.append(r)
            print(f"  Done in {time.time()-t0:.1f}s, "
                  f"chain_time/iter={r['chain_time_per_iter']*1e6:.1f}us, "
                  f"bmcs_overhead_ratio={r['overhead_ratio_bmcs']:.3f}", flush=True)

    # Aggregate by dimension
    summary = {}
    for d in dimensions:
        d_results = [r for r in all_results if r['d'] == d]
        summary[f"d{d}"] = {
            'chain_time_per_iter_us': float(np.mean([r['chain_time_per_iter'] for r in d_results]) * 1e6),
            'bmcs_overhead_ratio': float(np.mean([r['overhead_ratio_bmcs'] for r in d_results])),
            'mdcs_overhead_ratio': float(np.mean([r['overhead_ratio_mdcs'] for r in d_results])),
            'clt_width': float(np.mean([r['clt_width']['mean'] for r in d_results])),
            'bmcs_width': float(np.mean([r['bmcs_width']['mean'] for r in d_results])),
            'mdcs_width': float(np.mean([r['mdcs_width']['mean'] for r in d_results])),
            'cbcs_meeting_time': float(np.mean([
                r['cbcs_meeting_time']['median'] for r in d_results
                if r['cbcs_meeting_time']['median'] is not None
            ])) if any(r['cbcs_meeting_time']['median'] is not None for r in d_results) else None,
        }

    elapsed = time.time() - start_time
    output = {
        'experiment': 'exp5_scalability',
        'config': {'dimensions': dimensions, 'seeds': seeds, 'n_samples': 50000},
        'summary': summary,
        'details': all_results,
        'runtime_minutes': elapsed / 60,
    }

    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExperiment 5 completed in {elapsed/60:.1f} minutes")
    print("\nScalability Summary:")
    print(f"{'d':>5} {'chain_us':>10} {'bmcs_ratio':>12} {'mdcs_ratio':>12} {'clt_w':>8} {'bmcs_w':>8} {'cbcs_tau':>10}")
    for d in dimensions:
        s = summary[f"d{d}"]
        print(f"{d:>5} {s['chain_time_per_iter_us']:>10.1f} "
              f"{s['bmcs_overhead_ratio']:>12.3f} {s['mdcs_overhead_ratio']:>12.3f} "
              f"{s['clt_width']:>8.4f} {s['bmcs_width']:>8.4f} "
              f"{s['cbcs_meeting_time'] if s['cbcs_meeting_time'] else 'N/A':>10}")


if __name__ == '__main__':
    main()
