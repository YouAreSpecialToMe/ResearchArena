"""Experiment 2: Bayesian logistic regression on UCI datasets."""
import sys
import os
import json
import time
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.targets import BayesianLogisticRegression
from src.samplers import rwmh_sample, coupled_rwmh, compute_unbiased_estimator
from src.confidence_sequences import (
    clt_confidence_interval, fwsr_stopping_time,
    BatchMeansCS, CouplingCS, MartingaleCS, compute_ess
)


def load_dataset(name, data_dir):
    """Load and preprocess dataset."""
    fpath = os.path.join(data_dir, f'{name}.npz')
    if os.path.exists(fpath):
        data = np.load(fpath)
        return data['X'], data['y']

    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    if name == 'german_credit':
        data = fetch_openml(data_id=31, as_frame=True)
        df = data.data
        y = (data.target == '1').astype(float).values
    elif name == 'ionosphere':
        data = fetch_openml(data_id=59, as_frame=True)
        df = data.data
        y = (data.target == 'g').astype(float).values
    elif name == 'pima':
        data = fetch_openml(data_id=37, as_frame=True)
        df = data.data
        y = (data.target == '1').astype(float).values
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Encode categorical features
    X_parts = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            X_parts.append(le.fit_transform(df[col].astype(str)).astype(float))
        else:
            X_parts.append(df[col].values.astype(float))
    X = np.column_stack(X_parts)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    # Standardize
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-10] = 1.0
    X = (X - mu) / std

    # Add intercept
    X = np.hstack([np.ones((len(X), 1)), X])

    os.makedirs(data_dir, exist_ok=True)
    np.savez(fpath, X=X, y=y)
    return X, y


def compute_reference(target, d, n_ref=500000, seed=999):
    """Compute reference posterior mean via long MCMC run."""
    rng = np.random.default_rng(seed)
    mode = target.find_mode()
    h = 2.38 / np.sqrt(d) * 0.5
    chain, _ = rwmh_sample(target, mode, n_ref, h, burn_in=50000, rng=rng)
    return np.mean(chain, axis=0)


def run_dataset_seed(dataset_name, data_dir, seed, n_samples=100000, alpha=0.05):
    """Run all methods on one dataset with one seed."""
    X, y = load_dataset(dataset_name, data_dir)
    target = BayesianLogisticRegression(X, y)
    d = target.d
    rng = np.random.default_rng(seed)

    # Reference mean
    ref_mean = compute_reference(target, d, n_ref=300000, seed=seed * 7)

    # Initialize at mode
    mode = target.find_mode()
    h = 2.38 / np.sqrt(d) * 0.3

    # Run chain
    chain, accept_rate = rwmh_sample(target, mode, n_samples, h, burn_in=5000, rng=rng)

    # Monitor at these times
    monitor_times = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    monitor_times = [t for t in monitor_times if t <= n_samples]

    # Evaluate on first 5 coordinates
    n_coords = min(5, d)
    results_per_coord = []

    for coord in range(n_coords):
        f_values = chain[:, coord]
        true_val = ref_mean[coord]

        # CLT
        clt_widths = {}
        clt_covers = {}
        for t in monitor_times:
            c, hw = clt_confidence_interval(f_values[:t], alpha)
            clt_widths[str(t)] = float(hw)
            clt_covers[str(t)] = bool(abs(c - true_val) <= hw)

        # FWSR stopping time
        fwsr_stop = fwsr_stopping_time(f_values, epsilon=0.05, alpha=alpha)

        # BMCS
        bmcs = BatchMeansCS(alpha=alpha)
        bmcs_widths = {}
        bmcs_covers = {}
        bmcs_stop = n_samples
        prev_t = 0
        for t in monitor_times:
            bmcs.update(f_values[prev_t:t])
            c, hw = bmcs.get_cs()
            bmcs_widths[str(t)] = float(hw)
            bmcs_covers[str(t)] = bool(abs(c - true_val) <= hw)
            if hw < 0.05 and bmcs_stop == n_samples:
                bmcs_stop = t
            prev_t = t

        # MDCS
        mdcs = MartingaleCS(alpha=alpha)
        mdcs_widths = {}
        mdcs_covers = {}
        mt_set = set(monitor_times)
        for i in range(n_samples):
            mdcs.update(f_values[i])
            if (i + 1) in mt_set:
                c, hw = mdcs.get_cs()
                mdcs_widths[str(i + 1)] = float(hw)
                mdcs_covers[str(i + 1)] = bool(abs(c - true_val) <= hw)

        results_per_coord.append({
            'coord': coord,
            'true_val': float(true_val),
            'clt_widths': clt_widths,
            'clt_covers': clt_covers,
            'bmcs_widths': bmcs_widths,
            'bmcs_covers': bmcs_covers,
            'mdcs_widths': mdcs_widths,
            'mdcs_covers': mdcs_covers,
            'fwsr_stop': fwsr_stop,
            'bmcs_stop': bmcs_stop,
        })

    # CBCS: run coupled pairs
    n_pairs = 30
    meeting_times_list = []
    cbcs = CouplingCS(alpha=alpha)

    for pair_idx in range(n_pairs):
        pair_rng = np.random.default_rng(seed * 100000 + pair_idx)
        x0_c = mode + 0.1 * pair_rng.standard_normal(d)
        y0_c = mode + 0.1 * pair_rng.standard_normal(d)
        x_chain, y_chain, tau = coupled_rwmh(
            target, x0_c, y0_c, h, max_iter=5000, n_extra=300, rng=pair_rng
        )
        meeting_times_list.append(tau)
        if tau < 5000:
            h_val = compute_unbiased_estimator(
                x_chain, y_chain, tau, func=lambda x: x[0]
            )
            cbcs.add_estimator(float(h_val))

    cbcs_coverage = None
    cbcs_width = None
    if len(cbcs.estimators) >= 2:
        c, hw = cbcs.get_cs()
        cbcs_coverage = bool(abs(c - ref_mean[0]) <= hw)
        cbcs_width = float(hw)

    return {
        'dataset': dataset_name,
        'seed': seed,
        'n_samples': n_samples,
        'd': d,
        'accept_rate': accept_rate,
        'ess_coord0': float(compute_ess(chain[:, 0])),
        'coords': results_per_coord,
        'cbcs_coverage': cbcs_coverage,
        'cbcs_width': cbcs_width,
        'meeting_times': {
            'median': float(np.median(meeting_times_list)),
            'mean': float(np.mean(meeting_times_list)),
            'p95': float(np.percentile(meeting_times_list, 95)),
        },
    }


def main():
    start_time = time.time()
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    datasets = ['german_credit', 'ionosphere', 'pima']
    seeds = [42, 123, 456]

    all_results = {}
    for ds in datasets:
        print(f"Dataset: {ds}", flush=True)
        ds_results = []
        for seed in seeds:
            print(f"  seed={seed}...", flush=True)
            t0 = time.time()
            r = run_dataset_seed(ds, data_dir, seed)
            ds_results.append(r)
            print(f"  Done in {time.time()-t0:.1f}s, accept_rate={r['accept_rate']:.3f}, "
                  f"ESS={r['ess_coord0']:.0f}", flush=True)
        all_results[ds] = ds_results

    # Aggregate: per-dataset summary
    summary = {}
    for ds in datasets:
        ds_results = all_results[ds]

        # Average over seeds and coords
        methods_data = {'clt': [], 'bmcs': [], 'mdcs': []}
        fwsr_stops = []
        bmcs_stops = []

        for r in ds_results:
            for coord_data in r['coords']:
                for method in ['clt', 'bmcs', 'mdcs']:
                    final_t = str(r['n_samples'])
                    width = coord_data[f'{method}_widths'].get(final_t, np.nan)
                    cover = coord_data[f'{method}_covers'].get(final_t, True)
                    methods_data[method].append({'width': width, 'cover': cover})
                fwsr_stops.append(coord_data['fwsr_stop'])
                bmcs_stops.append(coord_data['bmcs_stop'])

        ds_summary = {}
        for method in ['clt', 'bmcs', 'mdcs']:
            widths = [d['width'] for d in methods_data[method]]
            covers = [d['cover'] for d in methods_data[method]]
            ds_summary[method] = {
                'mean_width': float(np.nanmean(widths)),
                'coverage': float(np.mean(covers)),
            }

        # CBCS summary
        cbcs_covers = [r['cbcs_coverage'] for r in ds_results if r['cbcs_coverage'] is not None]
        cbcs_widths = [r['cbcs_width'] for r in ds_results if r['cbcs_width'] is not None]
        ds_summary['cbcs'] = {
            'mean_width': float(np.mean(cbcs_widths)) if cbcs_widths else None,
            'coverage': float(np.mean(cbcs_covers)) if cbcs_covers else None,
        }

        ds_summary['fwsr_mean_stop'] = float(np.mean(fwsr_stops))
        ds_summary['bmcs_mean_stop'] = float(np.mean(bmcs_stops))
        ds_summary['savings_pct'] = float(
            (np.mean(fwsr_stops) - np.mean(bmcs_stops)) / np.mean(fwsr_stops) * 100
        ) if np.mean(fwsr_stops) > 0 else 0.0

        summary[ds] = ds_summary

    elapsed = time.time() - start_time
    output = {
        'experiment': 'exp2_logistic',
        'config': {'datasets': datasets, 'seeds': seeds, 'n_samples': 100000, 'alpha': 0.05},
        'summary': summary,
        'details': {ds: all_results[ds] for ds in datasets},
        'runtime_minutes': elapsed / 60,
    }

    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExperiment 2 completed in {elapsed/60:.1f} minutes")
    for ds, s in summary.items():
        print(f"\n{ds}:")
        for method in ['clt', 'bmcs', 'mdcs', 'cbcs']:
            print(f"  {method}: width={s[method]['mean_width']}, coverage={s[method]['coverage']}")
        print(f"  FWSR stop: {s['fwsr_mean_stop']:.0f}, BMCS stop: {s['bmcs_mean_stop']:.0f}, "
              f"savings: {s['savings_pct']:.1f}%")


if __name__ == '__main__':
    main()
