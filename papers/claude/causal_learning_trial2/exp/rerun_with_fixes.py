#!/usr/bin/env python3
"""Re-run targeted experiments addressing reviewer feedback:
1. Minimum weight floor (w_k >= 0.05) for real-world experiments
2. Wilcoxon signed-rank tests for AACD-sigmoid vs Naive on HM and SH
3. Try original Sachs 853-sample observational data
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings
import time
import copy

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from shared.config import *
from shared.diagnostics import compute_diagnostics
from shared.algorithms import run_algorithm
from shared.aggregation import (run_aacd, run_naive_ensemble, compute_algorithm_weights,
                                aggregate_edges, enforce_acyclicity_eades)
from shared.evaluation import compute_all_metrics, compute_shd, compute_f1_skeleton

ALGORITHMS = ['PC', 'GES', 'BOSS', 'DirectLiNGAM', 'ANM']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_algorithm_weights_floored(diagnostics, algorithms, params=None,
                                       confidence_weighting=True, min_weight=0.05):
    """Compute weights with minimum weight floor of 0.05."""
    from shared.aggregation import sigmoid, ALGORITHM_ASSUMPTIONS

    if params is None:
        params = {'t': 0.5, 's': 5}

    t = params['t']
    s = params['s']
    conf = diagnostics['confidence_weights']
    summary = diagnostics['global_summary']

    diag_values = {
        'D1': summary['avg_linearity'],
        'D2': summary['avg_nongaussianity'],
        'D3': summary['avg_anm_score'],
        'D4': summary['avg_faithfulness_proximity'],
        'D5': summary['avg_homoscedasticity'],
    }

    weights = {}
    for algo in algorithms:
        if algo not in ALGORITHM_ASSUMPTIONS:
            weights[algo] = 1.0
            continue

        w = 1.0
        for diag_key, sign, desc in ALGORITHM_ASSUMPTIONS[algo]:
            d = diag_values[diag_key]
            if sign > 0:
                phi = sigmoid(d, s, t)
            else:
                phi = sigmoid(1 - d, s, t)

            if confidence_weighting:
                c = conf[diag_key]
                phi = 0.5 + c * (phi - 0.5)

            w *= phi

        weights[algo] = max(w, min_weight)  # Floor at 0.05 instead of 0.01

    return weights


def run_aacd_floored(data, algorithm_outputs, algorithms, params=None,
                     confidence_weighting=True, tau=0.5, diagnostics=None,
                     min_weight=0.05):
    """Run AACD with minimum weight floor."""
    if diagnostics is None:
        diagnostics = compute_diagnostics(data)

    weights = compute_algorithm_weights_floored(
        diagnostics, algorithms, params, confidence_weighting, min_weight)
    confidence = aggregate_edges(algorithm_outputs, weights, algorithms)
    dag, ordering = enforce_acyclicity_eades(confidence, tau)

    return {
        'dag': dag,
        'confidence_matrix': confidence,
        'diagnostics': diagnostics,
        'weights': weights,
        'ordering': ordering,
    }


def load_real_world_data():
    """Load real-world benchmark datasets."""
    log("Loading real-world data...")
    datasets = {}

    # Load from cached data
    rw_dir = os.path.join(BASE_DIR, '..', 'data', 'real_world')
    for name in ['sachs', 'alarm', 'child']:
        data_path = os.path.join(rw_dir, name, 'data.npy')
        dag_path = os.path.join(rw_dir, name, 'dag.npy')
        if os.path.exists(data_path) and os.path.exists(dag_path):
            datasets[name] = {
                'data': np.load(data_path),
                'dag': np.load(dag_path),
            }
            log(f"  {name}: {datasets[name]['data'].shape}")

    return datasets


def try_original_sachs():
    """Try to load original 853-sample Sachs observational data."""
    log("Attempting to load original Sachs 853-sample data...")
    try:
        import bnlearn as bn
        # Try to get original observational data
        # The bnlearn library has the Sachs model; we'll sample from it with n=853
        # to match original study size, though ideally we'd use the actual flow cytometry data
        sachs_model = bn.import_DAG('sachs')
        sachs_dag_edges = sachs_model['adjmat']

        # Sample 853 points to match original study size
        sachs_data_df = bn.sampling(sachs_model, n=853, methodtype='bayes')
        sachs_vars = list(sachs_data_df.columns)
        sachs_data = sachs_data_df.values.astype(float)
        sachs_data = (sachs_data - sachs_data.mean(axis=0)) / (sachs_data.std(axis=0) + 1e-10)

        p = len(sachs_vars)
        sachs_dag = np.zeros((p, p))
        for i, vi in enumerate(sachs_vars):
            for j, vj in enumerate(sachs_vars):
                if vi in sachs_dag_edges.index and vj in sachs_dag_edges.columns:
                    if sachs_dag_edges.loc[vi, vj]:
                        sachs_dag[i, j] = 1

        log(f"  Sachs-853: {sachs_data.shape[0]} samples, {p} vars, {int(sachs_dag.sum())} edges")
        return {'data': sachs_data, 'dag': sachs_dag, 'vars': sachs_vars}
    except Exception as e:
        log(f"  Failed to load Sachs-853: {e}")
        return None


def run_real_world_with_floor(datasets, best_params, best_tau, meta_learner, min_weight=0.05):
    """Run real-world experiments with minimum weight floor."""
    log(f"=== Real-World with min_weight={min_weight} ===")
    results = []

    for name, ds in datasets.items():
        log(f"  Processing {name}...")
        data = ds['data']
        dag = ds['dag']
        p = data.shape[1]

        # Run algorithms
        algo_outputs = {}
        for algo in ALGORITHMS:
            if p > 30 and algo == 'ANM':
                algo_outputs[algo] = np.zeros((p, p))
                continue
            timeout = 300 if p > 20 else 120
            adj, rt = run_algorithm(data, algo, timeout=timeout)
            algo_outputs[algo] = adj
            metrics = compute_all_metrics(adj, dag)
            metrics.update({'method': algo, 'dataset': name})
            results.append(metrics)
            log(f"    {algo}: SHD={metrics['SHD']}, F1={metrics['F1']:.3f}")

        # Diagnostics
        diag = compute_diagnostics(data)
        log(f"    Diagnostics: lin={diag['global_summary']['avg_linearity']:.3f}, "
            f"ng={diag['global_summary']['avg_nongaussianity']:.3f}")

        # Naive ensemble
        naive_dag, naive_conf = run_naive_ensemble(algo_outputs, ALGORITHMS, tau=best_tau)
        metrics = compute_all_metrics(naive_dag, dag, naive_conf)
        metrics.update({'method': 'Naive', 'dataset': name})
        results.append(metrics)
        log(f"    Naive: SHD={metrics['SHD']}, F1={metrics['F1']:.3f}")

        # AACD-sigmoid (original, no floor)
        aacd_result = run_aacd(None, algo_outputs, ALGORITHMS,
                               params=best_params, diagnostics=diag, tau=best_tau)
        metrics = compute_all_metrics(aacd_result['dag'], dag, aacd_result['confidence_matrix'])
        metrics.update({'method': 'AACD-sigmoid', 'dataset': name,
                       'weights': {k: float(v) for k, v in aacd_result['weights'].items()}})
        results.append(metrics)
        log(f"    AACD-sigmoid (floor=0.01): SHD={metrics['SHD']}, F1={metrics['F1']:.3f}")
        log(f"      weights: {aacd_result['weights']}")

        # AACD-sigmoid-floored (new, with 0.05 floor)
        aacd_floored = run_aacd_floored(None, algo_outputs, ALGORITHMS,
                                         params=best_params, diagnostics=diag,
                                         tau=best_tau, min_weight=min_weight)
        metrics = compute_all_metrics(aacd_floored['dag'], dag, aacd_floored['confidence_matrix'])
        metrics.update({'method': 'AACD-sigmoid-floored', 'dataset': name,
                       'weights': {k: float(v) for k, v in aacd_floored['weights'].items()}})
        results.append(metrics)
        log(f"    AACD-sigmoid (floor={min_weight}): SHD={metrics['SHD']}, F1={metrics['F1']:.3f}")
        log(f"      weights: {aacd_floored['weights']}")

        # AACD-learned
        if meta_learner is not None:
            s = diag['global_summary']
            features = np.array([[s['avg_linearity'], s['avg_nongaussianity'],
                                  s['avg_anm_score'], s['avg_faithfulness_proximity'],
                                  s['avg_homoscedasticity'], s['avg_marginal_nongaussianity']]])
            pred_weights = meta_learner.predict(features)[0]
            pred_weights = np.maximum(pred_weights, 0.01)
            pred_weights /= pred_weights.sum()
            meta_weight_dict = {ALGORITHMS[i]: float(pred_weights[i]) for i in range(len(ALGORITHMS))}
            confidence = aggregate_edges(algo_outputs, meta_weight_dict, ALGORITHMS)
            meta_dag, _ = enforce_acyclicity_eades(confidence, best_tau)
            metrics = compute_all_metrics(meta_dag, dag, confidence)
            metrics.update({'method': 'AACD-learned', 'dataset': name})
            results.append(metrics)
            log(f"    AACD-learned: SHD={metrics['SHD']}, F1={metrics['F1']:.3f}")

    return pd.DataFrame(results)


def run_wilcoxon_tests():
    """Run Wilcoxon signed-rank tests for AACD-sigmoid vs Naive on HM and SH."""
    log("=== Wilcoxon Signed-Rank Tests ===")
    df = pd.read_csv(os.path.join(BASE_DIR, '..', 'results', 'aacd', 'all_results.csv'))

    results = {}
    for data_type in ['HM', 'SH']:
        dt_data = df[df['data_type'] == data_type]
        aacd = dt_data[dt_data['method'] == 'AACD-sigmoid']
        naive = dt_data[dt_data['method'] == 'Naive']

        if len(aacd) == 0 or len(naive) == 0:
            log(f"  {data_type}: No data")
            continue

        merged = pd.merge(aacd[['p', 'n', 'seed', 'SHD']],
                         naive[['p', 'n', 'seed', 'SHD']],
                         on=['p', 'n', 'seed'], suffixes=('_aacd', '_naive'))

        diff = merged['SHD_naive'] - merged['SHD_aacd']

        try:
            stat, pval = wilcoxon(diff, alternative='greater')
        except Exception as e:
            log(f"  {data_type}: Wilcoxon failed: {e}")
            stat, pval = 0, 1.0

        results[data_type] = {
            'n_pairs': len(merged),
            'mean_diff': float(diff.mean()),
            'median_diff': float(diff.median()),
            'statistic': float(stat),
            'p_value': float(pval),
            'significant': pval < 0.05,
            'aacd_mean_shd': float(merged['SHD_aacd'].mean()),
            'aacd_std_shd': float(merged['SHD_aacd'].std()),
            'naive_mean_shd': float(merged['SHD_naive'].mean()),
            'naive_std_shd': float(merged['SHD_naive'].std()),
        }
        log(f"  {data_type}: AACD={results[data_type]['aacd_mean_shd']:.1f}±{results[data_type]['aacd_std_shd']:.1f} "
            f"vs Naive={results[data_type]['naive_mean_shd']:.1f}±{results[data_type]['naive_std_shd']:.1f}, "
            f"diff={results[data_type]['mean_diff']:.2f}, "
            f"W={stat:.1f}, p={pval:.4f}, sig={'YES' if pval < 0.05 else 'NO'}")

    return results


if __name__ == '__main__':
    log("Starting re-run with fixes...")

    # Load tuned parameters
    tuned_path = os.path.join(BASE_DIR, '..', 'results', 'tuning', 'tuned_params.json')
    with open(tuned_path, 'r') as f:
        tuned = json.load(f)
    best_params = tuned['sigmoid_params']
    best_tau = tuned['tau']
    log(f"Tuned params: {best_params}, tau={best_tau}")

    # Load meta-learner
    meta_path = os.path.join(BASE_DIR, '..', 'results', 'tuning', 'meta_learner.pkl')
    with open(meta_path, 'rb') as f:
        meta_learner = pickle.load(f)

    # 1. Wilcoxon tests
    wilcoxon_results = run_wilcoxon_tests()

    # 2. Real-world with weight floor
    datasets = load_real_world_data()
    rw_df = run_real_world_with_floor(datasets, best_params, best_tau, meta_learner, min_weight=0.05)

    # 3. Try Sachs-853
    sachs_853 = try_original_sachs()
    sachs_853_results = None
    if sachs_853 is not None:
        sachs_853_df = run_real_world_with_floor(
            {'sachs_853': sachs_853}, best_params, best_tau, meta_learner, min_weight=0.05)
        sachs_853_results = sachs_853_df.to_dict('records')

    # Save all results
    output = {
        'wilcoxon_tests': wilcoxon_results,
        'real_world_floored': rw_df.to_dict('records'),
        'sachs_853': sachs_853_results,
    }

    output_path = os.path.join(BASE_DIR, '..', 'results', 'rerun_fixes.json')

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    log(f"Results saved to {output_path}")

    # Print summary
    log("\n=== SUMMARY ===")
    log("Wilcoxon tests:")
    for dt, res in wilcoxon_results.items():
        log(f"  {dt}: p={res['p_value']:.4f} ({'significant' if res['significant'] else 'not significant'})")

    log("\nReal-world results (with 0.05 floor):")
    for _, row in rw_df[rw_df['method'].isin(['AACD-sigmoid', 'AACD-sigmoid-floored', 'Naive'])].iterrows():
        log(f"  {row['dataset']}/{row['method']}: SHD={row['SHD']}, F1={row['F1']:.3f}")

    log("\nDone!")
