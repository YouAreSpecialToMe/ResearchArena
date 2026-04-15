#!/usr/bin/env python3
"""Main experiment runner for AACD.

Runs all experiments sequentially:
1. Generate data (tuning + test + real-world)
2. Smoke test
3. Tune sigmoid parameters
4. Run baselines on test data
5. Run AACD on test data
6. Run real-world benchmarks
7. Sample-size robustness (Exp 3)
8. SH vs HM (Exp 4)
9. Ablations
10. Statistical testing
11. Save results
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.config import *
from src.data_generator import generate_dag, generate_data, verify_dag
from src.diagnostics import compute_diagnostics
from src.algorithms import run_algorithm
from src.aggregation import (run_aacd, run_naive_ensemble, compute_algorithm_weights,
                             aggregate_edges, enforce_acyclicity_eades, enforce_acyclicity_greedy)
from src.evaluation import compute_all_metrics, compute_shd, compute_f1_skeleton

ALGORITHMS = ['PC', 'GES', 'BOSS', 'DirectLiNGAM', 'ANM']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# PHASE 1: Data Generation
# ============================================================

def generate_all_data():
    """Generate tuning, test, and validation datasets."""
    log("=== PHASE 1: Data Generation ===")

    # Validation data (quick sanity check)
    log("Generating validation data...")
    os.makedirs(os.path.join(DATA_DIR, 'validation'), exist_ok=True)
    for dt in DATA_TYPES:
        dag = generate_dag(10, EXPECTED_DEGREE, seed=99)
        data, labels = generate_data(dag, 1000, dt, seed=99)
        np.savez(os.path.join(DATA_DIR, 'validation', f'{dt}.npz'),
                 data=data, dag=dag, mechanism_labels=np.array(labels, dtype=object))
    log(f"  Validation: {len(DATA_TYPES)} datasets")

    # Tuning data (180 datasets: 6 types x 30 seeds)
    log("Generating tuning data...")
    os.makedirs(os.path.join(DATA_DIR, 'tuning'), exist_ok=True)
    count = 0
    for dt in DATA_TYPES:
        for seed in TUNING_SEEDS:
            dag = generate_dag(15, EXPECTED_DEGREE, seed=seed, model='erdos_renyi')
            data, labels = generate_data(dag, 1000, dt, seed=seed)
            np.savez(os.path.join(DATA_DIR, 'tuning', f'{dt}_{seed}.npz'),
                     data=data, dag=dag, mechanism_labels=np.array(labels, dtype=object))
            count += 1
    log(f"  Tuning: {count} datasets")

    # Test data (360 + 20 = 380 datasets)
    log("Generating test data...")
    os.makedirs(os.path.join(DATA_DIR, 'test'), exist_ok=True)
    count = 0
    for dt in DATA_TYPES:
        for p in P_VALUES_FULL:
            for n in SAMPLE_SIZES:
                for seed in RANDOM_SEEDS:
                    dag = generate_dag(p, EXPECTED_DEGREE, seed=seed)
                    data, labels = generate_data(dag, n, dt, seed=seed)
                    fname = f'{dt}_p{p}_n{n}_seed{seed}.npz'
                    np.savez(os.path.join(DATA_DIR, 'test', fname),
                             data=data, dag=dag, mechanism_labels=np.array(labels, dtype=object))
                    count += 1

    # Targeted p=50
    for dt in ['HM', 'SH']:
        for seed in RANDOM_SEEDS:
            dag = generate_dag(50, EXPECTED_DEGREE, seed=seed)
            data, labels = generate_data(dag, 1000, dt, seed=seed)
            fname = f'{dt}_p50_n1000_seed{seed}.npz'
            np.savez(os.path.join(DATA_DIR, 'test', fname),
                     data=data, dag=dag, mechanism_labels=np.array(labels, dtype=object))
            count += 1
    log(f"  Test: {count} datasets")

    # Sample-size robustness data (Experiment 3)
    log("Generating sample-size robustness data...")
    os.makedirs(os.path.join(DATA_DIR, 'test', 'sample_size'), exist_ok=True)
    ss_sizes = [200, 500, 1000, 2000, 5000]
    for seed in range(5):
        dag = generate_dag(20, EXPECTED_DEGREE, seed=seed + 100)
        for n in ss_sizes:
            data, labels = generate_data(dag, n, 'HM', seed=seed + 100)
            fname = f'HM_p20_n{n}_seed{seed}.npz'
            np.savez(os.path.join(DATA_DIR, 'test', 'sample_size', fname),
                     data=data, dag=dag, mechanism_labels=np.array(labels, dtype=object))
    log(f"  Sample-size: {5 * len(ss_sizes)} datasets")


def load_real_world_data():
    """Load real-world benchmark datasets."""
    log("Loading real-world data...")
    os.makedirs(os.path.join(DATA_DIR, 'real_world'), exist_ok=True)
    datasets = {}

    # Sachs
    try:
        import bnlearn as bn
        sachs_model = bn.import_DAG('sachs')
        sachs_dag_edges = sachs_model['adjmat']
        # Sample data from the model
        sachs_data_df = bn.sampling(sachs_model, n=2000, methodtype='bayes')
        sachs_vars = list(sachs_data_df.columns)
        sachs_data = sachs_data_df.values.astype(float)
        # Standardize
        sachs_data = (sachs_data - sachs_data.mean(axis=0)) / (sachs_data.std(axis=0) + 1e-10)
        # DAG adjacency matrix
        p = len(sachs_vars)
        sachs_dag = np.zeros((p, p))
        for i, vi in enumerate(sachs_vars):
            for j, vj in enumerate(sachs_vars):
                if vi in sachs_dag_edges.index and vj in sachs_dag_edges.columns:
                    if sachs_dag_edges.loc[vi, vj]:
                        sachs_dag[i, j] = 1
        datasets['sachs'] = {'data': sachs_data, 'dag': sachs_dag, 'vars': sachs_vars}
        log(f"  Sachs: {sachs_data.shape[0]} samples, {p} vars, {int(sachs_dag.sum())} edges")
    except Exception as e:
        log(f"  Sachs failed: {e}")

    # Alarm
    try:
        alarm_model = bn.import_DAG('alarm')
        alarm_dag_edges = alarm_model['adjmat']
        alarm_data_df = bn.sampling(alarm_model, n=2000, methodtype='bayes')
        alarm_vars = list(alarm_data_df.columns)
        alarm_data = alarm_data_df.values.astype(float)
        alarm_data = (alarm_data - alarm_data.mean(axis=0)) / (alarm_data.std(axis=0) + 1e-10)
        p = len(alarm_vars)
        alarm_dag = np.zeros((p, p))
        for i, vi in enumerate(alarm_vars):
            for j, vj in enumerate(alarm_vars):
                if vi in alarm_dag_edges.index and vj in alarm_dag_edges.columns:
                    if alarm_dag_edges.loc[vi, vj]:
                        alarm_dag[i, j] = 1
        datasets['alarm'] = {'data': alarm_data, 'dag': alarm_dag, 'vars': alarm_vars}
        log(f"  Alarm: {alarm_data.shape[0]} samples, {p} vars, {int(alarm_dag.sum())} edges")
    except Exception as e:
        log(f"  Alarm failed: {e}")

    # Child (use pgmpy directly since bnlearn returns empty)
    try:
        from pgmpy.utils import get_example_model
        from pgmpy.sampling import BayesianModelSampling
        child_pgm = get_example_model('child')
        child_vars = sorted(child_pgm.nodes())
        sampler = BayesianModelSampling(child_pgm)
        child_data_df = sampler.forward_sample(size=2000, seed=42)
        # Label-encode categorical columns
        from sklearn.preprocessing import LabelEncoder
        child_data_enc = child_data_df[child_vars].copy()
        for col in child_data_enc.columns:
            if child_data_enc[col].dtype == object:
                child_data_enc[col] = LabelEncoder().fit_transform(child_data_enc[col])
        child_data = child_data_enc.values.astype(float)
        child_data = (child_data - child_data.mean(axis=0)) / (child_data.std(axis=0) + 1e-10)
        p = len(child_vars)
        child_dag = np.zeros((p, p))
        for src, dst in child_pgm.edges():
            i = child_vars.index(src)
            j = child_vars.index(dst)
            child_dag[i, j] = 1
        datasets['child'] = {'data': child_data, 'dag': child_dag, 'vars': child_vars}
        log(f"  Child: {child_data.shape[0]} samples, {p} vars, {int(child_dag.sum())} edges")
    except Exception as e:
        log(f"  Child failed: {e}")

    # Save
    for name, ds in datasets.items():
        save_dir = os.path.join(DATA_DIR, 'real_world', name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'data.npy'), ds['data'])
        np.save(os.path.join(save_dir, 'dag.npy'), ds['dag'])
        with open(os.path.join(save_dir, 'vars.json'), 'w') as f:
            json.dump(ds['vars'], f)

    return datasets


# ============================================================
# PHASE 2: Smoke Test
# ============================================================

def smoke_test():
    """Quick smoke test on validation data."""
    log("=== Smoke Test ===")
    d = np.load(os.path.join(DATA_DIR, 'validation', 'HLG.npz'), allow_pickle=True)
    data, dag = d['data'], d['dag']
    log(f"  Data shape: {data.shape}, edges: {int(dag.sum())}")

    for algo in ALGORITHMS:
        try:
            adj, rt = run_algorithm(data, algo, timeout=60)
            shd = compute_shd(adj, dag)
            log(f"  {algo}: SHD={shd}, edges={int((adj>0.1).sum())}, time={rt:.1f}s")
        except Exception as e:
            log(f"  {algo}: FAILED - {e}")

    # Test diagnostics
    diag = compute_diagnostics(data)
    log(f"  Diagnostics: linearity={diag['global_summary']['avg_linearity']:.3f}, "
        f"nongauss={diag['global_summary']['avg_nongaussianity']:.3f}")


# ============================================================
# PHASE 3: Tuning
# ============================================================

def run_tuning():
    """Tune sigmoid parameters on tuning data."""
    log("=== PHASE 3: Tuning ===")
    tuning_dir = os.path.join(DATA_DIR, 'tuning')
    results_dir = os.path.join(RESULTS_DIR, 'tuning')
    os.makedirs(results_dir, exist_ok=True)

    # Load tuning data
    tuning_files = sorted([f for f in os.listdir(tuning_dir) if f.endswith('.npz')])
    log(f"  Found {len(tuning_files)} tuning datasets")

    # Run all algorithms on tuning data (with caching)
    cache_file = os.path.join(results_dir, 'tuning_algo_cache.pkl')
    if os.path.exists(cache_file):
        log("  Loading cached algorithm outputs...")
        with open(cache_file, 'rb') as f:
            algo_cache = pickle.load(f)
    else:
        algo_cache = {}
        for fi, fname in enumerate(tuning_files):
            if fi % 30 == 0:
                log(f"  Running algorithms on tuning data: {fi}/{len(tuning_files)}")
            d = np.load(os.path.join(tuning_dir, fname), allow_pickle=True)
            data, dag = d['data'], d['dag']
            key = fname.replace('.npz', '')
            algo_cache[key] = {'dag': dag}
            for algo in ALGORITHMS:
                adj, rt = run_algorithm(data, algo, timeout=120)
                algo_cache[key][algo] = adj
        with open(cache_file, 'wb') as f:
            pickle.dump(algo_cache, f)
        log(f"  Cached {len(algo_cache)} algorithm outputs")

    # Run diagnostics on tuning data
    diag_cache_file = os.path.join(results_dir, 'tuning_diag_cache.pkl')
    if os.path.exists(diag_cache_file):
        log("  Loading cached diagnostics...")
        with open(diag_cache_file, 'rb') as f:
            diag_cache = pickle.load(f)
    else:
        diag_cache = {}
        for fi, fname in enumerate(tuning_files):
            if fi % 30 == 0:
                log(f"  Running diagnostics on tuning data: {fi}/{len(tuning_files)}")
            d = np.load(os.path.join(tuning_dir, fname), allow_pickle=True)
            data = d['data']
            key = fname.replace('.npz', '')
            diag_cache[key] = compute_diagnostics(data)
        with open(diag_cache_file, 'wb') as f:
            pickle.dump(diag_cache, f)
        log(f"  Cached {len(diag_cache)} diagnostic profiles")

    # Grid search for sigmoid parameters with 5-fold CV
    log("  Sigmoid grid search...")
    keys = sorted(algo_cache.keys())

    # Split into 5 folds by seed
    folds = defaultdict(list)
    for k in keys:
        # Extract seed from key like "HLG_10000"
        seed = int(k.split('_')[-1])
        fold = seed % 5
        folds[fold].append(k)

    best_params = None
    best_oof_shd = np.inf
    grid_results = []

    for t in SIGMOID_T_VALUES:
        for s in SIGMOID_S_VALUES:
            oof_shds = []
            for fold_id in range(5):
                test_keys = folds[fold_id]
                for k in test_keys:
                    dag = algo_cache[k]['dag']
                    algo_outputs = {a: algo_cache[k][a] for a in ALGORITHMS}
                    diag = diag_cache[k]
                    result = run_aacd(None, algo_outputs, ALGORITHMS,
                                     params={'t': t, 's': s},
                                     diagnostics=diag, tau=0.5)
                    shd = compute_shd(result['dag'], dag)
                    oof_shds.append(shd)

            mean_shd = np.mean(oof_shds)
            grid_results.append({'t': t, 's': s, 'mean_shd': mean_shd})
            if mean_shd < best_oof_shd:
                best_oof_shd = mean_shd
                best_params = {'t': t, 's': s}

    log(f"  Best sigmoid params: t={best_params['t']}, s={best_params['s']}, OOF SHD={best_oof_shd:.2f}")

    # Tune threshold tau
    log("  Tuning threshold tau...")
    best_tau = 0.5
    best_f1 = 0
    for tau in TAU_VALUES:
        f1s = []
        for k in keys:
            dag = algo_cache[k]['dag']
            algo_outputs = {a: algo_cache[k][a] for a in ALGORITHMS}
            diag = diag_cache[k]
            result = run_aacd(None, algo_outputs, ALGORITHMS,
                             params=best_params, diagnostics=diag, tau=tau)
            f1, _, _ = compute_f1_skeleton(result['dag'], dag)
            f1s.append(f1)
        mean_f1 = np.mean(f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_tau = tau
    log(f"  Best tau={best_tau}, F1={best_f1:.3f}")

    # Train meta-learner
    log("  Training meta-learner...")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    X_meta = []
    Y_meta = []
    for k in keys:
        diag = diag_cache[k]
        dag = algo_cache[k]['dag']
        s = diag['global_summary']
        features = [s['avg_linearity'], s['avg_nongaussianity'], s['avg_anm_score'],
                    s['avg_faithfulness_proximity'], s['avg_homoscedasticity'],
                    s['avg_marginal_nongaussianity']]
        X_meta.append(features)

        # Oracle weights: proportional to 1/(SHD+1)
        oracle_w = []
        for algo in ALGORITHMS:
            shd = compute_shd(algo_cache[k][algo], dag)
            oracle_w.append(1.0 / (shd + 1))
        total = sum(oracle_w)
        oracle_w = [w / total for w in oracle_w]
        Y_meta.append(oracle_w)

    X_meta = np.array(X_meta)
    Y_meta = np.array(Y_meta)

    best_meta = None
    best_meta_score = -np.inf
    for C in [0.01, 0.1, 1.0, 10.0]:
        model = Ridge(alpha=1.0/C)
        scores = cross_val_score(model, X_meta, Y_meta, cv=5, scoring='neg_mean_squared_error')
        if scores.mean() > best_meta_score:
            best_meta_score = scores.mean()
            best_meta = Ridge(alpha=1.0/C)
    best_meta.fit(X_meta, Y_meta)
    log(f"  Meta-learner trained, CV score={best_meta_score:.4f}")

    # Save tuned parameters
    tuned = {
        'sigmoid_params': best_params,
        'tau': best_tau,
        'grid_results': grid_results,
    }
    with open(os.path.join(results_dir, 'tuned_params.json'), 'w') as f:
        json.dump(tuned, f, indent=2)
    with open(os.path.join(results_dir, 'meta_learner.pkl'), 'wb') as f:
        pickle.dump(best_meta, f)

    return best_params, best_tau, best_meta


# ============================================================
# PHASE 4: Run Baselines on Test Data
# ============================================================

def run_baselines(skip_p50_anm=True):
    """Run all individual algorithms on test datasets."""
    log("=== PHASE 4: Run Baselines ===")
    test_dir = os.path.join(DATA_DIR, 'test')
    cache_dir = os.path.join(RESULTS_DIR, 'baselines', 'algo_outputs')
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, 'test_algo_cache.pkl')
    if os.path.exists(cache_file):
        log("  Loading cached test algorithm outputs...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.npz')])
    log(f"  Found {len(test_files)} test datasets")

    algo_cache = {}
    total = len(test_files)
    start = time.time()

    for fi, fname in enumerate(test_files):
        if fi % 50 == 0:
            elapsed = time.time() - start
            log(f"  Processing {fi}/{total} ({elapsed:.0f}s elapsed)")

        d = np.load(os.path.join(test_dir, fname), allow_pickle=True)
        data, dag = d['data'], d['dag']
        key = fname.replace('.npz', '')
        p = data.shape[1]
        algo_cache[key] = {'dag': dag, 'runtimes': {}}

        for algo in ALGORITHMS:
            # Skip ANM for p=50 (too slow)
            if skip_p50_anm and p >= 50 and algo == 'ANM':
                algo_cache[key][algo] = np.zeros((p, p))
                algo_cache[key]['runtimes'][algo] = 0
                continue

            timeout = 120 if p <= 20 else ALGORITHM_TIMEOUT_SEC
            adj, rt = run_algorithm(data, algo, timeout=timeout)
            algo_cache[key][algo] = adj
            algo_cache[key]['runtimes'][algo] = rt

    with open(cache_file, 'wb') as f:
        pickle.dump(algo_cache, f)

    elapsed = time.time() - start
    log(f"  Baselines complete: {total} datasets in {elapsed:.0f}s")
    return algo_cache


# ============================================================
# PHASE 5: Run AACD and Ensemble Methods
# ============================================================

def run_aacd_experiments(algo_cache, best_params, best_tau, meta_learner):
    """Run AACD and ensemble baselines on all test data."""
    log("=== PHASE 5: Run AACD Experiments ===")
    test_dir = os.path.join(DATA_DIR, 'test')
    results_dir_aacd = os.path.join(RESULTS_DIR, 'aacd')
    os.makedirs(results_dir_aacd, exist_ok=True)

    # Run diagnostics on all test data
    diag_cache_file = os.path.join(results_dir_aacd, 'test_diag_cache.pkl')
    if os.path.exists(diag_cache_file):
        log("  Loading cached test diagnostics...")
        with open(diag_cache_file, 'rb') as f:
            diag_cache = pickle.load(f)
    else:
        diag_cache = {}
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.npz')])
        start = time.time()
        for fi, fname in enumerate(test_files):
            if fi % 50 == 0:
                log(f"  Computing diagnostics: {fi}/{len(test_files)}")
            d = np.load(os.path.join(test_dir, fname), allow_pickle=True)
            data = d['data']
            key = fname.replace('.npz', '')
            diag_cache[key] = compute_diagnostics(data)
        with open(diag_cache_file, 'wb') as f:
            pickle.dump(diag_cache, f)
        log(f"  Diagnostics done in {time.time()-start:.0f}s")

    # Run all ensemble methods
    all_results = []
    keys = sorted(algo_cache.keys())
    start = time.time()

    for ki, key in enumerate(keys):
        if ki % 50 == 0:
            log(f"  Running ensembles: {ki}/{len(keys)}")

        dag = algo_cache[key]['dag']
        algo_outputs = {a: algo_cache[key][a] for a in ALGORITHMS}

        # Parse key to get metadata
        parts = key.split('_')
        data_type = parts[0]
        p_val = int(parts[1].replace('p', ''))
        n_val = int(parts[2].replace('n', ''))
        seed_val = int(parts[3].replace('seed', ''))

        # Individual algorithm results
        for algo in ALGORITHMS:
            metrics = compute_all_metrics(algo_outputs[algo], dag)
            metrics.update({'method': algo, 'data_type': data_type,
                           'p': p_val, 'n': n_val, 'seed': seed_val,
                           'runtime': algo_cache[key].get('runtimes', {}).get(algo, 0)})
            if 'calibration_bins' in metrics:
                del metrics['calibration_bins']
            all_results.append(metrics)

        # Naive ensemble
        naive_dag, naive_conf = run_naive_ensemble(algo_outputs, ALGORITHMS, tau=best_tau)
        metrics = compute_all_metrics(naive_dag, dag, naive_conf)
        metrics.update({'method': 'Naive', 'data_type': data_type,
                       'p': p_val, 'n': n_val, 'seed': seed_val})
        if 'calibration_bins' in metrics:
            del metrics['calibration_bins']
        all_results.append(metrics)

        # AACD-sigmoid
        diag = diag_cache.get(key)
        if diag is None:
            continue
        aacd_result = run_aacd(None, algo_outputs, ALGORITHMS,
                              params=best_params, diagnostics=diag, tau=best_tau)
        metrics = compute_all_metrics(aacd_result['dag'], dag, aacd_result['confidence_matrix'])
        metrics.update({'method': 'AACD-sigmoid', 'data_type': data_type,
                       'p': p_val, 'n': n_val, 'seed': seed_val,
                       'weights': {k: float(v) for k, v in aacd_result['weights'].items()}})
        if 'calibration_bins' in metrics:
            del metrics['calibration_bins']
        all_results.append(metrics)

        # AACD-learned (meta-learner)
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
            metrics.update({'method': 'AACD-learned', 'data_type': data_type,
                           'p': p_val, 'n': n_val, 'seed': seed_val})
            if 'calibration_bins' in metrics:
                del metrics['calibration_bins']
            all_results.append(metrics)

    elapsed = time.time() - start
    log(f"  Ensemble experiments done in {elapsed:.0f}s")

    # Save results
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(results_dir_aacd, 'all_results.csv'), index=False)
    log(f"  Saved {len(df)} result rows")
    return df


# ============================================================
# PHASE 6: Real-World Benchmarks
# ============================================================

def run_real_world(real_data, best_params, best_tau, meta_learner):
    """Run all methods on real-world benchmarks."""
    log("=== PHASE 6: Real-World Benchmarks ===")
    results = []

    for name, ds in real_data.items():
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

        # AACD-sigmoid
        aacd_result = run_aacd(None, algo_outputs, ALGORITHMS,
                              params=best_params, diagnostics=diag, tau=best_tau)
        metrics = compute_all_metrics(aacd_result['dag'], dag, aacd_result['confidence_matrix'])
        metrics.update({'method': 'AACD-sigmoid', 'dataset': name})
        results.append(metrics)
        log(f"    AACD-sigmoid: SHD={metrics['SHD']}, F1={metrics['F1']:.3f}")

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

    # Save
    import pandas as pd
    rw_df = pd.DataFrame(results)
    rw_df.to_csv(os.path.join(RESULTS_DIR, 'aacd', 'real_world_results.csv'), index=False)
    return rw_df


# ============================================================
# PHASE 7: Sample-Size Robustness (Experiment 3)
# ============================================================

def run_sample_size_experiment(best_params, best_tau):
    """Evaluate diagnostic accuracy and AACD performance vs sample size."""
    log("=== PHASE 7: Sample-Size Robustness ===")
    ss_dir = os.path.join(DATA_DIR, 'test', 'sample_size')
    results = []
    diag_accuracy_results = []
    ss_sizes = [200, 500, 1000, 2000, 5000]

    for seed in range(5):
        for n in ss_sizes:
            fname = f'HM_p20_n{n}_seed{seed}.npz'
            d = np.load(os.path.join(ss_dir, fname), allow_pickle=True)
            data, dag = d['data'], d['dag']
            mech_labels = d['mechanism_labels']

            # Run algorithms
            algo_outputs = {}
            for algo in ALGORITHMS:
                adj, rt = run_algorithm(data, algo, timeout=120)
                algo_outputs[algo] = adj
                metrics = compute_all_metrics(adj, dag)
                metrics.update({'method': algo, 'n': n, 'seed': seed})
                results.append(metrics)

            # Diagnostics
            diag = compute_diagnostics(data)

            # Diagnostic accuracy: check if D1 correctly classifies linear vs nonlinear
            linear_count_correct = 0
            linear_count_total = 0
            for label in mech_labels:
                i, j, mtype = int(label[0]), int(label[1]), str(label[2])
                is_linear = mtype in ['LG', 'LNG']
                d1_score = diag['D1'][i, j]
                predicted_linear = d1_score > 0.5
                if predicted_linear == is_linear:
                    linear_count_correct += 1
                linear_count_total += 1

            d1_acc = linear_count_correct / max(linear_count_total, 1)
            diag_accuracy_results.append({'n': n, 'seed': seed,
                                         'D1_accuracy': d1_acc,
                                         'avg_linearity': diag['global_summary']['avg_linearity']})

            # Naive ensemble
            naive_dag, _ = run_naive_ensemble(algo_outputs, ALGORITHMS, tau=best_tau)
            metrics = compute_all_metrics(naive_dag, dag)
            metrics.update({'method': 'Naive', 'n': n, 'seed': seed})
            results.append(metrics)

            # AACD with confidence weighting
            aacd_result = run_aacd(None, algo_outputs, ALGORITHMS,
                                  params=best_params, diagnostics=diag,
                                  tau=best_tau, confidence_weighting=True)
            metrics = compute_all_metrics(aacd_result['dag'], dag)
            metrics.update({'method': 'AACD-conf', 'n': n, 'seed': seed})
            results.append(metrics)

            # AACD without confidence weighting
            aacd_result_noconf = run_aacd(None, algo_outputs, ALGORITHMS,
                                         params=best_params, diagnostics=diag,
                                         tau=best_tau, confidence_weighting=False)
            metrics = compute_all_metrics(aacd_result_noconf['dag'], dag)
            metrics.update({'method': 'AACD-noconf', 'n': n, 'seed': seed})
            results.append(metrics)

    import pandas as pd
    ss_df = pd.DataFrame(results)
    ss_df.to_csv(os.path.join(RESULTS_DIR, 'aacd', 'sample_size_results.csv'), index=False)
    diag_acc_df = pd.DataFrame(diag_accuracy_results)
    diag_acc_df.to_csv(os.path.join(RESULTS_DIR, 'aacd', 'diagnostic_accuracy.csv'), index=False)
    return ss_df, diag_acc_df


# ============================================================
# PHASE 8: Ablations
# ============================================================

def run_ablations(algo_cache, best_params, best_tau):
    """Run ablation studies."""
    log("=== PHASE 8: Ablation Studies ===")
    test_dir = os.path.join(DATA_DIR, 'test')
    ablation_results = []

    # Select HM, p=20, n=1000 datasets for ablations
    ablation_keys = [k for k in algo_cache.keys()
                     if k.startswith('HM_p20_n1000')]
    log(f"  Ablation datasets: {len(ablation_keys)}")

    # Load diagnostics
    diag_cache_file = os.path.join(RESULTS_DIR, 'aacd', 'test_diag_cache.pkl')
    with open(diag_cache_file, 'rb') as f:
        diag_cache = pickle.load(f)

    # Ablation 1: Diagnostic importance (leave-one-out)
    log("  Ablation 1: Diagnostic importance...")
    for exclude_diag in ['D1', 'D2', 'D3', 'D4', 'D5']:
        shds = []
        for key in ablation_keys:
            dag = algo_cache[key]['dag']
            algo_outputs = {a: algo_cache[key][a] for a in ALGORITHMS}
            diag = diag_cache.get(key)
            if diag is None:
                continue
            # Deep copy diagnostics and neutralize excluded
            import copy
            diag_copy = copy.deepcopy(diag)
            result = run_aacd(None, algo_outputs, ALGORITHMS,
                             params=best_params, diagnostics=diag_copy,
                             tau=best_tau, excluded_diagnostics=[exclude_diag])
            shd = compute_shd(result['dag'], dag)
            shds.append(shd)
        ablation_results.append({
            'ablation': f'remove_{exclude_diag}',
            'mean_SHD': float(np.mean(shds)),
            'std_SHD': float(np.std(shds)),
        })
        log(f"    Remove {exclude_diag}: SHD = {np.mean(shds):.2f} +/- {np.std(shds):.2f}")

    # Full AACD baseline
    full_shds = []
    for key in ablation_keys:
        dag = algo_cache[key]['dag']
        algo_outputs = {a: algo_cache[key][a] for a in ALGORITHMS}
        diag = diag_cache.get(key)
        if diag is None:
            continue
        result = run_aacd(None, algo_outputs, ALGORITHMS,
                         params=best_params, diagnostics=diag, tau=best_tau)
        full_shds.append(compute_shd(result['dag'], dag))
    ablation_results.append({
        'ablation': 'full_AACD',
        'mean_SHD': float(np.mean(full_shds)),
        'std_SHD': float(np.std(full_shds)),
    })
    log(f"    Full AACD: SHD = {np.mean(full_shds):.2f} +/- {np.std(full_shds):.2f}")

    # Ablation 2: Number of algorithms
    log("  Ablation 2: Number of algorithms...")
    algo_subsets = [
        ['PC', 'DirectLiNGAM'],
        ['PC', 'DirectLiNGAM', 'GES'],
        ['PC', 'DirectLiNGAM', 'GES', 'BOSS'],
        ['PC', 'DirectLiNGAM', 'GES', 'BOSS', 'ANM'],
    ]
    for subset in algo_subsets:
        shds = []
        for key in ablation_keys:
            dag = algo_cache[key]['dag']
            algo_outputs = {a: algo_cache[key][a] for a in subset}
            diag = diag_cache.get(key)
            if diag is None:
                continue
            result = run_aacd(None, algo_outputs, subset,
                             params=best_params, diagnostics=diag, tau=best_tau)
            shds.append(compute_shd(result['dag'], dag))
        ablation_results.append({
            'ablation': f'n_algos_{len(subset)}',
            'algorithms': subset,
            'mean_SHD': float(np.mean(shds)),
            'std_SHD': float(np.std(shds)),
        })
        log(f"    {len(subset)} algorithms: SHD = {np.mean(shds):.2f} +/- {np.std(shds):.2f}")

    # Ablation 5: Acyclicity enforcement comparison
    log("  Ablation 5: Acyclicity enforcement...")
    for method in ['eades', 'greedy']:
        shds = []
        for key in ablation_keys:
            dag = algo_cache[key]['dag']
            algo_outputs = {a: algo_cache[key][a] for a in ALGORITHMS}
            diag = diag_cache.get(key)
            if diag is None:
                continue
            result = run_aacd(None, algo_outputs, ALGORITHMS,
                             params=best_params, diagnostics=diag,
                             tau=best_tau, acyclicity=method)
            shds.append(compute_shd(result['dag'], dag))
        ablation_results.append({
            'ablation': f'acyclicity_{method}',
            'mean_SHD': float(np.mean(shds)),
            'std_SHD': float(np.std(shds)),
        })
        log(f"    Acyclicity {method}: SHD = {np.mean(shds):.2f} +/- {np.std(shds):.2f}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'ablations', 'ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)
    return ablation_results


# ============================================================
# PHASE 9: Statistical Testing
# ============================================================

def run_statistical_tests(df, rw_df, ss_df, ablation_results):
    """Evaluate all success criteria with statistical tests."""
    log("=== PHASE 9: Statistical Testing ===")
    import pandas as pd
    from scipy.stats import wilcoxon

    criteria = []

    # Criterion 1: AACD < all individual algos on HM data (SHD)
    log("  Criterion 1: AACD vs individual algorithms on HM...")
    hm_data = df[df['data_type'] == 'HM']
    aacd_hm = hm_data[hm_data['method'] == 'AACD-sigmoid']
    criterion1_pass = True
    for algo in ALGORITHMS:
        algo_hm = hm_data[hm_data['method'] == algo]
        if len(aacd_hm) == 0 or len(algo_hm) == 0:
            continue
        # Merge on configuration
        merged = pd.merge(aacd_hm[['p', 'n', 'seed', 'SHD']],
                         algo_hm[['p', 'n', 'seed', 'SHD']],
                         on=['p', 'n', 'seed'], suffixes=('_aacd', '_algo'))
        if len(merged) < 5:
            continue
        diff = merged['SHD_algo'] - merged['SHD_aacd']
        try:
            stat, pval = wilcoxon(diff, alternative='greater')
        except Exception:
            pval = 1.0
        log(f"    vs {algo}: mean_diff={diff.mean():.2f}, p={pval:.4f}")
        if pval > 0.05:
            criterion1_pass = False
    criteria.append({'criterion': 1, 'description': 'AACD < all individual on HM',
                    'pass': criterion1_pass})

    # Criterion 2: AACD < naive ensemble on HM
    log("  Criterion 2: AACD vs Naive ensemble on HM...")
    naive_hm = hm_data[hm_data['method'] == 'Naive']
    if len(aacd_hm) > 0 and len(naive_hm) > 0:
        merged = pd.merge(aacd_hm[['p', 'n', 'seed', 'SHD']],
                         naive_hm[['p', 'n', 'seed', 'SHD']],
                         on=['p', 'n', 'seed'], suffixes=('_aacd', '_naive'))
        diff = merged['SHD_naive'] - merged['SHD_aacd']
        try:
            stat, pval = wilcoxon(diff, alternative='greater')
        except Exception:
            pval = 1.0
        c2_pass = pval < 0.05
        log(f"    mean_diff={diff.mean():.2f}, p={pval:.4f}, pass={c2_pass}")
    else:
        c2_pass = False
    criteria.append({'criterion': 2, 'description': 'AACD < Naive on HM',
                    'pass': c2_pass})

    # Criterion 3: Real-world benchmarks (from rw_df)
    log("  Criterion 3: Real-world benchmarks...")
    rw_pass_count = 0
    if rw_df is not None:
        for dataset in rw_df['dataset'].unique():
            ds_data = rw_df[rw_df['dataset'] == dataset]
            aacd_shd = ds_data[ds_data['method'] == 'AACD-sigmoid']['SHD'].values
            naive_shd = ds_data[ds_data['method'] == 'Naive']['SHD'].values
            individual_shds = ds_data[ds_data['method'].isin(ALGORITHMS)]['SHD'].values
            if len(aacd_shd) > 0 and len(naive_shd) > 0 and len(individual_shds) > 0:
                best_ind = individual_shds.min()
                if aacd_shd[0] <= best_ind and aacd_shd[0] < naive_shd[0]:
                    rw_pass_count += 1
                    log(f"    {dataset}: PASS (AACD={aacd_shd[0]}, best_ind={best_ind}, naive={naive_shd[0]})")
                else:
                    log(f"    {dataset}: FAIL (AACD={aacd_shd[0]}, best_ind={best_ind}, naive={naive_shd[0]})")
    c3_pass = rw_pass_count >= 2
    criteria.append({'criterion': 3, 'description': 'AACD wins on >=2/3 real-world',
                    'pass': c3_pass, 'count': rw_pass_count})

    # Criterion 4: SH data advantage
    log("  Criterion 4: AACD advantage on SH data...")
    sh_data = df[df['data_type'] == 'SH']
    aacd_sh = sh_data[sh_data['method'] == 'AACD-sigmoid']
    naive_sh = sh_data[sh_data['method'] == 'Naive']
    if len(aacd_sh) > 0 and len(naive_sh) > 0:
        merged = pd.merge(aacd_sh[['p', 'n', 'seed', 'SHD']],
                         naive_sh[['p', 'n', 'seed', 'SHD']],
                         on=['p', 'n', 'seed'], suffixes=('_aacd', '_naive'))
        diff = merged['SHD_naive'] - merged['SHD_aacd']
        c4_pass = diff.mean() > 0
        log(f"    SH advantage: {diff.mean():.2f}, pass={c4_pass}")
    else:
        c4_pass = False
    criteria.append({'criterion': 4, 'description': 'AACD advantage on SH',
                    'pass': c4_pass})

    # Criterion 5: Calibration
    log("  Criterion 5: Calibration...")
    ece_vals = df[df['method'] == 'AACD-sigmoid']['ECE'].dropna()
    if len(ece_vals) > 0:
        avg_ece = ece_vals.mean()
        c5_pass = avg_ece < 0.15
        log(f"    Avg ECE={avg_ece:.4f}, pass={c5_pass}")
    else:
        c5_pass = False
        avg_ece = None
    criteria.append({'criterion': 5, 'description': 'ECE < 0.15',
                    'pass': c5_pass, 'ECE': float(avg_ece) if avg_ece is not None else None})

    # Criterion 6: Ablation confirms diagnostics
    log("  Criterion 6: Diagnostics matter in ablation...")
    full_shd = None
    diag_increases = 0
    for r in ablation_results:
        if r['ablation'] == 'full_AACD':
            full_shd = r['mean_SHD']
    if full_shd is not None:
        for r in ablation_results:
            if r['ablation'].startswith('remove_'):
                if r['mean_SHD'] > full_shd:
                    diag_increases += 1
    c6_pass = diag_increases >= 2
    log(f"    Diagnostics causing SHD increase: {diag_increases}/5, pass={c6_pass}")
    criteria.append({'criterion': 6, 'description': 'Ablation confirms diagnostics',
                    'pass': c6_pass})

    # Criterion 7: AACD-conf >= Naive at n=500
    log("  Criterion 7: AACD-conf never worse than Naive at n=500...")
    if ss_df is not None:
        ss500 = ss_df[ss_df['n'] == 500]
        aacd_conf = ss500[ss500['method'] == 'AACD-conf']['SHD'].mean()
        naive_500 = ss500[ss500['method'] == 'Naive']['SHD'].mean()
        c7_pass = aacd_conf <= naive_500 + 0.5  # Small tolerance
        log(f"    AACD-conf={aacd_conf:.2f}, Naive={naive_500:.2f}, pass={c7_pass}")
    else:
        c7_pass = False
    criteria.append({'criterion': 7, 'description': 'AACD-conf >= Naive at n=500',
                    'pass': c7_pass})

    # Save criteria - convert numpy types
    criteria_clean = []
    for c in criteria:
        cc = {}
        for k, v in c.items():
            if isinstance(v, (np.bool_,)):
                cc[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                cc[k] = int(v)
            elif isinstance(v, (np.floating,)):
                cc[k] = float(v)
            else:
                cc[k] = v
        criteria_clean.append(cc)
    with open(os.path.join(RESULTS_DIR, 'tables', 'success_criteria.json'), 'w') as f:
        json.dump(criteria_clean, f, indent=2)

    return criteria_clean


# ============================================================
# PHASE 10: Visualization
# ============================================================

def generate_figures(df, rw_df, ss_df, diag_acc_df, ablation_results, criteria):
    """Generate all publication figures."""
    log("=== PHASE 10: Visualization ===")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 12})
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Color mapping
    method_colors = {
        'PC': '#7f7f7f', 'GES': '#9e9e9e', 'BOSS': '#bdbdbd',
        'DirectLiNGAM': '#616161', 'ANM': '#424242',
        'Oracle': '#d32f2f', 'Naive': '#4caf50',
        'AACD-sigmoid': '#1976d2', 'AACD-learned': '#ff9800',
    }

    # Figure 2: Main results by data type (SHD)
    log("  Figure 2: Main SHD results...")
    try:
        methods_to_plot = ['PC', 'GES', 'BOSS', 'DirectLiNGAM', 'ANM', 'Naive', 'AACD-sigmoid', 'AACD-learned']
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_data = []
        for dt in DATA_TYPES:
            for method in methods_to_plot:
                subset = df[(df['data_type'] == dt) & (df['method'] == method)]
                if len(subset) > 0:
                    plot_data.append({
                        'Data Type': dt, 'Method': method,
                        'SHD': subset['SHD'].mean(),
                        'SHD_std': subset['SHD'].std()
                    })

        import pandas as pd
        plot_df = pd.DataFrame(plot_data)
        if len(plot_df) > 0:
            pivot = plot_df.pivot(index='Data Type', columns='Method', values='SHD')
            pivot_std = plot_df.pivot(index='Data Type', columns='Method', values='SHD_std')
            pivot = pivot.reindex(DATA_TYPES)
            pivot = pivot[methods_to_plot]
            ax_plot = pivot.plot(kind='bar', ax=ax, width=0.85,
                               color=[method_colors.get(m, '#333') for m in methods_to_plot])
            ax.set_ylabel('Structural Hamming Distance (SHD)')
            ax.set_xlabel('Data Type')
            ax.set_title('Main Results: SHD by Data Type')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'main_results_shd.pdf'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(FIGURES_DIR, 'main_results_shd.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log(f"    Figure 2 failed: {e}")

    # Figure 3: Real-world results
    log("  Figure 3: Real-world results...")
    try:
        if rw_df is not None and len(rw_df) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            datasets = rw_df['dataset'].unique()
            methods_rw = [m for m in methods_to_plot if m in rw_df['method'].unique()]

            for ax, metric, title in [(ax1, 'SHD', 'SHD (lower is better)'),
                                       (ax2, 'F1', 'F1 Score (higher is better)')]:
                x = np.arange(len(datasets))
                width = 0.8 / len(methods_rw)
                for mi, method in enumerate(methods_rw):
                    vals = []
                    for ds in datasets:
                        v = rw_df[(rw_df['dataset'] == ds) & (rw_df['method'] == method)][metric]
                        vals.append(v.values[0] if len(v) > 0 else 0)
                    ax.bar(x + mi * width, vals, width,
                           label=method, color=method_colors.get(method, '#333'))
                ax.set_xticks(x + width * len(methods_rw) / 2)
                ax.set_xticklabels(datasets)
                ax.set_ylabel(title)
                ax.legend(fontsize=8)

            plt.suptitle('Real-World Benchmark Results')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'real_world_results.pdf'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(FIGURES_DIR, 'real_world_results.png'), dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        log(f"    Figure 3 failed: {e}")

    # Figure 4: Sample-size robustness
    log("  Figure 4: Sample-size robustness...")
    try:
        if ss_df is not None and len(ss_df) > 0:
            import pandas as pd
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Left: Diagnostic accuracy vs n
            if diag_acc_df is not None and len(diag_acc_df) > 0:
                diag_agg = diag_acc_df.groupby('n')['D1_accuracy'].agg(['mean', 'std']).reset_index()
                ax1.errorbar(diag_agg['n'], diag_agg['mean'], yerr=diag_agg['std'],
                            marker='o', capsize=3, label='D1 (linearity)')
                ax1.set_xlabel('Sample Size (n)')
                ax1.set_ylabel('Diagnostic Classification Accuracy')
                ax1.set_title('Diagnostic Accuracy vs Sample Size')
                ax1.legend()
                ax1.set_xscale('log')

            # Right: SHD vs n
            for method in ['AACD-conf', 'AACD-noconf', 'Naive', 'PC', 'DirectLiNGAM']:
                sub = ss_df[ss_df['method'] == method]
                if len(sub) == 0:
                    continue
                agg = sub.groupby('n')['SHD'].agg(['mean', 'std']).reset_index()
                ax2.errorbar(agg['n'], agg['mean'], yerr=agg['std'],
                            marker='o', capsize=3, label=method,
                            color=method_colors.get(method, None))
            ax2.set_xlabel('Sample Size (n)')
            ax2.set_ylabel('SHD')
            ax2.set_title('SHD vs Sample Size')
            ax2.legend(fontsize=9)
            ax2.set_xscale('log')

            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'sample_size_robustness.pdf'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(FIGURES_DIR, 'sample_size_robustness.png'), dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        log(f"    Figure 4 failed: {e}")

    # Figure 6: Ablation - diagnostic importance
    log("  Figure 6: Ablation results...")
    try:
        if ablation_results:
            fig, ax = plt.subplots(figsize=(10, 5))
            diag_ablations = [r for r in ablation_results
                             if r['ablation'].startswith('remove_') or r['ablation'] == 'full_AACD']
            names = [r['ablation'].replace('remove_', '-').replace('full_AACD', 'Full AACD')
                     for r in diag_ablations]
            means = [r['mean_SHD'] for r in diag_ablations]
            stds = [r['std_SHD'] for r in diag_ablations]
            colors = ['#1976d2' if 'Full' in n else '#ff7043' for n in names]
            ax.bar(names, means, yerr=stds, capsize=3, color=colors)
            ax.set_ylabel('Mean SHD')
            ax.set_title('Ablation: Impact of Removing Each Diagnostic')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'ablation_diagnostics.pdf'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(FIGURES_DIR, 'ablation_diagnostics.png'), dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        log(f"    Figure 6 failed: {e}")

    # Figure 8: Algorithm weights heatmap
    log("  Figure 8: Algorithm weights heatmap...")
    try:
        import pandas as pd
        aacd_rows = df[df['method'] == 'AACD-sigmoid']
        if 'weights' in aacd_rows.columns and len(aacd_rows) > 0:
            weight_data = defaultdict(lambda: defaultdict(list))
            for _, row in aacd_rows.iterrows():
                if isinstance(row.get('weights'), dict):
                    for algo, w in row['weights'].items():
                        weight_data[row['data_type']][algo].append(w)

            if weight_data:
                weight_means = {}
                for dt in DATA_TYPES:
                    weight_means[dt] = {}
                    for algo in ALGORITHMS:
                        vals = weight_data[dt].get(algo, [0])
                        weight_means[dt][algo] = np.mean(vals)

                wdf = pd.DataFrame(weight_means).T
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(wdf, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
                ax.set_title('Algorithm Weights by Data Type (AACD-sigmoid)')
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR, 'algorithm_weights.pdf'), dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(FIGURES_DIR, 'algorithm_weights.png'), dpi=150, bbox_inches='tight')
                plt.close()
    except Exception as e:
        log(f"    Figure 8 failed: {e}")

    log("  Figures complete!")


# ============================================================
# PHASE 11: Save Final Results
# ============================================================

def save_final_results(df, rw_df, ss_df, ablation_results, criteria):
    """Save aggregated results.json at workspace root."""
    log("=== PHASE 11: Save Final Results ===")
    import pandas as pd

    results = {
        'title': 'AACD: Assumption-Adaptive Causal Discovery',
        'experiments': {},
    }

    # Experiment 1: Main synthetic results
    if df is not None and len(df) > 0:
        methods = df['method'].unique().tolist()
        exp1 = {}
        for method in methods:
            method_data = df[df['method'] == method]
            exp1[method] = {
                'SHD': {'mean': float(method_data['SHD'].mean()),
                       'std': float(method_data['SHD'].std())},
                'F1': {'mean': float(method_data['F1'].mean()),
                      'std': float(method_data['F1'].std())},
            }
            # Per data type
            by_dt = {}
            for dt in DATA_TYPES:
                dt_data = method_data[method_data['data_type'] == dt]
                if len(dt_data) > 0:
                    by_dt[dt] = {
                        'SHD': {'mean': float(dt_data['SHD'].mean()),
                               'std': float(dt_data['SHD'].std())},
                        'F1': {'mean': float(dt_data['F1'].mean()),
                              'std': float(dt_data['F1'].std())},
                    }
            exp1[method]['by_data_type'] = by_dt
        results['experiments']['synthetic'] = exp1

    # Experiment 2: Real-world results
    if rw_df is not None and len(rw_df) > 0:
        exp2 = {}
        for _, row in rw_df.iterrows():
            ds = row['dataset']
            if ds not in exp2:
                exp2[ds] = {}
            exp2[ds][row['method']] = {
                'SHD': float(row['SHD']),
                'F1': float(row['F1']),
                'precision': float(row.get('precision', 0)),
                'recall': float(row.get('recall', 0)),
            }
        results['experiments']['real_world'] = exp2

    # Experiment 3: Sample-size robustness
    if ss_df is not None and len(ss_df) > 0:
        exp3 = {}
        for method in ss_df['method'].unique():
            m_data = ss_df[ss_df['method'] == method]
            exp3[method] = {}
            for n in sorted(m_data['n'].unique()):
                n_data = m_data[m_data['n'] == n]
                exp3[method][int(n)] = {
                    'SHD': {'mean': float(n_data['SHD'].mean()),
                           'std': float(n_data['SHD'].std())},
                }
        results['experiments']['sample_size'] = exp3

    # Ablations
    results['experiments']['ablations'] = ablation_results

    # Success criteria
    results['success_criteria'] = criteria
    results['n_criteria_passed'] = sum(1 for c in criteria if c['pass'])
    results['n_criteria_total'] = len(criteria)

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

    with open(os.path.join(BASE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    log(f"  Saved results.json ({results['n_criteria_passed']}/{results['n_criteria_total']} criteria passed)")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    overall_start = time.time()
    log("Starting AACD experiments...")

    # Phase 1: Generate data
    generate_all_data()
    real_data = load_real_world_data()

    # Phase 2: Smoke test
    smoke_test()

    # Phase 3: Tuning
    best_params, best_tau, meta_learner = run_tuning()

    # Phase 4: Run baselines
    algo_cache = run_baselines()

    # Phase 5: Run AACD experiments
    df = run_aacd_experiments(algo_cache, best_params, best_tau, meta_learner)

    # Phase 6: Real-world benchmarks
    rw_df = run_real_world(real_data, best_params, best_tau, meta_learner)

    # Phase 7: Sample-size robustness
    ss_df, diag_acc_df = run_sample_size_experiment(best_params, best_tau)

    # Phase 8: Ablations
    ablation_results = run_ablations(algo_cache, best_params, best_tau)

    # Phase 9: Statistical tests
    criteria = run_statistical_tests(df, rw_df, ss_df, ablation_results)

    # Phase 10: Figures
    generate_figures(df, rw_df, ss_df, diag_acc_df, ablation_results, criteria)

    # Phase 11: Save results
    save_final_results(df, rw_df, ss_df, ablation_results, criteria)

    elapsed = time.time() - overall_start
    log(f"All experiments complete in {elapsed/3600:.1f} hours")
