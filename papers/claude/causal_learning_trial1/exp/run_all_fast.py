"""Optimized experiment runner with reduced grid for time budget."""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import (generate_dataset, generate_real_network_asia,
                                  generate_real_network_sachs, generate_real_network_alarm,
                                  generate_real_network_insurance)
from src.metrics import compute_metrics
from src.pc_p import run_pc_standard, run_pc_p, run_ges
from src.e_pc import run_epc
from src.utils import save_results


def run_single(config):
    """Run all methods on a single dataset."""
    ds = generate_dataset(**config)
    data = ds['data']
    true_dag = ds['true_dag']
    true_cpdag = ds['true_cpdag']
    num_true_edges = int(np.sum(((true_dag + true_dag.T) > 0).astype(int)) // 2)
    p = config['num_nodes']

    # Aggressive conditioning set limit for speed
    if p >= 50:
        max_cond = 2
    elif p >= 20:
        max_cond = 3
    else:
        max_cond = None

    result = {'config': config, 'num_true_edges': num_true_edges}

    # PC
    for alpha in [0.01, 0.05, 0.1]:
        t0 = time.time()
        try:
            est, info = run_pc_standard(data, alpha=alpha, max_cond_size=max_cond)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            result[f'PC_alpha{alpha}'] = m
        except Exception as e:
            result[f'PC_alpha{alpha}'] = {'error': str(e)}

    # PC-p
    for q in [0.05, 0.1, 0.2]:
        t0 = time.time()
        try:
            est, _, info = run_pc_p(data, q=q, max_cond_size=max_cond)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            result[f'PCp_q{q}'] = m
        except Exception as e:
            result[f'PCp_q{q}'] = {'error': str(e)}

    # GES (skip for p>=50 - too slow)
    if p <= 20:
        t0 = time.time()
        try:
            est, info = run_ges(data)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            result['GES'] = m
        except Exception as e:
            result['GES'] = {'error': str(e)}

    # E-PC calibrator
    for q in [0.05, 0.1, 0.2]:
        t0 = time.time()
        try:
            est, e_vals, info = run_epc(data, K=5, q=q, method='calibrator',
                                         seed=config['seed'], max_cond_size=max_cond)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            m['num_skeleton_edges'] = info['num_skeleton_edges']
            m['num_selected_edges'] = info['num_selected_edges']
            result[f'EPC_cal_q{q}'] = m
        except Exception as e:
            result[f'EPC_cal_q{q}'] = {'error': str(e)}

    # E-PC split-LR (only linear gaussian)
    if config['data_type'] == 'linear_gaussian':
        for q in [0.05, 0.1, 0.2]:
            t0 = time.time()
            try:
                est, e_vals, info = run_epc(data, K=5, q=q, method='split_lr',
                                             seed=config['seed'], max_cond_size=max_cond)
                m = compute_metrics(est, true_dag, true_cpdag)
                m['runtime'] = time.time() - t0
                m['num_skeleton_edges'] = info['num_skeleton_edges']
                m['num_selected_edges'] = info['num_selected_edges']
                result[f'EPC_slr_q{q}'] = m
            except Exception as e:
                result[f'EPC_slr_q{q}'] = {'error': str(e)}

    return result


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    t_start = time.time()

    print("=" * 60)
    print("E-PC EXPERIMENT SUITE (OPTIMIZED)")
    print("=" * 60)

    # Optimized grid:
    # Linear Gaussian: full set of node sizes but only degree=2
    # Plus degree=4 for p=10,20 only
    configs = []
    seeds = [42, 123, 456]

    # Core experiments: degree=2, all node sizes
    for gt in ['ER', 'SF']:
        for p in [10, 20, 50]:
            for n in [200, 500, 1000, 2000]:
                for seed in seeds:
                    configs.append({'graph_type': gt, 'num_nodes': p, 'avg_degree': 2,
                                    'n_samples': n, 'data_type': 'linear_gaussian', 'seed': seed})

    # Degree=4 for smaller graphs
    for gt in ['ER', 'SF']:
        for p in [10, 20]:
            for n in [500, 1000]:
                for seed in seeds:
                    configs.append({'graph_type': gt, 'num_nodes': p, 'avg_degree': 4,
                                    'n_samples': n, 'data_type': 'linear_gaussian', 'seed': seed})

    # Nonlinear: small graphs only
    for gt in ['ER', 'SF']:
        for p in [10, 20]:
            for n in [500, 1000]:
                for seed in seeds:
                    configs.append({'graph_type': gt, 'num_nodes': p, 'avg_degree': 2,
                                    'n_samples': n, 'data_type': 'nonlinear_additive', 'seed': seed})

    print(f"Total synthetic configs: {len(configs)}")

    # Load existing results to resume
    existing = []
    existing_names = set()
    if os.path.exists('results/synthetic_results.json'):
        try:
            with open('results/synthetic_results.json') as f:
                existing = json.load(f)
            for r in existing:
                c = r.get('config', {})
                name = f"{c.get('graph_type')}_{c.get('num_nodes')}_{c.get('avg_degree')}_{c.get('n_samples')}_{c.get('data_type')}_seed{c.get('seed')}"
                existing_names.add(name)
            print(f"Resuming from {len(existing)} existing results")
        except:
            pass

    all_results = existing

    for idx, config in enumerate(configs):
        name = f"{config['graph_type']}_{config['num_nodes']}_{config['avg_degree']}_{config['n_samples']}_{config['data_type']}_seed{config['seed']}"
        if name in existing_names:
            continue

        elapsed = time.time() - t_start
        print(f"[{idx+1}/{len(configs)}] {name} ({elapsed/60:.1f}min)")

        try:
            result = run_single(config)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({'config': config, 'error': str(e)})

        # Save periodically
        if (len(all_results) - len(existing)) % 10 == 0:
            save_results(all_results, 'results/synthetic_results.json')

    save_results(all_results, 'results/synthetic_results.json')
    print(f"\nSynthetic: {len(all_results)} results in {(time.time()-t_start)/60:.1f} min")

    # Real-world experiments
    print("\n=== REAL-WORLD EXPERIMENTS ===")
    real_results = {}
    for name, func in [('Asia', generate_real_network_asia),
                        ('Sachs', generate_real_network_sachs),
                        ('ALARM', generate_real_network_alarm),
                        ('Insurance', generate_real_network_insurance)]:
        print(f"  {name}...")
        res_list = []
        for seed in seeds:
            n_samp = 853 if name == 'Sachs' else 5000
            ds = func(n_samples=n_samp, seed=seed)
            data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
            p = data.shape[1]
            max_cond = 4 if p >= 30 else None

            r = {'name': name, 'seed': seed,
                 'num_true_edges': int(np.sum(((true_dag + true_dag.T) > 0).astype(int)) // 2)}

            for alpha in [0.01, 0.05, 0.1]:
                try:
                    t0 = time.time()
                    est, _ = run_pc_standard(data, alpha=alpha, max_cond_size=max_cond)
                    r[f'PC_alpha{alpha}'] = compute_metrics(est, true_dag, true_cpdag)
                    r[f'PC_alpha{alpha}']['runtime'] = time.time() - t0
                except Exception as e:
                    r[f'PC_alpha{alpha}'] = {'error': str(e)}

            for q in [0.05, 0.1, 0.2]:
                try:
                    t0 = time.time()
                    est, _, _ = run_pc_p(data, q=q, max_cond_size=max_cond)
                    r[f'PCp_q{q}'] = compute_metrics(est, true_dag, true_cpdag)
                    r[f'PCp_q{q}']['runtime'] = time.time() - t0
                except Exception as e:
                    r[f'PCp_q{q}'] = {'error': str(e)}

            try:
                t0 = time.time()
                est, _ = run_ges(data)
                r['GES'] = compute_metrics(est, true_dag, true_cpdag)
                r['GES']['runtime'] = time.time() - t0
            except Exception as e:
                r['GES'] = {'error': str(e)}

            for q in [0.05, 0.1, 0.2]:
                for method, tag in [('calibrator', 'EPC_cal'), ('split_lr', 'EPC_slr')]:
                    try:
                        t0 = time.time()
                        est, _, info = run_epc(data, K=5, q=q, method=method,
                                                seed=seed, max_cond_size=max_cond)
                        r[f'{tag}_q{q}'] = compute_metrics(est, true_dag, true_cpdag)
                        r[f'{tag}_q{q}']['runtime'] = time.time() - t0
                    except Exception as e:
                        r[f'{tag}_q{q}'] = {'error': str(e)}

            res_list.append(r)
        real_results[name] = res_list

    save_results(real_results, 'results/real_results.json')
    print(f"\nAll done in {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
