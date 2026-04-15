"""Comprehensive experiment runner for E-PC paper.

Runs all experiments: baselines, E-PC, ablations, anytime validity, scalability.
Saves logs and results to exp/ subdirectories.
"""

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
from src.pc_p import run_pc_standard, run_pc_p, run_ges, run_notears
from src.e_pc import run_epc, run_epc_anytime
from src.utils import save_results


SEEDS = [42, 123, 456, 789, 1024]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_max_cond(p):
    if p >= 50:
        return 2
    elif p >= 20:
        return 3
    return None


def run_all_methods_on_dataset(data, true_dag, true_cpdag, config, p_nodes, seed):
    """Run all methods on a single dataset and return results dict."""
    max_cond = get_max_cond(p_nodes)
    result = {}

    # PC-stable at multiple alpha
    for alpha in [0.01, 0.05, 0.1]:
        key = f'PC_alpha{alpha}'
        t0 = time.time()
        try:
            est, info = run_pc_standard(data, alpha=alpha, max_cond_size=max_cond)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            result[key] = m
        except Exception as e:
            result[key] = {'error': str(e), 'runtime': time.time() - t0}

    # PC-p with BY correction
    for q in [0.05, 0.1, 0.2]:
        key = f'PCp_q{q}'
        t0 = time.time()
        try:
            est, _, info = run_pc_p(data, q=q, max_cond_size=max_cond)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            m['num_before_by'] = info['num_remaining_edges_before_by']
            m['num_after_by'] = info['num_selected_edges']
            result[key] = m
        except Exception as e:
            result[key] = {'error': str(e), 'runtime': time.time() - t0}

    # GES (skip for dense or large graphs - hangs in causal-learn)
    if p_nodes <= 20 and config.get('avg_degree', 2) <= 2:
        key = 'GES'
        t0 = time.time()
        try:
            est, info = run_ges(data)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            result[key] = m
        except Exception as e:
            result[key] = {'error': str(e), 'runtime': time.time() - t0}

    # DirectLiNGAM / NOTEARS proxy
    if p_nodes <= 50:
        key = 'NOTEARS'
        t0 = time.time()
        try:
            est, info = run_notears(data, seed=seed)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            m['method_detail'] = info.get('method', 'unknown')
            result[key] = m
        except Exception as e:
            result[key] = {'error': str(e), 'runtime': time.time() - t0}

    # E-PC calibrator
    for q in [0.05, 0.1, 0.2]:
        key = f'EPC_cal_q{q}'
        t0 = time.time()
        try:
            est, e_vals, info = run_epc(data, q=q, method='calibrator',
                                         seed=seed, max_cond_size=max_cond)
            m = compute_metrics(est, true_dag, true_cpdag)
            m['runtime'] = time.time() - t0
            m['num_skeleton_edges'] = info['num_skeleton_edges']
            m['num_selected_edges'] = info['num_selected_edges']
            m['e_value_stats'] = info.get('e_value_stats', {})
            result[key] = m
        except Exception as e:
            result[key] = {'error': str(e), 'runtime': time.time() - t0}

    # E-PC split-LR (only linear gaussian)
    if config.get('data_type') == 'linear_gaussian' or config.get('name'):
        for q in [0.05, 0.1, 0.2]:
            key = f'EPC_slr_q{q}'
            t0 = time.time()
            try:
                est, e_vals, info = run_epc(data, q=q, method='split_lr',
                                             seed=seed, max_cond_size=max_cond)
                m = compute_metrics(est, true_dag, true_cpdag)
                m['runtime'] = time.time() - t0
                m['num_skeleton_edges'] = info['num_skeleton_edges']
                m['num_selected_edges'] = info['num_selected_edges']
                result[key] = m
            except Exception as e:
                result[key] = {'error': str(e), 'runtime': time.time() - t0}

    return result


def run_synthetic_experiments():
    """Run all methods on synthetic datasets."""
    print("=" * 60)
    print("SYNTHETIC EXPERIMENTS")
    print("=" * 60)

    configs = []

    # Core experiments: degree=2, all node sizes, linear gaussian
    for gt in ['ER', 'SF']:
        for p in [10, 20, 50]:
            for n in [200, 500, 1000, 2000]:
                for seed in SEEDS:
                    configs.append({'graph_type': gt, 'num_nodes': p, 'avg_degree': 2,
                                    'n_samples': n, 'data_type': 'linear_gaussian', 'seed': seed})

    # Degree=4 for smaller graphs
    for gt in ['ER', 'SF']:
        for p in [10, 20]:
            for n in [500, 1000]:
                for seed in SEEDS:
                    configs.append({'graph_type': gt, 'num_nodes': p, 'avg_degree': 4,
                                    'n_samples': n, 'data_type': 'linear_gaussian', 'seed': seed})

    # Nonlinear: small graphs only
    for gt in ['ER', 'SF']:
        for p in [10, 20]:
            for n in [500, 1000]:
                for seed in SEEDS:
                    configs.append({'graph_type': gt, 'num_nodes': p, 'avg_degree': 2,
                                    'n_samples': n, 'data_type': 'nonlinear_additive', 'seed': seed})

    print(f"Total synthetic configs: {len(configs)}")

    # Resume from existing results
    all_results = []
    existing_names = set()
    results_path = os.path.join(BASE_DIR, 'results/synthetic_results.json')
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                all_results = json.load(f)
            for r in all_results:
                c = r.get('config', {})
                n = f"{c.get('graph_type')}_{c.get('num_nodes')}_{c.get('avg_degree')}_{c.get('n_samples')}_{c.get('data_type')}_seed{c.get('seed')}"
                existing_names.add(n)
            print(f"Resuming from {len(all_results)} existing results")
        except Exception:
            all_results = []

    t_start = time.time()
    log_lines = []

    for idx, config in enumerate(configs):
        name = f"{config['graph_type']}_{config['num_nodes']}_{config['avg_degree']}_{config['n_samples']}_{config['data_type']}_seed{config['seed']}"
        if name in existing_names:
            continue
        elapsed = (time.time() - t_start) / 60
        msg = f"[{idx+1}/{len(configs)}] {name} ({elapsed:.1f}min)"
        print(msg)
        log_lines.append(msg)

        try:
            ds = generate_dataset(**config)
            data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
            num_true = int(np.sum(((true_dag + true_dag.T) > 0).astype(int)) // 2)
            result = run_all_methods_on_dataset(data, true_dag, true_cpdag,
                                                 config, config['num_nodes'], config['seed'])
            result['config'] = config
            result['num_true_edges'] = num_true
            all_results.append(result)
        except Exception as e:
            msg = f"  FAILED: {e}"
            print(msg)
            log_lines.append(msg)
            all_results.append({'config': config, 'error': str(e)})

        if (idx + 1) % 20 == 0:
            save_results(all_results, os.path.join(BASE_DIR, 'results/synthetic_results.json'))

    save_results(all_results, os.path.join(BASE_DIR, 'results/synthetic_results.json'))

    ensure_dir(os.path.join(BASE_DIR, 'exp/baselines/logs'))
    ensure_dir(os.path.join(BASE_DIR, 'exp/epc/logs'))
    with open(os.path.join(BASE_DIR, 'exp/baselines/logs/synthetic_run.log'), 'w') as f:
        f.write('\n'.join(log_lines))
    with open(os.path.join(BASE_DIR, 'exp/epc/logs/synthetic_run.log'), 'w') as f:
        f.write('\n'.join(log_lines))

    total_time = (time.time() - t_start) / 60
    print(f"\nSynthetic done: {len(all_results)} results in {total_time:.1f} min")
    return all_results


def run_real_experiments():
    """Run all methods on real-world benchmark networks."""
    print("\n" + "=" * 60)
    print("REAL-WORLD EXPERIMENTS")
    print("=" * 60)

    real_results = {}
    log_lines = []

    for name, func in [('Asia', generate_real_network_asia),
                        ('Sachs', generate_real_network_sachs),
                        ('ALARM', generate_real_network_alarm),
                        ('Insurance', generate_real_network_insurance)]:
        msg = f"  {name}..."
        print(msg)
        log_lines.append(msg)
        res_list = []

        for seed in SEEDS:
            n_samp = 853 if name == 'Sachs' else 5000
            ds = func(n_samples=n_samp, seed=seed)
            data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
            p_nodes = data.shape[1]

            result = run_all_methods_on_dataset(data, true_dag, true_cpdag,
                                                 ds['config'], p_nodes, seed)
            result['name'] = name
            result['seed'] = seed
            result['num_true_edges'] = int(np.sum(((true_dag + true_dag.T) > 0).astype(int)) // 2)
            res_list.append(result)
            msg = f"    seed={seed} done"
            print(msg)
            log_lines.append(msg)

        real_results[name] = res_list

    save_results(real_results, os.path.join(BASE_DIR, 'results/real_results.json'))

    ensure_dir(os.path.join(BASE_DIR, 'exp/baselines/logs'))
    with open(os.path.join(BASE_DIR, 'exp/baselines/logs/real_run.log'), 'w') as f:
        f.write('\n'.join(log_lines))

    print("Real-world done.")
    return real_results


def run_ablation_evalue_type():
    """Ablation: calibrator vs split-LR vs GRAAL e-values."""
    print("\n" + "=" * 60)
    print("ABLATION: E-VALUE TYPE")
    print("=" * 60)

    results = []
    log_lines = []

    for gt in ['ER', 'SF']:
        for p in [10, 20, 50]:
            for n in [500, 1000]:
                for seed in SEEDS:
                    config = {'graph_type': gt, 'num_nodes': p, 'avg_degree': 2,
                              'n_samples': n, 'data_type': 'linear_gaussian', 'seed': seed}
                    ds = generate_dataset(**config)
                    data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
                    max_cond = get_max_cond(p)

                    r = {'config': config}
                    for method, tag in [('calibrator', 'cal'), ('split_lr', 'slr'), ('graal', 'graal')]:
                        t0 = time.time()
                        try:
                            est, _, info = run_epc(data, q=0.1, method=method,
                                                    seed=seed, max_cond_size=max_cond)
                            m = compute_metrics(est, true_dag, true_cpdag)
                            m['runtime'] = time.time() - t0
                            m['e_value_stats'] = info.get('e_value_stats', {})
                            r[tag] = m
                        except Exception as e:
                            r[tag] = {'error': str(e)}

                    results.append(r)
                    msg = f"  {gt}_p{p}_n{n}_seed{seed}"
                    print(msg)
                    log_lines.append(msg)

    ensure_dir(os.path.join(BASE_DIR, 'exp/ablation_evalue_type/logs'))
    save_results(results, os.path.join(BASE_DIR, 'results/ablation_evalue_type.json'))
    with open(os.path.join(BASE_DIR, 'exp/ablation_evalue_type/logs/run.log'), 'w') as f:
        f.write('\n'.join(log_lines))
    print("Ablation (e-value type) done.")
    return results


def run_ablation_num_folds():
    """Ablation: number of folds K for anytime E-PC."""
    print("\n" + "=" * 60)
    print("ABLATION: NUMBER OF FOLDS K")
    print("=" * 60)

    results = []
    log_lines = []

    for gt in ['ER']:
        for p in [10, 20]:
            for n in [1000, 2000]:
                for seed in SEEDS:
                    config = {'graph_type': gt, 'num_nodes': p, 'avg_degree': 2,
                              'n_samples': n, 'data_type': 'linear_gaussian', 'seed': seed}
                    ds = generate_dataset(**config)
                    data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
                    max_cond = get_max_cond(p)

                    r = {'config': config}
                    for K in [2, 3, 5, 8, 10]:
                        t0 = time.time()
                        try:
                            results_by_fold = run_epc_anytime(
                                data, K=K, q=0.1, method='calibrator',
                                seed=seed, max_cond_size=max_cond)
                            est_graph, _, fold_info = results_by_fold[K]
                            m = compute_metrics(est_graph, true_dag, true_cpdag)
                            m['runtime'] = time.time() - t0
                            r[f'K{K}'] = m
                        except Exception as e:
                            r[f'K{K}'] = {'error': str(e)}

                    results.append(r)
                    msg = f"  {gt}_p{p}_n{n}_seed{seed}"
                    print(msg)
                    log_lines.append(msg)

    ensure_dir(os.path.join(BASE_DIR, 'exp/ablation_num_folds/logs'))
    save_results(results, os.path.join(BASE_DIR, 'results/ablation_num_folds.json'))
    with open(os.path.join(BASE_DIR, 'exp/ablation_num_folds/logs/run.log'), 'w') as f:
        f.write('\n'.join(log_lines))
    print("Ablation (num folds) done.")
    return results


def run_anytime_validity():
    """Verify anytime validity: FDR control at every fold stopping point."""
    print("\n" + "=" * 60)
    print("ANYTIME VALIDITY EXPERIMENT")
    print("=" * 60)

    results = []
    log_lines = []
    K = 10

    for gt in ['ER', 'SF']:
        for p in [10, 20, 50]:
            for seed in SEEDS:
                config = {'graph_type': gt, 'num_nodes': p, 'avg_degree': 2,
                          'n_samples': 1000, 'data_type': 'linear_gaussian', 'seed': seed}
                ds = generate_dataset(**config)
                data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
                max_cond = get_max_cond(p)

                t0 = time.time()
                try:
                    results_by_fold = run_epc_anytime(
                        data, K=K, q=0.1, method='calibrator',
                        seed=seed, max_cond_size=max_cond)

                    fold_metrics = {}
                    for k in range(1, K + 1):
                        est_graph, edge_evals, fold_info = results_by_fold[k]
                        m = compute_metrics(est_graph, true_dag, true_cpdag)
                        m['num_selected'] = fold_info['num_selected']
                        m['num_skeleton'] = fold_info['num_skeleton']
                        fold_metrics[str(k)] = m

                    r = {'config': config, 'fold_metrics': fold_metrics,
                         'runtime': time.time() - t0}
                except Exception as e:
                    r = {'config': config, 'error': str(e), 'runtime': time.time() - t0}

                results.append(r)
                msg = f"  {gt}_p{p}_seed{seed} ({time.time()-t0:.1f}s)"
                print(msg)
                log_lines.append(msg)

    ensure_dir(os.path.join(BASE_DIR, 'exp/ablation_anytime/logs'))
    save_results(results, os.path.join(BASE_DIR, 'results/anytime_validity.json'))
    with open(os.path.join(BASE_DIR, 'exp/ablation_anytime/logs/run.log'), 'w') as f:
        f.write('\n'.join(log_lines))
    print("Anytime validity done.")
    return results


def run_scalability():
    """Scalability analysis: runtime vs graph size."""
    print("\n" + "=" * 60)
    print("SCALABILITY EXPERIMENT")
    print("=" * 60)

    results = []
    log_lines = []

    for p in [10, 20, 50, 100]:
        for seed in SEEDS[:3]:
            config = {'graph_type': 'ER', 'num_nodes': p, 'avg_degree': 2,
                      'n_samples': 1000, 'data_type': 'linear_gaussian', 'seed': seed}
            ds = generate_dataset(**config)
            data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
            max_cond = get_max_cond(p)

            r = {'config': config}

            t0 = time.time()
            try:
                est, _ = run_pc_standard(data, alpha=0.05, max_cond_size=max_cond)
                r['PC_time'] = time.time() - t0
                r['PC'] = compute_metrics(est, true_dag, true_cpdag)
            except Exception as e:
                r['PC_time'] = time.time() - t0
                r['PC'] = {'error': str(e)}

            t0 = time.time()
            try:
                est, _, _ = run_pc_p(data, q=0.1, max_cond_size=max_cond)
                r['PCp_time'] = time.time() - t0
                r['PCp'] = compute_metrics(est, true_dag, true_cpdag)
            except Exception as e:
                r['PCp_time'] = time.time() - t0
                r['PCp'] = {'error': str(e)}

            t0 = time.time()
            try:
                est, _, _ = run_epc(data, q=0.1, method='calibrator',
                                     seed=seed, max_cond_size=max_cond)
                r['EPC_cal_time'] = time.time() - t0
                r['EPC_cal'] = compute_metrics(est, true_dag, true_cpdag)
            except Exception as e:
                r['EPC_cal_time'] = time.time() - t0
                r['EPC_cal'] = {'error': str(e)}

            t0 = time.time()
            try:
                est, _, _ = run_epc(data, q=0.1, method='split_lr',
                                     seed=seed, max_cond_size=max_cond)
                r['EPC_slr_time'] = time.time() - t0
                r['EPC_slr'] = compute_metrics(est, true_dag, true_cpdag)
            except Exception as e:
                r['EPC_slr_time'] = time.time() - t0
                r['EPC_slr'] = {'error': str(e)}

            if p <= 20:
                t0 = time.time()
                try:
                    est, _ = run_ges(data)
                    r['GES_time'] = time.time() - t0
                    r['GES'] = compute_metrics(est, true_dag, true_cpdag)
                except Exception as e:
                    r['GES_time'] = time.time() - t0
                    r['GES'] = {'error': str(e)}

            results.append(r)
            msg = f"  p={p} seed={seed}: PC={r.get('PC_time',0):.2f}s EPC_cal={r.get('EPC_cal_time',0):.2f}s"
            print(msg)
            log_lines.append(msg)

    ensure_dir(os.path.join(BASE_DIR, 'exp/scalability/logs'))
    save_results(results, os.path.join(BASE_DIR, 'results/scalability.json'))
    with open(os.path.join(BASE_DIR, 'exp/scalability/logs/run.log'), 'w') as f:
        f.write('\n'.join(log_lines))
    print("Scalability done.")
    return results


def main():
    os.chdir(BASE_DIR)
    ensure_dir('results')
    ensure_dir('figures')

    t_start = time.time()
    print("=" * 60)
    print("E-PC FULL EXPERIMENT SUITE")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    run_synthetic_experiments()
    run_real_experiments()
    run_ablation_evalue_type()
    run_ablation_num_folds()
    run_anytime_validity()
    run_scalability()

    total = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total:.1f} minutes")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
