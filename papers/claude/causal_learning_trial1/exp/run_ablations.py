"""Ablation studies: e-value type, number of folds K, and anytime validity."""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import generate_dataset
from src.metrics import compute_metrics
from src.e_pc import run_epc, run_epc_anytime
from src.utils import save_results


def ablation_evalue_type():
    """Compare calibrator-based vs split-LR e-values."""
    print("\n=== ABLATION: E-value Construction Method ===")
    results = []

    for num_nodes in [10, 20, 50]:
        for n_samples in [500, 1000]:
            for seed in [42, 123, 456]:
                ds = generate_dataset('ER', num_nodes, 2, n_samples, 'linear_gaussian', seed=seed)
                data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
                max_cond = min(num_nodes - 2, 4) if num_nodes >= 50 else None

                config = {'num_nodes': num_nodes, 'n_samples': n_samples, 'seed': seed}

                for method in ['calibrator', 'split_lr']:
                    t0 = time.time()
                    try:
                        est, e_vals, info = run_epc(data, K=5, q=0.1, method=method,
                                                     seed=seed, max_cond_size=max_cond)
                        metrics = compute_metrics(est, true_dag, true_cpdag)
                        metrics['runtime'] = time.time() - t0
                        metrics['method'] = method
                        metrics.update(config)
                        results.append(metrics)
                        print(f"  p={num_nodes} n={n_samples} seed={seed} {method}: "
                              f"FDR={metrics['FDR']:.3f} TPR={metrics['TPR']:.3f} F1={metrics['F1']:.3f}")
                    except Exception as e:
                        results.append({**config, 'method': method, 'error': str(e)})

    save_results(results, 'results/analysis/ablation_evalue_type.json')
    print(f"  Saved {len(results)} results")
    return results


def ablation_num_folds():
    """Study effect of number of folds K."""
    print("\n=== ABLATION: Number of Folds K ===")
    results = []

    for num_nodes in [10, 20]:
        for n_samples in [1000, 2000]:
            for seed in [42, 123, 456]:
                ds = generate_dataset('ER', num_nodes, 2, n_samples, 'linear_gaussian', seed=seed)
                data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']

                config = {'num_nodes': num_nodes, 'n_samples': n_samples, 'seed': seed}

                for K in [2, 3, 5, 8, 10]:
                    t0 = time.time()
                    try:
                        est, e_vals, info = run_epc(data, K=K, q=0.1, method='calibrator',
                                                     seed=seed)
                        metrics = compute_metrics(est, true_dag, true_cpdag)
                        metrics['runtime'] = time.time() - t0
                        metrics['K'] = K
                        metrics.update(config)
                        results.append(metrics)
                        print(f"  p={num_nodes} n={n_samples} seed={seed} K={K}: "
                              f"FDR={metrics['FDR']:.3f} TPR={metrics['TPR']:.3f} F1={metrics['F1']:.3f}")
                    except Exception as e:
                        results.append({**config, 'K': K, 'error': str(e)})

    save_results(results, 'results/analysis/ablation_num_folds.json')
    print(f"  Saved {len(results)} results")
    return results


def ablation_anytime_validity():
    """Verify anytime-valid property: FDR control at every stopping point."""
    print("\n=== ABLATION: Anytime Validity ===")
    results = []

    for num_nodes in [10, 20, 50]:
        for seed in [42, 123, 456]:
            ds = generate_dataset('ER', num_nodes, 2, 1000, 'linear_gaussian', seed=seed)
            data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
            max_cond = min(num_nodes - 2, 4) if num_nodes >= 50 else None

            config = {'num_nodes': num_nodes, 'n_samples': 1000, 'seed': seed}

            try:
                t0 = time.time()
                results_by_fold = run_epc_anytime(data, K=10, q=0.1, method='calibrator',
                                                   seed=seed, max_cond_size=max_cond)
                runtime = time.time() - t0

                for k_prime, (est_graph, edge_e_vals) in results_by_fold.items():
                    metrics = compute_metrics(est_graph, true_dag, true_cpdag)
                    metrics['K_prime'] = k_prime
                    metrics['runtime'] = runtime
                    metrics.update(config)
                    results.append(metrics)

                print(f"  p={num_nodes} seed={seed}: done in {runtime:.1f}s")
            except Exception as e:
                print(f"  p={num_nodes} seed={seed}: FAILED - {e}")
                results.append({**config, 'error': str(e)})

    save_results(results, 'results/analysis/anytime_validity.json')
    print(f"  Saved {len(results)} results")
    return results


def scalability_analysis():
    """Measure runtime vs graph size for all methods."""
    print("\n=== SCALABILITY ANALYSIS ===")
    results = []

    from src.pc_p import run_pc_standard, run_pc_p, run_ges

    for num_nodes in [10, 20, 50, 100]:
        ds = generate_dataset('ER', num_nodes, 2, 1000, 'linear_gaussian', seed=42)
        data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
        max_cond = min(num_nodes - 2, 3) if num_nodes >= 50 else None

        config = {'num_nodes': num_nodes, 'n_samples': 1000}
        print(f"  p={num_nodes}:")

        # PC-stable
        t0 = time.time()
        try:
            est, _ = run_pc_standard(data, alpha=0.05, max_cond_size=max_cond)
            rt = time.time() - t0
            m = compute_metrics(est, true_dag, true_cpdag)
            results.append({**config, 'method': 'PC', 'runtime': rt, **m})
            print(f"    PC: {rt:.2f}s")
        except Exception as e:
            results.append({**config, 'method': 'PC', 'error': str(e)})

        # PC-p
        t0 = time.time()
        try:
            est, _, _ = run_pc_p(data, q=0.1, max_cond_size=max_cond)
            rt = time.time() - t0
            m = compute_metrics(est, true_dag, true_cpdag)
            results.append({**config, 'method': 'PC-p', 'runtime': rt, **m})
            print(f"    PC-p: {rt:.2f}s")
        except Exception as e:
            results.append({**config, 'method': 'PC-p', 'error': str(e)})

        # E-PC calibrator
        t0 = time.time()
        try:
            est, _, _ = run_epc(data, K=5, q=0.1, method='calibrator', seed=42, max_cond_size=max_cond)
            rt = time.time() - t0
            m = compute_metrics(est, true_dag, true_cpdag)
            results.append({**config, 'method': 'E-PC (cal)', 'runtime': rt, **m})
            print(f"    E-PC (cal): {rt:.2f}s")
        except Exception as e:
            results.append({**config, 'method': 'E-PC (cal)', 'error': str(e)})

        # E-PC split-LR
        t0 = time.time()
        try:
            est, _, _ = run_epc(data, K=5, q=0.1, method='split_lr', seed=42, max_cond_size=max_cond)
            rt = time.time() - t0
            m = compute_metrics(est, true_dag, true_cpdag)
            results.append({**config, 'method': 'E-PC (slr)', 'runtime': rt, **m})
            print(f"    E-PC (slr): {rt:.2f}s")
        except Exception as e:
            results.append({**config, 'method': 'E-PC (slr)', 'error': str(e)})

        # GES
        t0 = time.time()
        try:
            est, _ = run_ges(data)
            rt = time.time() - t0
            m = compute_metrics(est, true_dag, true_cpdag)
            results.append({**config, 'method': 'GES', 'runtime': rt, **m})
            print(f"    GES: {rt:.2f}s")
        except Exception as e:
            results.append({**config, 'method': 'GES', 'error': str(e)})

    save_results(results, 'results/analysis/scalability.json')
    print(f"  Saved {len(results)} results")
    return results


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    t_start = time.time()

    ablation_evalue_type()
    ablation_num_folds()
    ablation_anytime_validity()
    scalability_analysis()

    print(f"\nAll ablations done in {(time.time()-t_start)/60:.1f} min")
