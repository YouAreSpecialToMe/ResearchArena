"""Scalability analysis with aggressive cond set limits."""

import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import generate_dataset
from src.metrics import compute_metrics
from src.pc_p import run_pc_standard, run_pc_p, run_ges
from src.e_pc import run_epc
from src.utils import save_results


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("=== SCALABILITY ANALYSIS ===")
    results = []

    for p in [10, 20, 50, 100]:
        ds = generate_dataset('ER', p, 2, 1000, 'linear_gaussian', seed=42)
        data, true_dag, true_cpdag = ds['data'], ds['true_dag'], ds['true_cpdag']
        max_cond = min(p - 2, 2) if p >= 50 else (3 if p >= 20 else None)
        config = {'num_nodes': p, 'n_samples': 1000}

        print(f"  p={p} (max_cond={max_cond}):")

        for method_name, runner in [
            ('PC', lambda: run_pc_standard(data, alpha=0.05, max_cond_size=max_cond)),
            ('PC-p', lambda: run_pc_p(data, q=0.1, max_cond_size=max_cond)[:1]),
            ('E-PC (cal)', lambda: run_epc(data, K=5, q=0.1, method='calibrator', seed=42, max_cond_size=max_cond)[:1]),
            ('E-PC (slr)', lambda: run_epc(data, K=5, q=0.1, method='split_lr', seed=42, max_cond_size=max_cond)[:1]),
            ('GES', lambda: run_ges(data)),
        ]:
            t0 = time.time()
            try:
                res = runner()
                est = res[0]
                rt = time.time() - t0
                m = compute_metrics(est, true_dag, true_cpdag)
                results.append({**config, 'method': method_name, 'runtime': rt, **m})
                print(f"    {method_name}: {rt:.2f}s")
            except Exception as e:
                rt = time.time() - t0
                print(f"    {method_name}: FAILED ({rt:.1f}s) - {e}")
                results.append({**config, 'method': method_name, 'runtime': rt, 'error': str(e)})

    save_results(results, 'results/analysis/scalability.json')
    print("Done!")


if __name__ == '__main__':
    main()
