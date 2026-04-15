"""Ablation studies for CBCS components."""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.targets import GaussianTarget
from src.samplers import rwmh_sample, coupled_rwmh, compute_unbiased_estimator
from src.confidence_sequences import CouplingCS, BatchMeansCS


def crn_coupling(target, x0, y0, step_size, max_iter=5000, rng=None):
    """Common Random Numbers coupling (simpler baseline)."""
    if rng is None:
        rng = np.random.default_rng()
    d = len(x0)
    h = step_size

    x, y = x0.copy(), y0.copy()
    log_px = target.log_density(x)
    log_py = target.log_density(y)
    x_chain = [x.copy()]
    y_chain = []

    # First step: only X
    z = rng.standard_normal(d)
    x_prop = x + h * z
    lp = target.log_density(x_prop)
    if np.log(rng.random()) < lp - log_px:
        x = x_prop
        log_px = lp
    x_chain.append(x.copy())

    for t in range(1, max_iter):
        # Same random numbers for both
        z = rng.standard_normal(d)
        u = rng.random()

        x_prop = x + h * z
        y_prop = y + h * z

        lp_x = target.log_density(x_prop)
        if np.log(u) < lp_x - log_px:
            x = x_prop
            log_px = lp_x

        lp_y = target.log_density(y_prop)
        if np.log(u) < lp_y - log_py:
            y = y_prop
            log_py = lp_y

        x_chain.append(x.copy())
        y_chain.append(y.copy())

        if np.allclose(x, y):
            # Continue X
            for _ in range(300):
                x_prop = x + h * rng.standard_normal(d)
                lp = target.log_density(x_prop)
                if np.log(rng.random()) < lp - log_px:
                    x = x_prop
                    log_px = lp
                x_chain.append(x.copy())
            return x_chain, y_chain, t + 1

    return x_chain, y_chain, max_iter


def main():
    start_time = time.time()
    seeds = [42, 123, 456]
    d = 10
    target = GaussianTarget(d)
    h = 2.38 / np.sqrt(d)
    true_mean = 0.0
    alpha = 0.05
    n_pairs = 100

    results = {}

    # B1: Coupling mechanism comparison
    print("Ablation B1: Coupling mechanism...", flush=True)
    for coupling_name, coupling_fn in [('maximal', coupled_rwmh), ('crn', crn_coupling)]:
        meeting_times = []
        estimators = []
        for seed in seeds:
            for pair_idx in range(n_pairs):
                rng = np.random.default_rng(seed * 100000 + pair_idx)
                x0 = rng.standard_normal(d)
                y0 = rng.standard_normal(d)
                xc, yc, tau = coupling_fn(target, x0, y0, h, max_iter=5000, rng=rng)
                meeting_times.append(tau)
                if tau < 5000:
                    hv = compute_unbiased_estimator(xc, yc, tau, lambda x: x[0])
                    estimators.append(float(hv))

        results[f'B1_{coupling_name}'] = {
            'median_tau': float(np.median(meeting_times)),
            'mean_tau': float(np.mean(meeting_times)),
            'p95_tau': float(np.percentile(meeting_times, 95)),
            'n_converged': int(np.sum(np.array(meeting_times) < 5000)),
            'estimator_mean': float(np.mean(estimators)) if estimators else None,
            'estimator_std': float(np.std(estimators)) if estimators else None,
        }
    print(f"  Maximal: median_tau={results['B1_maximal']['median_tau']:.0f}")
    print(f"  CRN: median_tau={results['B1_crn']['median_tau']:.0f}")

    # B2: Burn-in parameter k
    print("Ablation B2: Burn-in parameter k...", flush=True)
    for k_frac_name, k_frac in [('quarter', 0.25), ('half', 0.5), ('three_quarter', 0.75)]:
        estimators = []
        variances = []
        for seed in seeds:
            for pair_idx in range(n_pairs):
                rng = np.random.default_rng(seed * 100000 + pair_idx)
                x0 = rng.standard_normal(d)
                y0 = rng.standard_normal(d)
                xc, yc, tau = coupled_rwmh(target, x0, y0, h, max_iter=5000, n_extra=500, rng=rng)
                if tau < 5000:
                    k = max(1, int(tau * k_frac))
                    hv = compute_unbiased_estimator(xc, yc, tau, lambda x: x[0], burn_in_k=k)
                    estimators.append(float(hv))

        results[f'B2_k_{k_frac_name}'] = {
            'mean': float(np.mean(estimators)) if estimators else None,
            'std': float(np.std(estimators)) if estimators else None,
            'n': len(estimators),
        }
        print(f"  k={k_frac_name}: mean={results[f'B2_k_{k_frac_name}']['mean']:.4f}, "
              f"std={results[f'B2_k_{k_frac_name}']['std']:.4f}")

    # B3: Coupled pairs vs single long chain (equal compute budget)
    print("Ablation B3: Coupling vs single chain...", flush=True)
    # CBCS: n_pairs coupled pairs
    total_cbcs_iters = 0
    cbcs = CouplingCS(alpha=alpha)
    for seed in seeds:
        for pair_idx in range(50):
            rng = np.random.default_rng(seed * 100000 + pair_idx)
            x0 = rng.standard_normal(d)
            y0 = rng.standard_normal(d)
            xc, yc, tau = coupled_rwmh(target, x0, y0, h, max_iter=5000, n_extra=300, rng=rng)
            total_cbcs_iters += len(xc) + len(yc)
            if tau < 5000:
                hv = compute_unbiased_estimator(xc, yc, tau, lambda x: x[0])
                cbcs.add_estimator(float(hv))

    cbcs_c, cbcs_hw = cbcs.get_cs()

    # BMCS: single chain with same total iterations
    bmcs = BatchMeansCS(alpha=alpha)
    rng = np.random.default_rng(42)
    chain, _ = rwmh_sample(target, rng.standard_normal(d), total_cbcs_iters,
                           h, burn_in=1000, rng=rng)
    bmcs.update(chain[:, 0])
    bmcs_c, bmcs_hw = bmcs.get_cs()

    results['B3_equal_compute'] = {
        'total_iters': total_cbcs_iters,
        'cbcs_width': float(cbcs_hw),
        'cbcs_center': float(cbcs_c),
        'bmcs_width': float(bmcs_hw),
        'bmcs_center': float(bmcs_c),
        'cbcs_n_estimators': len(cbcs.estimators),
    }
    print(f"  Total iters: {total_cbcs_iters}")
    print(f"  CBCS width: {cbcs_hw:.4f}, BMCS width: {bmcs_hw:.4f}")

    elapsed = time.time() - start_time
    output = {
        'experiment': 'ablation_cbcs',
        'config': {'d': d, 'seeds': seeds, 'n_pairs': n_pairs, 'alpha': alpha},
        'results': results,
        'runtime_minutes': elapsed / 60,
    }

    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nCBCS Ablations completed in {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
