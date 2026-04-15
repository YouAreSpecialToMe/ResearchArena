"""Experiment 6: Ablation studies for each novel component."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from src.adaquantcs import AdaQuantCS
from src.utils import (
    generate_stream, true_quantile, get_initial_range,
    check_anytime_coverage, save_results, SEEDS, CHECKPOINTS_100K,
)

N = 100_000
ALPHA = 0.05
K = 50
DIST = 'gaussian'
P = 0.5


def run_ablation(dist_name, p, seeds, **kwargs):
    """Run AdaQuantCS with given settings and return metrics."""
    tq = true_quantile(dist_name, p)
    init_range = get_initial_range(dist_name)

    all_widths = []
    all_covered = []
    all_memory = []

    for seed in seeds:
        stream = generate_stream(dist_name, N, seed)
        grid_k = kwargs.get('k', K)
        other_kwargs = {key: val for key, val in kwargs.items() if key != 'k'}
        aq = AdaQuantCS(p=p, k=grid_k, alpha=ALPHA,
                         initial_range=init_range, **other_kwargs)

        checkpoint_set = set(CHECKPOINTS_100K)
        widths = []
        for i, x in enumerate(stream):
            aq.update(x)
            t = i + 1
            if t in checkpoint_set:
                lo, hi = aq.get_ci()
                widths.append(hi - lo)

        all_widths.append(widths)
        final_lo, final_hi = aq.get_ci()
        all_covered.append(final_lo <= tq <= final_hi)
        all_memory.append(aq.memory_usage())

    all_widths = np.array(all_widths)
    return {
        'ci_width_mean': np.mean(all_widths, axis=0).tolist(),
        'ci_width_std': np.std(all_widths, axis=0).tolist(),
        'final_width_mean': float(np.mean(all_widths[:, -1])),
        'final_width_std': float(np.std(all_widths[:, -1])),
        'coverage': float(np.mean(all_covered)),
        'memory_mean': float(np.mean(all_memory)),
        'checkpoints': CHECKPOINTS_100K,
    }


def main():
    print("=" * 60)
    print("Experiment 6: Ablation Studies")
    print("=" * 60)

    start_time = time.time()
    all_results = {}

    # Ablation A: Fixed vs adaptive grid
    print("\nAblation A: Fixed vs Adaptive Grid")
    baseline_adaptive = run_ablation(DIST, P, SEEDS, grid_type='adaptive')
    ablation_fixed = run_ablation(DIST, P, SEEDS, grid_type='fixed')

    if ablation_fixed['final_width_mean'] > 0:
        improvement = (ablation_fixed['final_width_mean'] - baseline_adaptive['final_width_mean']) / ablation_fixed['final_width_mean'] * 100
    else:
        improvement = 0.0  # Fixed grid collapsed
    print(f"  Adaptive: {baseline_adaptive['final_width_mean']:.4f}±{baseline_adaptive['final_width_std']:.4f}")
    print(f"  Fixed:    {ablation_fixed['final_width_mean']:.4f}±{ablation_fixed['final_width_std']:.4f}")
    print(f"  Improvement: {improvement:.1f}%")

    all_results['ablation_A_grid_type'] = {
        'adaptive': baseline_adaptive,
        'fixed': ablation_fixed,
        'improvement_pct': improvement,
    }

    # Also test on Student-t (heavier tails)
    print("\n  Also on Student-t:")
    baseline_adaptive_t = run_ablation('student_t', P, SEEDS, grid_type='adaptive')
    ablation_fixed_t = run_ablation('student_t', P, SEEDS, grid_type='fixed')
    if ablation_fixed_t['final_width_mean'] > 0:
        improvement_t = (ablation_fixed_t['final_width_mean'] - baseline_adaptive_t['final_width_mean']) / ablation_fixed_t['final_width_mean'] * 100
    else:
        improvement_t = 0.0
    print(f"  Adaptive: {baseline_adaptive_t['final_width_mean']:.4f}, Fixed: {ablation_fixed_t['final_width_mean']:.4f}, Improvement: {improvement_t:.1f}%")

    all_results['ablation_A_grid_type']['student_t_adaptive'] = baseline_adaptive_t
    all_results['ablation_A_grid_type']['student_t_fixed'] = ablation_fixed_t
    all_results['ablation_A_grid_type']['student_t_improvement_pct'] = improvement_t

    # Ablation B: Grid size sensitivity
    print("\nAblation B: Grid Size Sensitivity")
    k_values = [5, 10, 20, 50, 100, 200]
    ablation_b = {}
    for k in k_values:
        res = run_ablation(DIST, P, SEEDS, k=k)
        print(f"  k={k:3d}: width={res['final_width_mean']:.4f}±{res['final_width_std']:.4f}, "
              f"cov={res['coverage']:.2f}, mem={res['memory_mean']:.0f}")
        ablation_b[f'k={k}'] = res

    # Also on Student-t
    ablation_b_t = {}
    for k in k_values:
        res = run_ablation('student_t', P, SEEDS, k=k)
        ablation_b_t[f'k={k}'] = res

    all_results['ablation_B_grid_size'] = {
        'gaussian': ablation_b,
        'student_t': ablation_b_t,
        'k_values': k_values,
    }

    # Ablation C: Confidence sequence type
    print("\nAblation C: Confidence Sequence Type")
    cs_types = ['bernstein', 'hoeffding', 'wald']
    ablation_c = {}
    for cs in cs_types:
        res = run_ablation(DIST, P, SEEDS, cs_type=cs)
        print(f"  {cs:10s}: width={res['final_width_mean']:.4f}±{res['final_width_std']:.4f}, cov={res['coverage']:.2f}")
        ablation_c[cs] = res

    all_results['ablation_C_cs_type'] = ablation_c

    # Ablation D: Epoch schedule
    print("\nAblation D: Epoch Schedule")
    schedules = ['doubling', 'tripling', 'fixed']
    ablation_d = {}
    for sched in schedules:
        res = run_ablation(DIST, P, SEEDS, epoch_schedule=sched)
        print(f"  {sched:10s}: width={res['final_width_mean']:.4f}±{res['final_width_std']:.4f}, "
              f"cov={res['coverage']:.2f}, mem={res['memory_mean']:.0f}")
        ablation_d[sched] = res

    all_results['ablation_D_epoch_schedule'] = ablation_d

    # Ablation E: Multi-epoch intersection
    print("\nAblation E: Multi-Epoch Intersection")
    with_intersection = run_ablation(DIST, P, SEEDS, use_intersection=True)
    without_intersection = run_ablation(DIST, P, SEEDS, use_intersection=False)
    print(f"  With intersection:    width={with_intersection['final_width_mean']:.4f}±{with_intersection['final_width_std']:.4f}")
    print(f"  Without intersection: width={without_intersection['final_width_mean']:.4f}±{without_intersection['final_width_std']:.4f}")

    all_results['ablation_E_intersection'] = {
        'with_intersection': with_intersection,
        'without_intersection': without_intersection,
    }

    elapsed = time.time() - start_time
    all_results['runtime_seconds'] = elapsed
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    outpath = os.path.join(os.path.dirname(__file__), 'results.json')
    save_results(all_results, outpath)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
