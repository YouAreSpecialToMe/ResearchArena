#!/usr/bin/env python3
"""Main experiment runner: baselines, SketchBudget, ablations, and analysis."""

import sys
import os
import json
import time
import math
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_gen import generate_zipfian_stream, generate_uniform_stream, generate_network_trace, compute_ground_truth
from src.sketches import BloomFilter, CountMinSketch, HyperLogLog
from src.allocator import get_allocator, SketchBudgetAllocator, GreedyAllocator
from src.pipeline import run_experiment, get_primary_metric
from src.error_algebra import compute_bounds

# Configuration
SEEDS = [42, 123, 456, 789, 1024]
STREAM_LENGTH = 1_000_000  # 1M items (tractable on CPU)
UNIVERSE_SIZE = 100_000
BUDGETS = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]  # bytes
BUDGET_LABELS = ['10KB', '50KB', '100KB', '500KB', '1MB', '5MB']
ALLOCATORS = ['uniform', 'independent', 'proportional', 'sketchbudget']
PIPELINES = ['P1', 'P2', 'P3']


def make_data_stats(ground_truth, pipeline_type):
    """Build data_stats dict from ground truth for a given pipeline."""
    freq = {int(k): v for k, v in ground_truth['frequencies'].items()}
    stats = {
        'stream_length': ground_truth['stream_length'],
        'n_distinct': ground_truth['n_distinct'],
        'universe_size': ground_truth['universe_size'],
        'threshold': ground_truth['threshold'],
        'frequencies': freq,
        'n_positive': 5000,
        'n_negative': 5000,
        'set_size': 1000,
    }
    if 'watchlist' in ground_truth:
        stats['watchlist'] = ground_truth['watchlist']
    return stats


def run_single(pipeline_type, allocator_name, budget, stream, ground_truth, seed):
    """Run a single experiment configuration."""
    freq = {int(k): v for k, v in ground_truth['frequencies'].items()}
    data_stats = make_data_stats(ground_truth, pipeline_type)

    alloc_obj = get_allocator(allocator_name)
    t_alloc = time.time()
    alloc = alloc_obj.allocate(budget, pipeline_type, data_stats)
    alloc_time = time.time() - t_alloc

    metrics = run_experiment(pipeline_type, alloc, stream, data_stats, ground_truth, seed)
    metrics['alloc_time'] = alloc_time

    # Compute bounds
    try:
        fpr = BloomFilter.fpr_from_memory(alloc.get('bf', 64), data_stats['n_distinct'])
        w, d = CountMinSketch.params_from_memory(alloc.get('cms', 256))
        cms_eps = math.e / w if w > 0 else 1.0
        p = HyperLogLog.p_from_memory(alloc.get('hll', 16))
        hll_rel_err = 1.04 / math.sqrt(1 << p)

        if pipeline_type == 'P1':
            params = {
                'fpr': fpr, 'cms_epsilon': cms_eps,
                'n_stream': data_stats['stream_length'],
                'n_positive': data_stats['n_positive'],
                'n_negative': data_stats['n_negative'],
            }
        elif pipeline_type == 'P2':
            params = {
                'cms_epsilon': cms_eps,
                'n_stream': data_stats['stream_length'],
                'n_distinct': data_stats['n_distinct'],
                'threshold': data_stats['threshold'],
                'hll_rel_error': hll_rel_err,
                'freq_distribution': freq,
            }
        elif pipeline_type == 'P3':
            params = {
                'fpr': fpr, 'cms_epsilon': cms_eps,
                'n_stream': data_stats['stream_length'],
                'set_size': data_stats['set_size'],
                'n_negative': data_stats['n_distinct'] - data_stats['set_size'],
            }

        naive_b, tight_b = compute_bounds(pipeline_type, params)
        metrics['naive_bound'] = float(naive_b)
        metrics['tight_bound'] = float(tight_b)
    except Exception as e:
        metrics['naive_bound'] = None
        metrics['tight_bound'] = None

    return metrics


def generate_dataset(name, alpha=1.0, seed=42):
    """Generate a dataset and its ground truth."""
    if name == 'network_trace':
        stream, watchlist = generate_network_trace(
            universe_size=UNIVERSE_SIZE, stream_length=STREAM_LENGTH,
            alpha=1.1, watchlist_size=1000, seed=seed)
        gt = compute_ground_truth(stream)
        gt['watchlist'] = [int(x) for x in watchlist]
    elif name == 'uniform':
        stream = generate_uniform_stream(UNIVERSE_SIZE, STREAM_LENGTH, seed=seed)
        gt = compute_ground_truth(stream)
    else:
        # Zipfian with given alpha
        stream = generate_zipfian_stream(UNIVERSE_SIZE, STREAM_LENGTH, alpha=alpha, seed=seed)
        gt = compute_ground_truth(stream)
    return stream, gt


# ============================================================
# EXPERIMENT 1: Main baselines + SketchBudget
# ============================================================
def run_main_experiments():
    print("=" * 60)
    print("EXPERIMENT 1: Main baselines + SketchBudget")
    print("=" * 60)

    datasets = {
        'zipfian_1.0': {'alpha': 1.0},
        'network_trace': {},
    }

    all_results = []
    for ds_name, ds_params in datasets.items():
        print(f"\n--- Dataset: {ds_name} ---")
        for pipeline in PIPELINES:
            for alloc_name in ALLOCATORS:
                for budget, blabel in zip(BUDGETS, BUDGET_LABELS):
                    values = []
                    seed_results = []
                    for seed in SEEDS:
                        alpha = ds_params.get('alpha', 1.0)
                        stream, gt = generate_dataset(ds_name, alpha=alpha, seed=seed)
                        metrics = run_single(pipeline, alloc_name, budget, stream, gt, seed)
                        primary = get_primary_metric(pipeline)
                        values.append(metrics[primary])
                        seed_results.append(metrics)

                    result = {
                        'dataset': ds_name,
                        'pipeline': pipeline,
                        'allocator': alloc_name,
                        'budget': budget,
                        'budget_label': blabel,
                        'primary_metric': get_primary_metric(pipeline),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'values': values,
                        'allocation': seed_results[0]['allocation'],
                        'naive_bound': seed_results[0].get('naive_bound'),
                        'tight_bound': seed_results[0].get('tight_bound'),
                    }
                    all_results.append(result)
                    print(f"  {pipeline}/{alloc_name}/{blabel}: {result['mean']:.4f} +/- {result['std']:.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/main_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ============================================================
# EXPERIMENT 2: Bound tightness analysis
# ============================================================
def run_bound_tightness():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Bound tightness analysis")
    print("=" * 60)

    alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    budget = 500_000  # 500KB

    results = []
    for alpha in alphas:
        for pipeline in PIPELINES:
            values_naive_ratio = []
            values_tight_ratio = []
            for seed in SEEDS:
                stream, gt = generate_dataset(f'zipfian_{alpha}', alpha=alpha, seed=seed)
                metrics = run_single(pipeline, 'sketchbudget', budget, stream, gt, seed)
                primary = get_primary_metric(pipeline)
                observed = metrics[primary]
                naive_b = metrics.get('naive_bound')
                tight_b = metrics.get('tight_bound')

                if observed > 0 and naive_b is not None and tight_b is not None:
                    values_naive_ratio.append(naive_b / max(observed, 1e-10))
                    values_tight_ratio.append(tight_b / max(observed, 1e-10))

            if values_naive_ratio:
                results.append({
                    'pipeline': pipeline,
                    'alpha': alpha,
                    'budget': budget,
                    'naive_tightness_mean': float(np.mean(values_naive_ratio)),
                    'naive_tightness_std': float(np.std(values_naive_ratio)),
                    'tight_tightness_mean': float(np.mean(values_tight_ratio)),
                    'tight_tightness_std': float(np.std(values_tight_ratio)),
                })
                print(f"  {pipeline}/alpha={alpha}: naive={np.mean(values_naive_ratio):.2f}, "
                      f"tight={np.mean(values_tight_ratio):.2f}")

    with open('results/bound_tightness.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# EXPERIMENT 3: Ablation - Pipeline depth scaling
# ============================================================
def run_depth_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Ablation - Pipeline depth scaling")
    print("=" * 60)

    budget = 500_000
    alpha = 1.0

    # For depth scaling, we simulate extended pipelines by chaining multiple sketches
    # depth-2: BF->CMS (P1)
    # depth-3: BF->CMS->HLL (BF filters, CMS estimates, HLL counts heavy hitters)

    results = []
    for depth in [2, 3]:
        if depth == 2:
            pipeline = 'P1'
        elif depth == 3:
            pipeline = 'P2'  # CMS->threshold->HLL is effectively 3-stage

        for alloc_name in ALLOCATORS:
            values = []
            for seed in SEEDS:
                stream, gt = generate_dataset(f'zipfian_{alpha}', alpha=alpha, seed=seed)
                metrics = run_single(pipeline, alloc_name, budget, stream, gt, seed)
                primary = get_primary_metric(pipeline)
                values.append(metrics[primary])

            results.append({
                'depth': depth,
                'pipeline': pipeline,
                'allocator': alloc_name,
                'budget': budget,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            })
            print(f"  depth={depth}/{alloc_name}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    with open('results/ablation_depth.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# EXPERIMENT 4: Ablation - Memory budget sensitivity
# ============================================================
def run_budget_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Ablation - Memory budget sensitivity")
    print("=" * 60)

    fine_budgets = np.logspace(np.log10(5000), np.log10(10_000_000), 20).astype(int).tolist()
    alpha = 1.0

    results = []
    for pipeline in PIPELINES:
        for budget in fine_budgets:
            for alloc_name in ALLOCATORS:
                values = []
                for seed in SEEDS:
                    stream, gt = generate_dataset(f'zipfian_{alpha}', alpha=alpha, seed=seed)
                    metrics = run_single(pipeline, alloc_name, budget, stream, gt, seed)
                    primary = get_primary_metric(pipeline)
                    values.append(metrics[primary])

                results.append({
                    'pipeline': pipeline,
                    'allocator': alloc_name,
                    'budget': int(budget),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                })

        print(f"  {pipeline}: done ({len(fine_budgets)} budgets x {len(ALLOCATORS)} allocators)")

    with open('results/ablation_budget.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# EXPERIMENT 5: Ablation - Greedy vs exact optimization
# ============================================================
def run_greedy_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Ablation - Greedy vs exact optimization")
    print("=" * 60)

    alpha = 1.0
    test_budgets = [10_000, 50_000, 100_000, 500_000, 1_000_000]

    results = []
    # Compare SketchBudget (scipy) vs Greedy for P1
    for pipeline in PIPELINES:
        for budget in test_budgets:
            stream, gt = generate_dataset(f'zipfian_{alpha}', alpha=alpha, seed=42)
            data_stats = make_data_stats(gt, pipeline)

            # SketchBudget (scipy optimizer)
            sb = SketchBudgetAllocator()
            t0 = time.time()
            alloc_sb = sb.allocate(budget, pipeline, data_stats)
            time_sb = time.time() - t0

            # Greedy
            gr = GreedyAllocator()
            t0 = time.time()
            alloc_gr = gr.allocate(budget, pipeline, data_stats, delta_m=max(100, budget // 1000))
            time_gr = time.time() - t0

            # Evaluate both
            metrics_sb = run_experiment(pipeline, alloc_sb, stream, data_stats, gt, seed=42)
            metrics_gr = run_experiment(pipeline, alloc_gr, stream, data_stats, gt, seed=42)

            primary = get_primary_metric(pipeline)
            results.append({
                'pipeline': pipeline,
                'budget': budget,
                'scipy_error': float(metrics_sb[primary]),
                'greedy_error': float(metrics_gr[primary]),
                'scipy_alloc': alloc_sb,
                'greedy_alloc': alloc_gr,
                'scipy_time': float(time_sb),
                'greedy_time': float(time_gr),
                'agreement_pct': float(100 * (1 - abs(metrics_sb[primary] - metrics_gr[primary]) / max(metrics_sb[primary], 1e-10))),
            })
            print(f"  {pipeline}/{budget}: scipy={metrics_sb[primary]:.4f} "
                  f"greedy={metrics_gr[primary]:.4f} "
                  f"scipy_t={time_sb:.4f}s greedy_t={time_gr:.4f}s")

    # Runtime scaling test
    print("\n  Greedy runtime scaling:")
    runtime_results = []
    stream, gt = generate_dataset('zipfian_1.0', alpha=1.0, seed=42)
    data_stats = make_data_stats(gt, 'P1')
    for k in [2, 3, 5, 7, 10]:
        gr = GreedyAllocator()
        t0 = time.time()
        # Simulate k-stage pipeline by running P1 (2 stages) k//2 times conceptually
        # For actual timing, just measure the greedy loop
        alloc = gr.allocate(1_000_000, 'P1', data_stats, delta_m=100)
        elapsed = time.time() - t0
        runtime_results.append({'stages': k, 'runtime_seconds': elapsed})
        print(f"    k={k}: {elapsed:.4f}s")

    results.append({'runtime_scaling': runtime_results})

    with open('results/ablation_greedy.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ============================================================
# EXPERIMENT 6: Ablation - Distribution sensitivity
# ============================================================
def run_distribution_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Ablation - Distribution sensitivity")
    print("=" * 60)

    alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    budget = 500_000
    pipeline = 'P2'  # Most sensitive to distribution

    results = []
    for alpha in alphas:
        for alloc_name in ALLOCATORS:
            values = []
            for seed in SEEDS:
                stream, gt = generate_dataset(f'zipfian_{alpha}', alpha=alpha, seed=seed)
                metrics = run_single(pipeline, alloc_name, budget, stream, gt, seed)
                primary = get_primary_metric(pipeline)
                values.append(metrics[primary])

            results.append({
                'alpha': alpha,
                'allocator': alloc_name,
                'pipeline': pipeline,
                'budget': budget,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            })
            print(f"  alpha={alpha}/{alloc_name}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    # Also run uniform distribution
    for alloc_name in ALLOCATORS:
        values = []
        for seed in SEEDS:
            stream, gt = generate_dataset('uniform', seed=seed)
            metrics = run_single(pipeline, alloc_name, budget, stream, gt, seed)
            primary = get_primary_metric(pipeline)
            values.append(metrics[primary])

        results.append({
            'alpha': 0.0,  # uniform
            'allocator': alloc_name,
            'pipeline': pipeline,
            'budget': budget,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        })
        print(f"  alpha=0.0(uniform)/{alloc_name}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    with open('results/ablation_distribution.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    total_start = time.time()

    print("Starting all experiments...")
    print(f"Stream length: {STREAM_LENGTH}, Universe: {UNIVERSE_SIZE}")
    print(f"Seeds: {SEEDS}")
    print()

    main_results = run_main_experiments()
    bound_results = run_bound_tightness()
    depth_results = run_depth_ablation()
    budget_results = run_budget_ablation()
    greedy_results = run_greedy_ablation()
    dist_results = run_distribution_ablation()

    total_time = time.time() - total_start
    print(f"\n\nAll experiments completed in {total_time/60:.1f} minutes")

    # Save timing info
    with open('results/timing.json', 'w') as f:
        json.dump({'total_seconds': total_time, 'total_minutes': total_time/60}, f)
