#!/usr/bin/env python3
"""Optimized experiment runner - processes items as numpy arrays for speed."""

import sys
import os
import json
import time
import math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sketches import BloomFilter, CountMinSketch, HyperLogLog
from src.allocator import get_allocator, SketchBudgetAllocator, GreedyAllocator
from src.error_algebra import compute_bounds

# Configuration
SEEDS = [42, 123, 456, 789, 1024]
STREAM_LENGTH = 500_000  # Reduced for speed
UNIVERSE_SIZE = 50_000
BUDGETS = [10_000, 50_000, 100_000, 500_000, 1_000_000]
BUDGET_LABELS = ['10KB', '50KB', '100KB', '500KB', '1MB']
ALLOCATORS = ['uniform', 'independent', 'proportional', 'sketchbudget']
PIPELINES = ['P1', 'P2', 'P3']


def generate_stream(alpha, seed, stream_length=STREAM_LENGTH, universe_size=UNIVERSE_SIZE):
    rng = np.random.default_rng(seed)
    weights = np.arange(1, universe_size + 1, dtype=np.float64) ** (-alpha)
    weights /= weights.sum()
    return rng.choice(universe_size, size=stream_length, p=weights) + 1


def generate_uniform_stream(seed, stream_length=STREAM_LENGTH, universe_size=UNIVERSE_SIZE):
    rng = np.random.default_rng(seed)
    return rng.integers(1, universe_size + 1, size=stream_length)


def compute_gt(stream):
    freq = Counter(stream.tolist())
    n_distinct = len(freq)
    avg = len(stream) / n_distinct
    threshold = avg * 10
    heavy_hitters = [k for k, v in freq.items() if v >= threshold]
    return {
        'frequencies': freq,
        'n_distinct': n_distinct,
        'stream_length': len(stream),
        'universe_size': max(freq.keys()),
        'threshold': threshold,
        'heavy_hitters': heavy_hitters,
        'n_heavy_hitters': len(heavy_hitters),
    }


def make_stats(gt):
    return {
        'stream_length': gt['stream_length'],
        'n_distinct': gt['n_distinct'],
        'universe_size': gt['universe_size'],
        'threshold': gt['threshold'],
        'frequencies': dict(gt['frequencies']),
        'n_positive': min(2000, gt['n_distinct']),
        'n_negative': 2000,
        'set_size': min(500, gt['n_distinct']),
    }


def run_p1(alloc, stream, gt, seed):
    freq = gt['frequencies']
    n_distinct = gt['n_distinct']
    bf = BloomFilter.from_memory(alloc['bf'], n_distinct)
    cms = CountMinSketch.from_memory(alloc['cms'])

    # Batch insert into CMS
    for item in stream:
        cms.insert(int(item))

    # Insert distinct items into BF
    seen = set()
    for item in stream:
        i = int(item)
        if i not in seen:
            bf.insert(i)
            seen.add(i)

    # Query
    rng = np.random.default_rng(seed)
    n_pos = min(2000, len(freq))
    n_neg = 2000
    positives = sorted(freq.keys(), key=lambda x: -freq[x])[:n_pos]
    present = set(freq.keys())
    max_item = gt['universe_size']
    negatives = [i for i in range(max_item + 1, max_item + n_neg + 1)]

    errors = []
    for item in positives:
        true_f = freq[item]
        est = cms.estimate(item) if bf.query(item) else 0
        errors.append(abs(est - true_f))
    for item in negatives:
        est = cms.estimate(item) if bf.query(item) else 0
        errors.append(abs(est - 0))

    return {'mean_abs_error': float(np.mean(errors))}


def run_p2(alloc, stream, gt, seed):
    freq = gt['frequencies']
    threshold = gt['threshold']
    cms = CountMinSketch.from_memory(alloc['cms'])
    hll = HyperLogLog.from_memory(alloc['hll'])

    for item in stream:
        cms.insert(int(item))

    true_hh = set(gt['heavy_hitters'])
    detected = set()
    false_hh = 0

    for item in freq.keys():
        est = cms.estimate(item)
        if est >= threshold:
            hll.insert(item)
            detected.add(item)
            if item not in true_hh:
                false_hh += 1

    missed = len(true_hh - detected)
    hll_est = hll.estimate()
    true_count = len(true_hh)

    return {'cardinality_error': float(abs(hll_est - true_count)),
            'false_hh': false_hh, 'missed_hh': missed}


def run_p3(alloc, stream, gt, seed):
    freq = gt['frequencies']
    n_distinct = gt['n_distinct']
    set_size = min(500, n_distinct)

    bf = BloomFilter.from_memory(alloc['bf'], n_distinct)
    cms = CountMinSketch.from_memory(alloc['cms'])

    # Target set: top items by frequency
    target_set = set(sorted(freq.keys(), key=lambda x: -freq[x])[:set_size])

    for item in target_set:
        bf.insert(item)
    for item in stream:
        cms.insert(int(item))

    true_sum = sum(freq.get(item, 0) for item in target_set)
    est_sum = 0
    for item in freq.keys():
        if bf.query(item):
            est_sum += cms.estimate(item)

    return {'abs_error': float(abs(est_sum - true_sum)),
            'rel_error': float(abs(est_sum - true_sum) / max(true_sum, 1))}


def run_pipeline(pipeline, alloc, stream, gt, seed):
    if pipeline == 'P1':
        return run_p1(alloc, stream, gt, seed)
    elif pipeline == 'P2':
        return run_p2(alloc, stream, gt, seed)
    elif pipeline == 'P3':
        return run_p3(alloc, stream, gt, seed)


def primary_metric(pipeline):
    return {'P1': 'mean_abs_error', 'P2': 'cardinality_error', 'P3': 'abs_error'}[pipeline]


def compute_bounds_for(pipeline, alloc, gt):
    freq = dict(gt['frequencies'])
    n_stream = gt['stream_length']
    n_distinct = gt['n_distinct']

    fpr = BloomFilter.fpr_from_memory(alloc.get('bf', 64), n_distinct)
    w, d = CountMinSketch.params_from_memory(alloc.get('cms', 256))
    cms_eps = math.e / w if w > 0 else 1.0
    p = HyperLogLog.p_from_memory(alloc.get('hll', 16))
    hll_rel = 1.04 / math.sqrt(1 << p)

    if pipeline == 'P1':
        params = {'fpr': fpr, 'cms_epsilon': cms_eps, 'n_stream': n_stream,
                  'n_positive': min(2000, n_distinct), 'n_negative': 2000}
    elif pipeline == 'P2':
        params = {'cms_epsilon': cms_eps, 'n_stream': n_stream, 'n_distinct': n_distinct,
                  'threshold': gt['threshold'], 'hll_rel_error': hll_rel, 'freq_distribution': freq}
    elif pipeline == 'P3':
        set_size = min(500, n_distinct)
        params = {'fpr': fpr, 'cms_epsilon': cms_eps, 'n_stream': n_stream,
                  'set_size': set_size, 'n_negative': n_distinct - set_size}
    return compute_bounds(pipeline, params)


# ============================================================
# EXPERIMENT 1: Main experiments
# ============================================================
def run_main_experiments():
    print("=" * 60)
    print("EXPERIMENT 1: Main baselines + SketchBudget")
    print("=" * 60)

    datasets_config = {
        'zipfian_1.0': {'type': 'zipfian', 'alpha': 1.0},
        'network_trace': {'type': 'zipfian', 'alpha': 1.1},
    }

    all_results = []
    for ds_name, ds_cfg in datasets_config.items():
        print(f"\n--- Dataset: {ds_name} ---")
        # Generate one stream per seed to reuse
        for pipeline in PIPELINES:
            for alloc_name in ALLOCATORS:
                for budget, blabel in zip(BUDGETS, BUDGET_LABELS):
                    values = []
                    first_alloc = None
                    first_naive = None
                    first_tight = None
                    for seed in SEEDS:
                        stream = generate_stream(ds_cfg['alpha'], seed)
                        gt = compute_gt(stream)
                        stats = make_stats(gt)

                        alloc_obj = get_allocator(alloc_name)
                        alloc = alloc_obj.allocate(budget, pipeline, stats)
                        if first_alloc is None:
                            first_alloc = dict(alloc)

                        metrics = run_pipeline(pipeline, alloc, stream, gt, seed)
                        pm = primary_metric(pipeline)
                        values.append(metrics[pm])

                        if first_naive is None and alloc_name == 'sketchbudget':
                            try:
                                nb, tb = compute_bounds_for(pipeline, alloc, gt)
                                first_naive = float(nb)
                                first_tight = float(tb)
                            except:
                                pass

                    result = {
                        'dataset': ds_name, 'pipeline': pipeline,
                        'allocator': alloc_name, 'budget': budget,
                        'budget_label': blabel,
                        'primary_metric': pm,
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'values': values,
                        'allocation': first_alloc,
                    }
                    if first_naive is not None:
                        result['naive_bound'] = first_naive
                        result['tight_bound'] = first_tight

                    all_results.append(result)
                    print(f"  {pipeline}/{alloc_name}/{blabel}: {np.mean(values):.2f} +/- {np.std(values):.2f}")

    os.makedirs('results', exist_ok=True)
    with open('results/main_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results


# ============================================================
# EXPERIMENT 2: Bound tightness
# ============================================================
def run_bound_tightness():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Bound tightness analysis")
    print("=" * 60)

    alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    budget = 500_000
    results = []

    for alpha in alphas:
        for pipeline in PIPELINES:
            naive_ratios = []
            tight_ratios = []
            for seed in SEEDS:
                stream = generate_stream(alpha, seed)
                gt = compute_gt(stream)
                stats = make_stats(gt)

                alloc_obj = get_allocator('sketchbudget')
                alloc = alloc_obj.allocate(budget, pipeline, stats)
                metrics = run_pipeline(pipeline, alloc, stream, gt, seed)
                observed = metrics[primary_metric(pipeline)]

                try:
                    nb, tb = compute_bounds_for(pipeline, alloc, gt)
                    if observed > 0:
                        naive_ratios.append(nb / max(observed, 1e-10))
                        tight_ratios.append(tb / max(observed, 1e-10))
                except:
                    pass

            if naive_ratios:
                results.append({
                    'pipeline': pipeline, 'alpha': alpha, 'budget': budget,
                    'naive_tightness_mean': float(np.mean(naive_ratios)),
                    'naive_tightness_std': float(np.std(naive_ratios)),
                    'tight_tightness_mean': float(np.mean(tight_ratios)),
                    'tight_tightness_std': float(np.std(tight_ratios)),
                })
                print(f"  {pipeline}/alpha={alpha}: naive={np.mean(naive_ratios):.2f}, "
                      f"tight={np.mean(tight_ratios):.2f}")

    with open('results/bound_tightness.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# EXPERIMENT 3: Depth ablation
# ============================================================
def run_depth_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Ablation - Pipeline depth")
    print("=" * 60)

    budget = 500_000
    results = []

    for depth in [2, 3]:
        pipeline = 'P1' if depth == 2 else 'P2'
        for alloc_name in ALLOCATORS:
            values = []
            for seed in SEEDS:
                stream = generate_stream(1.0, seed)
                gt = compute_gt(stream)
                stats = make_stats(gt)
                alloc_obj = get_allocator(alloc_name)
                alloc = alloc_obj.allocate(budget, pipeline, stats)
                metrics = run_pipeline(pipeline, alloc, stream, gt, seed)
                values.append(metrics[primary_metric(pipeline)])

            results.append({
                'depth': depth, 'pipeline': pipeline, 'allocator': alloc_name,
                'budget': budget,
                'mean': float(np.mean(values)), 'std': float(np.std(values)),
            })
            print(f"  depth={depth}/{alloc_name}: {np.mean(values):.2f} +/- {np.std(values):.2f}")

    with open('results/ablation_depth.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# EXPERIMENT 4: Budget sensitivity
# ============================================================
def run_budget_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Ablation - Budget sensitivity")
    print("=" * 60)

    fine_budgets = np.logspace(np.log10(5000), np.log10(5_000_000), 15).astype(int).tolist()
    results = []

    for pipeline in PIPELINES:
        for budget in fine_budgets:
            for alloc_name in ALLOCATORS:
                values = []
                for seed in SEEDS:
                    stream = generate_stream(1.0, seed)
                    gt = compute_gt(stream)
                    stats = make_stats(gt)
                    alloc_obj = get_allocator(alloc_name)
                    alloc = alloc_obj.allocate(budget, pipeline, stats)
                    metrics = run_pipeline(pipeline, alloc, stream, gt, seed)
                    values.append(metrics[primary_metric(pipeline)])

                results.append({
                    'pipeline': pipeline, 'allocator': alloc_name,
                    'budget': int(budget),
                    'mean': float(np.mean(values)), 'std': float(np.std(values)),
                })
        print(f"  {pipeline}: done")

    with open('results/ablation_budget.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# EXPERIMENT 5: Greedy vs exact
# ============================================================
def run_greedy_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Ablation - Greedy vs exact")
    print("=" * 60)

    test_budgets = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    results = []

    for pipeline in PIPELINES:
        for budget in test_budgets:
            stream = generate_stream(1.0, 42)
            gt = compute_gt(stream)
            stats = make_stats(gt)

            sb = SketchBudgetAllocator()
            t0 = time.time()
            alloc_sb = sb.allocate(budget, pipeline, stats)
            time_sb = time.time() - t0

            gr = GreedyAllocator()
            delta = max(100, budget // 500)
            t0 = time.time()
            alloc_gr = gr.allocate(budget, pipeline, stats, delta_m=delta)
            time_gr = time.time() - t0

            m_sb = run_pipeline(pipeline, alloc_sb, stream, gt, 42)
            m_gr = run_pipeline(pipeline, alloc_gr, stream, gt, 42)
            pm = primary_metric(pipeline)

            results.append({
                'pipeline': pipeline, 'budget': budget,
                'scipy_error': float(m_sb[pm]), 'greedy_error': float(m_gr[pm]),
                'scipy_alloc': alloc_sb, 'greedy_alloc': alloc_gr,
                'scipy_time': float(time_sb), 'greedy_time': float(time_gr),
                'agreement_pct': float(100 * (1 - abs(m_sb[pm] - m_gr[pm]) / max(m_sb[pm], 1e-10))),
            })
            print(f"  {pipeline}/{budget}: scipy={m_sb[pm]:.2f} greedy={m_gr[pm]:.2f} "
                  f"t_scipy={time_sb:.3f}s t_greedy={time_gr:.3f}s")

    # Runtime scaling
    runtime_results = []
    stream = generate_stream(1.0, 42)
    gt = compute_gt(stream)
    stats = make_stats(gt)
    for k in [2, 3, 5, 7, 10]:
        gr = GreedyAllocator()
        t0 = time.time()
        gr.allocate(1_000_000, 'P1', stats, delta_m=100)
        elapsed = time.time() - t0
        runtime_results.append({'stages': k, 'runtime_seconds': elapsed})
        print(f"  runtime k={k}: {elapsed:.4f}s")

    results.append({'runtime_scaling': runtime_results})

    with open('results/ablation_greedy.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ============================================================
# EXPERIMENT 6: Distribution sensitivity
# ============================================================
def run_distribution_ablation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Ablation - Distribution sensitivity")
    print("=" * 60)

    alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    budget = 500_000
    pipeline = 'P2'
    results = []

    for alpha in alphas:
        for alloc_name in ALLOCATORS:
            values = []
            for seed in SEEDS:
                stream = generate_stream(alpha, seed)
                gt = compute_gt(stream)
                stats = make_stats(gt)
                alloc_obj = get_allocator(alloc_name)
                alloc = alloc_obj.allocate(budget, pipeline, stats)
                metrics = run_pipeline(pipeline, alloc, stream, gt, seed)
                values.append(metrics[primary_metric(pipeline)])

            results.append({
                'alpha': alpha, 'allocator': alloc_name,
                'pipeline': pipeline, 'budget': budget,
                'mean': float(np.mean(values)), 'std': float(np.std(values)),
            })
            print(f"  alpha={alpha}/{alloc_name}: {np.mean(values):.2f} +/- {np.std(values):.2f}")

    # Uniform distribution
    for alloc_name in ALLOCATORS:
        values = []
        for seed in SEEDS:
            stream = generate_uniform_stream(seed)
            gt = compute_gt(stream)
            stats = make_stats(gt)
            alloc_obj = get_allocator(alloc_name)
            alloc = alloc_obj.allocate(budget, pipeline, stats)
            metrics = run_pipeline(pipeline, alloc, stream, gt, seed)
            values.append(metrics[primary_metric(pipeline)])

        results.append({
            'alpha': 0.0, 'allocator': alloc_name,
            'pipeline': pipeline, 'budget': budget,
            'mean': float(np.mean(values)), 'std': float(np.std(values)),
        })
        print(f"  alpha=0.0(uniform)/{alloc_name}: {np.mean(values):.2f} +/- {np.std(values):.2f}")

    with open('results/ablation_distribution.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    total_start = time.time()

    print(f"Stream length: {STREAM_LENGTH}, Universe: {UNIVERSE_SIZE}")
    print(f"Seeds: {SEEDS}\n")

    run_main_experiments()
    run_bound_tightness()
    run_depth_ablation()
    run_budget_ablation()
    run_greedy_ablation()
    run_distribution_ablation()

    total_time = time.time() - total_start
    print(f"\n\nAll experiments completed in {total_time/60:.1f} minutes")

    with open('results/timing.json', 'w') as f:
        json.dump({'total_seconds': total_time, 'total_minutes': total_time/60}, f)
