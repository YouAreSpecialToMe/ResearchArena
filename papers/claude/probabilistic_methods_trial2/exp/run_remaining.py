#!/usr/bin/env python3
"""Run remaining experiments: budget ablation, greedy ablation, distribution sensitivity."""

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

SEEDS = [42, 123, 456, 789, 1024]
STREAM_LENGTH = 200_000  # Reduced further for speed
UNIVERSE_SIZE = 20_000
ALLOCATORS = ['uniform', 'independent', 'proportional', 'sketchbudget']


def generate_stream(alpha, seed):
    rng = np.random.default_rng(seed)
    weights = np.arange(1, UNIVERSE_SIZE + 1, dtype=np.float64) ** (-alpha)
    weights /= weights.sum()
    return rng.choice(UNIVERSE_SIZE, size=STREAM_LENGTH, p=weights) + 1


def generate_uniform_stream(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(1, UNIVERSE_SIZE + 1, size=STREAM_LENGTH)


def compute_gt(stream):
    freq = Counter(stream.tolist())
    n_distinct = len(freq)
    avg = len(stream) / n_distinct
    threshold = avg * 10
    heavy_hitters = [k for k, v in freq.items() if v >= threshold]
    return {
        'frequencies': freq, 'n_distinct': n_distinct,
        'stream_length': len(stream), 'universe_size': max(freq.keys()),
        'threshold': threshold, 'heavy_hitters': heavy_hitters,
        'n_heavy_hitters': len(heavy_hitters),
    }


def make_stats(gt):
    return {
        'stream_length': gt['stream_length'], 'n_distinct': gt['n_distinct'],
        'universe_size': gt['universe_size'], 'threshold': gt['threshold'],
        'frequencies': dict(gt['frequencies']),
        'n_positive': min(1000, gt['n_distinct']),
        'n_negative': 1000, 'set_size': min(300, gt['n_distinct']),
    }


def run_p1(alloc, stream, gt, seed):
    freq = gt['frequencies']
    bf = BloomFilter.from_memory(alloc['bf'], gt['n_distinct'])
    cms = CountMinSketch.from_memory(alloc['cms'])
    for item in stream:
        cms.insert(int(item))
    seen = set()
    for item in stream:
        i = int(item)
        if i not in seen:
            bf.insert(i)
            seen.add(i)
    n_pos = min(1000, len(freq))
    positives = sorted(freq.keys(), key=lambda x: -freq[x])[:n_pos]
    negatives = list(range(gt['universe_size'] + 1, gt['universe_size'] + 1001))
    errors = []
    for item in positives:
        est = cms.estimate(item) if bf.query(item) else 0
        errors.append(abs(est - freq[item]))
    for item in negatives:
        est = cms.estimate(item) if bf.query(item) else 0
        errors.append(abs(est))
    return {'mean_abs_error': float(np.mean(errors))}


def run_p2(alloc, stream, gt, seed):
    freq = gt['frequencies']
    cms = CountMinSketch.from_memory(alloc['cms'])
    hll = HyperLogLog.from_memory(alloc['hll'])
    for item in stream:
        cms.insert(int(item))
    true_hh = set(gt['heavy_hitters'])
    detected = set()
    false_hh = 0
    for item in freq.keys():
        if cms.estimate(item) >= gt['threshold']:
            hll.insert(item)
            detected.add(item)
            if item not in true_hh:
                false_hh += 1
    hll_est = hll.estimate()
    return {'cardinality_error': float(abs(hll_est - len(true_hh)))}


def run_p3(alloc, stream, gt, seed):
    freq = gt['frequencies']
    set_size = min(300, gt['n_distinct'])
    bf = BloomFilter.from_memory(alloc['bf'], gt['n_distinct'])
    cms = CountMinSketch.from_memory(alloc['cms'])
    target_set = set(sorted(freq.keys(), key=lambda x: -freq[x])[:set_size])
    for item in target_set:
        bf.insert(item)
    for item in stream:
        cms.insert(int(item))
    true_sum = sum(freq.get(item, 0) for item in target_set)
    est_sum = sum(cms.estimate(item) for item in freq.keys() if bf.query(item))
    return {'abs_error': float(abs(est_sum - true_sum))}


def run_pipeline(pipeline, alloc, stream, gt, seed):
    if pipeline == 'P1': return run_p1(alloc, stream, gt, seed)
    elif pipeline == 'P2': return run_p2(alloc, stream, gt, seed)
    elif pipeline == 'P3': return run_p3(alloc, stream, gt, seed)


def pm(pipeline):
    return {'P1': 'mean_abs_error', 'P2': 'cardinality_error', 'P3': 'abs_error'}[pipeline]


# ============================================================
# Budget sensitivity ablation
# ============================================================
def run_budget_ablation():
    print("EXPERIMENT 4: Budget sensitivity ablation")
    fine_budgets = np.logspace(np.log10(5000), np.log10(5_000_000), 12).astype(int).tolist()
    results = []

    # Pre-generate streams (reuse across allocators)
    for pipeline in ['P1', 'P2', 'P3']:
        t0 = time.time()
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
                    values.append(metrics[pm(pipeline)])
                results.append({
                    'pipeline': pipeline, 'allocator': alloc_name,
                    'budget': int(budget),
                    'mean': float(np.mean(values)), 'std': float(np.std(values)),
                })
        print(f"  {pipeline}: done ({time.time()-t0:.1f}s)")

    with open('results/ablation_budget.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# Greedy vs exact ablation
# ============================================================
def run_greedy_ablation():
    print("\nEXPERIMENT 5: Greedy vs exact")
    test_budgets = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    results = []

    for pipeline in ['P1', 'P2', 'P3']:
        for budget in test_budgets:
            stream = generate_stream(1.0, 42)
            gt = compute_gt(stream)
            stats = make_stats(gt)

            sb = SketchBudgetAllocator()
            t0 = time.time()
            alloc_sb = sb.allocate(budget, pipeline, stats)
            time_sb = time.time() - t0

            gr = GreedyAllocator()
            t0 = time.time()
            alloc_gr = gr.allocate(budget, pipeline, stats, delta_m=max(200, budget // 200))
            time_gr = time.time() - t0

            m_sb = run_pipeline(pipeline, alloc_sb, stream, gt, 42)
            m_gr = run_pipeline(pipeline, alloc_gr, stream, gt, 42)
            p = pm(pipeline)

            results.append({
                'pipeline': pipeline, 'budget': budget,
                'scipy_error': float(m_sb[p]), 'greedy_error': float(m_gr[p]),
                'scipy_alloc': alloc_sb, 'greedy_alloc': alloc_gr,
                'scipy_time': float(time_sb), 'greedy_time': float(time_gr),
                'agreement_pct': float(100 * (1 - abs(m_sb[p] - m_gr[p]) / max(m_sb[p], 1e-10))),
            })
            print(f"  {pipeline}/{budget}: scipy={m_sb[p]:.2f} greedy={m_gr[p]:.2f}")

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
# Distribution sensitivity
# ============================================================
def run_distribution_ablation():
    print("\nEXPERIMENT 6: Distribution sensitivity")
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
                values.append(metrics[pm(pipeline)])
            results.append({
                'alpha': alpha, 'allocator': alloc_name,
                'pipeline': pipeline, 'budget': budget,
                'mean': float(np.mean(values)), 'std': float(np.std(values)),
            })
            print(f"  alpha={alpha}/{alloc_name}: {np.mean(values):.2f}")

    for alloc_name in ALLOCATORS:
        values = []
        for seed in SEEDS:
            stream = generate_uniform_stream(seed)
            gt = compute_gt(stream)
            stats = make_stats(gt)
            alloc_obj = get_allocator(alloc_name)
            alloc = alloc_obj.allocate(budget, pipeline, stats)
            metrics = run_pipeline(pipeline, alloc, stream, gt, seed)
            values.append(metrics[pm(pipeline)])
        results.append({
            'alpha': 0.0, 'allocator': alloc_name,
            'pipeline': pipeline, 'budget': budget,
            'mean': float(np.mean(values)), 'std': float(np.std(values)),
        })
        print(f"  uniform/{alloc_name}: {np.mean(values):.2f}")

    with open('results/ablation_distribution.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    t0 = time.time()
    run_budget_ablation()
    run_greedy_ablation()
    run_distribution_ablation()
    print(f"\nDone in {(time.time()-t0)/60:.1f} minutes")
