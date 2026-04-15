"""Run experiments on NASA HTTP access logs."""

import gzip
import re
import json
import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sketches import BloomFilter, CountMinSketch, HyperLogLog
from src.allocator import get_allocator
from src.pipeline import run_experiment, get_primary_metric

def parse_nasa_logs(filepath):
    """Parse NASA HTTP access logs, extract hostnames as stream items."""
    items = []
    pattern = re.compile(r'^(\S+)')
    with gzip.open(filepath, 'rt', errors='replace') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                items.append(m.group(1))
    return items

def prepare_data(items, max_items=1000000):
    """Convert string items to integers and compute ground truth."""
    # Map hostnames to integers
    item_map = {}
    counter = 0
    stream = []
    for item in items[:max_items]:
        if item not in item_map:
            item_map[item] = counter
            counter += 1
        stream.append(item_map[item])

    freq = Counter(stream)
    n_distinct = len(freq)
    stream_length = len(stream)

    # Heavy hitter threshold: items with freq >= 10x average
    avg_freq = stream_length / n_distinct
    threshold = max(int(avg_freq * 10), 10)
    heavy_hitters = [item for item, f in freq.items() if f >= threshold]

    # Watchlist: top 1000 items by frequency
    top_items = sorted(freq.keys(), key=lambda x: -freq[x])[:1000]

    ground_truth = {
        'frequencies': {int(k): int(v) for k, v in freq.items()},
        'universe_size': counter,
        'heavy_hitters': [int(x) for x in heavy_hitters],
        'watchlist': [int(x) for x in top_items],
    }

    data_stats = {
        'stream_length': stream_length,
        'n_distinct': n_distinct,
        'threshold': threshold,
        'n_positive': min(5000, n_distinct),
        'n_negative': min(5000, counter - n_distinct + 5000),
        'set_size': 1000,
        'frequencies': {int(k): int(v) for k, v in freq.items()},
    }

    return np.array(stream), ground_truth, data_stats

def run_nasa_experiments():
    nasa_path = '/tmp/nasa_jul95.gz'
    if not os.path.exists(nasa_path):
        print("NASA data not found, exiting")
        return

    print("Parsing NASA HTTP logs...")
    items = parse_nasa_logs(nasa_path)
    print(f"  Total log entries: {len(items)}")

    print("Preparing data (using first 1M items)...")
    stream, ground_truth, data_stats = prepare_data(items, max_items=1000000)
    print(f"  Stream length: {data_stats['stream_length']}")
    print(f"  Distinct items: {data_stats['n_distinct']}")
    print(f"  Heavy hitters: {len(ground_truth['heavy_hitters'])} (threshold={data_stats['threshold']})")

    budgets = [10000, 50000, 100000, 500000, 1000000]
    allocators = ['uniform', 'independent', 'proportional', 'sketchbudget']
    pipelines = ['P1', 'P2', 'P3']
    seeds = [42, 123, 456, 789, 1024]

    results = []

    for pipeline in pipelines:
        primary_metric = get_primary_metric(pipeline)
        for budget in budgets:
            for alloc_name in allocators:
                alloc_obj = get_allocator(alloc_name)
                alloc = alloc_obj.allocate(budget, pipeline, data_stats)

                values = []
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    # Shuffle stream slightly for different seeds
                    idx = rng.permutation(len(stream))
                    stream_seed = stream[idx]

                    metrics = run_experiment(pipeline, alloc, stream_seed,
                                           data_stats, ground_truth, seed)
                    values.append(metrics[primary_metric])

                result = {
                    'dataset': 'nasa_http',
                    'pipeline': pipeline,
                    'allocator': alloc_name,
                    'budget': budget,
                    'budget_label': f'{budget//1000}KB' if budget < 1000000 else f'{budget//1000000}MB',
                    'primary_metric': primary_metric,
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': [float(v) for v in values],
                    'allocation': alloc,
                }

                # Add bounds for sketchbudget
                if alloc_name == 'sketchbudget':
                    from src.error_algebra import compute_bounds
                    try:
                        bounds = compute_bounds(pipeline, alloc, data_stats)
                        result['naive_bound'] = bounds.get('naive_bound', None)
                        result['tight_bound'] = bounds.get('tight_bound', None)
                    except:
                        pass

                results.append(result)
                print(f"  {pipeline}/{alloc_name}/{budget//1000}KB: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'nasa_experiments.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n=== NASA HTTP Results Summary (500KB budget) ===")
    for pipeline in pipelines:
        print(f"\n{pipeline}:")
        for alloc_name in allocators:
            for r in results:
                if r['pipeline'] == pipeline and r['allocator'] == alloc_name and r['budget'] == 500000:
                    print(f"  {alloc_name:15s}: {r['mean']:.4f} +/- {r['std']:.4f}")

if __name__ == '__main__':
    run_nasa_experiments()
