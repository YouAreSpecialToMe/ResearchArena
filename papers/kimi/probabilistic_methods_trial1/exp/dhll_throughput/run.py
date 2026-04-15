#!/usr/bin/env python3
"""
D-HLL Throughput and Memory Evaluation.
Measures processing speed and memory usage.
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import time
from pathlib import Path
from dhll import DecayingHyperLogLog
from hll import HyperLogLog
from sliding_hll import SlidingHyperLogLog

def measure_throughput(sketch_class, n_elements, **sketch_kwargs):
    """Measure throughput of a sketch class."""
    sketch = sketch_class(**sketch_kwargs)
    
    start = time.time()
    for i in range(n_elements):
        sketch.add(f"item_{i}", timestamp=i)
    elapsed = time.time() - start
    
    throughput = n_elements / elapsed
    
    return {
        "throughput": throughput,
        "elements": n_elements,
        "time_seconds": elapsed,
        "memory_bytes": sketch.memory_bytes()
    }

if __name__ == "__main__":
    Path("exp/dhll_throughput").mkdir(exist_ok=True)
    
    n_elements = 1_000_000
    seeds = [42, 123, 999, 456, 789]
    
    results = {}
    
    # Measure D-HLL throughput
    print("Measuring D-HLL throughput...")
    dhll_results = []
    for seed in seeds:
        r = measure_throughput(DecayingHyperLogLog, n_elements, precision=11, decay_rate=0.01, seed=seed)
        dhll_results.append(r)
        print(f"  Seed {seed}: {r['throughput']:.0f} items/sec, {r['memory_bytes']} bytes")
    
    results["dhll"] = {
        "throughput_mean": float(np.mean([r['throughput'] for r in dhll_results])),
        "throughput_std": float(np.std([r['throughput'] for r in dhll_results])),
        "memory_bytes": dhll_results[0]['memory_bytes']
    }
    
    # Measure Standard HLL throughput
    print("\nMeasuring Standard HLL throughput...")
    hll_results = []
    for seed in seeds:
        sketch = HyperLogLog(precision=11, seed=seed)
        start = time.time()
        for i in range(n_elements):
            sketch.add(f"item_{i}")
        elapsed = time.time() - start
        hll_results.append({
            "throughput": n_elements / elapsed,
            "memory_bytes": sketch.memory_bytes()
        })
        print(f"  Seed {seed}: {hll_results[-1]['throughput']:.0f} items/sec")
    
    results["standard_hll"] = {
        "throughput_mean": float(np.mean([r['throughput'] for r in hll_results])),
        "throughput_std": float(np.std([r['throughput'] for r in hll_results])),
        "memory_bytes": hll_results[0]['memory_bytes']
    }
    
    # Measure Sliding HLL throughput
    print("\nMeasuring Sliding HLL throughput...")
    sliding_results = []
    for seed in seeds:
        sketch = SlidingHyperLogLog(precision=11, window_size=1000, seed=seed)
        start = time.time()
        for i in range(n_elements):
            sketch.add(f"item_{i}", timestamp=i)
        elapsed = time.time() - start
        sliding_results.append({
            "throughput": n_elements / elapsed,
            "memory_bytes": sketch.memory_bytes()
        })
        print(f"  Seed {seed}: {sliding_results[-1]['throughput']:.0f} items/sec")
    
    results["sliding_hll"] = {
        "throughput_mean": float(np.mean([r['throughput'] for r in sliding_results])),
        "throughput_std": float(np.std([r['throughput'] for r in sliding_results])),
        "memory_bytes_mean": float(np.mean([r['memory_bytes'] for r in sliding_results])),
        "memory_bytes_std": float(np.std([r['memory_bytes'] for r in sliding_results]))
    }
    
    with open("exp/dhll_throughput/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to exp/dhll_throughput/results.json")
