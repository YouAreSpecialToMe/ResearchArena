#!/usr/bin/env python3
"""
Baseline: Standard HyperLogLog (landmark model).
All elements weighted equally - no decay.
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import time
from pathlib import Path
from data_generator import load_stream
from hll import HyperLogLog

def run_baseline_hll(dataset_name, seeds=[42, 123, 999], precision=11):
    """Run standard HLL experiment on a dataset."""
    print(f"Running Standard HLL on {dataset_name}...")
    
    results = {}
    ids, timestamps = load_stream(dataset_name)
    
    # For standard HLL, we estimate total unique elements
    true_unique = len(np.unique(ids))
    
    for seed in seeds:
        hll = HyperLogLog(precision=precision, seed=seed)
        
        start = time.time()
        for i in range(len(ids)):
            hll.add(f"elem_{ids[i]}")
        elapsed = time.time() - start
        
        estimate = hll.cardinality()
        rel_error = abs(estimate - true_unique) / true_unique
        
        results[f"seed_{seed}"] = {
            "estimate": float(estimate),
            "true_unique": int(true_unique),
            "relative_error": float(rel_error),
            "time_seconds": elapsed,
            "throughput": len(ids) / elapsed
        }
        print(f"  Seed {seed}: est={estimate:.1f}, true={true_unique}, error={rel_error*100:.2f}%")
    
    # Aggregate statistics
    errors = [results[f"seed_{s}"]["relative_error"] for s in seeds]
    throughputs = [results[f"seed_{s}"]["throughput"] for s in seeds]
    
    return {
        "dataset": dataset_name,
        "method": "Standard HLL",
        "per_seed": results,
        "relative_error_mean": float(np.mean(errors)),
        "relative_error_std": float(np.std(errors)),
        "throughput_mean": float(np.mean(throughputs)),
        "throughput_std": float(np.std(throughputs)),
        "memory_bytes": HyperLogLog(precision=precision).memory_bytes()
    }

if __name__ == "__main__":
    Path("exp/baseline_hll").mkdir(exist_ok=True)
    
    all_results = {}
    for dataset in ["uniform", "zipfian", "bursty"]:
        all_results[dataset] = run_baseline_hll(dataset)
    
    with open("exp/baseline_hll/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to exp/baseline_hll/results.json")
