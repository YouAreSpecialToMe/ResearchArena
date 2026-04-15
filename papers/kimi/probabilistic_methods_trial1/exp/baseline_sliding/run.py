#!/usr/bin/env python3
"""
Baseline: Sliding Window HyperLogLog.
Uses fixed-size landmark windows.
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import time
from pathlib import Path
from data_generator import load_stream, load_ground_truth
from sliding_hll import SlidingHyperLogLog

def run_sliding_hll(dataset_name, window_size=1000, seeds=[42, 123, 999], precision=11, lambda_val=0.01):
    """Run sliding window HLL experiment."""
    print(f"Running Sliding HLL on {dataset_name}...")
    
    query_times, ground_truths = load_ground_truth(dataset_name, lambda_val)
    ids, timestamps = load_stream(dataset_name)
    
    results = {}
    
    for seed in seeds:
        shll = SlidingHyperLogLog(precision=precision, window_size=window_size, seed=seed)
        
        # Add all elements
        start = time.time()
        for i in range(len(ids)):
            shll.add(f"elem_{ids[i]}", timestamps[i])
        add_time = time.time() - start
        
        # Query at different times
        estimates = []
        for qt in query_times:
            est = shll.cardinality(qt, window_duration=window_size)
            estimates.append(est)
        
        # Calculate errors against decayed ground truth
        errors = []
        for est, truth in zip(estimates, ground_truths):
            if truth > 0:
                rel_error = abs(est - truth) / truth
                errors.append(rel_error)
        
        results[f"seed_{seed}"] = {
            "query_times": query_times,
            "estimates": [float(e) for e in estimates],
            "ground_truths": ground_truths,
            "relative_errors": [float(e) for e in errors],
            "mean_relative_error": float(np.mean(errors)) if errors else None,
            "add_time_seconds": add_time,
            "throughput": len(ids) / add_time,
            "memory_bytes": shll.memory_bytes(),
            "num_windows": shll.num_windows()
        }
        print(f"  Seed {seed}: mean_error={np.mean(errors)*100:.2f}%, throughput={len(ids)/add_time:.0f} items/s")
    
    # Aggregate
    errors = [results[f"seed_{s}"]["mean_relative_error"] for s in seeds]
    throughputs = [results[f"seed_{s}"]["throughput"] for s in seeds]
    memories = [results[f"seed_{s}"]["memory_bytes"] for s in seeds]
    
    return {
        "dataset": dataset_name,
        "method": "Sliding Window HLL",
        "lambda": lambda_val,
        "window_size": window_size,
        "per_seed": results,
        "relative_error_mean": float(np.mean(errors)),
        "relative_error_std": float(np.std(errors)),
        "throughput_mean": float(np.mean(throughputs)),
        "throughput_std": float(np.std(throughputs)),
        "memory_bytes_mean": float(np.mean(memories)),
        "memory_bytes_std": float(np.std(memories))
    }

if __name__ == "__main__":
    Path("exp/baseline_sliding").mkdir(exist_ok=True)
    
    all_results = {}
    for dataset in ["uniform", "zipfian", "bursty"]:
        all_results[dataset] = run_sliding_hll(dataset, lambda_val=0.01)
    
    with open("exp/baseline_sliding/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to exp/baseline_sliding/results.json")
