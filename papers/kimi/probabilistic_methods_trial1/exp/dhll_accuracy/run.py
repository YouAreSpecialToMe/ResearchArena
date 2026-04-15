#!/usr/bin/env python3
"""
D-HLL Accuracy vs Decay Rate experiment.
Tests D-HLL across different lambda values and datasets.
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import time
from pathlib import Path
from data_generator import load_stream, load_ground_truth
from dhll import DecayingHyperLogLog

def run_dhll_accuracy(dataset_name, lambda_val, seeds=[42, 123, 999], precision=11):
    """Run D-HLL experiment with given decay rate."""
    print(f"  Running D-HLL on {dataset_name} with lambda={lambda_val}...")
    
    query_times, ground_truths = load_ground_truth(dataset_name, lambda_val)
    ids, timestamps = load_stream(dataset_name)
    
    results = {}
    
    for seed in seeds:
        dhll = DecayingHyperLogLog(precision=precision, decay_rate=lambda_val, seed=seed)
        
        # Add all elements
        start = time.time()
        for i in range(len(ids)):
            dhll.add(f"elem_{ids[i]}", timestamps[i])
        add_time = time.time() - start
        
        # Query at specified times
        estimates = []
        for qt in query_times:
            est = dhll.cardinality(qt)
            estimates.append(est)
        
        # Calculate errors
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
            "memory_bytes": dhll.memory_bytes(),
            "stats": dhll.get_stats()
        }
    
    # Aggregate across seeds
    errors = [results[f"seed_{s}"]["mean_relative_error"] for s in seeds]
    throughputs = [results[f"seed_{s}"]["throughput"] for s in seeds]
    memories = [results[f"seed_{s}"]["memory_bytes"] for s in seeds]
    
    print(f"    Mean relative error: {np.mean(errors)*100:.2f}% ± {np.std(errors)*100:.2f}%")
    
    return {
        "dataset": dataset_name,
        "lambda": lambda_val,
        "per_seed": results,
        "relative_error_mean": float(np.mean(errors)),
        "relative_error_std": float(np.std(errors)),
        "throughput_mean": float(np.mean(throughputs)),
        "throughput_std": float(np.std(throughputs)),
        "memory_bytes_mean": float(np.mean(memories)),
        "memory_bytes_std": float(np.std(memories))
    }

if __name__ == "__main__":
    Path("exp/dhll_accuracy").mkdir(exist_ok=True)
    
    lambda_values = [0.001, 0.01, 0.05, 0.1]
    datasets = ["uniform", "zipfian", "bursty"]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        all_results[dataset] = {}
        for lam in lambda_values:
            all_results[dataset][f"lambda_{lam}"] = run_dhll_accuracy(dataset, lam)
    
    with open("exp/dhll_accuracy/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to exp/dhll_accuracy/results.json")
