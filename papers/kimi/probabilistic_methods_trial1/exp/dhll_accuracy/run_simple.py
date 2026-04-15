#!/usr/bin/env python3
"""
D-HLL Accuracy experiment - simplified for speed.
Tests D-HLL on one dataset at a time.
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
    print(f"Running D-HLL on {dataset_name} with lambda={lambda_val}...")
    
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
            "mean_relative_error": float(np.mean(errors)) if errors else None,
            "add_time_seconds": add_time,
            "throughput": len(ids) / add_time,
            "memory_bytes": dhll.memory_bytes(),
        }
    
    # Aggregate across seeds
    errors = [results[f"seed_{s}"]["mean_relative_error"] for s in seeds]
    throughputs = [results[f"seed_{s}"]["throughput"] for s in seeds]
    
    print(f"  Mean relative error: {np.mean(errors)*100:.2f}% ± {np.std(errors)*100:.2f}%")
    
    return {
        "dataset": dataset_name,
        "lambda": lambda_val,
        "per_seed": results,
        "relative_error_mean": float(np.mean(errors)),
        "relative_error_std": float(np.std(errors)),
        "throughput_mean": float(np.mean(throughputs)),
        "throughput_std": float(np.std(throughputs)),
        "memory_bytes": results["seed_42"]["memory_bytes"]
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="uniform")
    parser.add_argument("--lambda_val", type=float, default=0.01)
    args = parser.parse_args()
    
    Path("exp/dhll_accuracy").mkdir(exist_ok=True)
    
    result = run_dhll_accuracy(args.dataset, args.lambda_val)
    
    # Append to results file
    results_file = Path("exp/dhll_accuracy/results.json")
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    if args.dataset not in all_results:
        all_results[args.dataset] = {}
    all_results[args.dataset][f"lambda_{args.lambda_val}"] = result
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to exp/dhll_accuracy/results.json")
