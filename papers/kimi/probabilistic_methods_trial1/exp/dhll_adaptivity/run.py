#!/usr/bin/env python3
"""
D-HLL Adaptivity to Concept Drift.
Tests response to sudden cardinality changes (bursty stream).
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
from pathlib import Path
from data_generator import load_stream, load_ground_truth
from dhll import DecayingHyperLogLog
from sliding_hll import SlidingHyperLogLog

def run_adaptivity_experiment(lambda_val=0.02, window_size=500, precision=11, seeds=[42, 123, 999]):
    """Compare D-HLL vs Sliding HLL on bursty stream."""
    print("Running adaptivity experiment on bursty stream...")
    
    dataset_name = "bursty"
    query_times, ground_truths = load_ground_truth(dataset_name, lambda_val)
    ids, timestamps = load_stream(dataset_name)
    
    results = {"dhll": {}, "sliding": {}}
    
    for seed in seeds:
        # D-HLL
        dhll = DecayingHyperLogLog(precision=precision, decay_rate=lambda_val, seed=seed)
        for i in range(len(ids)):
            dhll.add(f"elem_{ids[i]}", timestamps[i])
        
        dhll_estimates = [dhll.cardinality(qt) for qt in query_times]
        dhll_errors = [abs(est - truth) / truth if truth > 0 else 0 
                       for est, truth in zip(dhll_estimates, ground_truths)]
        
        # Sliding HLL
        sliding = SlidingHyperLogLog(precision=precision, window_size=window_size, seed=seed)
        for i in range(len(ids)):
            sliding.add(f"elem_{ids[i]}", timestamps[i])
        
        sliding_estimates = [sliding.cardinality(qt, window_duration=window_size) for qt in query_times]
        sliding_errors = [abs(est - truth) / truth if truth > 0 else 0 
                          for est, truth in zip(sliding_estimates, ground_truths)]
        
        results["dhll"][f"seed_{seed}"] = {
            "query_times": query_times,
            "estimates": [float(e) for e in dhll_estimates],
            "ground_truths": ground_truths,
            "relative_errors": [float(e) for e in dhll_errors],
            "mean_error": float(np.mean(dhll_errors))
        }
        results["sliding"][f"seed_{seed}"] = {
            "query_times": query_times,
            "estimates": [float(e) for e in sliding_estimates],
            "ground_truths": ground_truths,
            "relative_errors": [float(e) for e in sliding_errors],
            "mean_error": float(np.mean(sliding_errors))
        }
        
        print(f"  Seed {seed}: D-HLL error={np.mean(dhll_errors)*100:.2f}%, Sliding error={np.mean(sliding_errors)*100:.2f}%")
    
    # Aggregate
    dhll_errors = [results["dhll"][f"seed_{s}"]["mean_error"] for s in seeds]
    sliding_errors = [results["sliding"][f"seed_{s}"]["mean_error"] for s in seeds]
    
    return {
        "dataset": dataset_name,
        "lambda": lambda_val,
        "window_size": window_size,
        "dhll": {
            "error_mean": float(np.mean(dhll_errors)),
            "error_std": float(np.std(dhll_errors))
        },
        "sliding": {
            "error_mean": float(np.mean(sliding_errors)),
            "error_std": float(np.std(sliding_errors))
        },
        "per_seed": results
    }

if __name__ == "__main__":
    Path("exp/dhll_adaptivity").mkdir(exist_ok=True)
    
    result = run_adaptivity_experiment(lambda_val=0.02, window_size=500)
    
    with open("exp/dhll_adaptivity/results.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nResults saved to exp/dhll_adaptivity/results.json")
    print(f"\nSummary:")
    print(f"  D-HLL: {result['dhll']['error_mean']*100:.2f}% ± {result['dhll']['error_std']*100:.2f}%")
    print(f"  Sliding: {result['sliding']['error_mean']*100:.2f}% ± {result['sliding']['error_std']*100:.2f}%")
