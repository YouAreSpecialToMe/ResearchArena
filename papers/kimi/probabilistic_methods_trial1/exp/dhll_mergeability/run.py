#!/usr/bin/env python3
"""
D-HLL Mergeability Evaluation.
Verifies distributed computation correctness.
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import time
from pathlib import Path
from data_generator import load_stream, load_ground_truth
from dhll import DecayingHyperLogLog

def run_mergeability_experiment(dataset_name, n_shards=10, lambda_val=0.01, seeds=[42, 123, 999], precision=11):
    """Test merging sketches from distributed sources."""
    print(f"Running mergeability test on {dataset_name} (lambda={lambda_val})...")
    
    query_times, ground_truths = load_ground_truth(dataset_name, lambda_val)
    ids, timestamps = load_stream(dataset_name)
    
    results = {}
    
    for seed in seeds:
        # Centralized: one sketch for all data
        centralized = DecayingHyperLogLog(precision=precision, decay_rate=lambda_val, seed=seed)
        for i in range(len(ids)):
            centralized.add(f"elem_{ids[i]}", timestamps[i])
        
        # Distributed: multiple sketches merged
        shards = [DecayingHyperLogLog(precision=precision, decay_rate=lambda_val, seed=seed) for _ in range(n_shards)]
        for i in range(len(ids)):
            shard_idx = i % n_shards
            shards[shard_idx].add(f"elem_{ids[i]}", timestamps[i])
        
        # Merge all shards
        merged = shards[0]
        for i in range(1, n_shards):
            merged = merged.merge(shards[i])
        
        # Compare estimates
        estimates_central = []
        estimates_merged = []
        for qt in query_times:
            estimates_central.append(centralized.cardinality(qt))
            estimates_merged.append(merged.cardinality(qt))
        
        # Calculate merge error
        merge_errors = []
        for est_c, est_m in zip(estimates_central, estimates_merged):
            if est_c > 0:
                merge_error = abs(est_m - est_c) / est_c
                merge_errors.append(merge_error)
        
        results[f"seed_{seed}"] = {
            "query_times": query_times,
            "estimates_centralized": [float(e) for e in estimates_central],
            "estimates_merged": [float(e) for e in estimates_merged],
            "merge_errors": [float(e) for e in merge_errors],
            "mean_merge_error": float(np.mean(merge_errors)) if merge_errors else None
        }
        print(f"  Seed {seed}: mean merge error = {np.mean(merge_errors)*100:.2f}%")
    
    # Aggregate
    merge_errors = [results[f"seed_{s}"]["mean_merge_error"] for s in seeds]
    
    return {
        "dataset": dataset_name,
        "lambda": lambda_val,
        "n_shards": n_shards,
        "per_seed": results,
        "merge_error_mean": float(np.mean(merge_errors)),
        "merge_error_std": float(np.std(merge_errors))
    }

if __name__ == "__main__":
    Path("exp/dhll_mergeability").mkdir(exist_ok=True)
    
    all_results = {}
    
    # Test with different lambda values
    for lambda_val in [0.001, 0.01, 0.05, 0.1]:
        all_results[f"lambda_{lambda_val}"] = run_mergeability_experiment("uniform", lambda_val=lambda_val)
    
    with open("exp/dhll_mergeability/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to exp/dhll_mergeability/results.json")
