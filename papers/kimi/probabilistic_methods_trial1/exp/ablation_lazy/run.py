#!/usr/bin/env python3
"""
Ablation: Lazy vs Eager Decay.
Verifies lazy decay does not significantly impact accuracy.
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import time
from pathlib import Path
from data_generator import load_stream, load_ground_truth
from dhll import DecayingHyperLogLog, EagerDecayingHyperLogLog

def run_ablation(dataset_name, lambda_val=0.01, seeds=[42, 123, 999], precision=11):
    """Compare lazy vs eager decay."""
    print(f"Running lazy vs eager ablation on {dataset_name}...")
    
    query_times, ground_truths = load_ground_truth(dataset_name, lambda_val)
    ids, timestamps = load_stream(dataset_name)
    
    results = {"lazy": {}, "eager": {}}
    
    for seed in seeds:
        # Lazy decay
        lazy = DecayingHyperLogLog(precision=precision, decay_rate=lambda_val, seed=seed)
        start = time.time()
        for i in range(len(ids)):
            lazy.add(f"elem_{ids[i]}", timestamps[i])
        lazy_time = time.time() - start
        
        lazy_estimates = [lazy.cardinality(qt) for qt in query_times]
        lazy_errors = [abs(est - truth) / truth for est, truth in zip(lazy_estimates, ground_truths) if truth > 0]
        
        # Eager decay
        eager = EagerDecayingHyperLogLog(precision=precision, decay_rate=lambda_val, seed=seed)
        start = time.time()
        for i in range(len(ids)):
            eager.add(f"elem_{ids[i]}", timestamps[i])
        eager_time = time.time() - start
        
        eager_estimates = [eager.cardinality(qt) for qt in query_times]
        eager_errors = [abs(est - truth) / truth for est, truth in zip(eager_estimates, ground_truths) if truth > 0]
        
        results["lazy"][f"seed_{seed}"] = {
            "mean_error": float(np.mean(lazy_errors)),
            "time_seconds": lazy_time,
            "throughput": len(ids) / lazy_time
        }
        results["eager"][f"seed_{seed}"] = {
            "mean_error": float(np.mean(eager_errors)),
            "time_seconds": eager_time,
            "throughput": len(ids) / eager_time
        }
        
        print(f"  Seed {seed}: lazy_error={np.mean(lazy_errors)*100:.2f}%, eager_error={np.mean(eager_errors)*100:.2f}%")
        print(f"           lazy_time={lazy_time:.2f}s, eager_time={eager_time:.2f}s")
    
    # Aggregate
    lazy_errors = [results["lazy"][f"seed_{s}"]["mean_error"] for s in seeds]
    eager_errors = [results["eager"][f"seed_{s}"]["mean_error"] for s in seeds]
    lazy_throughputs = [results["lazy"][f"seed_{s}"]["throughput"] for s in seeds]
    eager_throughputs = [results["eager"][f"seed_{s}"]["throughput"] for s in seeds]
    
    # Difference in accuracy
    error_diff = [abs(l - e) for l, e in zip(lazy_errors, eager_errors)]
    
    return {
        "dataset": dataset_name,
        "lambda": lambda_val,
        "lazy": {
            "error_mean": float(np.mean(lazy_errors)),
            "error_std": float(np.std(lazy_errors)),
            "throughput_mean": float(np.mean(lazy_throughputs)),
            "throughput_std": float(np.std(lazy_throughputs))
        },
        "eager": {
            "error_mean": float(np.mean(eager_errors)),
            "error_std": float(np.std(eager_errors)),
            "throughput_mean": float(np.mean(eager_throughputs)),
            "throughput_std": float(np.std(eager_throughputs))
        },
        "accuracy_difference": {
            "mean": float(np.mean(error_diff)),
            "std": float(np.std(error_diff))
        },
        "per_seed": results
    }

if __name__ == "__main__":
    Path("exp/ablation_lazy").mkdir(exist_ok=True)
    
    result = run_ablation("uniform", lambda_val=0.01)
    
    with open("exp/ablation_lazy/results.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nResults saved to exp/ablation_lazy/results.json")
    print(f"\nSummary:")
    print(f"  Lazy error: {result['lazy']['error_mean']*100:.2f}% ± {result['lazy']['error_std']*100:.2f}%")
    print(f"  Eager error: {result['eager']['error_mean']*100:.2f}% ± {result['eager']['error_std']*100:.2f}%")
    print(f"  Difference: {result['accuracy_difference']['mean']*100:.4f}% ± {result['accuracy_difference']['std']*100:.4f}%")
    print(f"  Speedup: {result['eager']['throughput_mean'] / result['lazy']['throughput_mean']:.2f}x")
