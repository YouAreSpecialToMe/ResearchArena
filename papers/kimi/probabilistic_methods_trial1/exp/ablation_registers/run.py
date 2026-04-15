#!/usr/bin/env python3
"""
Ablation: Register Count (m) Scaling.
Verifies error scales as O(1/√m).
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
from pathlib import Path
from scipy import stats
from data_generator import load_stream, load_ground_truth
from dhll import DecayingHyperLogLog

def run_register_scaling(dataset_name="uniform", lambda_val=0.01, seeds=[42, 123, 999]):
    """Test error scaling with register count."""
    print(f"Running register scaling experiment on {dataset_name}...")
    
    query_times, ground_truths = load_ground_truth(dataset_name, lambda_val)
    ids, timestamps = load_stream(dataset_name)
    
    # Test different register counts (precision p = log2(m))
    precisions = [8, 9, 10, 11, 12]
    results = {}
    
    for p in precisions:
        m = 1 << p
        print(f"  Testing m={m} (p={p})...")
        
        errors = []
        for seed in seeds:
            dhll = DecayingHyperLogLog(precision=p, decay_rate=lambda_val, seed=seed)
            for i in range(len(ids)):
                dhll.add(f"elem_{ids[i]}", timestamps[i])
            
            # Calculate errors
            errs = []
            for qt, truth in zip(query_times, ground_truths):
                est = dhll.cardinality(qt)
                if truth > 0:
                    errs.append(abs(est - truth) / truth)
            
            errors.append(np.mean(errs))
        
        results[f"m_{m}"] = {
            "precision": p,
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "memory_bytes": DecayingHyperLogLog(precision=p, decay_rate=lambda_val).memory_bytes()
        }
        print(f"    Error: {np.mean(errors)*100:.2f}% ± {np.std(errors)*100:.2f}%")
    
    # Fit error = c / sqrt(m)
    m_values = [1 << p for p in precisions]
    error_means = [results[f"m_{m}"]["error_mean"] for m in m_values]
    inv_sqrt_m = [1.0 / np.sqrt(m) for m in m_values]
    
    # Linear regression: error = c * (1/sqrt(m))
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_sqrt_m, error_means)
    
    return {
        "dataset": dataset_name,
        "lambda": lambda_val,
        "per_m": results,
        "scaling_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value)
        },
        "m_values": m_values,
        "error_means": error_means,
        "inv_sqrt_m": inv_sqrt_m
    }

if __name__ == "__main__":
    Path("exp/ablation_registers").mkdir(exist_ok=True)
    
    result = run_register_scaling("uniform", lambda_val=0.01)
    
    with open("exp/ablation_registers/results.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nResults saved to exp/ablation_registers/results.json")
    print(f"\nScaling fit: error = {result['scaling_fit']['slope']:.4f}/sqrt(m) + {result['scaling_fit']['intercept']:.6f}")
    print(f"R² = {result['scaling_fit']['r_squared']:.4f}")
