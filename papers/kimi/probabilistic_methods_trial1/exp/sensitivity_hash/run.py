#!/usr/bin/env python3
"""
Sensitivity Analysis: Hash Function Quality.
Verifies robustness to hash function choice.
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import hashlib
import mmh3
from pathlib import Path
from data_generator import load_stream, load_ground_truth
from dhll import DecayingHyperLogLog

class DHLLOtherHash(DecayingHyperLogLog):
    """D-HLL with different hash functions."""
    
    def __init__(self, precision=11, decay_rate=0.01, seed=0, hash_type="mmh3"):
        super().__init__(precision, decay_rate, seed)
        self.hash_type = hash_type
    
    def _hash(self, item):
        if isinstance(item, str):
            item = item.encode('utf-8')
        
        if self.hash_type == "mmh3":
            return mmh3.hash64(item, self.seed)[0] & 0xFFFFFFFFFFFFFFFF
        elif self.hash_type == "sha256":
            h = hashlib.sha256(item).digest()[:8]
            return int.from_bytes(h, 'little')
        elif self.hash_type == "simple":
            # Simple multiplicative hash
            val = sum(b * (i + 1) for i, b in enumerate(item))
            return (val * 2654435761) & 0xFFFFFFFFFFFFFFFF
        else:
            raise ValueError(f"Unknown hash type: {self.hash_type}")
    
    def add(self, item, timestamp):
        if isinstance(item, str):
            item = item.encode('utf-8')
        
        self.current_time = max(self.current_time, timestamp)
        
        hash_val = self._hash(item)
        j = hash_val & (self.m - 1)
        w = hash_val >> self.p
        rank = self._rho(w)
        
        self._apply_lazy_decay(j, timestamp)
        
        if rank > self.registers[j]:
            self.registers[j] = float(rank)
            self.timestamps[j] = timestamp

def run_hash_sensitivity(dataset_name="uniform", lambda_val=0.01, seeds=[42, 123, 999], precision=11):
    """Test D-HLL with different hash functions."""
    print(f"Running hash sensitivity test on {dataset_name}...")
    
    query_times, ground_truths = load_ground_truth(dataset_name, lambda_val)
    ids, timestamps = load_stream(dataset_name)
    
    hash_types = ["mmh3", "sha256", "simple"]
    results = {}
    
    for hash_type in hash_types:
        print(f"  Testing {hash_type}...")
        errors = []
        
        for seed in seeds:
            dhll = DHLLOtherHash(precision=precision, decay_rate=lambda_val, seed=seed, hash_type=hash_type)
            for i in range(len(ids)):
                dhll.add(f"elem_{ids[i]}", timestamps[i])
            
            errs = []
            for qt, truth in zip(query_times, ground_truths):
                est = dhll.cardinality(qt)
                if truth > 0:
                    errs.append(abs(est - truth) / truth)
            
            errors.append(np.mean(errs))
        
        results[hash_type] = {
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors))
        }
        print(f"    Error: {np.mean(errors)*100:.2f}% ± {np.std(errors)*100:.2f}%")
    
    return {
        "dataset": dataset_name,
        "lambda": lambda_val,
        "hash_types": results
    }

if __name__ == "__main__":
    Path("exp/sensitivity_hash").mkdir(exist_ok=True)
    
    result = run_hash_sensitivity("uniform", lambda_val=0.01)
    
    with open("exp/sensitivity_hash/results.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nResults saved to exp/sensitivity_hash/results.json")
