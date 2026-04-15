#!/usr/bin/env python3
"""
Run all experiments sequentially to fix the issues identified in self-review.
This script runs experiments one by one to avoid GPU memory issues.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

# Add shared module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exp/shared'))

# Experiment configurations
EXPERIMENTS = [
    # CIFAR-100 CE (seeds 123, 456 - seed 42 already done)
    {"script": "exp/cifar100_ce/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "123", "--epochs", "200"]},
    {"script": "exp/cifar100_ce/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "456", "--epochs", "200"]},
    
    # CIFAR-100 SupCon (all seeds - need to rerun with fixed implementation)
    {"script": "exp/cifar100_supcon/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200"]},
    {"script": "exp/cifar100_supcon/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "123", "--epochs", "200"]},
    {"script": "exp/cifar100_supcon/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "456", "--epochs", "200"]},
    
    # CIFAR-100 ETF-SCL (all seeds with fixed lambda_etf=0.05)
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200", "--lambda_etf", "0.05"]},
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "123", "--epochs", "200", "--lambda_etf", "0.05"]},
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "456", "--epochs", "200", "--lambda_etf", "0.05"]},
    
    # CIFAR-100 BCL (all seeds)
    {"script": "exp/cifar100_bcl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200"]},
    {"script": "exp/cifar100_bcl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "123", "--epochs", "200"]},
    {"script": "exp/cifar100_bcl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "456", "--epochs", "200"]},
    
    # CIFAR-10 experiments (CE, SupCon, ETF-SCL)
    {"script": "exp/cifar100_ce/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200"]},
    {"script": "exp/cifar100_ce/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "123", "--epochs", "200"]},
    {"script": "exp/cifar100_ce/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "456", "--epochs", "200"]},
    
    {"script": "exp/cifar100_supcon/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200"]},
    {"script": "exp/cifar100_supcon/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "123", "--epochs", "200"]},
    {"script": "exp/cifar100_supcon/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "456", "--epochs", "200"]},
    
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200", "--lambda_etf", "0.05"]},
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "123", "--epochs", "200", "--lambda_etf", "0.05"]},
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar10", "--imbalance_factor", "100", "--seed", "456", "--epochs", "200", "--lambda_etf", "0.05"]},
    
    # Ablation studies (on CIFAR-100)
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200", "--lambda_etf", "0.0"], "name": "ablation_no_etf"},
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200", "--lambda_etf", "0.05", "--alpha", "0.0"], "name": "ablation_no_adaptive"},
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200", "--lambda_etf", "0.05", "--beta", "0.0"], "name": "ablation_no_temp"},
    {"script": "exp/cifar100_etfscl/run.py", "args": ["--dataset", "cifar100", "--imbalance_factor", "100", "--seed", "42", "--epochs", "200", "--lambda_etf", "0.05", "--alpha", "0.0", "--beta", "0.0"], "name": "ablation_etf_only"},
]

def run_experiment(script, args, exp_idx, total):
    """Run a single experiment."""
    cmd = ["python", script] + args
    exp_name = f"{script} {' '.join(args)}"
    
    print(f"\n{'='*80}")
    print(f"Experiment {exp_idx}/{total}: {exp_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Completed in {elapsed/60:.1f} minutes")
            return True
        else:
            print(f"✗ Failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 1 hour")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("Starting experiment sequence...")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    
    success_count = 0
    fail_count = 0
    
    for idx, exp in enumerate(EXPERIMENTS, 1):
        script = exp["script"]
        args = exp["args"]
        
        if run_experiment(script, args, idx, len(EXPERIMENTS)):
            success_count += 1
        else:
            fail_count += 1
        
        # Small delay between experiments to let GPU memory clear
        time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Successful: {success_count}/{len(EXPERIMENTS)}")
    print(f"Failed: {fail_count}/{len(EXPERIMENTS)}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
