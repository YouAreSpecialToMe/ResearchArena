#!/usr/bin/env python3
"""
Run experiments sequentially to avoid resource contention.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

# Reduced epochs for faster completion
EXPERIMENTS = [
    # Baseline: Cross-Entropy
    ('CE', 'exp/cifar100_ce/run.py', [
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 42',
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 123',
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 456',
    ]),
    # Baseline: SupCon
    ('SupCon', 'exp/cifar100_supcon/run.py', [
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 42',
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 123',
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 456',
    ]),
    # Baseline: BCL
    ('BCL', 'exp/cifar100_bcl/run.py', [
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 42',
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 123',
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 456',
    ]),
    # Main Method: ETF-SCL
    ('ETF-SCL', 'exp/cifar100_etfscl/run.py', [
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 42',
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 123',
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 456',
    ]),
    # Ablation: No ETF
    ('Ablation-No-ETF', 'exp/ablation_no_etf/run.py', [
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 42',
    ]),
    # Ablation: No Adaptive
    ('Ablation-No-Adaptive', 'exp/ablation_no_adaptive/run.py', [
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 42',
    ]),
    # Ablation: No Temp
    ('Ablation-No-Temp', 'exp/ablation_no_temp/run.py', [
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 42',
    ]),
    # Ablation: ETF Only
    ('Ablation-ETF-Only', 'exp/ablation_etf_only/run.py', [
        '--dataset cifar100 --imbalance_factor 100 --epochs 100 --seed 42',
    ]),
]


def run_experiment(name, script, args_list, output_dir='results'):
    """Run all seeds for one experiment type sequentially."""
    print(f"\n{'='*80}")
    print(f"Running {name}")
    print(f"{'='*80}")
    
    for i, args in enumerate(args_list):
        cmd = f"python -u {script} {args} --output_dir {output_dir}"
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running {name} - Seed {i+1}/{len(args_list)}")
        print(f"Command: {cmd}")
        
        start = time.time()
        result = subprocess.run(cmd, shell=True)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ Completed in {elapsed/60:.1f} minutes")
        else:
            print(f"✗ Failed with code {result.returncode}")
    
    print(f"\n{name} completed!")


def main():
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    total_start = time.time()
    
    for name, script, args_list in EXPERIMENTS:
        run_experiment(name, script, args_list, output_dir)
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_elapsed/3600:.2f} hours")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
