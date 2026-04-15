#!/usr/bin/env python
"""
Master script to run all LASER-SCL experiments.
Executes experiments in dependency order with proper parallelization.
"""
import os
import sys
import subprocess
import json
import time
from pathlib import Path


def run_command(cmd, log_file=None):
    """Run a command and optionally log output."""
    print(f"Running: {cmd}")
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    results_dir = './results'
    logs_dir = './logs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Shared path
    shared_path = os.path.abspath('./exp/shared')
    
    # Keep track of all results
    all_results = {}
    start_time = time.time()
    
    # ============================================================
    # PHASE 1: ELP Validation (preliminary evidence)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: ELP Validation")
    print("="*60)
    
    elp_cmd = f"cd {shared_path} && python ../elp_validation/run.py --dataset cifar10 --noise_rate 0.4 --seed 42 --save_dir {results_dir}"
    elp_log = f"{logs_dir}/elp_validation.log"
    
    if run_command(elp_cmd, elp_log):
        print("✓ ELP validation completed")
    else:
        print("✗ ELP validation failed")
    
    # ============================================================
    # PHASE 2: Main Experiments (CIFAR-10, 40% noise)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: CIFAR-10 with 40% Symmetric Noise")
    print("="*60)
    
    # Methods to run with 3 seeds
    methods_cifar10 = [
        ('supcon', 'Vanilla SupCon'),
        ('supcon_lr', 'SupCon + Loss Reweighting'),
        ('supcon_il', 'SupCon + Inverse Loss'),
        ('laser_scl', 'LASER-SCL (Full)'),
    ]
    
    seeds = [42, 123, 456]
    
    for method, name in methods_cifar10:
        print(f"\n--- {name} ---")
        for seed in seeds:
            cmd = f"cd {shared_path} && python train.py --method {method} --dataset cifar10 --noise_rate 0.4 --seed {seed} --epochs 500 --save_dir {results_dir}"
            log_file = f"{logs_dir}/{method}_cifar10_n40_s{seed}.log"
            
            if run_command(cmd, log_file):
                print(f"  ✓ Seed {seed} completed")
            else:
                print(f"  ✗ Seed {seed} failed")
    
    # ============================================================
    # PHASE 3: CIFAR-100 Main Experiments
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 3: CIFAR-100 with 40% Symmetric Noise")
    print("="*60)
    
    methods_cifar100 = [
        ('supcon', 'Vanilla SupCon'),
        ('supcon_lr', 'SupCon + Loss Reweighting'),
        ('laser_scl', 'LASER-SCL (Full)'),
    ]
    
    for method, name in methods_cifar100:
        print(f"\n--- {name} ---")
        for seed in seeds:
            cmd = f"cd {shared_path} && python train.py --method {method} --dataset cifar100 --noise_rate 0.4 --seed {seed} --epochs 500 --save_dir {results_dir}"
            log_file = f"{logs_dir}/{method}_cifar100_n40_s{seed}.log"
            
            if run_command(cmd, log_file):
                print(f"  ✓ Seed {seed} completed")
            else:
                print(f"  ✗ Seed {seed} failed")
    
    # ============================================================
    # PHASE 4: Ablation Studies (CIFAR-100)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 4: Ablation Studies on CIFAR-100")
    print("="*60)
    
    ablations = [
        ('ablation_no_curriculum', 'No Curriculum'),
        ('ablation_no_elp', 'No ELP (Current Loss)'),
        ('ablation_static', 'Static Weighting'),
    ]
    
    ablation_seeds = [42, 123]
    
    for method, name in ablations:
        print(f"\n--- {name} ---")
        for seed in ablation_seeds:
            cmd = f"cd {shared_path} && python train.py --method {method} --dataset cifar100 --noise_rate 0.4 --seed {seed} --epochs 500 --save_dir {results_dir}"
            log_file = f"{logs_dir}/{method}_cifar100_n40_s{seed}.log"
            
            if run_command(cmd, log_file):
                print(f"  ✓ Seed {seed} completed")
            else:
                print(f"  ✗ Seed {seed} failed")
    
    # ============================================================
    # PHASE 5: Additional Noise Levels (if time permits)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 5: Additional Noise Levels on CIFAR-10")
    print("="*60)
    
    # Quick runs with 1 seed for 20% and 60% noise
    additional_noise = [0.2, 0.6]
    
    for noise_rate in additional_noise:
        print(f"\n--- Noise Rate: {noise_rate*100}% ---")
        for method in ['supcon', 'laser_scl']:
            cmd = f"cd {shared_path} && python train.py --method {method} --dataset cifar10 --noise_rate {noise_rate} --seed 42 --epochs 500 --save_dir {results_dir}"
            log_file = f"{logs_dir}/{method}_cifar10_n{int(noise_rate*100)}_s42.log"
            
            if run_command(cmd, log_file):
                print(f"  ✓ {method} completed")
            else:
                print(f"  ✗ {method} failed")
    
    elapsed = (time.time() - start_time) / 3600
    print(f"\n{'='*60}")
    print(f"All experiments completed in {elapsed:.2f} hours")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
