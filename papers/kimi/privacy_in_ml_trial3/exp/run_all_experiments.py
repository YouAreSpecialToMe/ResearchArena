#!/usr/bin/env python3
"""
Master script to run all experiments for FedSecure-CL.
Runs experiments in parallel where possible to maximize GPU utilization.
"""

import os
import sys
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Experiment configuration
EXPERIMENTS = [
    # Format: (name, script_path, [list of (seed, dataset) configs])
    
    # Baseline 1: Standard FCL - 3 seeds CIFAR-10, 1 seed CIFAR-100
    ('baseline_fcl_cifar10', 'exp/baseline_fcl/run.py', [(42, 'cifar10'), (123, 'cifar10'), (456, 'cifar10')]),
    ('baseline_fcl_cifar100', 'exp/baseline_fcl/run.py', [(42, 'cifar100')]),
    
    # Baseline 2: FCL + AT - 3 seeds CIFAR-10, 1 seed CIFAR-100
    ('baseline_fcl_at_cifar10', 'exp/baseline_fcl_at/run.py', [(42, 'cifar10'), (123, 'cifar10'), (456, 'cifar10')]),
    ('baseline_fcl_at_cifar100', 'exp/baseline_fcl_at/run.py', [(42, 'cifar100')]),
    
    # Baseline 3: FCL + DP - 3 seeds CIFAR-10 only
    ('baseline_fcl_dp', 'exp/baseline_fcl_dp/run.py', [(42, 'cifar10'), (123, 'cifar10'), (456, 'cifar10')]),
    
    # Baseline 4: FCL + DP + AT - 1 seed CIFAR-10
    ('baseline_fcl_dp_at', 'exp/baseline_fcl_dp_at/run.py', [(42, 'cifar10')]),
    
    # FedSecure-CL - 3 seeds CIFAR-10, 1 seed CIFAR-100
    ('fedsecure_cl_cifar10', 'exp/fedsecure_cl/run.py', [(42, 'cifar10'), (123, 'cifar10'), (456, 'cifar10')]),
    ('fedsecure_cl_cifar100', 'exp/fedsecure_cl/run.py', [(42, 'cifar100')]),
    
    # Ablation: No Privacy Reg - 2 seeds CIFAR-10
    ('ablation_no_privacy', 'exp/fedsecure_cl/run.py', [(42, 'cifar10'), (123, 'cifar10')]),
    
    # Ablation: No Grad Noise - 2 seeds CIFAR-10
    ('ablation_no_grad_noise', 'exp/fedsecure_cl/run.py', [(42, 'cifar10'), (123, 'cifar10')]),
    
    # Ablation: No Adv Training - 2 seeds CIFAR-10
    ('ablation_no_adv', 'exp/fedsecure_cl/run.py', [(42, 'cifar10'), (123, 'cifar10')]),
]


def run_experiment(name, script_path, seed, dataset=None, ablation=None):
    """Run a single experiment."""
    cmd = ['python', script_path, '--seed', str(seed)]
    
    if dataset:
        cmd.extend(['--dataset', dataset])
    
    if ablation:
        cmd.extend(['--ablation', ablation])
    
    print(f"\n[START] {name} (seed={seed}, dataset={dataset})")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[DONE] {name} (seed={seed}) in {elapsed/60:.1f} min")
            return True, name, seed, elapsed
        else:
            print(f"[ERROR] {name} (seed={seed})")
            print(f"STDOUT: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
            print(f"STDERR: {result.stderr[-500:] if len(result.stderr) > 500 else result.stderr}")
            return False, name, seed, elapsed
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name} (seed={seed})")
        return False, name, seed, 7200
    except Exception as e:
        print(f"[EXCEPTION] {name} (seed={seed}): {e}")
        return False, name, seed, 0


def run_all_experiments():
    """Run all experiments."""
    print("=" * 60)
    print("FedSecure-CL: Running All Experiments")
    print("=" * 60)
    
    results = []
    
    # Run experiments sequentially to avoid memory issues
    for exp_name, script_path, configs in EXPERIMENTS:
        for seed, dataset in configs:
            # Determine ablation type
            ablation = None
            if 'ablation_no_privacy' in exp_name:
                ablation = 'no_privacy'
            elif 'ablation_no_grad_noise' in exp_name:
                ablation = 'no_grad_noise'
            elif 'ablation_no_adv' in exp_name:
                ablation = 'no_adv'
            
            success, name, s, elapsed = run_experiment(
                exp_name, script_path, seed, dataset, ablation
            )
            results.append({
                'name': name,
                'seed': s,
                'dataset': dataset,
                'success': success,
                'time': elapsed
            })
            
            # Save progress
            with open('results/experiment_progress.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    total_time = sum(r['time'] for r in results)
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}/{total}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    
    return results


if __name__ == '__main__':
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    run_all_experiments()
