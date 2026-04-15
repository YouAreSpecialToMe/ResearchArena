#!/usr/bin/env python3
"""
Master script to run all LASER-SCL experiments efficiently.
Prioritizes critical experiments given time constraints.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Add shared to path
sys.path.insert(0, 'exp/shared')

# Configuration
EXPERIMENTS = [
    # Priority 1: Main CIFAR-100 experiments (critical for success criterion)
    {'method': 'supcon', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 1},
    {'method': 'supcon', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 123, 'epochs': 500, 'priority': 1},
    {'method': 'supcon', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 456, 'epochs': 500, 'priority': 1},
    
    {'method': 'supcon_lr', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 1},
    {'method': 'supcon_lr', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 123, 'epochs': 500, 'priority': 1},
    {'method': 'supcon_lr', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 456, 'epochs': 500, 'priority': 1},
    
    {'method': 'laser_scl', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 1},
    {'method': 'laser_scl', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 123, 'epochs': 500, 'priority': 1},
    {'method': 'laser_scl', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 456, 'epochs': 500, 'priority': 1},
    
    # Priority 2: CIFAR-10 experiments (secondary validation)
    {'method': 'supcon', 'dataset': 'cifar10', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 2},
    {'method': 'supcon_lr', 'dataset': 'cifar10', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 2},
    {'method': 'laser_scl', 'dataset': 'cifar10', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 2},
    
    # Priority 3: Ablation studies on CIFAR-100
    {'method': 'ablation_no_curriculum', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 3},
    {'method': 'ablation_no_curriculum', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 123, 'epochs': 500, 'priority': 3},
    {'method': 'ablation_no_elp', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 3},
    {'method': 'ablation_no_elp', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 123, 'epochs': 500, 'priority': 3},
    {'method': 'ablation_static', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 42, 'epochs': 500, 'priority': 3},
    {'method': 'ablation_static', 'dataset': 'cifar100', 'noise': 0.4, 'seed': 123, 'epochs': 500, 'priority': 3},
]

RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

def run_experiment(exp):
    """Run a single experiment."""
    method = exp['method']
    dataset = exp['dataset']
    noise = exp['noise']
    seed = exp['seed']
    epochs = exp['epochs']
    
    result_file = f"{method}_{dataset}_n{int(noise*100)}_s{seed}.json"
    result_path = os.path.join(RESULTS_DIR, result_file)
    
    # Skip if already completed
    if os.path.exists(result_path):
        print(f"Skipping {result_file} - already exists")
        return True
    
    log_file = os.path.join(LOGS_DIR, f"{method}_{dataset}_n{int(noise*100)}_s{seed}.log")
    
    cmd = [
        'python', 'exp/shared/train.py',
        '--method', method,
        '--dataset', dataset,
        '--noise_rate', str(noise),
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--save_dir', RESULTS_DIR
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {method} on {dataset} (noise={noise}, seed={seed})")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    with open(log_file, 'w') as f:
        f.write(f"Starting: {' '.join(cmd)}\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end='')
        
        process.wait()
    
    elapsed = (time.time() - start_time) / 60
    
    if process.returncode == 0 and os.path.exists(result_path):
        print(f"\n✓ Completed in {elapsed:.1f} minutes")
        return True
    else:
        print(f"\n✗ Failed after {elapsed:.1f} minutes")
        return False

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    print("LASER-SCL Experiment Runner")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Logs dir: {LOGS_DIR}")
    
    # Group by priority
    priorities = sorted(set(e['priority'] for e in EXPERIMENTS))
    
    total_start = time.time()
    completed = 0
    failed = 0
    
    for priority in priorities:
        exps = [e for e in EXPERIMENTS if e['priority'] == priority]
        print(f"\n\n{'#'*80}")
        print(f"# Priority {priority}: {len(exps)} experiments")
        print(f"{'#'*80}")
        
        for exp in exps:
            if run_experiment(exp):
                completed += 1
            else:
                failed += 1
            
            # Check elapsed time
            elapsed_hours = (time.time() - total_start) / 3600
            print(f"\nTotal elapsed: {elapsed_hours:.2f} hours")
            
            # If we're running out of time, focus only on priority 1
            if elapsed_hours > 6.5 and priority > 1:
                print("\n⚠ Approaching time limit - stopping after priority 1 experiments")
                break
        
        if elapsed_hours > 6.5 and priority == 1:
            print("\n⚠ Approaching time limit - stopping")
            break
    
    total_elapsed = (time.time() - total_start) / 3600
    print(f"\n\n{'='*80}")
    print(f"Experiment run complete!")
    print(f"Completed: {completed}, Failed: {failed}")
    print(f"Total time: {total_elapsed:.2f} hours")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
