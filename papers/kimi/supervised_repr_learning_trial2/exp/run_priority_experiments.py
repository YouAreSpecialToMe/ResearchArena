#!/usr/bin/env python3
"""
Sequential experiment runner focusing on priority 1 experiments.
Runs one experiment at a time to avoid GPU memory contention.
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# Priority 1: Main CIFAR-100 experiments (critical for success criterion)
# These are the minimum experiments needed to validate the hypothesis
EXPERIMENTS = [
    # SupCon vanilla - 3 seeds
    {'method': 'supcon', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 42, 'epochs': 500},
    {'method': 'supcon', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 123, 'epochs': 500},
    {'method': 'supcon', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 456, 'epochs': 500},
    
    # SupCon + Loss Reweighting - 3 seeds
    {'method': 'supcon_lr', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 42, 'epochs': 500},
    {'method': 'supcon_lr', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 123, 'epochs': 500},
    {'method': 'supcon_lr', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 456, 'epochs': 500},
    
    # LASER-SCL - 3 seeds
    {'method': 'laser_scl', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 42, 'epochs': 500},
    {'method': 'laser_scl', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 123, 'epochs': 500},
    {'method': 'laser_scl', 'dataset': 'cifar100', 'noise_rate': 0.4, 'seed': 456, 'epochs': 500},
]

def run_single_experiment(exp, exp_num, total):
    """Run a single experiment and return results."""
    method = exp['method']
    dataset = exp['dataset']
    noise = exp['noise_rate']
    seed = exp['seed']
    epochs = exp['epochs']
    
    result_file = f"{method}_{dataset}_n{int(noise*100)}_s{seed}.json"
    result_path = os.path.join(RESULTS_DIR, result_file)
    log_file = os.path.join(LOGS_DIR, f"{method}_{dataset}_n{int(noise*100)}_s{seed}.log")
    
    # Skip if already completed
    if os.path.exists(result_path):
        print(f"[{exp_num}/{total}] Skipping {result_file} - already exists")
        return True, 0
    
    print(f"\n{'='*80}")
    print(f"[{exp_num}/{total}] Running: {method} on {dataset}")
    print(f"  Noise: {noise}, Seed: {seed}, Epochs: {epochs}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'exp/shared/train.py',
        '--method', method,
        '--dataset', dataset,
        '--noise_rate', str(noise),
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--save_dir', RESULTS_DIR,
        '--batch_size', '256',
        '--num_workers', '4'
    ]
    
    start_time = time.time()
    
    # Run experiment with real-time output
    with open(log_file, 'w') as log_f:
        log_f.write(f"Experiment: {exp}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write("="*80 + "\n\n")
        log_f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            log_f.write(line)
            log_f.flush()
            # Print progress for key epochs
            if 'Epoch' in line and ('/500' in line or 'Test Acc' in line):
                print(line.strip())
    
    process.wait()
    elapsed_min = (time.time() - start_time) / 60
    
    success = (process.returncode == 0 and os.path.exists(result_path))
    
    if success:
        # Load and display final accuracy
        try:
            with open(result_path) as f:
                result = json.load(f)
            final_acc = result.get('final_accuracy', 'N/A')
            print(f"\n✓ Completed in {elapsed_min:.1f} min - Final Acc: {final_acc:.2f}%")
        except:
            print(f"\n✓ Completed in {elapsed_min:.1f} min")
    else:
        print(f"\n✗ Failed after {elapsed_min:.1f} min")
    
    return success, elapsed_min

def print_summary(completed, failed, total_time):
    """Print experiment summary."""
    print(f"\n\n{'='*80}")
    print(f"EXPERIMENT RUN SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {completed}, Failed: {failed}")
    print(f"Total time: {total_time/60:.2f} hours")
    
    # Show results
    print(f"\nResults:")
    for exp in EXPERIMENTS:
        method = exp['method']
        dataset = exp['dataset']
        noise = exp['noise_rate']
        seed = exp['seed']
        result_file = f"{method}_{dataset}_n{int(noise*100)}_s{seed}.json"
        result_path = os.path.join(RESULTS_DIR, result_file)
        
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    result = json.load(f)
                acc = result.get('final_accuracy', 'N/A')
                print(f"  {result_file}: {acc:.2f}%")
            except:
                print(f"  {result_file}: ERROR")
        else:
            print(f"  {result_file}: NOT RUN")
    
    print(f"{'='*80}\n")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    print("="*80)
    print("LASER-SCL Priority Experiment Runner")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Logs directory: {LOGS_DIR}")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Estimated time: ~30-40 hours (will run as many as possible in 8h limit)")
    print("="*80)
    
    start_time = time.time()
    completed = 0
    failed = 0
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        # Check remaining time
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours > 7.5:  # Stop if less than 30 min remaining
            print(f"\n⚠ Approaching time limit ({elapsed_hours:.1f}h elapsed). Stopping.")
            break
        
        success, _ = run_single_experiment(exp, i, len(EXPERIMENTS))
        if success:
            completed += 1
        else:
            failed += 1
        
        # Print progress
        elapsed_hours = (time.time() - start_time) / 3600
        remaining_hours = 8 - elapsed_hours
        print(f"\nProgress: {i}/{len(EXPERIMENTS)} - Elapsed: {elapsed_hours:.2f}h - Remaining: {remaining_hours:.2f}h")
    
    total_time = (time.time() - start_time) / 60  # in minutes
    print_summary(completed, failed, total_time)

if __name__ == '__main__':
    main()
