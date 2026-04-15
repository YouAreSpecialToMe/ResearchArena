#!/usr/bin/env python3
"""
Minimal experiments for FedSecure-CL.
Uses 3 rounds, 2 clients, 1 epoch for ultra-fast completion.
"""
import os
import sys
import json
import subprocess
import time

# Experiments to run
EXPERIMENTS = [
    ('baseline_fcl', 'standard', 'cifar10', 42, {}),
    ('baseline_fcl', 'standard', 'cifar10', 123, {}),
    ('baseline_fcl', 'standard', 'cifar10', 456, {}),
    
    ('baseline_fcl_at', 'fcl_at', 'cifar10', 42, {'use_adversarial': True}),
    ('baseline_fcl_at', 'fcl_at', 'cifar10', 123, {'use_adversarial': True}),
    ('baseline_fcl_at', 'fcl_at', 'cifar10', 456, {'use_adversarial': True}),
    
    ('baseline_fcl_dp', 'fcl_dp', 'cifar10', 42, {}),
    ('baseline_fcl_dp', 'fcl_dp', 'cifar10', 123, {}),
    ('baseline_fcl_dp', 'fcl_dp', 'cifar10', 456, {}),
    
    ('fedsecure_cl', 'fedsecure', 'cifar10', 42, {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': True}),
    ('fedsecure_cl', 'fedsecure', 'cifar10', 123, {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': True}),
    ('fedsecure_cl', 'fedsecure', 'cifar10', 456, {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': True}),
    
    ('ablation_no_privacy', 'fedsecure', 'cifar10', 42, {'use_adversarial': True, 'use_privacy_reg': False, 'use_grad_noise': True}),
    ('ablation_no_privacy', 'fedsecure', 'cifar10', 123, {'use_adversarial': True, 'use_privacy_reg': False, 'use_grad_noise': True}),
    
    ('ablation_no_grad_noise', 'fedsecure', 'cifar10', 42, {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': False}),
    ('ablation_no_grad_noise', 'fedsecure', 'cifar10', 123, {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': False}),
    
    ('ablation_no_adv', 'fedsecure', 'cifar10', 42, {'use_adversarial': False, 'use_privacy_reg': True, 'use_grad_noise': True}),
    ('ablation_no_adv', 'fedsecure', 'cifar10', 123, {'use_adversarial': False, 'use_privacy_reg': True, 'use_grad_noise': True}),
]


def run_single(exp_name, exp_type, dataset, seed, extra_args):
    """Run a single experiment."""
    cmd = [
        'python', 'exp/shared/fast_trainer.py',
        '--experiment_name', exp_name,
        '--experiment_type', exp_type,
        '--dataset', dataset,
        '--seed', str(seed),
        '--global_rounds', '3',
        '--local_epochs', '1',
        '--num_clients', '2',
        '--batch_size', '256',
    ]
    
    for k, v in extra_args.items():
        if isinstance(v, bool) and v:
            cmd.append(f'--{k}')
        elif not isinstance(v, bool):
            cmd.extend([f'--{k}', str(v)])
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            return True, elapsed, None
        else:
            return False, elapsed, result.stderr[-300:]
    except subprocess.TimeoutExpired:
        return False, 150, "Timeout"


def main():
    print("="*60)
    print("FedSecure-CL: Minimal Experiment Runner")
    print("Settings: 3 rounds, 2 clients, 1 local epoch")
    print("="*60)
    
    os.makedirs('results/models', exist_ok=True)
    
    results = []
    start_total = time.time()
    
    for i, (name, exp_type, dataset, seed, extra) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {name} (seed={seed})")
        sys.stdout.flush()
        
        success, elapsed, error = run_single(name, exp_type, dataset, seed, extra)
        
        results.append({
            'name': name,
            'dataset': dataset,
            'seed': seed,
            'success': success,
            'time': elapsed,
            'error': error
        })
        
        status = "✓" if success else "✗"
        print(f"  {status} {elapsed:.0f}s")
        sys.stdout.flush()
        
        # Save progress
        with open('results/experiment_progress.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    total_time = time.time() - start_total
    
    print("\n" + "="*60)
    successful = sum(1 for r in results if r['success'])
    print(f"Completed: {successful}/{len(results)} successful")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if successful > 0:
        print("\nGenerating visualizations...")
        subprocess.run(['python', 'exp/shared/visualize_results.py'])


if __name__ == '__main__':
    main()
