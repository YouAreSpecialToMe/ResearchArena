#!/usr/bin/env python3
"""
Fast experiment runner for FedSecure-CL.
Runs all key experiments with reduced settings for timely completion.
"""

import os
import sys
import subprocess
import json
import time
from collections import defaultdict

sys.path.insert(0, 'exp/shared')

EXPERIMENTS = [
    # (name, type, dataset, seeds, extra_args)
    ('baseline_fcl', 'standard', 'cifar10', [42, 123, 456], {}),
    ('baseline_fcl_at', 'fcl_at', 'cifar10', [42, 123, 456], {'use_adversarial': True}),
    ('baseline_fcl_dp', 'fcl_dp', 'cifar10', [42, 123, 456], {}),
    ('fedsecure_cl', 'fedsecure', 'cifar10', [42, 123, 456], {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': True}),
    ('ablation_no_privacy', 'fedsecure', 'cifar10', [42, 123], {'use_adversarial': True, 'use_privacy_reg': False, 'use_grad_noise': True}),
    ('ablation_no_grad_noise', 'fedsecure', 'cifar10', [42, 123], {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': False}),
    ('ablation_no_adv', 'fedsecure', 'cifar10', [42, 123], {'use_adversarial': False, 'use_privacy_reg': True, 'use_grad_noise': True}),
    ('baseline_fcl_cifar100', 'standard', 'cifar100', [42], {}),
    ('baseline_fcl_at_cifar100', 'fcl_at', 'cifar100', [42], {'use_adversarial': True}),
    ('fedsecure_cl_cifar100', 'fedsecure', 'cifar100', [42], {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': True}),
]


def run_experiment(name, exp_type, dataset, seed, extra_args):
    """Run a single experiment."""
    cmd = [
        'python', 'exp/shared/fast_trainer.py',
        '--experiment_name', name,
        '--experiment_type', exp_type,
        '--dataset', dataset,
        '--seed', str(seed),
        '--global_rounds', '15' if dataset == 'cifar10' else '10',
        '--local_epochs', '2',
        '--num_clients', '5',
    ]
    
    for key, val in extra_args.items():
        if isinstance(val, bool):
            if val:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(val)])
    
    print(f"\n{'='*60}")
    print(f"Running: {name} (seed={seed}, dataset={dataset})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"[SUCCESS] {name} seed={seed} in {elapsed/60:.1f} min")
        return True, elapsed
    else:
        print(f"[FAILED] {name} seed={seed}")
        print(f"STDERR: {result.stderr[-500:]}")
        return False, elapsed


def main():
    print("="*60)
    print("FedSecure-CL Fast Experiment Runner")
    print("="*60)
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    results = []
    total_start = time.time()
    
    for name, exp_type, dataset, seeds, extra_args in EXPERIMENTS:
        for seed in seeds:
            success, elapsed = run_experiment(name, exp_type, dataset, seed, extra_args)
            results.append({
                'name': name,
                'dataset': dataset,
                'seed': seed,
                'success': success,
                'time': elapsed
            })
            
            # Save progress
            with open('results/experiment_progress.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("All Experiments Completed!")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"Successful: {successful}/{len(results)}")
    print(f"Total time: {total_time/3600:.2f} hours")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    subprocess.run(['python', 'exp/shared/visualize_results.py'])
    
    print("\nDone!")


if __name__ == '__main__':
    main()
