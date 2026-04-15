#!/usr/bin/env python3
"""
Run all experiments efficiently with minimal settings.
"""
import os
import sys
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Experiments to run: (name, type, dataset, seed, extra_args)
EXPERIMENTS = [
    # CIFAR-10 main experiments (3 seeds)
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
    
    # Ablations (2 seeds)
    ('ablation_no_privacy', 'fedsecure', 'cifar10', 42, {'use_adversarial': True, 'use_privacy_reg': False, 'use_grad_noise': True}),
    ('ablation_no_privacy', 'fedsecure', 'cifar10', 123, {'use_adversarial': True, 'use_privacy_reg': False, 'use_grad_noise': True}),
    
    ('ablation_no_grad_noise', 'fedsecure', 'cifar10', 42, {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': False}),
    ('ablation_no_grad_noise', 'fedsecure', 'cifar10', 123, {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': False}),
    
    ('ablation_no_adv', 'fedsecure', 'cifar10', 42, {'use_adversarial': False, 'use_privacy_reg': True, 'use_grad_noise': True}),
    ('ablation_no_adv', 'fedsecure', 'cifar10', 123, {'use_adversarial': False, 'use_privacy_reg': True, 'use_grad_noise': True}),
    
    # CIFAR-100 (1 seed)
    ('baseline_fcl_cifar100', 'standard', 'cifar100', 42, {}),
    ('baseline_fcl_at_cifar100', 'fcl_at', 'cifar100', 42, {'use_adversarial': True}),
    ('fedsecure_cl_cifar100', 'fedsecure', 'cifar100', 42, {'use_adversarial': True, 'use_privacy_reg': True, 'use_grad_noise': True}),
]


def run_single(exp_name, exp_type, dataset, seed, extra_args):
    """Run a single experiment."""
    cmd = [
        'python', 'exp/shared/fast_trainer.py',
        '--experiment_name', exp_name,
        '--experiment_type', exp_type,
        '--dataset', dataset,
        '--seed', str(seed),
        '--global_rounds', '5',
        '--local_epochs', '1',
        '--num_clients', '3',
        '--batch_size', '256',
    ]
    
    for k, v in extra_args.items():
        if isinstance(v, bool) and v:
            cmd.append(f'--{k}')
        elif not isinstance(v, bool):
            cmd.extend([f'--{k}', str(v)])
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            return True, elapsed, None
        else:
            return False, elapsed, result.stderr[-500:]
    except subprocess.TimeoutExpired:
        return False, 240, "Timeout"


def main():
    print("="*60)
    print("FedSecure-CL: Fast Experiment Runner")
    print("="*60)
    
    os.makedirs('results/models', exist_ok=True)
    
    results = []
    start_total = time.time()
    
    # Run sequentially to avoid memory issues
    for i, (name, exp_type, dataset, seed, extra) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Running {name} (seed={seed}, dataset={dataset})")
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
        
        # Save progress
        with open('results/experiment_progress.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    total_time = time.time() - start_total
    
    print("\n" + "="*60)
    print("Completed!")
    print(f"Successful: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"Total time: {total_time/3600:.2f} hours")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    subprocess.run(['python', 'exp/shared/visualize_results.py'])


if __name__ == '__main__':
    main()
