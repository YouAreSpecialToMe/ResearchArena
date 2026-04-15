#!/usr/bin/env python3
"""
Master script to run all CASS-ViM experiments.
Runs experiments sequentially due to single GPU constraint.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

# Experiment configuration
EXPERIMENTS = [
    # Format: (model_type, seeds, description)
    ('vmamba', [42, 123, 456], 'VMamba baseline (fixed 4-direction)'),
    ('localmamba', [42, 123, 456], 'LocalMamba baseline (fixed per-layer directions)'),
    ('cassvim_4d', [42, 123, 456], 'CASS-ViM with 4 directions (gradient-based)'),
    ('cassvim_8d', [42, 123, 456], 'CASS-ViM with 8 directions (gradient-based)'),
    ('random_selection', [42, 123, 456], 'Ablation: Random direction selection'),
    ('fixed_perlayer', [42, 123, 456], 'Ablation: Fixed per-layer directions'),
]

RESULTS_DIR = './results'
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'


def run_experiment(model, seed, exp_idx, total_exps):
    """Run a single experiment."""
    exp_name = f'{model}_seed{seed}'
    log_file = os.path.join(LOG_DIR, f'{exp_name}.log')
    
    print(f'\n{"="*70}')
    print(f'Experiment {exp_idx}/{total_exps}: {model} (seed={seed})')
    print(f'{"="*70}')
    
    cmd = [
        'python', 'src/train.py',
        '--model', model,
        '--seed', str(seed),
        '--epochs', '100',
        '--batch_size', '128',
        '--lr', '1e-3',
        '--weight_decay', '0.05',
        '--save_dir', CHECKPOINT_DIR,
        '--exp_name', exp_name
    ]
    
    print(f'Command: {" ".join(cmd)}')
    print(f'Log file: {log_file}')
    
    start_time = time.time()
    
    # Run experiment
    with open(log_file, 'w') as f:
        f.write(f'Starting experiment: {exp_name}\n')
        f.write(f'Time: {datetime.now().isoformat()}\n')
        f.write(f'Command: {" ".join(cmd)}\n')
        f.write('='*70 + '\n\n')
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
    
    elapsed = time.time() - start_time
    
    if process.returncode != 0:
        print(f'ERROR: Experiment {exp_name} failed!')
        return None
    
    # Load results
    results_file = os.path.join(CHECKPOINT_DIR, f'{exp_name}_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f'\nCompleted: Best acc = {results["best_test_acc"]:.2f}%, Time = {elapsed/60:.1f} min')
        return results
    else:
        print(f'ERROR: Results file not found: {results_file}')
        return None


def aggregate_results():
    """Aggregate results from all experiments."""
    print('\n' + '='*70)
    print('Aggregating results...')
    print('='*70)
    
    all_results = {}
    
    for model, seeds, desc in EXPERIMENTS:
        model_results = []
        
        for seed in seeds:
            exp_name = f'{model}_seed{seed}'
            results_file = os.path.join(CHECKPOINT_DIR, f'{exp_name}_results.json')
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                model_results.append({
                    'seed': seed,
                    'best_acc': results['best_test_acc'],
                    'final_acc': results['final_test_acc'],
                    'train_time': results['train_time_minutes']
                })
        
        if model_results:
            best_accs = [r['best_acc'] for r in model_results]
            final_accs = [r['final_acc'] for r in model_results]
            train_times = [r['train_time'] for r in model_results]
            
            all_results[model] = {
                'description': desc,
                'seeds': [r['seed'] for r in model_results],
                'best_acc_mean': sum(best_accs) / len(best_accs),
                'best_acc_std': (sum((x - sum(best_accs)/len(best_accs))**2 for x in best_accs) / len(best_accs))**0.5,
                'final_acc_mean': sum(final_accs) / len(final_accs),
                'final_acc_std': (sum((x - sum(final_accs)/len(final_accs))**2 for x in final_accs) / len(final_accs))**0.5,
                'avg_train_time': sum(train_times) / len(train_times),
                'individual_results': model_results
            }
    
    # Save aggregated results
    output_file = os.path.join(RESULTS_DIR, 'aggregated_results.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print('\n' + '='*70)
    print('RESULTS SUMMARY')
    print('='*70)
    print(f'{"Model":<25} {"Best Acc (%)":>20} {"Time (min)":>15}')
    print('-'*70)
    
    for model, results in all_results.items():
        mean = results['best_acc_mean']
        std = results['best_acc_std']
        time = results['avg_train_time']
        print(f'{model:<25} {mean:>8.2f} ± {std:<8.2f} {time:>12.1f}')
    
    print('='*70)
    
    # Check success criteria
    print('\nSUCCESS CRITERIA EVALUATION:')
    print('-'*70)
    
    if 'vmamba' in all_results and 'cassvim_4d' in all_results:
        vmamba_acc = all_results['vmamba']['best_acc_mean']
        cassvim_acc = all_results['cassvim_4d']['best_acc_mean']
        diff = cassvim_acc - vmamba_acc
        status = 'PASS' if abs(diff) <= 1.0 else 'FAIL'
        print(f'1. CASS-ViM within 1% of VMamba: {diff:+.2f}% [{status}]')
    
    if 'localmamba' in all_results and 'cassvim_4d' in all_results:
        localmamba_acc = all_results['localmamba']['best_acc_mean']
        cassvim_acc = all_results['cassvim_4d']['best_acc_mean']
        diff = cassvim_acc - localmamba_acc
        status = 'PASS' if diff > 0 else 'FAIL'
        print(f'2. CASS-ViM outperforms LocalMamba: {diff:+.2f}% [{status}]')
    
    if 'random_selection' in all_results and 'cassvim_4d' in all_results:
        random_acc = all_results['random_selection']['best_acc_mean']
        cassvim_acc = all_results['cassvim_4d']['best_acc_mean']
        diff = cassvim_acc - random_acc
        status = 'PASS' if diff >= 0.5 else 'FAIL'
        print(f'3. Gradient > Random by >=0.5%: {diff:+.2f}% [{status}]')
    
    if 'cassvim_4d' in all_results and 'cassvim_8d' in all_results:
        acc_4d = all_results['cassvim_4d']['best_acc_mean']
        acc_8d = all_results['cassvim_8d']['best_acc_mean']
        diff = abs(acc_4d - acc_8d)
        status = 'PASS' if diff <= 1.0 else 'FAIL'
        print(f'4. 4D vs 8D similar performance: diff={diff:.2f}% [{status}]')
    
    print('='*70)
    
    return all_results


def main():
    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print('='*70)
    print('CASS-ViM Experiment Runner')
    print('='*70)
    print(f'Start time: {datetime.now().isoformat()}')
    print(f'Total experiments: {sum(len(seeds) for _, seeds, _ in EXPERIMENTS)}')
    print('='*70)
    
    # Calculate total experiments
    total_exps = sum(len(seeds) for _, seeds, _ in EXPERIMENTS)
    exp_idx = 0
    
    # Run all experiments
    for model, seeds, desc in EXPERIMENTS:
        for seed in seeds:
            exp_idx += 1
            result = run_experiment(model, seed, exp_idx, total_exps)
            
            if result is None:
                print(f'Warning: Experiment {model} seed {seed} failed or incomplete')
    
    # Aggregate results
    all_results = aggregate_results()
    
    print('\n' + '='*70)
    print('All experiments completed!')
    print(f'End time: {datetime.now().isoformat()}')
    print('='*70)


if __name__ == '__main__':
    main()
