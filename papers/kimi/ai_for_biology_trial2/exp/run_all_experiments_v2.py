#!/usr/bin/env python3
"""
Run all experiments for CROSS-GRN with fixed implementations.
Executes training and evaluation for:
1. CROSS-GRN main model (3 seeds)
2. scMultiomeGRN baseline (3 seeds)
3. XATGRN baseline (3 seeds)
4. Ablation studies (3 seeds each)
5. Correlation baseline
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import subprocess
import json
import os
import time
import argparse


def run_command(cmd, log_file=None):
    """Run a command and optionally log output."""
    print(f"Running: {cmd}")
    if log_file:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def run_experiment(exp_name, script_path, args_dict, log_dir='exp/logs'):
    """Run a single experiment."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Build command
    cmd = f"cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01 && python {script_path}"
    for key, value in args_dict.items():
        cmd += f" --{key} {value}"
    
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    success = run_command(cmd, log_file)
    
    if success:
        print(f"  ✓ {exp_name} completed")
    else:
        print(f"  ✗ {exp_name} failed (see {log_file})")
    
    return success


def run_all_experiments(parallel=False, quick=False):
    """Run all experiments."""
    os.makedirs('exp/logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    seeds = [42, 43, 44]
    epochs = 30 if quick else 50
    batch_size = 64
    
    results = {}
    
    # =========================================================================
    # 1. Simple baselines (fast)
    # =========================================================================
    print("\n" + "="*70)
    print("1. Running simple baselines (correlation, cosine)")
    print("="*70)
    
    # Correlation baseline
    print("\nRunning correlation baseline...")
    run_experiment(
        'correlation_baseline',
        'exp/simple_baselines.py',
        {'method': 'correlation', 'output': 'exp/correlation_baseline/results.json'},
        log_dir='exp/logs'
    )
    
    # Cosine baseline
    print("\nRunning cosine baseline...")
    run_experiment(
        'cosine_baseline',
        'exp/simple_baselines.py',
        {'method': 'cosine', 'output': 'exp/cosine_baseline/results.json'},
        log_dir='exp/logs'
    )
    
    # =========================================================================
    # 2. CROSS-GRN main model (3 seeds)
    # =========================================================================
    print("\n" + "="*70)
    print("2. Training CROSS-GRN (3 seeds)")
    print("="*70)
    
    crossgrn_results = []
    for seed in seeds:
        print(f"\n  Training CROSS-GRN with seed {seed}...")
        output_path = f'exp/crossgrn_main/results_s{seed}.json'
        model_path = f'models/crossgrn_s{seed}.pt'
        
        success = run_experiment(
            f'crossgrn_s{seed}',
            'exp/train_fixed.py',
            {
                'seed': seed,
                'epochs': epochs,
                'batch_size': batch_size,
                'output': output_path,
                'model_path': model_path,
                'use_cell_type_cond': 1,
                'use_asymmetric': 1,
                'predict_sign': 1,
                'variant': 'full'
            },
            log_dir='exp/logs'
        )
        
        if success and os.path.exists(output_path):
            with open(output_path) as f:
                crossgrn_results.append(json.load(f))
    
    results['crossgrn'] = crossgrn_results
    
    # =========================================================================
    # 3. scMultiomeGRN baseline (3 seeds)
    # =========================================================================
    print("\n" + "="*70)
    print("3. Training scMultiomeGRN baseline (3 seeds)")
    print("="*70)
    
    scmulti_results = []
    for seed in seeds:
        print(f"\n  Training scMultiomeGRN with seed {seed}...")
        output_path = f'exp/scmultiomegrn_baseline/results_s{seed}.json'
        model_path = f'models/scmultiomegrn_s{seed}.pt'
        
        success = run_experiment(
            f'scmultiomegrn_s{seed}',
            'exp/run_baseline_scmultiomegrn.py',
            {
                'seed': seed,
                'epochs': epochs,
                'batch_size': batch_size,
                'output': output_path,
                'model_path': model_path
            },
            log_dir='exp/logs'
        )
        
        if success and os.path.exists(output_path):
            with open(output_path) as f:
                scmulti_results.append(json.load(f))
    
    results['scmultiomegrn'] = scmulti_results
    
    # =========================================================================
    # 4. XATGRN baseline (3 seeds)
    # =========================================================================
    print("\n" + "="*70)
    print("4. Training XATGRN baseline (3 seeds)")
    print("="*70)
    
    xatgrn_results = []
    for seed in seeds:
        print(f"\n  Training XATGRN with seed {seed}...")
        output_path = f'exp/xatgrn_baseline/results_s{seed}.json'
        model_path = f'models/xatgrn_s{seed}.pt'
        
        success = run_experiment(
            f'xatgrn_s{seed}',
            'exp/run_baseline_xatgrn.py',
            {
                'seed': seed,
                'epochs': epochs,
                'batch_size': batch_size,
                'output': output_path,
                'model_path': model_path
            },
            log_dir='exp/logs'
        )
        
        if success and os.path.exists(output_path):
            with open(output_path) as f:
                xatgrn_results.append(json.load(f))
    
    results['xatgrn'] = xatgrn_results
    
    # =========================================================================
    # 5. Ablation studies
    # =========================================================================
    print("\n" + "="*70)
    print("5. Running ablation studies")
    print("="*70)
    
    ablations = [
        ('symmetric', {'use_asymmetric': 0, 'use_cell_type_cond': 1, 'predict_sign': 1}),
        ('no_celltype', {'use_asymmetric': 1, 'use_cell_type_cond': 0, 'predict_sign': 1}),
        ('no_sign', {'use_asymmetric': 1, 'use_cell_type_cond': 1, 'predict_sign': 0}),
    ]
    
    for abl_name, abl_config in ablations:
        print(f"\n  Running ablation: {abl_name}")
        abl_results = []
        
        for seed in seeds:
            output_path = f'exp/ablation_{abl_name}/results_s{seed}.json'
            model_path = f'models/ablation_{abl_name}_s{seed}.pt'
            
            success = run_experiment(
                f'ablation_{abl_name}_s{seed}',
                'exp/train_fixed.py',
                {
                    'seed': seed,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'output': output_path,
                    'model_path': model_path,
                    'variant': abl_name,
                    **abl_config
                },
                log_dir='exp/logs'
            )
            
            if success and os.path.exists(output_path):
                with open(output_path) as f:
                    abl_results.append(json.load(f))
        
        results[f'ablation_{abl_name}'] = abl_results
    
    # =========================================================================
    # 6. Aggregate results
    # =========================================================================
    print("\n" + "="*70)
    print("6. Aggregating results")
    print("="*70)
    
    aggregate_results(results)
    
    return results


def aggregate_results(results):
    """Aggregate results from all experiments."""
    
    def compute_stats(values):
        """Compute mean and std."""
        if not values:
            return {'mean': 0, 'std': 0, 'values': []}
        import numpy as np
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    aggregated = {}
    
    for method, method_results in results.items():
        if not method_results:
            continue
        
        metrics = {}
        for metric in ['auroc', 'auprc', 'sign_accuracy', 'pearson_r']:
            values = [r['metrics'].get(metric, 0) for r in method_results if 'metrics' in r]
            if values:
                metrics[metric] = compute_stats(values)
        
        aggregated[method] = {
            'metrics': metrics,
            'n_seeds': len(method_results)
        }
    
    # Load simple baselines
    for baseline in ['correlation', 'cosine']:
        baseline_path = f'exp/{baseline}_baseline/results.json'
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline_data = json.load(f)
                aggregated[baseline] = {
                    'metrics': {k: {'mean': v, 'std': 0, 'values': [v]} 
                               for k, v in baseline_data.get('metrics', {}).items()},
                    'n_seeds': 1
                }
    
    # Save aggregated results
    with open('results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print("\n" + "="*70)
    print("AGGREGATED RESULTS")
    print("="*70)
    print(f"{'Method':<20} {'AUROC':<20} {'AUPRC':<20} {'Sign Acc':<20}")
    print("-"*70)
    
    for method, data in aggregated.items():
        metrics = data['metrics']
        auroc = metrics.get('auroc', {})
        auprc = metrics.get('auprc', {})
        sign_acc = metrics.get('sign_accuracy', {})
        
        auroc_str = f"{auroc.get('mean', 0):.4f}±{auroc.get('std', 0):.4f}" if auroc else "N/A"
        auprc_str = f"{auprc.get('mean', 0):.4f}±{auprc.get('std', 0):.4f}" if auprc else "N/A"
        sign_str = f"{sign_acc.get('mean', 0):.4f}±{sign_acc.get('std', 0):.4f}" if sign_acc else "N/A"
        
        print(f"{method:<20} {auroc_str:<20} {auprc_str:<20} {sign_str:<20}")
    
    print("\nResults saved to results.json")
    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run with fewer epochs for testing')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    args = parser.parse_args()
    
    start_time = time.time()
    
    results = run_all_experiments(parallel=args.parallel, quick=args.quick)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"All experiments completed in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
