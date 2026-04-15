#!/usr/bin/env python3
"""Master script to run all experiments and aggregate results."""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Experiment configuration
SEEDS = [42, 123, 456]
EXPERIMENTS = {
    'baseline_ldm': {
        'name': 'Standard LDM',
        'path': 'exp/baseline_ldm/run.py',
        'epochs_vae': 30,
        'epochs_diff': 50,
        'depends_on': None
    },
    'baseline_cat': {
        'name': 'CAT-Style Adaptive Tokenization',
        'path': 'exp/baseline_cat/run.py',
        'epochs_vae': 30,
        'epochs_diff': 50,
        'depends_on': None
    },
    'baseline_tcaq': {
        'name': 'TCAQ-Style 2D Quantization',
        'path': 'exp/baseline_tcaq/run.py',
        'depends_on': 'baseline_ldm'
    },
    'ualq_diff': {
        'name': 'UALQ-Diff (Ours)',
        'path': 'exp/ualq_diff/run.py',
        'epochs_stage1': 20,
        'epochs_stage2': 35,
        'epochs_stage3': 10,
        'depends_on': None
    }
}


def run_experiment(exp_key, seed):
    """Run a single experiment with given seed."""
    exp = EXPERIMENTS[exp_key]
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']} (seed={seed})")
    print(f"{'='*60}")
    
    cmd = [
        'python', exp['path'],
        '--seed', str(seed),
        '--device', 'cuda'
    ]
    
    # Add experiment-specific args
    if 'epochs_vae' in exp:
        cmd.extend(['--epochs_vae', str(exp['epochs_vae'])])
    if 'epochs_diff' in exp:
        cmd.extend(['--epochs_diff', str(exp['epochs_diff'])])
    if 'epochs_stage1' in exp:
        cmd.extend(['--epochs_stage1', str(exp['epochs_stage1'])])
    if 'epochs_stage2' in exp:
        cmd.extend(['--epochs_stage2', str(exp['epochs_stage2'])])
    if 'epochs_stage3' in exp:
        cmd.extend(['--epochs_stage3', str(exp['epochs_stage3'])])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: {exp['name']} failed with seed {seed}")
        return False, elapsed
    
    return True, elapsed


def aggregate_results():
    """Aggregate results from all experiments."""
    results = {}
    
    for exp_key, exp in EXPERIMENTS.items():
        exp_results = []
        
        for seed in SEEDS:
            result_file = f'exp/{exp_key}/results_seed{seed}.json'
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    exp_results.append(data)
        
        if exp_results:
            # Compute mean and std across seeds
            metrics_to_aggregate = [
                'fid', 'is_mean', 'time_per_image_ms', 'peak_memory_gb',
                'avg_bitwidth', 'estimated_bops_g'
            ]
            
            aggregated = {
                'name': exp['name'],
                'num_seeds': len(exp_results),
                'metrics': {}
            }
            
            for metric in metrics_to_aggregate:
                values = [r.get(metric, 0) for r in exp_results if metric in r]
                if values:
                    aggregated['metrics'][metric] = {
                        'mean': sum(values) / len(values),
                        'std': (sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5,
                        'values': values
                    }
            
            # Special fields
            if 'avg_token_density' in exp_results[0]:
                aggregated['token_density'] = {
                    'mean': sum(r.get('avg_token_density', 0) for r in exp_results) / len(exp_results),
                    'std': 0
                }
            
            if 'effective_tokens_per_image' in exp_results[0]:
                aggregated['effective_tokens'] = {
                    'mean': sum(r.get('effective_tokens_per_image', 0) for r in exp_results) / len(exp_results),
                    'std': 0
                }
            
            results[exp_key] = aggregated
    
    return results


def generate_summary(results):
    """Generate summary table."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<35} {'FID':<15} {'IS':<15} {'Time(ms)':<12} {'Mem(GB)':<10}")
    print("-"*80)
    
    for exp_key, data in results.items():
        name = data['name']
        metrics = data['metrics']
        
        fid = metrics.get('fid', {})
        is_score = metrics.get('is_mean', {})
        time_ms = metrics.get('time_per_image_ms', {})
        mem = metrics.get('peak_memory_gb', {})
        
        fid_str = f"{fid.get('mean', 0):.3f}±{fid.get('std', 0):.3f}" if fid else "N/A"
        is_str = f"{is_score.get('mean', 0):.2f}±{is_score.get('std', 0):.2f}" if is_score else "N/A"
        time_str = f"{time_ms.get('mean', 0):.1f}±{time_ms.get('std', 0):.1f}" if time_ms else "N/A"
        mem_str = f"{mem.get('mean', 0):.2f}±{mem.get('std', 0):.2f}" if mem else "N/A"
        
        print(f"{name:<35} {fid_str:<15} {is_str:<15} {time_str:<12} {mem_str:<10}")
    
    print("\n" + "="*80)
    print("EFFICIENCY METRICS")
    print("="*80)
    
    print(f"\n{'Method':<35} {'Avg Bits':<15} {'BOPs(G)':<15} {'Tokens/img':<15}")
    print("-"*80)
    
    for exp_key, data in results.items():
        name = data['name']
        bits = data['metrics'].get('avg_bitwidth', {}).get('mean', 16.0)
        bops = data['metrics'].get('estimated_bops_g', {}).get('mean', 0)
        tokens = data.get('effective_tokens', {}).get('mean', 64)
        
        bits_str = f"{bits:.2f}" if bits else "16 (FP)"
        bops_str = f"{bops:.2f}" if bops else "N/A"
        tokens_str = f"{tokens:.1f}" if tokens else "64 (fixed)"
        
        print(f"{name:<35} {bits_str:<15} {bops_str:<15} {tokens_str:<15}")
    
    print()


def main():
    """Main execution."""
    total_start = time.time()
    
    # Create output directories
    os.makedirs('exp', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("="*80)
    print("UALQ-DIFF EXPERIMENT SUITE")
    print("="*80)
    print(f"Seeds: {SEEDS}")
    print(f"Experiments: {list(EXPERIMENTS.keys())}")
    
    # Determine execution order based on dependencies
    execution_order = ['baseline_ldm', 'baseline_cat', 'baseline_tcaq', 'ualq_diff']
    
    # Track results
    all_success = True
    timing_log = {}
    
    for exp_key in execution_order:
        exp = EXPERIMENTS[exp_key]
        timing_log[exp_key] = []
        
        # Run for each seed
        for seed in SEEDS:
            success, elapsed = run_experiment(exp_key, seed)
            timing_log[exp_key].append({'seed': seed, 'time': elapsed, 'success': success})
            
            if not success:
                all_success = False
                print(f"WARNING: {exp['name']} failed for seed {seed}")
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS...")
    print("="*80)
    
    results = aggregate_results()
    
    # Generate summary
    generate_summary(results)
    
    # Save aggregated results
    final_results = {
        'experiments': results,
        'timing_log': timing_log,
        'total_runtime_minutes': (time.time() - total_start) / 60,
        'seeds': SEEDS,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: results.json")
    print(f"Total runtime: {final_results['total_runtime_minutes']:.1f} minutes")
    
    return all_success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
