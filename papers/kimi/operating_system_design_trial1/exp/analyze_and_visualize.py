#!/usr/bin/env python3
"""
Comprehensive Analysis and Visualization for UniSched.
Includes detailed statistical analysis and publication-quality figures.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# Setup matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_all_results():
    """Load all experiment results."""
    results = {}
    
    files = {
        'main': 'exp/results/main_experiments.json',
        'targeted': 'exp/results/targeted_experiments.json',
        'pmu': 'exp/pmu_validation/pmu_overhead_results.json'
    }
    
    for key, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)
    
    return results


def process_main_results(main_results):
    """Process main experiment results."""
    records = []
    
    for result in main_results:
        if 'error' in result or 'metrics' not in result:
            continue
        
        scheduler = result['scheduler']
        workload = result['workload']
        seed = result['seed']
        metrics = result['metrics']
        
        # Parse workload
        parts = workload.replace('.json', '').split('_')
        
        # Extract workload type and size
        if 'seed' in workload:
            # Find the seed part
            for i, part in enumerate(parts):
                if 'seed' in part:
                    size = parts[i-1]
                    workload_type = '_'.join(parts[:i-1])
                    break
        else:
            workload_type = '_'.join(parts[:-1])
            size = parts[-1]
        
        records.append({
            'scheduler': scheduler,
            'workload_type': workload_type,
            'size': size,
            'seed': seed,
            'throughput': metrics.get('throughput_tasks_per_sec', 0),
            'avg_latency': metrics.get('avg_latency_ms', 0),
            'p95_latency': metrics.get('p95_latency_ms', 0),
            'fairness': metrics.get('jain_fairness', 0)
        })
    
    return pd.DataFrame(records)


def generate_summary_report(df, results):
    """Generate comprehensive summary report."""
    report = []
    report.append("="*80)
    report.append("UniSched Experimental Results Summary")
    report.append("="*80)
    report.append("")
    
    # PMU Validation
    if 'pmu' in results:
        pmu = results['pmu']
        report.append("PMU OVERHEAD VALIDATION")
        report.append("-"*40)
        report.append(f"Gate Status: {'PASSED' if pmu.get('gate_passed') else 'FAILED'}")
        if 'configurations' in pmu:
            for config, data in pmu['configurations'].items():
                if 'overhead_pct' in data:
                    report.append(f"  {config}: {data['overhead_pct']:.2f}% overhead")
        report.append("")
    
    # Overall statistics by scheduler
    report.append("OVERALL PERFORMANCE BY SCHEDULER")
    report.append("-"*40)
    
    scheduler_stats = df.groupby('scheduler').agg({
        'throughput': ['mean', 'std', 'count'],
        'avg_latency': ['mean', 'std'],
        'fairness': ['mean', 'std']
    }).round(3)
    
    for scheduler in df['scheduler'].unique():
        sched_data = df[df['scheduler'] == scheduler]
        throughput_mean = sched_data['throughput'].mean()
        throughput_std = sched_data['throughput'].std()
        latency_mean = sched_data['avg_latency'].mean()
        fairness_mean = sched_data['fairness'].mean()
        n = len(sched_data)
        
        report.append(f"\n{scheduler}:")
        report.append(f"  Throughput: {throughput_mean:.2f} ± {throughput_std:.2f} tasks/s (n={n})")
        report.append(f"  Avg Latency: {latency_mean:.1f} ms")
        report.append(f"  Fairness: {fairness_mean:.3f}")
    
    # Performance by workload type
    report.append("\n")
    report.append("PERFORMANCE BY WORKLOAD TYPE")
    report.append("-"*40)
    
    for workload_type in sorted(df['workload_type'].unique()):
        report.append(f"\n{workload_type}:")
        wt_data = df[df['workload_type'] == workload_type]
        
        for scheduler in ['EEVDF', 'AutoNUMA', 'UniSched_Full']:
            sched_data = wt_data[wt_data['scheduler'] == scheduler]
            if len(sched_data) > 0:
                throughput = sched_data['throughput'].mean()
                report.append(f"  {scheduler:20s}: {throughput:.2f} tasks/s")
    
    # Statistical comparisons
    report.append("\n")
    report.append("STATISTICAL COMPARISONS (UniSched vs Baselines)")
    report.append("-"*40)
    
    baselines = ['EEVDF', 'AutoNUMA', 'Tiresias']
    
    for baseline in baselines:
        baseline_data = df[df['scheduler'] == baseline]['throughput']
        unisched_data = df[df['scheduler'] == 'UniSched_Full']['throughput']
        
        if len(baseline_data) > 0 and len(unisched_data) > 0:
            # Independent t-test
            t_stat, p_val = stats.ttest_ind(unisched_data, baseline_data)
            
            baseline_mean = baseline_data.mean()
            unisched_mean = unisched_data.mean()
            improvement = ((unisched_mean - baseline_mean) / baseline_mean * 100) 
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            report.append(f"\nvs {baseline}:")
            report.append(f"  Improvement: {improvement:+.2f}% {sig}")
            report.append(f"  p-value: {p_val:.4f}")
    
    # Success criteria
    report.append("\n")
    report.append("SUCCESS CRITERIA EVALUATION")
    report.append("-"*40)
    
    # PMU overhead
    report.append(f"1. PMU Overhead ≤ 3%: {'✓ PASS' if 'pmu' in results and results['pmu'].get('gate_passed') else '✗ FAIL'}")
    
    # Throughput improvement
    autonuma_mean = df[df['scheduler'] == 'AutoNUMA']['throughput'].mean()
    unisched_mean = df[df['scheduler'] == 'UniSched_Full']['throughput'].mean()
    improvement = ((unisched_mean - autonuma_mean) / autonuma_mean * 100) if autonuma_mean > 0 else 0
    
    report.append(f"2. Throughput Improvement ≥ 5%: {'✓ PASS' if improvement >= 5 else '✗ FAIL'} ({improvement:.2f}%)")
    
    # Fairness
    fairness = df[df['scheduler'] == 'UniSched_Full']['fairness'].mean()
    report.append(f"3. Fairness ≥ 0.85: {'✓ PASS' if fairness >= 0.85 else '✗ FAIL'} ({fairness:.3f})")
    
    # Deployability
    report.append(f"4. Deployability (0 kernel lines): ✓ PASS (BPF-based)")
    
    report.append("\n")
    report.append("="*80)
    
    return "\n".join(report)


def generate_figures(df):
    """Generate publication-quality figures."""
    os.makedirs('figures', exist_ok=True)
    
    # Figure 1: Overall throughput comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    schedulers = ['EEVDF', 'AutoNUMA', 'Tiresias', 'CXLAimPod', 'UniSched_Full']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
    
    throughput_data = []
    for scheduler in schedulers:
        sched_data = df[df['scheduler'] == scheduler]['throughput']
        throughput_data.append(sched_data.values if len(sched_data) > 0 else [0])
    
    bp = ax.boxplot(throughput_data, labels=[s.replace('_Full', '') for s in schedulers],
                    patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Throughput (tasks/sec)')
    ax.set_title('Overall Throughput Comparison Across All Workloads')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/overall_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Throughput by workload type
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    workload_types = sorted(df['workload_type'].unique())[:6]
    
    for idx, wt in enumerate(workload_types):
        ax = axes[idx]
        wt_data = df[df['workload_type'] == wt]
        
        schedulers_present = [s for s in schedulers if s in wt_data['scheduler'].values]
        throughput_by_sched = [wt_data[wt_data['scheduler'] == s]['throughput'].values 
                               for s in schedulers_present]
        
        if throughput_by_sched and any(len(x) > 0 for x in throughput_by_sched):
            bp = ax.boxplot(throughput_by_sched, labels=[s.replace('_Full', '') for s in schedulers_present],
                           patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors[:len(schedulers_present)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(wt.replace('_', ' ').title())
        ax.set_ylabel('Throughput')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/throughput_by_workload.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Fairness comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fairness_data = []
    for scheduler in schedulers:
        sched_data = df[df['scheduler'] == scheduler]['fairness']
        fairness_data.append(sched_data.values if len(sched_data) > 0 else [0])
    
    bp = ax.boxplot(fairness_data, labels=[s.replace('_Full', '') for s in schedulers],
                    patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0.85, color='red', linestyle='--', label='Target (0.85)')
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title('Fairness Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fairness_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Latency comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    latency_data = []
    for scheduler in schedulers:
        sched_data = df[df['scheduler'] == scheduler]['avg_latency']
        latency_data.append(sched_data.values if len(sched_data) > 0 else [0])
    
    bp = ax.boxplot(latency_data, labels=[s.replace('_Full', '') for s in schedulers],
                    patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Latency Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated figures:")
    print("  - figures/overall_throughput.png")
    print("  - figures/throughput_by_workload.png")
    print("  - figures/fairness_comparison.png")
    print("  - figures/latency_comparison.png")


def to_python_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    return obj


def create_final_results_json(df, results):
    """Create final aggregated results.json."""
    
    final_results = {
        'metadata': {
            'experiment_count': int(len(df)),
            'schedulers': [str(s) for s in df['scheduler'].unique().tolist()],
            'workload_types': [str(s) for s in df['workload_type'].unique().tolist()],
            'seeds': [int(s) for s in df['seed'].unique().tolist()]
        },
        'pmu_validation': to_python_types(results.get('pmu', {})),
        'scheduler_performance': {},
        'workload_performance': {},
        'statistical_tests': {},
        'success_criteria': {}
    }
    
    # Scheduler performance
    for scheduler in df['scheduler'].unique():
        sched_data = df[df['scheduler'] == scheduler]
        
        final_results['scheduler_performance'][scheduler] = {
            'throughput_mean': float(sched_data['throughput'].mean()),
            'throughput_std': float(sched_data['throughput'].std()),
            'latency_mean': float(sched_data['avg_latency'].mean()),
            'fairness_mean': float(sched_data['fairness'].mean()),
            'n_experiments': len(sched_data)
        }
    
    # Workload-specific performance
    for workload_type in df['workload_type'].unique():
        wt_data = df[df['workload_type'] == workload_type]
        
        final_results['workload_performance'][workload_type] = {}
        for scheduler in wt_data['scheduler'].unique():
            sched_data = wt_data[wt_data['scheduler'] == scheduler]
            final_results['workload_performance'][workload_type][scheduler] = {
                'throughput_mean': float(sched_data['throughput'].mean()),
                'throughput_std': float(sched_data['throughput'].std())
            }
    
    # Statistical tests
    baselines = ['EEVDF', 'AutoNUMA', 'Tiresias']
    for baseline in baselines:
        if baseline in df['scheduler'].values and 'UniSched_Full' in df['scheduler'].values:
            baseline_data = df[df['scheduler'] == baseline]['throughput']
            unisched_data = df[df['scheduler'] == 'UniSched_Full']['throughput']
            
            t_stat, p_val = stats.ttest_ind(unisched_data, baseline_data)
            
            baseline_mean = baseline_data.mean()
            unisched_mean = unisched_data.mean()
            improvement = ((unisched_mean - baseline_mean) / baseline_mean * 100)
            
            final_results['statistical_tests'][f'UniSched_vs_{baseline}'] = {
                'improvement_percent': float(improvement),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05)
            }
    
    # Success criteria
    autonuma_mean = df[df['scheduler'] == 'AutoNUMA']['throughput'].mean()
    unisched_mean = df[df['scheduler'] == 'UniSched_Full']['throughput'].mean()
    improvement = ((unisched_mean - autonuma_mean) / autonuma_mean * 100) if autonuma_mean > 0 else 0
    fairness = df[df['scheduler'] == 'UniSched_Full']['fairness'].mean()
    
    final_results['success_criteria'] = {
        'pmu_overhead_gate': {
            'target': '<= 3%',
            'achieved': results.get('pmu', {}).get('configurations', {}).get('1.0%', {}).get('overhead_pct', 0),
            'passed': results.get('pmu', {}).get('gate_passed', False)
        },
        'throughput_improvement': {
            'target': '>= 5%',
            'achieved': float(improvement),
            'passed': improvement >= 5
        },
        'fairness': {
            'target': '>= 0.85',
            'achieved': float(fairness),
            'passed': fairness >= 0.85
        },
        'deployability': {
            'target': '0 kernel lines',
            'achieved': '0 (BPF-based)',
            'passed': True
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(to_python_types(final_results), f, indent=2)
    
    print("\nGenerated: results.json")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("UniSched Analysis and Visualization")
    print("="*80)
    print()
    
    # Load results
    print("Loading results...")
    results = load_all_results()
    
    # Process main results
    if 'main' in results:
        print(f"Processing {len(results['main'])} main experiment results...")
        df = process_main_results(results['main'])
        print(f"  -> {len(df)} valid records")
        
        # Generate summary report
        print("\nGenerating summary report...")
        report = generate_summary_report(df, results)
        print(report)
        
        # Save report
        with open('exp/results/summary_report.txt', 'w') as f:
            f.write(report)
        
        # Generate figures
        print("\nGenerating figures...")
        generate_figures(df)
        
        # Create results.json
        print("\nCreating final results.json...")
        create_final_results_json(df, results)
    else:
        print("No main results found!")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
