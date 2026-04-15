#!/usr/bin/env python3
"""
Analysis and Visualization for WattSched Experiments

Aggregates results from all experiments, computes statistics, and generates
publication-quality figures.
"""

import json
import os
import numpy as np
from scipy import stats
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_native(v) for v in obj]
    return obj


def load_results(exp_dir: str) -> Dict:
    """Load results.json from an experiment directory."""
    path = os.path.join(exp_dir, 'results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def compute_energy_savings(baseline_energy: float, method_energy: float) -> float:
    """Compute percentage energy savings."""
    return (baseline_energy - method_energy) / baseline_energy * 100


def compute_performance_degradation(baseline_time: float, method_time: float) -> float:
    """Compute percentage performance degradation."""
    return (method_time - baseline_time) / baseline_time * 100


def perform_ttest(baseline_values: List[float], method_values: List[float]) -> tuple:
    """Perform paired t-test."""
    t_stat, p_value = stats.ttest_rel(baseline_values, method_values)
    return t_stat, p_value


def analyze_all_results(exp_base_dir: str = 'exp') -> Dict:
    """Analyze and compare all experiment results."""
    
    # Load all results
    schedulers = {
        'EEVDF': load_results(os.path.join(exp_base_dir, 'baseline_eevdf')),
        'SimpleEnergy': load_results(os.path.join(exp_base_dir, 'baseline_simple_energy')),
        'WattSched': load_results(os.path.join(exp_base_dir, 'wattsched')),
        'WattSched_NoClassify': load_results(os.path.join(exp_base_dir, 'wattsched_no_classify')),
        'WattSched_NoTopology': load_results(os.path.join(exp_base_dir, 'wattsched_no_topology')),
    }
    
    # Filter out missing results
    schedulers = {k: v for k, v in schedulers.items() if v is not None}
    
    if not schedulers:
        print("No results found!")
        return {}
    
    analysis = {
        'schedulers': list(schedulers.keys()),
        'comparisons': {},
        'ablation_contributions': {},
        'success_criteria': {}
    }
    
    # Find common experiments
    experiments = set()
    for scheduler_name, results in schedulers.items():
        if 'experiments' in results:
            for exp in results['experiments']:
                experiments.add(exp['experiment'])
    
    # Compare each method against EEVDF baseline
    baseline_name = 'EEVDF'
    if baseline_name in schedulers:
        baseline_results = schedulers[baseline_name]
        baseline_by_exp = {e['experiment']: e for e in baseline_results['experiments']}
        
        for scheduler_name in ['SimpleEnergy', 'WattSched']:
            if scheduler_name not in schedulers:
                continue
                
            method_results = schedulers[scheduler_name]
            method_by_exp = {e['experiment']: e for e in method_results['experiments']}
            
            for exp_name in experiments:
                if exp_name not in baseline_by_exp or exp_name not in method_by_exp:
                    continue
                
                baseline = baseline_by_exp[exp_name]
                method = method_by_exp[exp_name]
                
                energy_savings = compute_energy_savings(
                    baseline['energy_mean_joules'],
                    method['energy_mean_joules']
                )
                perf_degradation = compute_performance_degradation(
                    baseline['time_mean_seconds'],
                    method['time_mean_seconds']
                )
                
                # Extract raw values for t-test
                baseline_energies = [r['total_energy_joules'] for r in baseline['raw_results']]
                method_energies = [r['total_energy_joules'] for r in method['raw_results']]
                t_stat, p_value = perform_ttest(baseline_energies, method_energies)
                
                key = f"{scheduler_name}_vs_EEVDF_{exp_name}"
                analysis['comparisons'][key] = {
                    'experiment': exp_name,
                    'method': scheduler_name,
                    'baseline': baseline_name,
                    'energy_savings_percent': energy_savings,
                    'performance_degradation_percent': perf_degradation,
                    'ttest_t_stat': t_stat,
                    'ttest_p_value': p_value,
                    'statistically_significant': p_value < 0.05
                }
    
    # Ablation analysis: contribution of each component
    if 'WattSched' in schedulers:
        full_results = schedulers['WattSched']
        full_by_exp = {e['experiment']: e for e in full_results['experiments']}
        
        for ablation_name in ['WattSched_NoClassify', 'WattSched_NoTopology']:
            if ablation_name not in schedulers:
                continue
            
            ablation_results = schedulers[ablation_name]
            ablation_by_exp = {e['experiment']: e for e in ablation_results['experiments']}
            
            for exp_name in ['mixed_workload_small', 'mixed_workload_large']:
                if exp_name not in full_by_exp or exp_name not in ablation_by_exp:
                    continue
                
                full = full_by_exp[exp_name]
                ablation = ablation_by_exp[exp_name]
                
                contribution = compute_energy_savings(
                    ablation['energy_mean_joules'],
                    full['energy_mean_joules']
                )
                
                component = 'classification' if 'NoClassify' in ablation_name else 'topology'
                key = f"{component}_{exp_name}"
                analysis['ablation_contributions'][key] = {
                    'component': component,
                    'experiment': exp_name,
                    'contribution_percent': contribution,
                    'full_energy': full['energy_mean_joules'],
                    'ablation_energy': ablation['energy_mean_joules']
                }
    
    # Check success criteria
    if 'WattSched' in schedulers and 'EEVDF' in schedulers:
        wattsched = schedulers['WattSched']
        eevdf = schedulers['EEVDF']
        
        wattsched_by_exp = {e['experiment']: e for e in wattsched['experiments']}
        eevdf_by_exp = {e['experiment']: e for e in eevdf['experiments']}
        
        # Check mixed workload energy reduction >= 15%
        mixed_exp = 'mixed_workload_large'
        if mixed_exp in wattsched_by_exp and mixed_exp in eevdf_by_exp:
            energy_savings = compute_energy_savings(
                eevdf_by_exp[mixed_exp]['energy_mean_joules'],
                wattsched_by_exp[mixed_exp]['energy_mean_joules']
            )
            analysis['success_criteria']['energy_reduction_15pct'] = {
                'target': '>= 15%',
                'achieved': energy_savings,
                'passed': energy_savings >= 15
            }
        
        # Check performance degradation < 5%
        if mixed_exp in wattsched_by_exp and mixed_exp in eevdf_by_exp:
            perf_degradation = compute_performance_degradation(
                eevdf_by_exp[mixed_exp]['time_mean_seconds'],
                wattsched_by_exp[mixed_exp]['time_mean_seconds']
            )
            analysis['success_criteria']['performance_overhead_5pct'] = {
                'target': '< 5%',
                'achieved': perf_degradation,
                'passed': perf_degradation < 5
            }
    
    return analysis


def generate_figures(analysis: Dict, output_dir: str = 'figures'):
    """Generate publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Energy comparison across schedulers
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect data for mixed workload
    schedulers = ['EEVDF', 'SimpleEnergy', 'WattSched']
    energies = []
    errors = []
    
    exp_base = 'exp'
    for scheduler in schedulers:
        results = load_results(os.path.join(exp_base, scheduler.lower().replace(' ', '_')))
        if results:
            for exp in results['experiments']:
                if exp['experiment'] == 'mixed_workload_large':
                    energies.append(exp['energy_mean_joules'])
                    errors.append(exp['energy_std_joules'])
    
    if energies:
        x = range(len(schedulers))
        ax.bar(x, energies, yerr=errors, capsize=5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(schedulers)
        ax.set_ylabel('Energy (Joules)')
        ax.set_title('Energy Consumption: Mixed Workload (32 processes)')
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        if len(energies) >= 2:
            baseline = energies[0]
            for i, e in enumerate(energies[1:], 1):
                savings = (baseline - e) / baseline * 100
                ax.text(i, e + errors[i] + 5, f'{savings:.1f}%\nsavings', 
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure1_energy_comparison.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'figure1_energy_comparison.png'), dpi=300)
    plt.close()
    
    # Figure 2: Energy vs Performance tradeoff
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Map scheduler names to directory names
    scheduler_dirs = {
        'EEVDF': 'baseline_eevdf',
        'SimpleEnergy': 'baseline_simple_energy',
        'WattSched': 'wattsched'
    }
    
    scheduler_data = {}
    for scheduler in schedulers:
        dir_name = scheduler_dirs.get(scheduler, scheduler.lower())
        results = load_results(os.path.join(exp_base, dir_name))
        if results:
            for exp in results['experiments']:
                if exp['experiment'] == 'mixed_workload_large':
                    scheduler_data[scheduler] = {
                        'energy': exp['energy_mean_joules'],
                        'time': exp['time_mean_seconds'],
                        'energy_std': exp['energy_std_joules'],
                        'time_std': exp['time_std_seconds']
                    }
    
    if scheduler_data and 'EEVDF' in scheduler_data:
        # Normalize to EEVDF baseline
        baseline_energy = scheduler_data['EEVDF']['energy']
        baseline_time = scheduler_data['EEVDF']['time']
        
        for scheduler, data in scheduler_data.items():
            norm_energy = data['energy'] / baseline_energy * 100
            norm_time = data['time'] / baseline_time * 100
            ax.scatter(norm_time, norm_energy, s=200, alpha=0.7, label=scheduler)
        
        ax.axhline(100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(100, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Execution Time (normalized, %)')
        ax.set_ylabel('Energy Consumption (normalized, %)')
        ax.set_title('Energy-Performance Tradeoff (lower-left is better)')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure2_energy_performance_tradeoff.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'figure2_energy_performance_tradeoff.png'), dpi=300)
    plt.close()
    
    # Figure 3: Ablation study
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ablation_labels = []
    ablation_values = []
    
    for key, data in analysis.get('ablation_contributions', {}).items():
        if 'mixed_workload_large' in key:
            component = data['component'].capitalize()
            ablation_labels.append(component)
            ablation_values.append(data['contribution_percent'])
    
    if ablation_values:
        x = range(len(ablation_labels))
        bars = ax.bar(x, ablation_values, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_labels)
        ax.set_ylabel('Energy Savings Contribution (%)')
        ax.set_title('Ablation Study: Component Contributions (vs Full WattSched)')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, ablation_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure3_ablation_study.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'figure3_ablation_study.png'), dpi=300)
    plt.close()
    
    # Figure 4: Workload type breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy by workload type
    workload_types = ['cpu_only', 'memory_only', 'mixed_workload_small']
    workload_labels = ['CPU-bound', 'Memory-bound', 'Mixed']
    
    for ax, metric in zip(axes, ['energy', 'time']):
        data = {wt: {'EEVDF': [], 'WattSched': []} for wt in workload_types}
        
        for scheduler in ['EEVDF', 'WattSched']:
            dir_name = scheduler_dirs.get(scheduler, scheduler.lower())
            results = load_results(os.path.join(exp_base, dir_name))
            if results:
                for exp in results['experiments']:
                    if exp['experiment'] in workload_types:
                        if metric == 'energy':
                            data[exp['experiment']][scheduler] = exp['energy_mean_joules']
                        else:
                            data[exp['experiment']][scheduler] = exp['time_mean_seconds']
        
        x = np.arange(len(workload_labels))
        width = 0.35
        
        eevdf_vals = [data[wt]['EEVDF'] for wt in workload_types]
        wattsched_vals = [data[wt]['WattSched'] for wt in workload_types]
        
        ax.bar(x - width/2, eevdf_vals, width, label='EEVDF', alpha=0.8)
        ax.bar(x + width/2, wattsched_vals, width, label='WattSched', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(workload_labels)
        ax.set_ylabel('Energy (Joules)' if metric == 'energy' else 'Time (seconds)')
        ax.set_title(f'{"Energy" if metric == "energy" else "Execution Time"} by Workload Type')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure4_workload_breakdown.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'figure4_workload_breakdown.png'), dpi=300)
    plt.close()
    
    print(f"Figures saved to {output_dir}/")


def main():
    """Run analysis and generate figures."""
    print("Analyzing results...")
    analysis = analyze_all_results()
    
    # Save analysis (convert numpy types to native Python types)
    analysis_native = convert_to_native(analysis)
    with open('results/processed/analysis.json', 'w') as f:
        json.dump(analysis_native, f, indent=2)
    print("Analysis saved to results/processed/analysis.json")
    
    # Print summary
    print("\n" + "="*60)
    print("WATTsched EXPERIMENT SUMMARY")
    print("="*60)
    
    print("\n--- Success Criteria ---")
    for criterion, data in analysis.get('success_criteria', {}).items():
        status = "PASS" if data['passed'] else "FAIL"
        print(f"  {criterion}: {data['achieved']:.2f}% (target: {data['target']}) [{status}]")
    
    print("\n--- Energy Savings vs EEVDF ---")
    for key, data in analysis.get('comparisons', {}).items():
        if 'WattSched_vs_EEVDF' in key and 'mixed' in key:
            sig = "***" if data['statistically_significant'] else ""
            print(f"  {data['experiment']}:")
            print(f"    Energy savings: {data['energy_savings_percent']:.2f}% {sig}")
            print(f"    Performance overhead: {data['performance_degradation_percent']:.2f}%")
    
    print("\n--- Ablation Contributions ---")
    for key, data in analysis.get('ablation_contributions', {}).items():
        if 'mixed' in key:
            print(f"  {data['component'].capitalize()}: {data['contribution_percent']:.2f}% contribution")
    
    print("\n--- Statistical Significance ---")
    for key, data in analysis.get('comparisons', {}).items():
        if 'WattSched' in key and 'mixed' in key:
            print(f"  {data['method']} vs {data['baseline']} ({data['experiment']}):")
            print(f"    t-statistic: {data['ttest_t_stat']:.4f}")
            print(f"    p-value: {data['ttest_p_value']:.4f}")
    
    print("\n" + "="*60)
    
    # Generate figures
    print("\nGenerating figures...")
    generate_figures(analysis)
    
    return analysis


if __name__ == '__main__':
    os.makedirs('results/processed', exist_ok=True)
    main()
