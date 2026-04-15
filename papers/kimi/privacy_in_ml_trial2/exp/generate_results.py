#!/usr/bin/env python
"""Generate final results and visualizations."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_results(path):
    """Load results from JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def generate_privacy_utility_plot():
    """Generate privacy-utility frontier plot."""
    exp_dir = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01'
    
    # Load all results
    baseline = load_results(os.path.join(exp_dir, 'exp/baseline_unpruned/results.json'))
    magnitude = load_results(os.path.join(exp_dir, 'exp/magnitude_pruning/results.json'))
    hybrid = load_results(os.path.join(exp_dir, 'exp/hybrid_pruning/results.json'))
    g3p = load_results(os.path.join(exp_dir, 'exp/g3p/results.json'))
    taylor = load_results(os.path.join(exp_dir, 'exp/taylor_pruning/results.json'))
    mia = load_results(os.path.join(exp_dir, 'exp/mia_threshold/results.json'))
    
    if not all([baseline, mia]):
        print("Missing required results files")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Unpruned baseline
    baseline_acc = baseline['test_acc']['mean']
    baseline_auc = mia.get('unpruned', {}).get('mia_auc', {}).get('mean', 0.7)
    ax1.scatter(baseline_acc, baseline_auc, s=150, marker='*', 
               label='Unpruned', color='black', zorder=10)
    
    # Magnitude Pruning
    if magnitude and 'sparsity_levels' in magnitude:
        for sparsity in magnitude['sparsity_levels']:
            sp_key = f'sparsity_{int(sparsity*100)}'
            if sp_key in magnitude:
                acc = magnitude[sp_key]['test_acc']['mean']
                std = magnitude[sp_key]['test_acc']['std']
                auc = mia.get(f'magnitude_{int(sparsity*100)}', {}).get('mia_auc', {}).get('mean', 0)
                auc_std = mia.get(f'magnitude_{int(sparsity*100)}', {}).get('mia_auc', {}).get('std', 0)
                ax1.errorbar(acc, auc, xerr=std, yerr=auc_std, 
                           fmt='o', markersize=8, label=f'Magnitude ({int(sparsity*100)}%)', 
                           capsize=3)
    
    # Hybrid
    if hybrid and 'sparsity_levels' in hybrid:
        for sparsity in hybrid['sparsity_levels']:
            sp_key = f'sparsity_{int(sparsity*100)}'
            if sp_key in hybrid:
                acc = hybrid[sp_key]['test_acc']['mean']
                std = hybrid[sp_key]['test_acc']['std']
                auc = mia.get(f'hybrid_{int(sparsity*100)}', {}).get('mia_auc', {}).get('mean', 0)
                auc_std = mia.get(f'hybrid_{int(sparsity*100)}', {}).get('mia_auc', {}).get('std', 0)
                ax1.errorbar(acc, auc, xerr=std, yerr=auc_std, 
                           fmt='s', markersize=8, label=f'Hybrid ({int(sparsity*100)}%)', 
                           capsize=3)
    
    # G3P
    if g3p and 'sparsity_levels' in g3p:
        for sparsity in g3p['sparsity_levels']:
            sp_key = f'sparsity_{int(sparsity*100)}'
            if sp_key in g3p:
                acc = g3p[sp_key]['test_acc']['mean']
                std = g3p[sp_key]['test_acc']['std']
                auc = mia.get(f'g3p_{int(sparsity*100)}', {}).get('mia_auc', {}).get('mean', 0)
                auc_std = mia.get(f'g3p_{int(sparsity*100)}', {}).get('mia_auc', {}).get('std', 0)
                ax1.errorbar(acc, auc, xerr=std, yerr=auc_std, 
                           fmt='^', markersize=10, label=f'G3P ({int(sparsity*100)}%)', 
                           capsize=3, linewidth=2)
    
    ax1.set_xlabel('Test Accuracy (%)', fontsize=12)
    ax1.set_ylabel('MIA AUC (lower is better)', fontsize=12)
    ax1.set_title('Privacy-Utility Trade-off', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 0.8)
    
    # Sparsity vs MIA AUC
    methods = ['Magnitude', 'Hybrid', 'G3P']
    sparsities = [30, 50, 70]
    colors = ['C0', 'C1', 'C2']
    markers = ['o', 's', '^']
    
    for method, color, marker in zip(methods, colors, markers):
        aucs = []
        auc_stds = []
        for sparsity in sparsities:
            key = f'{method.lower()}_{sparsity}'
            auc = mia.get(key, {}).get('mia_auc', {}).get('mean', 0)
            auc_std = mia.get(key, {}).get('mia_auc', {}).get('std', 0)
            aucs.append(auc)
            auc_stds.append(auc_std)
        
        ax2.errorbar(sparsities, aucs, yerr=auc_stds, 
                    fmt=marker, markersize=8, label=method, 
                    color=color, capsize=3, linewidth=2)
        ax2.plot(sparsities, aucs, color=color, alpha=0.5, linewidth=1)
    
    # Add baseline
    ax2.axhline(y=baseline_auc, color='black', linestyle='--', 
               label='Unpruned', linewidth=2)
    
    ax2.set_xlabel('Sparsity (%)', fontsize=12)
    ax2.set_ylabel('MIA AUC (lower is better)', fontsize=12)
    ax2.set_title('Privacy Protection vs Sparsity', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 0.8)
    
    plt.tight_layout()
    
    figures_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'privacy_utility.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'privacy_utility.png'), bbox_inches='tight')
    print(f"Saved: {os.path.join(figures_dir, 'privacy_utility.pdf')}")
    plt.close()


def generate_accuracy_plot():
    """Generate accuracy vs sparsity plot."""
    exp_dir = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01'
    
    baseline = load_results(os.path.join(exp_dir, 'exp/baseline_unpruned/results.json'))
    magnitude = load_results(os.path.join(exp_dir, 'exp/magnitude_pruning/results.json'))
    hybrid = load_results(os.path.join(exp_dir, 'exp/hybrid_pruning/results.json'))
    g3p = load_results(os.path.join(exp_dir, 'exp/g3p/results.json'))
    
    if not baseline:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Baseline
    baseline_acc = baseline['test_acc']['mean']
    ax.axhline(y=baseline_acc, color='black', linestyle='--', 
              label='Unpruned', linewidth=2)
    
    # Methods
    methods_data = [
        ('Magnitude', magnitude, 'C0', 'o'),
        ('Hybrid', hybrid, 'C1', 's'),
        ('G3P', g3p, 'C2', '^')
    ]
    
    for name, results, color, marker in methods_data:
        if not results or 'sparsity_levels' not in results:
            continue
        
        sparsities = []
        accs = []
        acc_stds = []
        
        for sparsity in results['sparsity_levels']:
            sp_key = f'sparsity_{int(sparsity*100)}'
            if sp_key in results:
                sparsities.append(sparsity * 100)
                accs.append(results[sp_key]['test_acc']['mean'])
                acc_stds.append(results[sp_key]['test_acc']['std'])
        
        ax.errorbar(sparsities, accs, yerr=acc_stds, 
                   fmt=marker, markersize=10, label=name, 
                   color=color, capsize=3, linewidth=2)
        ax.plot(sparsities, accs, color=color, alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Utility Preservation vs Sparsity', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figures_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'accuracy_sparsity.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'accuracy_sparsity.png'), bbox_inches='tight')
    print(f"Saved: {os.path.join(figures_dir, 'accuracy_sparsity.pdf')}")
    plt.close()


def generate_summary_table():
    """Generate summary table of results."""
    exp_dir = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01'
    
    baseline = load_results(os.path.join(exp_dir, 'exp/baseline_unpruned/results.json'))
    magnitude = load_results(os.path.join(exp_dir, 'exp/magnitude_pruning/results.json'))
    hybrid = load_results(os.path.join(exp_dir, 'exp/hybrid_pruning/results.json'))
    g3p = load_results(os.path.join(exp_dir, 'exp/g3p/results.json'))
    mia = load_results(os.path.join(exp_dir, 'exp/mia_threshold/results.json'))
    
    if not all([baseline, mia]):
        print("Missing required results files")
        return None
    
    summary = {
        'experiment': 'G3P Summary',
        'unpruned': {
            'test_acc': baseline['test_acc'],
            'mia_auc': mia.get('unpruned', {}).get('mia_auc', {})
        }
    }
    
    # Add pruning methods
    for method, results in [('magnitude', magnitude), ('hybrid', hybrid), ('g3p', g3p)]:
        if not results:
            continue
        
        summary[method] = {}
        for sparsity in results.get('sparsity_levels', []):
            sp_key = f'sparsity_{int(sparsity*100)}'
            mia_key = f'{method}_{int(sparsity*100)}'
            
            if sp_key in results:
                summary[method][f'{int(sparsity*100)}%'] = {
                    'test_acc': results[sp_key]['test_acc'],
                    'mia_auc': mia.get(mia_key, {}).get('mia_auc', {})
                }
    
    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    
    # Compare G3P vs Magnitude at 50% sparsity
    if g3p and magnitude and mia:
        g3p_50 = mia.get('g3p_50', {}).get('mia_auc', {}).get('values', [])
        mag_50 = mia.get('magnitude_50', {}).get('mia_auc', {}).get('values', [])
        
        if len(g3p_50) == len(mag_50) and len(g3p_50) >= 3:
            t_stat, p_value = stats.ttest_rel(mag_50, g3p_50)
            print(f"\nG3P vs Magnitude (50% sparsity):")
            print(f"  G3P MIA AUC: {np.mean(g3p_50):.4f} ± {np.std(g3p_50):.4f}")
            print(f"  Magnitude MIA AUC: {np.mean(mag_50):.4f} ± {np.std(mag_50):.4f}")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"  ✓ Statistically significant (p < 0.05)")
            else:
                print(f"  ✗ Not statistically significant (p >= 0.05)")
        
        # Compare G3P vs Hybrid at 50% sparsity
        hybrid_50 = mia.get('hybrid_50', {}).get('mia_auc', {}).get('values', [])
        
        if len(g3p_50) == len(hybrid_50) and len(g3p_50) >= 3:
            t_stat, p_value = stats.ttest_rel(hybrid_50, g3p_50)
            print(f"\nG3P vs Hybrid (50% sparsity):")
            print(f"  G3P MIA AUC: {np.mean(g3p_50):.4f} ± {np.std(g3p_50):.4f}")
            print(f"  Hybrid MIA AUC: {np.mean(hybrid_50):.4f} ± {np.std(hybrid_50):.4f}")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"  ✓ Statistically significant (p < 0.05)")
            else:
                print(f"  ✗ Not statistically significant (p >= 0.05)")
    
    # Save summary
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to: {results_path}")
    
    return summary


def main():
    """Generate all results and visualizations."""
    print("="*70)
    print("GENERATING RESULTS AND VISUALIZATIONS")
    print("="*70)
    
    # Create figures directory
    figures_dir = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating privacy-utility plots...")
    generate_privacy_utility_plot()
    
    print("\nGenerating accuracy plots...")
    generate_accuracy_plot()
    
    print("\nGenerating summary table...")
    generate_summary_table()
    
    print("\n" + "="*70)
    print("RESULTS GENERATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
