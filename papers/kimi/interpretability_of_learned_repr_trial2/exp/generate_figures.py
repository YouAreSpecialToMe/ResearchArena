"""
Generate figures for the paper.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)


def load_results():
    """Load experiment results."""
    with open('results/all_experiments.json', 'r') as f:
        return json.load(f)


def plot_curvature_semantics_correlation(results, output_dir='figures'):
    """Figure 2: Curvature-Semantics Correlation."""
    os.makedirs(output_dir, exist_ok=True)
    
    exp1_results = [r for r in results.get('experiment_1', []) if 'correlation' in r]
    
    if not exp1_results:
        print("No data for curvature-semantics plot")
        return
    
    # Group by model and dataset
    models = list(set(r['model'] for r in exp1_results))
    datasets = list(set(r['dataset'] for r in exp1_results))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, dataset in enumerate(['cifar10', 'cifar100']):
        ax = axes[idx]
        data = [r for r in exp1_results if r['dataset'] == dataset]
        
        if not data:
            continue
        
        correlations = [r['correlation'] for r in data]
        p_values = [r['p_value'] for r in data]
        
        # Box plot
        bp = ax.boxplot([correlations], labels=['All Models'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('Spearman Correlation')
        ax.set_title(f'{dataset.upper()}: Curvature-Semantics Correlation')
        ax.set_ylim(-0.5, 0.5)
        
        # Add significance indicators
        sig_count = sum(1 for p in p_values if p < 0.05)
        ax.text(0.5, 0.95, f'Significant: {sig_count}/{len(p_values)}', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_curvature_semantics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_curvature_semantics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to {output_dir}")


def plot_feature_comparison(results, output_dir='figures'):
    """Figure 3: Feature Comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    exp3_results = results.get('experiment_3', [])
    
    if not exp3_results:
        print("No data for feature comparison plot")
        return
    
    # Aggregate by dataset
    datasets = list(set(r['dataset'] for r in exp3_results))
    methods = ['linear_accuracy', 'sae_accuracy', 'curvature_accuracy']
    method_names = ['Linear (PCA)', 'SAE', 'Curvature']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        means = []
        stds = []
        for dataset in datasets:
            values = [r[method] for r in exp3_results if r['dataset'] == dataset]
            means.append(np.mean(values))
            stds.append(np.std(values) if len(values) > 1 else 0)
        
        ax.bar(x + i*width, means, width, yerr=stds, label=name, capsize=5)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Feature Comparison: Linear vs SAE vs Curvature')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_feature_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig3_feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to {output_dir}")


def plot_intervention_results(results, output_dir='figures'):
    """Figure 4: Intervention Results."""
    os.makedirs(output_dir, exist_ok=True)
    
    exp4_results = results.get('experiment_4', [])
    
    if not exp4_results:
        print("No data for intervention plot")
        return
    
    selectivities = [r['selectivity'] for r in exp4_results if 'selectivity' in r]
    
    if not selectivities:
        print("No selectivity data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.bar(['Curvature-Based\nIntervention'], [np.mean(selectivities)], 
           yerr=[np.std(selectivities)], capsize=10, color='steelblue')
    ax.axhline(y=0.7, color='r', linestyle='--', label='Target (0.7)')
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Random Baseline (0.5)')
    
    ax.set_ylabel('Selectivity Score')
    ax.set_title('Intervention Selectivity')
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add text annotation
    mean_sel = np.mean(selectivities)
    ax.text(0, mean_sel + 0.1, f'Mean: {mean_sel:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_intervention.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig4_intervention.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4 to {output_dir}")


def plot_ablation(results, output_dir='figures'):
    """Figure 5: Ablation Study."""
    os.makedirs(output_dir, exist_ok=True)
    
    exp5_results = results.get('experiment_5', [])
    
    if not exp5_results:
        print("No data for ablation plot")
        return
    
    # Average across seeds
    methods = ['full', 'pca_only', 'sff_only', 'random']
    method_names = ['Full Method', 'PCA Only', 'SFF Only', 'Random']
    
    means = []
    stds = []
    for method in methods:
        values = [r[method] for r in exp5_results if method in r]
        means.append(np.mean(values) if values else 0)
        stds.append(np.std(values) if len(values) > 1 else 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['green', 'blue', 'orange', 'gray'])
    
    ax.set_xlabel('Method Variant')
    ax.set_ylabel('Mean Curvature')
    ax.set_title('Ablation Study: Component Contributions')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig5_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 5 to {output_dir}")


def plot_scaling(results, output_dir='figures'):
    """Figure 6: Scaling Analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    exp6_results = results.get('experiment_6', {})
    
    if not exp6_results:
        print("No data for scaling plot")
        return
    
    sample_sizes = []
    times = []
    
    for key, value in exp6_results.items():
        if key.startswith('n'):
            n = int(key[1:])
            sample_sizes.append(n)
            times.append(value['time'])
    
    if not sample_sizes:
        print("No scaling data")
        return
    
    # Sort by sample size
    sorted_data = sorted(zip(sample_sizes, times))
    sample_sizes, times = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(sample_sizes, times, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Scaling: Time vs Sample Size')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig6_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 6 to {output_dir}")


def generate_summary_table(results, output_dir='results'):
    """Generate summary table of results."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary = []
    
    # Experiment 1 summary
    exp1 = results.get('experiment_1', [])
    if exp1:
        sig_count = sum(1 for r in exp1 if r.get('significant', False))
        summary.append(f"Experiment 1 (Curvature-Semantics): {sig_count}/{len(exp1)} significant correlations")
    
    # Experiment 2 summary
    exp2 = results.get('experiment_2', [])
    if exp2:
        sig_count = sum(1 for r in exp2 if r.get('significant', False))
        summary.append(f"Experiment 2 (Language Boundaries): {sig_count}/{len(exp2)} significant tests")
    
    # Experiment 3 summary
    exp3 = results.get('experiment_3', [])
    if exp3:
        avg_improvement = np.mean([r['improvement_over_linear'] for r in exp3])
        summary.append(f"Experiment 3 (Feature Comparison): Avg improvement {avg_improvement:.4f}")
    
    # Experiment 4 summary
    exp4 = results.get('experiment_4', [])
    if exp4:
        avg_selectivity = np.mean([r['selectivity'] for r in exp4 if 'selectivity' in r])
        summary.append(f"Experiment 4 (Intervention): Avg selectivity {avg_selectivity:.3f}")
    
    with open(f'{output_dir}/summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    print("\nSummary:")
    print('\n'.join(summary))


def main():
    print("Generating figures...")
    
    # Load results
    results = load_results()
    
    # Generate figures
    plot_curvature_semantics_correlation(results)
    plot_feature_comparison(results)
    plot_intervention_results(results)
    plot_ablation(results)
    plot_scaling(results)
    
    # Generate summary
    generate_summary_table(results)
    
    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
