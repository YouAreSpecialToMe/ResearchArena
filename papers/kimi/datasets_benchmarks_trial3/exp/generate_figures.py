"""
Generate publication-quality figures for the paper.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def set_plot_style():
    """Set consistent plot style."""
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['figure.dpi'] = 150


def plot_discriminative_power_over_time(results_dict, output_path):
    """Figure 2: Discriminative power over time for all methods."""
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Extract data for each method
    methods = {
        'Static': results_dict.get('baseline_static'),
        'Stratified': results_dict.get('baseline_stratified'),
        'Fluid': results_dict.get('baseline_fluid'),
        'DynaScale': results_dict.get('dynascale_full')
    }
    
    colors = {'Static': '#e74c3c', 'Stratified': '#f39c12', 
              'Fluid': '#3498db', 'DynaScale': '#2ecc71'}
    
    times = [0, 3, 6, 9, 12]
    
    # Plot 1: Kendall's tau
    ax = axes[0]
    for method_name, data in methods.items():
        if data and 'aggregated' in data:
            agg = data['aggregated']
            taus = [x['kendall_tau']['mean'] for x in agg]
            stds = [x['kendall_tau']['std'] for x in agg]
            ax.errorbar(times, taus, yerr=stds, label=method_name, 
                       marker='o', color=colors.get(method_name, 'gray'),
                       capsize=3, linewidth=2, markersize=6)
    
    ax.set_xlabel('Simulated Month')
    ax.set_ylabel("Kendall's τ")
    ax.set_title("Ranking Correlation Over Time")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.0])
    
    # Plot 2: Pairwise Accuracy
    ax = axes[1]
    for method_name, data in methods.items():
        if data and 'aggregated' in data:
            agg = data['aggregated']
            accs = [x['pairwise_accuracy']['mean'] for x in agg]
            stds = [x['pairwise_accuracy']['std'] for x in agg]
            ax.errorbar(times, accs, yerr=stds, label=method_name,
                       marker='s', color=colors.get(method_name, 'gray'),
                       capsize=3, linewidth=2, markersize=6)
    
    ax.set_xlabel('Simulated Month')
    ax.set_ylabel('Pairwise Accuracy')
    ax.set_title("Ranking Accuracy Over Time")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_fisher_information(results_dict, output_path):
    """Figure 3: Fisher information comparison."""
    set_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    methods = {
        'Static': results_dict.get('baseline_static'),
        'Stratified': results_dict.get('baseline_stratified'),
        'DynaScale': results_dict.get('dynascale_full')
    }
    
    colors = {'Static': '#e74c3c', 'Stratified': '#f39c12', 'DynaScale': '#2ecc71'}
    times = [0, 3, 6, 9, 12]
    
    for method_name, data in methods.items():
        if data and 'aggregated' in data:
            agg = data['aggregated']
            if 'expected_fisher_info' in agg[0]:
                fishers = [x['expected_fisher_info']['mean'] for x in agg]
                stds = [x['expected_fisher_info']['std'] for x in agg]
                ax.errorbar(times, fishers, yerr=stds, label=method_name,
                           marker='o', color=colors.get(method_name, 'gray'),
                           capsize=3, linewidth=2, markersize=6)
    
    ax.set_xlabel('Simulated Month')
    ax.set_ylabel('Expected Fisher Information')
    ax.set_title('Measurement Precision Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add theoretical maximum line
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Approx. Maximum')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_summary(results_dict, output_path):
    """Figure 4: Ablation study summary."""
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Ablation 1: Selection Method
    ax = axes[0]
    ab_sel = results_dict.get('ablation_selection', {}).get('summary', {})
    if ab_sel:
        methods = ['Wasserstein', 'Bin-Matching']
        means = [ab_sel.get('wasserstein_mean_tau', 0), ab_sel.get('bin_matching_mean_tau', 0)]
        stds = [ab_sel.get('wasserstein_std_tau', 0), ab_sel.get('bin_matching_std_tau', 0)]
        
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, capsize=5, color=['#2ecc71', '#95a5a6'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Kendall's τ")
        ax.set_title('Selection Method Comparison')
        ax.set_ylim([0.94, 0.96])
        ax.grid(True, alpha=0.3, axis='y')
    
    # Ablation 2: Difficulty Target
    ax = axes[1]
    ab_diff = results_dict.get('ablation_difficulty', {}).get('summary', {})
    if ab_diff:
        methods = ['Fisher-Optimal', 'Uniform']
        means = [ab_diff.get('fisher_optimal_mean_fisher', 0), ab_diff.get('uniform_mean_fisher', 0)]
        stds = [ab_diff.get('fisher_optimal_mean_acc', 0)*10, ab_diff.get('uniform_mean_acc', 0)*10]  # Scale for visibility
        
        x = np.arange(len(methods))
        ax.bar(x, means, capsize=5, color=['#2ecc71', '#95a5a6'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Expected Fisher Information')
        ax.set_title('Difficulty Target Comparison')
        ax.set_ylim([85, 95])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ability_difficulty_evolution(output_path):
    """Figure 5: Ability distribution evolution over time."""
    set_plot_style()
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for t in range(5):
        abilities = np.load(f'data/population/abilities_t{t}.npy')
        ax = axes[t]
        ax.hist(abilities, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_title(f'Month {t*3}')
        ax.set_xlabel('Ability (logits)')
        if t == 0:
            ax.set_ylabel('Count')
        ax.set_xlim([-3, 4])
        ax.set_ylim([0, 8])
    
    plt.suptitle('Model Ability Distribution Evolution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_main_results_table(results_dict, output_path):
    """Create a table figure with main results."""
    set_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Method', "Kendall's τ", 'Pairwise Acc.', 'Fisher Info.', 'Stability']
    
    rows = []
    for method_name, key in [('Static', 'baseline_static'),
                             ('Stratified', 'baseline_stratified'),
                             ('Fluid', 'baseline_fluid'),
                             ('DynaScale', 'dynascale_full')]:
        data = results_dict.get(key)
        if data and 'aggregated' in data:
            agg = data['aggregated']
            tau = f"{agg[-1]['kendall_tau']['mean']:.4f} ± {agg[-1]['kendall_tau']['std']:.4f}"
            acc = f"{agg[-1]['pairwise_accuracy']['mean']:.4f} ± {agg[-1]['pairwise_accuracy']['std']:.4f}"
            
            fisher = "-"
            if 'expected_fisher_info' in agg[-1]:
                fisher = f"{agg[-1]['expected_fisher_info']['mean']:.1f}"
            
            stability = "-"
            if 'summary' in data and 'mean_stability' in data['summary']:
                stability = f"{data['summary']['mean_stability']:.4f}"
            
            rows.append([method_name, tau, acc, fisher, stability])
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                    cellLoc='center', colColours=['#3498db']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Main Results Comparison (Final Time Point)', fontsize=14, pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures():
    """Generate all figures."""
    print("=" * 60)
    print("Generating Figures")
    print("=" * 60)
    
    os.makedirs('figures', exist_ok=True)
    
    # Load all results
    results = {
        'baseline_static': load_json('exp/baseline_static/results.json'),
        'baseline_stratified': load_json('exp/baseline_stratified/results.json'),
        'baseline_fluid': load_json('exp/baseline_fluid/results.json'),
        'dynascale_full': load_json('exp/dynascale_full/results.json'),
        'ablation_selection': load_json('exp/ablation_selection/results.json'),
        'ablation_difficulty': load_json('exp/ablation_difficulty/results.json'),
        'ablation_frequency': load_json('exp/ablation_frequency/results.json'),
    }
    
    # Generate figures
    plot_discriminative_power_over_time(results, 'figures/discriminative_power_over_time.pdf')
    plot_discriminative_power_over_time(results, 'figures/discriminative_power_over_time.png')
    
    plot_fisher_information(results, 'figures/fisher_information.pdf')
    plot_fisher_information(results, 'figures/fisher_information.png')
    
    plot_ablation_summary(results, 'figures/ablation_summary.pdf')
    plot_ablation_summary(results, 'figures/ablation_summary.png')
    
    plot_ability_difficulty_evolution('figures/ability_evolution.pdf')
    plot_ability_difficulty_evolution('figures/ability_evolution.png')
    
    plot_main_results_table(results, 'figures/main_results_table.pdf')
    plot_main_results_table(results, 'figures/main_results_table.png')
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_figures()
