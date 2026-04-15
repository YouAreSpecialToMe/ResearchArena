"""
Generate publication-quality figures for the paper.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create figures directory
Path('figures').mkdir(exist_ok=True)

def load_results():
    with open('results.json', 'r') as f:
        return json.load(f)

def figure_1_robustness_comparison(results):
    """Figure 1: Robustness metrics comparison across methods."""
    if 'evaluations' not in results or 'robustness' not in results['evaluations']:
        print("Skipping Figure 1: No robustness evaluation data")
        return
    
    robustness = results['evaluations']['robustness']
    
    methods = ['TopK', 'JumpReLU', 'Denoising', 'RobustSAE']
    method_keys = ['topk', 'jumprelu', 'denoising', 'robust']
    
    # Extract metrics
    jaccard = [robustness[k]['population_attack']['jaccard_stability_mean'] for k in method_keys]
    jaccard_std = [robustness[k]['population_attack']['jaccard_stability_std'] for k in method_keys]
    
    rep_stab = [robustness[k]['population_attack']['representation_stability_mean'] for k in method_keys]
    rep_std = [robustness[k]['population_attack']['representation_stability_std'] for k in method_keys]
    
    asr = [robustness[k]['individual_attack']['attack_success_rate_mean'] * 100 for k in method_keys]
    asr_std = [robustness[k]['individual_attack']['attack_success_rate_std'] * 100 for k in method_keys]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Jaccard Stability
    x = np.arange(len(methods))
    axes[0].bar(x, jaccard, yerr=jaccard_std, capsize=5, alpha=0.8)
    axes[0].set_ylabel('Jaccard Stability', fontsize=12)
    axes[0].set_title('Population Attack Resistance', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0].set_ylim([0.8, 1.0])
    axes[0].axhline(y=jaccard[0], color='gray', linestyle='--', alpha=0.5, label='TopK baseline')
    axes[0].legend()
    
    # Representation Stability
    axes[1].bar(x, rep_stab, yerr=rep_std, capsize=5, alpha=0.8)
    axes[1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1].set_title('Representation Stability', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha='right')
    axes[1].set_ylim([0.9, 1.0])
    axes[1].axhline(y=rep_stab[0], color='gray', linestyle='--', alpha=0.5, label='TopK baseline')
    axes[1].legend()
    
    # Attack Success Rate (lower is better)
    axes[2].bar(x, asr, yerr=asr_std, capsize=5, alpha=0.8, color='coral')
    axes[2].set_ylabel('Attack Success Rate (%)', fontsize=12)
    axes[2].set_title('Individual Feature Attack', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15, ha='right')
    axes[2].axhline(y=asr[0], color='gray', linestyle='--', alpha=0.5, label='TopK baseline')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('figures/figure1_robustness_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/figure1_robustness_comparison.png', bbox_inches='tight', dpi=300)
    print("Figure 1 saved: figures/figure1_robustness_comparison.{pdf,png}")
    plt.close()

def figure_2_fvu_vs_robustness(results):
    """Figure 2: FVU vs Jaccard stability tradeoff."""
    if 'baselines' not in results or 'method' not in results:
        print("Skipping Figure 2: Missing data")
        return
    
    # Collect data
    methods_data = []
    
    # Baselines
    for baseline_name in ['topk_baseline', 'jumprelu_baseline', 'denoising_baseline']:
        if baseline_name in results['baselines']:
            fvu = results['baselines'][baseline_name]['fvu']['mean']
            methods_data.append(('TopK' if 'topk' in baseline_name else 
                               'JumpReLU' if 'jumprelu' in baseline_name else
                               'Denoising', fvu, 'baseline'))
    
    # RobustSAE
    if 'robustsae_full' in results['method']:
        fvu = results['method']['robustsae_full']['fvu']['mean']
        methods_data.append(('RobustSAE', fvu, 'method'))
    
    # Get robustness metrics
    if 'evaluations' in results and 'robustness' in results['evaluations']:
        robustness = results['evaluations']['robustness']
        jaccard_map = {
            'TopK': robustness['topk']['population_attack']['jaccard_stability_mean'],
            'JumpReLU': robustness['jumprelu']['population_attack']['jaccard_stability_mean'],
            'Denoising': robustness['denoising']['population_attack']['jaccard_stability_mean'],
            'RobustSAE': robustness['robust']['population_attack']['jaccard_stability_mean'],
        }
    else:
        jaccard_map = {}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'TopK': 'C0', 'JumpReLU': 'C1', 'Denoising': 'C2', 'RobustSAE': 'C3'}
    markers = {'baseline': 'o', 'method': '*'}
    
    for name, fvu, mtype in methods_data:
        if name in jaccard_map:
            jaccard = jaccard_map[name]
            ax.scatter(fvu, jaccard, s=300 if mtype == 'method' else 200, 
                      c=colors.get(name, 'gray'), marker=markers.get(mtype, 'o'),
                      label=name, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Fraction of Variance Unexplained (FVU)', fontsize=12)
    ax.set_ylabel('Jaccard Stability', fontsize=12)
    ax.set_title('Reconstruction-Robustness Tradeoff', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/figure2_fvu_vs_robustness.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/figure2_fvu_vs_robustness.png', bbox_inches='tight', dpi=300)
    print("Figure 2 saved: figures/figure2_fvu_vs_robustness.{pdf,png}")
    plt.close()

def figure_3_proxy_validation(results):
    """Figure 3: Proxy score vs empirical robustness."""
    if 'evaluations' not in results or 'proxy_validation' not in results['evaluations']:
        print("Skipping Figure 3: No proxy validation data")
        return
    
    proxy = results['evaluations']['proxy_validation']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create annotation text
    textstr = '\n'.join([
        'Proxy Validation Results',
        '',
        f"Spearman r = {proxy['spearman_r']:.4f} (p={proxy['spearman_p']:.4f})",
        f"Pearson r = {proxy['pearson_r']:.4f} (p={proxy['pearson_p']:.4f})",
        f"Kendall τ = {proxy['kendall_tau']:.4f} (p={proxy['kendall_p']:.4f})",
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Indicate success threshold
    ax.axvline(x=0.6, color='green', linestyle='--', alpha=0.5, label='Success threshold (r=0.6)')
    
    # Plot correlation values
    correlations = ['Spearman', 'Pearson', 'Kendall']
    values = [proxy['spearman_r'], proxy['pearson_r'], proxy['kendall_tau']]
    colors = ['C0', 'C1', 'C2']
    
    x_pos = np.arange(len(correlations))
    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(correlations)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Unsupervised Proxy Validation', fontsize=13, fontweight='bold')
    ax.set_ylim([0, max(values) * 1.2])
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/figure3_proxy_validation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/figure3_proxy_validation.png', bbox_inches='tight', dpi=300)
    print("Figure 3 saved: figures/figure3_proxy_validation.{pdf,png}")
    plt.close()

def figure_4_lambda_ablation(results):
    """Figure 4: Effect of consistency loss weight (lambda)."""
    if 'ablations' not in results or 'lambda_sweep' not in results['ablations']:
        print("Skipping Figure 4: No lambda ablation data")
        return
    
    lambda_data = results['ablations']['lambda_sweep']
    
    lambdas = []
    fvus = []
    dead_features = []
    
    for lambda_val in sorted([float(k) for k in lambda_data.keys()]):
        key = str(lambda_val)
        if key in lambda_data:
            lambdas.append(lambda_val)
            fvus.append(lambda_data[key]['fvu'])
            dead_features.append(lambda_data[key]['dead_features_pct'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # FVU vs lambda
    axes[0].plot(lambdas, fvus, 'o-', linewidth=2, markersize=8, color='C0')
    axes[0].set_xlabel('Consistency Loss Weight (λ)', fontsize=12)
    axes[0].set_ylabel('FVU', fontsize=12)
    axes[0].set_title('Reconstruction Quality vs λ', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Dead features vs lambda
    axes[1].plot(lambdas, dead_features, 's-', linewidth=2, markersize=8, color='C1')
    axes[1].set_xlabel('Consistency Loss Weight (λ)', fontsize=12)
    axes[1].set_ylabel('Dead Features (%)', fontsize=12)
    axes[1].set_title('Feature Death vs λ', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('figures/figure4_lambda_ablation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/figure4_lambda_ablation.png', bbox_inches='tight', dpi=300)
    print("Figure 4 saved: figures/figure4_lambda_ablation.{pdf,png}")
    plt.close()

def generate_summary_table(results):
    """Generate LaTeX and CSV tables."""
    
    # Main results table
    table_data = []
    
    # Header
    header = ['Method', 'FVU', 'L0 Sparsity', 'Dead Features (%)', 
              'Jaccard Stability', 'Attack Success Rate (%)']
    
    # TopK
    if 'topk_baseline' in results.get('baselines', {}):
        b = results['baselines']['topk_baseline']
        r = results['evaluations']['robustness']['topk']
        table_data.append([
            'TopK SAE',
            f"{b['fvu']['mean']:.4f} ± {b['fvu']['std']:.4f}",
            f"{b['l0_sparsity']['mean']:.1f}",
            f"{b['dead_features_pct']['mean']:.1f}",
            f"{r['population_attack']['jaccard_stability_mean']:.4f}",
            f"{r['individual_attack']['attack_success_rate_mean']*100:.2f}"
        ])
    
    # JumpReLU
    if 'jumprelu_baseline' in results.get('baselines', {}):
        b = results['baselines']['jumprelu_baseline']
        r = results['evaluations']['robustness']['jumprelu']
        table_data.append([
            'JumpReLU SAE',
            f"{b['fvu']['mean']:.4f} ± {b['fvu']['std']:.4f}",
            f"{b['l0_sparsity']['mean']:.1f}",
            f"{b['dead_features_pct']['mean']:.1f}",
            f"{r['population_attack']['jaccard_stability_mean']:.4f}",
            f"{r['individual_attack']['attack_success_rate_mean']*100:.2f}"
        ])
    
    # Denoising
    if 'denoising_baseline' in results.get('baselines', {}):
        b = results['baselines']['denoising_baseline']
        r = results['evaluations']['robustness']['denoising']
        table_data.append([
            'Denoising SAE',
            f"{b['fvu']['mean']:.4f} ± {b['fvu']['std']:.4f}",
            f"{b['l0_sparsity']['mean']:.1f}",
            f"{b['dead_features_pct']['mean']:.1f}",
            f"{r['population_attack']['jaccard_stability_mean']:.4f}",
            f"{r['individual_attack']['attack_success_rate_mean']*100:.2f}"
        ])
    
    # RobustSAE
    if 'robustsae_full' in results.get('method', {}):
        b = results['method']['robustsae_full']
        r = results['evaluations']['robustness']['robust']
        table_data.append([
            '\\textbf{RobustSAE (Ours)}',
            f"\\textbf{{{b['fvu']['mean']:.4f} ± {b['fvu']['std']:.4f}}}",
            f"{b['l0_sparsity']['mean']:.1f}",
            f"{b['dead_features_pct']['mean']:.1f}",
            f"\\textbf{{{r['population_attack']['jaccard_stability_mean']:.4f}}}",
            f"{r['individual_attack']['attack_success_rate_mean']*100:.2f}"
        ])
    
    # Save CSV
    import csv
    with open('figures/table_main_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table_data)
    print("Table saved: figures/table_main_results.csv")
    
    # Generate LaTeX
    latex = []
    latex.append('\\begin{table}[t]')
    latex.append('\\centering')
    latex.append('\\caption{Main Results: RobustSAE vs Baselines}')
    latex.append('\\label{tab:main_results}')
    latex.append('\\begin{tabular}{lccccc}')
    latex.append('\\toprule')
    latex.append(' & '.join(header) + ' \\\\\n')
    latex.append('\\midrule')
    for row in table_data:
        latex.append(' & '.join(row) + ' \\\\\n')
    latex.append('\\bottomrule')
    latex.append('\\end{tabular}')
    latex.append('\\end{table}')
    
    with open('figures/table_main_results.tex', 'w') as f:
        f.write('\n'.join(latex))
    print("Table saved: figures/table_main_results.tex")

def main():
    print("Loading results...")
    results = load_results()
    
    print("\nGenerating figures...")
    figure_1_robustness_comparison(results)
    figure_2_fvu_vs_robustness(results)
    figure_3_proxy_validation(results)
    figure_4_lambda_ablation(results)
    
    print("\nGenerating tables...")
    generate_summary_table(results)
    
    print("\nAll figures and tables generated successfully!")

if __name__ == '__main__':
    main()
