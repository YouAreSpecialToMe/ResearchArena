"""Create publication-ready figures for CAGER results."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def load_json_safe(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None


def create_cgas_comparison_figure():
    """Create bar plot comparing C-GAS scores across methods and dimensionality."""
    results = load_json_safe('exp/synthetic/cgas/results_fixed.json')
    if not results:
        print("Synthetic results not found")
        return
    
    summary = results.get('summary', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['sae', 'pca', 'random']
    method_labels = ['SAE', 'PCA', 'Random']
    overcompletes = ['1x', '4x', '16x']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    x = np.arange(len(overcompletes))
    width = 0.25
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = []
        stds = []
        for oc in overcompletes:
            if method in summary and oc in summary[method]:
                means.append(summary[method][oc].get('cgas_mean', 0))
                stds.append(summary[method][oc].get('cgas_std', 0))
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, label=label, color=color, 
               yerr=stds, capsize=5, alpha=0.8)
    
    # Add horizontal line at 0.75 threshold
    ax.axhline(y=0.75, color='gray', linestyle='--', linewidth=1.5, 
               label='Threshold (0.75)', alpha=0.7)
    
    ax.set_xlabel('Dictionary Size', fontsize=12)
    ax.set_ylabel('C-GAS Score', fontsize=12)
    ax.set_title('C-GAS Comparison: SAE vs Baselines (Synthetic Task)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(overcompletes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('figures/cgas_comparison_fixed.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/cgas_comparison_fixed.png")
    plt.close()


def create_dimensionality_penalty_figure():
    """Visualize the effect of dimensionality penalty."""
    results = load_json_safe('exp/synthetic/cgas/results_fixed.json')
    if not results:
        return
    
    summary = results.get('summary', {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['sae', 'random']
    method_labels = ['SAE', 'Random']
    colors = ['#2ecc71', '#e74c3c']
    overcompletes = ['1x', '4x', '16x']
    
    # Plot 1: Penalized vs Unpenalized C-GAS
    x = np.arange(len(overcompletes))
    width = 0.35
    
    for method, label, color in zip(methods, method_labels, colors):
        pen_means = []
        unpen_means = []
        for oc in overcompletes:
            if method in summary and oc in summary[method]:
                pen_means.append(summary[method][oc].get('cgas_mean', 0))
                unpen_means.append(summary[method][oc].get('cgas_unpenalized_mean', 
                                                            summary[method][oc].get('cgas_mean', 0)))
        
        if pen_means and unpen_means:
            offset = 0 if method == 'sae' else width
            ax1.bar(x + offset - width/2, unpen_means, width/2, label=f'{label} (unpenalized)', 
                   color=color, alpha=0.4)
            ax1.bar(x + offset, pen_means, width/2, label=f'{label} (penalized)', 
                   color=color, alpha=0.9)
    
    ax1.set_xlabel('Dictionary Size', fontsize=11)
    ax1.set_ylabel('C-GAS Score', fontsize=11)
    ax1.set_title('Effect of Dimensionality Penalty', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(overcompletes)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: C-GAS vs Recovery Rate
    all_results = results.get('all_results', [])
    
    for method, label, color in zip(methods, method_labels, colors):
        cgas_vals = [r['cgas'] for r in all_results if r['method'] == method]
        recovery_vals = [r['recovery_rate'] for r in all_results if r['method'] == method]
        
        if cgas_vals and recovery_vals:
            ax2.scatter(cgas_vals, recovery_vals, label=label, color=color, s=50, alpha=0.6)
    
    # Add correlation line
    all_cgas = [r['cgas'] for r in all_results]
    all_recovery = [r['recovery_rate'] for r in all_results]
    if all_cgas and all_recovery:
        z = np.polyfit(all_cgas, all_recovery, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_cgas), max(all_cgas), 100)
        ax2.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1)
        
        # Compute correlation
        from scipy.stats import pearsonr
        r, pval = pearsonr(all_cgas, all_recovery)
        ax2.text(0.05, 0.95, f'r = {r:.3f}\np = {pval:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('C-GAS Score', fontsize=11)
    ax2.set_ylabel('Ground-Truth Recovery Rate', fontsize=11)
    ax2.set_title('C-GAS vs Ground-Truth Recovery', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/dimensionality_penalty_effect.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/dimensionality_penalty_effect.png")
    plt.close()


def create_validation_impact_figure():
    """Visualize impact of multi-method validation."""
    validation_results = load_json_safe('exp/synthetic/validation/results.json')
    if not validation_results:
        print("Validation results not found")
        return
    
    validated_atlas = validation_results.get('validated_atlas', {})
    
    # Count checks passed per feature
    check_counts = [v['checks_passed'] for v in validated_atlas.values()]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bar chart of checks passed
    features = list(validated_atlas.keys())
    colors = ['#e74c3c' if c < 2 else '#2ecc71' for c in check_counts]
    
    ax.barh(features, check_counts, color=colors, alpha=0.8)
    ax.axvline(x=2, color='black', linestyle='--', linewidth=2, label='Validation threshold (>=2)')
    
    ax.set_xlabel('Number of Checks Passed', fontsize=12)
    ax.set_ylabel('Ground-Truth Feature', fontsize=12)
    ax.set_title('Multi-Method Validation Results', fontsize=14)
    ax.set_xlim(0, 4)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Add validation status
    for i, (feat, checks) in enumerate(zip(features, check_counts)):
        status = 'VALIDATED' if checks >= 2 else 'REJECTED'
        color = 'green' if checks >= 2 else 'red'
        ax.text(checks + 0.1, i, status, va='center', fontsize=9, color=color)
    
    plt.tight_layout()
    plt.savefig('figures/validation_impact.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/validation_impact.png")
    plt.close()


def create_layerwise_heatmap():
    """Create heatmap of C-GAS across layers for IOI task."""
    ioi_results = load_json_safe('exp/ioi/cgas/results_fixed.json')
    if not ioi_results:
        print("IOI results not found, skipping heatmap")
        return
    
    layerwise = ioi_results.get('layerwise_results', {})
    if not layerwise:
        print("No layerwise results available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = sorted([int(k) for k in layerwise.keys()])
    cgas_values = [layerwise[str(l)].get('cgas', 0) for l in layers]
    
    # Create heatmap data
    data = np.array(cgas_values).reshape(1, -1)
    
    im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('C-GAS Score', rotation=270, labelpad=20)
    
    # Set ticks
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks([0])
    ax.set_yticklabels(['SAE 1x'])
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_title('Layer-wise C-GAS Analysis (IOI Task)', fontsize=14)
    
    # Add value annotations
    for i, (layer, cgas) in enumerate(zip(layers, cgas_values)):
        ax.text(i, 0, f'{cgas:.3f}', ha='center', va='center', 
               color='white' if cgas < 0.5 else 'black', fontsize=10)
    
    # Highlight known IOI layers
    for ioi_layer in [8, 9, 10]:
        if ioi_layer in layers:
            idx = layers.index(ioi_layer)
            ax.axvline(x=idx-0.5, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=idx+0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('figures/layerwise_heatmap.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/layerwise_heatmap.png")
    plt.close()


def create_summary_table():
    """Create summary table of results."""
    results = load_json_safe('results_final_fixed.json')
    if not results:
        return
    
    print("\n" + "="*80)
    print("SUMMARY TABLE: C-GAS Results (Fixed Version)")
    print("="*80)
    
    print(f"\n{'Method':<15} {'1x':<15} {'4x':<15} {'16x':<15}")
    print("-" * 60)
    
    for task, task_results in results.get('results_by_task', {}).items():
        if 'summary' not in task_results:
            continue
        
        print(f"\n{task.upper()} Task:")
        summary = task_results['summary']
        
        for method in ['sae', 'pca', 'random', 'oracle']:
            if method not in summary:
                continue
            
            row = [method.upper()]
            for oc in ['1x', '4x', '16x']:
                if oc in summary[method]:
                    mean = summary[method][oc].get('cgas_mean', 0)
                    std = summary[method][oc].get('cgas_std', 0)
                    row.append(f"{mean:.3f}±{std:.3f}")
                else:
                    row.append("-")
            
            print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
    
    print("\n" + "="*80)


def main():
    print("="*60)
    print("Creating Figures (FIXED Results)")
    print("="*60)
    
    os.makedirs('figures', exist_ok=True)
    
    print("\nCreating C-GAS comparison figure...")
    create_cgas_comparison_figure()
    
    print("\nCreating dimensionality penalty effect figure...")
    create_dimensionality_penalty_figure()
    
    print("\nCreating validation impact figure...")
    create_validation_impact_figure()
    
    print("\nCreating layer-wise heatmap...")
    create_layerwise_heatmap()
    
    print("\nCreating summary table...")
    create_summary_table()
    
    print("\n" + "="*60)
    print("Figure generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
