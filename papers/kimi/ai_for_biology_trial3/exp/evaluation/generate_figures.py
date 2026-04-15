"""
Generate publication-quality figures for the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


def load_results(path='results/all_experiments.json'):
    """Load experiment results."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_property_prediction_comparison(results, output_path='figures/figure2_property_prediction.pdf'):
    """Figure 2: Property prediction R² comparison."""
    
    models = []
    r2_means = []
    r2_stds = []
    
    for name, data in results.items():
        if 'metrics' in data:
            models.append(name.replace('_', ' ').title())
            r2_means.append(data['metrics']['test_r2']['mean'])
            r2_stds.append(data['metrics']['test_r2']['std'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(models))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    # Add DMPNN baseline reference line (from Liu et al. 2025)
    ax.axhline(y=0.65, color='red', linestyle='--', linewidth=2, label='DMPNN Baseline (Liu et al. 2025)')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Permeability Prediction Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(r2_means, r2_stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_study(results, output_path='figures/figure4_ablation.pdf'):
    """Figure 4: Ablation study results."""
    
    # Extract full model and ablations
    full_r2 = results.get('strucvae_full', {}).get('metrics', {}).get('test_r2', {}).get('mean', 0)
    
    ablations = {
        'Full Model': full_r2,
        'No Structure': results.get('ablation_no_structure', {}).get('metrics', {}).get('test_r2', {}).get('mean', 0),
        'Late Fusion': results.get('ablation_late_fusion', {}).get('metrics', {}).get('test_r2', {}).get('mean', 0),
        'No Disentanglement': results.get('ablation_no_disentangle', {}).get('metrics', {}).get('test_r2', {}).get('mean', 0),
    }
    
    # Compute relative drop
    baseline = full_r2
    drops = {k: (baseline - v) * 100 for k, v in ablations.items()}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Absolute performance
    names = list(ablations.keys())
    values = list(ablations.values())
    colors = ['#2ca02c' if v == full_r2 else '#d62728' for v in values]
    
    ax1.barh(names, values, color=colors)
    ax1.set_xlabel('R² Score', fontsize=12)
    ax1.set_title('Absolute Performance', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 1.0])
    
    # Add value labels
    for i, (name, val) in enumerate(zip(names, values)):
        ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    
    # Performance drop
    drop_names = [k for k in drops.keys() if k != 'Full Model']
    drop_values = [drops[k] for k in drop_names]
    
    ax2.barh(drop_names, drop_values, color='#d62728')
    ax2.set_xlabel('Performance Drop (R² %)', fontsize=12)
    ax2.set_title('Relative to Full Model', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for i, (name, val) in enumerate(zip(drop_names, drop_values)):
        ax2.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_curves(results, output_path='figures/training_curves.pdf'):
    """Plot training curves for each model."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        if idx >= 6:
            break
        
        ax = axes[idx]
        
        # Plot first seed's training history
        if 'seeds' in data and len(data['seeds']) > 0:
            seed_data = data['seeds'][0]
            if 'history' in seed_data:
                history = seed_data['history']
                epochs = [h['epoch'] for h in history]
                train_r2 = [h.get('train_r2', 0) for h in history]
                val_r2 = [h.get('val_r2', 0) for h in history]
                
                ax.plot(epochs, train_r2, label='Train R²', alpha=0.7)
                ax.plot(epochs, val_r2, label='Val R²', alpha=0.7)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('R² Score')
                ax.set_title(name.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_results_table(results, output_path='figures/table1_main_results.csv'):
    """Generate main results table."""
    
    rows = []
    for name, data in results.items():
        if 'metrics' in data:
            metrics = data['metrics']
            rows.append({
                'Model': name.replace('_', ' ').title(),
                'R²': f"{metrics['test_r2']['mean']:.4f} ± {metrics['test_r2']['std']:.4f}",
                'MAE': f"{metrics['test_mae']['mean']:.4f} ± {metrics['test_mae']['std']:.4f}",
                'Pearson r': f"{metrics['test_pearson']['mean']:.4f} ± {metrics['test_pearson']['std']:.4f}"
            })
    
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print("\nMain Results Table:")
    print(df.to_string(index=False))


def main():
    print("Generating figures...")
    
    if not os.path.exists('results/all_experiments.json'):
        print("Error: results/all_experiments.json not found")
        return
    
    results = load_results()
    
    # Generate figures
    plot_property_prediction_comparison(results)
    plot_ablation_study(results)
    plot_training_curves(results)
    generate_results_table(results)
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
