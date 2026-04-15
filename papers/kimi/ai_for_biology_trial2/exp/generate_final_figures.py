#!/usr/bin/env python3
"""Generate final figures for CROSS-GRN experiments."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_results():
    """Load experiment results."""
    with open('results.json') as f:
        return json.load(f)


def plot_method_comparison(results, output_path='figures/figure_1_comparison.pdf'):
    """Figure 1: Bar plot comparing methods with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Methods to plot
    methods = ['correlation', 'cosine', 'random', 'crossgrn', 'scmultiomegrn', 'xatgrn']
    labels = ['Correlation', 'Cosine', 'Random', 'CROSS-GRN', 'scMultiomeGRN', 'XATGRN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    aurocs = []
    auroc_errs = []
    auprcs = []
    auprc_errs = []
    
    for method in methods:
        if method in results:
            data = results[method]
            auroc = data['metrics']['auroc']
            auprc = data['metrics']['auprc']
            
            if isinstance(auroc, dict):
                aurocs.append(auroc['mean'])
                auroc_errs.append(auroc['std'])
                auprcs.append(auprc['mean'])
                auprc_errs.append(auprc['std'])
            else:
                aurocs.append(auroc)
                auroc_errs.append(0)
                auprcs.append(auprc)
                auprc_errs.append(0)
    
    # AUROC plot
    x = np.arange(len(labels))
    axes[0].bar(x, aurocs, yerr=auroc_errs, color=colors, capsize=5, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    axes[0].set_title('GRN Inference Performance (AUROC)', fontsize=13, fontweight='bold')
    axes[0].legend()
    
    # AUPRC plot
    axes[1].bar(x, auprcs, yerr=auprc_errs, color=colors, capsize=5, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('AUPRC', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_title('GRN Inference Performance (AUPRC)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()


def plot_ablation_study(results, output_path='figures/figure_2_ablation.pdf'):
    """Figure 2: Ablation study results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ablations = ['crossgrn', 'ablation_no_celltype', 'ablation_rna_only']
    labels = ['CROSS-GRN\n(Full)', 'No Cell-Type\nConditioning', 'RNA Only\n(No ATAC)']
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    aurocs = []
    auroc_errs = []
    auprcs = []
    
    for abl in ablations:
        if abl in results:
            data = results[abl]
            auroc = data['metrics']['auroc']
            auprc = data['metrics']['auprc']
            
            if isinstance(auroc, dict):
                aurocs.append(auroc['mean'])
                auroc_errs.append(auroc['std'])
                auprcs.append(auprc['mean'])
            else:
                aurocs.append(auroc)
                auroc_errs.append(0)
                auprcs.append(auprc)
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars = ax.bar(x, aurocs, width, yerr=auroc_errs, color=colors, capsize=5, 
                   edgecolor='black', linewidth=1.2, label='AUROC')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, aurocs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Ablation Study: Impact of Model Components', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()


def plot_sign_prediction(results, output_path='figures/figure_3_sign_prediction.pdf'):
    """Figure 3: Sign prediction accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['correlation', 'cosine', 'crossgrn', 'scmultiomegrn', 'xatgrn']
    labels = ['Correlation', 'Cosine', 'CROSS-GRN', 'scMultiomeGRN', 'XATGRN']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    
    sign_accs = []
    sign_acc_errs = []
    
    for method in methods:
        if method in results:
            data = results[method]
            sign_acc = data['metrics']['sign_accuracy']
            
            if isinstance(sign_acc, dict):
                sign_accs.append(sign_acc['mean'])
                sign_acc_errs.append(sign_acc['std'])
            else:
                sign_accs.append(sign_acc)
                sign_acc_errs.append(0)
    
    x = np.arange(len(labels))
    ax.bar(x, sign_accs, yerr=sign_acc_errs, color=colors, capsize=5, 
           edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Sign Prediction Accuracy', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target (70%)')
    ax.set_title('Sign Prediction Accuracy (Activation/Repression)', fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()


def create_summary_table(results, output_path='figures/table_1_results.md'):
    """Create a markdown table with results."""
    
    table = """# CROSS-GRN Experimental Results

## Table 1: Main Results Comparison

| Method | AUROC | AUPRC | Sign Accuracy | n_seeds |
|--------|-------|-------|---------------|---------|
"""
    
    methods = [
        ('random', 'Random'),
        ('correlation', 'Correlation'),
        ('cosine', 'Cosine Similarity'),
        ('scmultiomegrn', 'scMultiomeGRN'),
        ('xatgrn', 'XATGRN'),
        ('crossgrn', '**CROSS-GRN**'),
    ]
    
    for key, name in methods:
        if key in results:
            data = results[key]
            metrics = data['metrics']
            
            auroc = metrics['auroc']
            auprc = metrics['auprc']
            sign_acc = metrics['sign_accuracy']
            n_seeds = len(data.get('seeds', [1]))
            
            if isinstance(auroc, dict):
                auroc_str = f"{auroc['mean']:.4f} ± {auroc['std']:.4f}"
                auprc_str = f"{auprc['mean']:.4f} ± {auprc['std']:.4f}"
                sign_str = f"{sign_acc['mean']:.4f} ± {sign_acc['std']:.4f}"
            else:
                auroc_str = f"{auroc:.4f}"
                auprc_str = f"{auprc:.4f}"
                sign_str = f"{sign_acc:.4f}"
            
            table += f"| {name} | {auroc_str} | {auprc_str} | {sign_str} | {n_seeds} |\n"
    
    table += """
## Table 2: Ablation Study Results

| Variant | AUROC | AUPRC | Sign Accuracy |
|---------|-------|-------|---------------|
"""
    
    ablations = [
        ('crossgrn', 'CROSS-GRN (Full)'),
        ('ablation_no_celltype', 'No Cell-Type Conditioning'),
        ('ablation_rna_only', 'RNA Only (No ATAC)'),
    ]
    
    for key, name in ablations:
        if key in results:
            data = results[key]
            metrics = data['metrics']
            
            auroc = metrics['auroc']
            auprc = metrics['auprc']
            sign_acc = metrics['sign_accuracy']
            
            if isinstance(auroc, dict):
                auroc_str = f"{auroc['mean']:.4f} ± {auroc['std']:.4f}"
                auprc_str = f"{auprc['mean']:.4f} ± {auprc['std']:.4f}"
                sign_str = f"{sign_acc['mean']:.4f} ± {sign_acc['std']:.4f}"
            else:
                auroc_str = f"{auroc:.4f}"
                auprc_str = f"{auprc:.4f}"
                sign_str = f"{sign_acc:.4f}"
            
            table += f"| {name} | {auroc_str} | {auprc_str} | {sign_str} |\n"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"Saved table to {output_path}")


def main():
    print("Generating figures...")
    
    results = load_results()
    
    os.makedirs('figures', exist_ok=True)
    
    # Generate figures
    plot_method_comparison(results)
    plot_ablation_study(results)
    plot_sign_prediction(results)
    create_summary_table(results)
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
