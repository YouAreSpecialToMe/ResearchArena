#!/usr/bin/env python3
"""
Generate figures for CellStratCP paper.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_results():
    """Load all experimental results."""
    with open('results.json') as f:
        return json.load(f)

def plot_coverage_comparison(results):
    """Figure 1: Coverage comparison between methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prepare data
    cellstratcp = [r for r in results.get('cellstratcp', []) if 'marginal_coverage' in r]
    standard_cp = [r for r in results.get('standard_cp', []) if 'marginal_coverage' in r]
    
    # Coverage by dataset
    datasets = sorted(set(r['dataset'] for r in cellstratcp))
    
    cs_cov = []
    sc_cov = []
    
    for ds in datasets:
        cs_vals = [r['marginal_coverage'] for r in cellstratcp if r['dataset'] == ds]
        sc_vals = [r['marginal_coverage'] for r in standard_cp if r['dataset'] == ds]
        cs_cov.append(np.mean(cs_vals) if cs_vals else 0)
        sc_cov.append(np.mean(sc_vals) if sc_vals else 0)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    axes[0].bar(x - width/2, sc_cov, width, label='Standard CP', alpha=0.8)
    axes[0].bar(x + width/2, cs_cov, width, label='CellStratCP', alpha=0.8)
    axes[0].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
    axes[0].set_ylabel('Marginal Coverage')
    axes[0].set_title('Coverage by Dataset')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=0, ha='center')
    axes[0].legend()
    axes[0].set_ylim([0.8, 1.0])
    
    # Coverage discrepancy
    cs_disc = [r['max_coverage_discrepancy'] for r in cellstratcp if 'max_coverage_discrepancy' in r]
    sc_disc = [r['max_coverage_discrepancy'] for r in standard_cp if 'max_coverage_discrepancy' in r]
    
    axes[1].boxplot([sc_disc, cs_disc], labels=['Standard CP', 'CellStratCP'])
    axes[1].set_ylabel('Max Coverage Discrepancy')
    axes[1].set_title('Coverage Discrepancy Distribution')
    
    plt.tight_layout()
    plt.savefig('figures/coverage_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/coverage_comparison.pdf', bbox_inches='tight')
    print("Saved figures/coverage_comparison.png")
    plt.close()

def plot_conditional_coverage(results):
    """Figure 2: Conditional coverage by cell type."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    cellstratcp = results.get('cellstratcp', [])
    
    # Get PBMC results
    pbmc_results = [r for r in cellstratcp if r['dataset'] == 'pbmc_processed']
    
    if pbmc_results:
        # Aggregate conditional coverage across seeds
        cell_types = ['B cell', 'Monocyte', 'NK']
        coverages = {ct: [] for ct in cell_types}
        
        for r in pbmc_results:
            cond_cov = r.get('conditional_coverage', {})
            for ct in cell_types:
                if ct in cond_cov:
                    coverages[ct].append(cond_cov[ct])
        
        means = [np.mean(coverages[ct]) if coverages[ct] else 0 for ct in cell_types]
        stds = [np.std(coverages[ct]) if coverages[ct] else 0 for ct in cell_types]
        
        axes[0].bar(cell_types, means, yerr=stds, capsize=5, alpha=0.8)
        axes[0].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
        axes[0].set_ylabel('Coverage')
        axes[0].set_title('CellStratCP: Conditional Coverage (PBMC)')
        axes[0].set_ylim([0.8, 1.0])
        axes[0].legend()
    
    # Synthetic datasets
    for idx, ds in enumerate(['synthetic_d30_s42', 'synthetic_d50_s42', 'synthetic_d70_s42']):
        ds_results = [r for r in cellstratcp if r['dataset'] == ds]
        
        if ds_results:
            cell_types = [f'CT_{i}' for i in range(6)]
            coverages = {ct: [] for ct in cell_types}
            
            for r in ds_results:
                cond_cov = r.get('conditional_coverage', {})
                for ct in cell_types:
                    if ct in cond_cov:
                        coverages[ct].append(cond_cov[ct])
            
            means = [np.mean(coverages[ct]) if coverages[ct] else 0 for ct in cell_types]
            stds = [np.std(coverages[ct]) if coverages[ct] else 0 for ct in cell_types]
            
            axes[idx+1].bar(range(6), means, yerr=stds, capsize=3, alpha=0.8)
            axes[idx+1].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
            axes[idx+1].set_ylabel('Coverage')
            axes[idx+1].set_title(f'Conditional Coverage ({ds.split("_")[1]} dropout)')
            axes[idx+1].set_xticks(range(6))
            axes[idx+1].set_xticklabels([f'CT{i}' for i in range(6)])
            axes[idx+1].set_ylim([0.7, 1.05])
            axes[idx+1].legend()
    
    plt.tight_layout()
    plt.savefig('figures/conditional_coverage.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/conditional_coverage.pdf', bbox_inches='tight')
    print("Saved figures/conditional_coverage.png")
    plt.close()

def plot_ablations(results):
    """Figure 3: Ablation study results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ablations = results.get('ablations', {})
    
    # No Mondrian vs Full
    no_mondrian = ablations.get('no_mondrian', [])
    cellstratcp = results.get('cellstratcp', [])
    
    if no_mondrian and cellstratcp:
        nm_disc = [r['max_coverage_discrepancy'] for r in no_mondrian if 'max_coverage_discrepancy' in r]
        cs_disc = [r['max_coverage_discrepancy'] for r in cellstratcp if 'max_coverage_discrepancy' in r]
        
        axes[0].boxplot([nm_disc, cs_disc], labels=['No Mondrian', 'CellStratCP'])
        axes[0].set_ylabel('Max Coverage Discrepancy')
        axes[0].set_title('Effect of Mondrian Stratification')
    
    # Residual scores vs ZINB
    residual = ablations.get('residual_scores', [])
    
    if residual and cellstratcp:
        res_width = [r['mean_interval_width'] for r in residual if 'mean_interval_width' in r]
        cs_width = [r['mean_interval_width'] for r in cellstratcp if 'mean_interval_width' in r]
        
        axes[1].boxplot([res_width, cs_width], labels=['Residual Scores', 'ZINB Scores'])
        axes[1].set_ylabel('Mean Interval Width')
        axes[1].set_title('ZINB vs Residual Scores')
    
    # Coverage comparison across methods
    methods_data = []
    methods_labels = []
    
    if standard_cp := results.get('standard_cp', []):
        methods_data.append([r['marginal_coverage'] for r in standard_cp if 'marginal_coverage' in r])
        methods_labels.append('Standard CP')
    
    if cellstratcp:
        methods_data.append([r['marginal_coverage'] for r in cellstratcp if 'marginal_coverage' in r])
        methods_labels.append('CellStratCP')
    
    if methods_data:
        axes[2].boxplot(methods_data, labels=methods_labels)
        axes[2].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
        axes[2].set_ylabel('Marginal Coverage')
        axes[2].set_title('Method Comparison')
        axes[2].legend()
        axes[2].set_ylim([0.8, 1.0])
    
    plt.tight_layout()
    plt.savefig('figures/ablation_studies.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/ablation_studies.pdf', bbox_inches='tight')
    print("Saved figures/ablation_studies.png")
    plt.close()

def plot_ood_detection(results):
    """Figure 4: OOD detection performance."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ood = results.get('ood_detection', [])
    
    if ood:
        aurocs = [r['auroc'] for r in ood if 'auroc' in r]
        fprs = [r['fpr_at_95_tpr'] for r in ood if 'fpr_at_95_tpr' in r]
        
        ax.scatter(fprs, aurocs, alpha=0.6, s=50)
        ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='AUROC = 0.85')
        ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='FPR@95TPR = 0.2')
        ax.set_xlabel('FPR at 95% TPR')
        ax.set_ylabel('AUROC')
        ax.set_title('OOD Detection Performance')
        ax.legend()
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0.5, 1.0])
        
        # Add stats text
        stats_text = f"Mean AUROC: {np.mean(aurocs):.3f}\nMean FPR@95TPR: {np.mean(fprs):.3f}"
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, 
                ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig('figures/ood_detection.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/ood_detection.pdf', bbox_inches='tight')
    print("Saved figures/ood_detection.png")
    plt.close()

def create_summary_table(results):
    """Create a summary table CSV."""
    rows = []
    
    for method_name, key in [('CellStratCP', 'cellstratcp'), ('Standard CP', 'standard_cp')]:
        data = results.get(key, [])
        if not data:
            continue
        
        covs = [r['marginal_coverage'] for r in data if 'marginal_coverage' in r]
        widths = [r['mean_interval_width'] for r in data if 'mean_interval_width' in r]
        discs = [r['max_coverage_discrepancy'] for r in data if 'max_coverage_discrepancy' in r]
        
        rows.append({
            'Method': method_name,
            'Coverage (mean)': f"{np.mean(covs):.3f}" if covs else 'N/A',
            'Coverage (std)': f"{np.std(covs):.3f}" if covs else 'N/A',
            'Width (mean)': f"{np.mean(widths):.3f}" if widths else 'N/A',
            'Width (std)': f"{np.std(widths):.3f}" if widths else 'N/A',
            'Discrepancy (mean)': f"{np.mean(discs):.3f}" if discs else 'N/A',
            'Discrepancy (std)': f"{np.std(discs):.3f}" if discs else 'N/A',
        })
    
    df = pd.DataFrame(rows)
    df.to_csv('figures/table1_main_results.csv', index=False)
    print("Saved figures/table1_main_results.csv")
    
    return df

def main():
    """Generate all figures."""
    print("="*60)
    print("GENERATING FIGURES FOR CELLSTRATCP")
    print("="*60)
    
    os.makedirs('figures', exist_ok=True)
    
    results = load_results()
    
    print(f"\nLoaded results:")
    print(f"  CellStratCP: {len(results.get('cellstratcp', []))} experiments")
    print(f"  Standard CP: {len(results.get('standard_cp', []))} experiments")
    print(f"  Ablations: {sum(len(v) for v in results.get('ablations', {}).values())} experiments")
    print(f"  OOD: {len(results.get('ood_detection', []))} experiments")
    
    print("\nGenerating figures...")
    plot_coverage_comparison(results)
    plot_conditional_coverage(results)
    plot_ablations(results)
    plot_ood_detection(results)
    
    print("\nGenerating summary table...")
    df = create_summary_table(results)
    print("\n" + str(df))
    
    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
