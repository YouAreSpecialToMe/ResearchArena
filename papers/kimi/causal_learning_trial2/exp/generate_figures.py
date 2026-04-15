"""
Generate publication-quality figures for SPICED paper.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def load_results(filename):
    """Load results from JSON file."""
    for subdir in ['synthetic', 'ablations', 'real_world']:
        filepath = os.path.join(PROJECT_ROOT, f"results/{subdir}", filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    return None


def plot_sample_efficiency():
    """Figure 1: Sample efficiency comparison (SHD vs N)."""
    print("Generating Figure 1: Sample Efficiency...")
    
    spiced = load_results("spiced_knn_results.json")
    notears = load_results("notears_results.json")
    pc = load_results("pc_results.json")
    
    if not spiced or not notears:
        print("  Missing data, skipping...")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    mechanisms = ['linear_gaussian', 'linear_nongaussian']
    colors = {'SPICED': '#1f77b4', 'NOTEARS': '#d62728', 'PC': '#ff7f0e'}
    
    for idx, mechanism in enumerate(mechanisms):
        ax = axes[idx]
        
        for method_name, results in [('SPICED', spiced), ('NOTEARS', notears), ('PC', pc)]:
            if not results:
                continue
                
            # Group by sample size
            by_N = defaultdict(list)
            for r in results:
                if r['mechanism'] == mechanism and r.get('n_nodes') in [10, 20]:
                    by_N[r['n_samples']].append(r['shd'])
            
            N_values = sorted(by_N.keys())
            means = [np.mean(by_N[N]) for N in N_values]
            stds = [np.std(by_N[N]) for N in N_values]
            
            ax.errorbar(N_values, means, yerr=stds, label=method_name, 
                       marker='o', color=colors.get(method_name, 'gray'), capsize=3)
        
        ax.set_xlabel('Sample Size (N)', fontsize=11)
        ax.set_ylabel('Structural Hamming Distance (SHD)', fontsize=11)
        ax.set_title(f'{mechanism.replace("_", " ").title()}', fontsize=12)
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(f"{PROJECT_ROOT}/figures", exist_ok=True)
    plt.savefig(f"{PROJECT_ROOT}/figures/figure1_sample_efficiency.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{PROJECT_ROOT}/figures/figure1_sample_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved to figures/figure1_sample_efficiency.pdf")


def plot_scalability():
    """Figure 2: Scalability analysis (runtime vs n_nodes)."""
    print("Generating Figure 2: Scalability...")
    
    spiced = load_results("spiced_knn_results.json")
    notears = load_results("notears_results.json")
    pc = load_results("pc_results.json")
    
    if not spiced:
        print("  Missing data, skipping...")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    colors = {'SPICED': '#1f77b4', 'NOTEARS': '#d62728', 'PC': '#ff7f0e'}
    markers = {'SPICED': 'o', 'NOTEARS': 's', 'PC': '^'}
    
    for method_name, results in [('SPICED', spiced), ('NOTEARS', notears), ('PC', pc)]:
        if not results:
            continue
        
        # Group by graph size
        by_size = defaultdict(list)
        for r in results:
            by_size[r.get('n_nodes', 10)].append(r['runtime'])
        
        sizes = sorted(by_size.keys())
        means = [np.mean(by_size[s]) for s in sizes]
        stds = [np.std(by_size[s]) for s in sizes]
        
        ax.errorbar(sizes, means, yerr=stds, label=method_name,
                   marker=markers.get(method_name, 'o'), 
                   color=colors.get(method_name, 'gray'), capsize=3)
    
    ax.axhline(y=300, color='red', linestyle='--', alpha=0.5, label='5 min threshold')
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title('Scalability Analysis', fontsize=12)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT}/figures/figure2_scalability.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{PROJECT_ROOT}/figures/figure2_scalability.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved to figures/figure2_scalability.pdf")


def plot_ablations():
    """Figure 3: Ablation study results."""
    print("Generating Figure 3: Ablation Studies...")
    
    spiced_full = load_results("spiced_knn_results.json")
    spiced_no_constraints = load_results("spiced_no_constraints.json")
    spiced_no_init = load_results("spiced_no_it_init.json")
    spiced_kernel = load_results("spiced_kernel_mi.json")
    
    if not spiced_full:
        print("  Missing data, skipping...")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ablation 1: k-NN vs Kernel
    ax = axes[0]
    if spiced_kernel:
        knn_shd = [r['shd'] for r in spiced_full if 'shd' in r][:50]
        kernel_shd = [r['shd'] for r in spiced_kernel if 'shd' in r][:50]
        
        ax.bar(['k-NN MI', 'Kernel MI'], 
               [np.mean(knn_shd), np.mean(kernel_shd)],
               yerr=[np.std(knn_shd), np.std(kernel_shd)],
               capsize=5, color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel('SHD', fontsize=11)
        ax.set_title('Phase 1: MI Estimation Method', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Ablation 2: Structural constraints
    ax = axes[1]
    if spiced_no_constraints:
        full_shd = [r['shd'] for r in spiced_full if 'shd' in r][:50]
        no_constr_shd = [r['shd'] for r in spiced_no_constraints if 'shd' in r]
        
        ax.bar(['With Constraints', 'Without'], 
               [np.mean(full_shd), np.mean(no_constr_shd)],
               yerr=[np.std(full_shd), np.std(no_constr_shd)],
               capsize=5, color=['#1f77b4', '#d62728'])
        ax.set_ylabel('SHD', fontsize=11)
        ax.set_title('Phase 2: Structural Constraints', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Ablation 3: IT initialization
    ax = axes[2]
    if spiced_no_init:
        it_shd = [r['shd'] for r in spiced_full if 'shd' in r][:50]
        random_shd = [r['shd'] for r in spiced_no_init if 'shd' in r]
        
        ax.bar(['IT Init', 'Random Init'], 
               [np.mean(it_shd), np.mean(random_shd)],
               yerr=[np.std(it_shd), np.std(random_shd)],
               capsize=5, color=['#1f77b4', '#2ca02c'])
        ax.set_ylabel('SHD', fontsize=11)
        ax.set_title('Phase 3: Initialization Method', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT}/figures/figure3_ablations.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{PROJECT_ROOT}/figures/figure3_ablations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved to figures/figure3_ablations.pdf")


def plot_n50_results():
    """Figure 4: n=50 scalability results."""
    print("Generating Figure 4: n=50 Results...")
    
    spiced = load_results("spiced_knn_results.json")
    notears = load_results("notears_results.json")
    
    if not spiced:
        print("  Missing data, skipping...")
        return
    
    # Filter for n=50
    spiced_n50 = [r for r in spiced if r.get('n_nodes') == 50]
    notears_n50 = [r for r in notears if r.get('n_nodes') == 50] if notears else []
    
    if not spiced_n50:
        print("  No n=50 data, skipping...")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # SHD comparison
    ax = axes[0]
    spiced_shd = [r['shd'] for r in spiced_n50 if 'shd' in r]
    notears_shd = [r['shd'] for r in notears_n50 if 'shd' in r]
    
    if spiced_shd and notears_shd:
        ax.bar(['SPICED', 'NOTEARS'], 
               [np.mean(spiced_shd), np.mean(notears_shd)],
               yerr=[np.std(spiced_shd), np.std(notears_shd)],
               capsize=5, color=['#1f77b4', '#d62728'])
        ax.set_ylabel('SHD', fontsize=11)
        ax.set_title('Accuracy on n=50 Graphs', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Runtime comparison
    ax = axes[1]
    spiced_rt = [r['runtime'] for r in spiced_n50 if 'runtime' in r]
    notears_rt = [r['runtime'] for r in notears_n50 if 'runtime' in r]
    
    if spiced_rt:
        bars = ax.bar(['SPICED', 'NOTEARS'] if notears_rt else ['SPICED'], 
               [np.mean(spiced_rt), np.mean(notears_rt)] if notears_rt else [np.mean(spiced_rt)],
               yerr=[np.std(spiced_rt), np.std(notears_rt)] if notears_rt else [np.std(spiced_rt)],
               capsize=5, color=['#1f77b4', '#d62728'][:2 if notears_rt else 1])
        ax.axhline(y=300, color='red', linestyle='--', alpha=0.5, label='5 min threshold')
        ax.set_ylabel('Runtime (seconds)', fontsize=11)
        ax.set_title('Runtime on n=50 Graphs', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        median_rt = np.median(spiced_rt)
        ax.text(0.5, 0.9, f'Median: {median_rt:.1f}s ({median_rt/60:.1f} min)',
               transform=ax.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT}/figures/figure4_n50_results.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{PROJECT_ROOT}/figures/figure4_n50_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved to figures/figure4_n50_results.pdf")


def main():
    print("=" * 60)
    print("Generating Figures for SPICED Paper")
    print("=" * 60)
    
    plot_sample_efficiency()
    plot_scalability()
    plot_ablations()
    plot_n50_results()
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
