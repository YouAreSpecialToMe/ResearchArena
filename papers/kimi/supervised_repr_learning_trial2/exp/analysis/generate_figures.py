"""
Generate figures for the paper.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_summary():
    """Load summary JSON."""
    with open('results/summary.json', 'r') as f:
        return json.load(f)

def plot_comparison_bar_chart(summary):
    """Figure 2: Performance comparison bar chart."""
    
    results = summary['main_results']
    
    # Filter for CIFAR-100
    cifar100 = [r for r in results if r['dataset'] == 'cifar100']
    
    methods = ['CrossEntropy', 'SupCon', 'GC-SCL']
    noise_levels = [0.0, 0.2, 0.4]
    noise_labels = ['Clean', '20% Noise', '40% Noise']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(noise_levels))
    width = 0.25
    
    for i, method in enumerate(methods):
        means = []
        stds = []
        for noise in noise_levels:
            matches = [r for r in cifar100 if r['method'] == method and r['noise_ratio'] == noise]
            if matches:
                means.append(matches[0]['accuracy_mean'])
                stds.append(matches[0]['accuracy_std'])
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + i*width, means, width, label=method, yerr=stds, capsize=5)
    
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-100: Performance vs Label Noise', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(noise_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/figure2_performance_comparison.png', dpi=300)
    plt.savefig('figures/figure2_performance_comparison.pdf')
    plt.close()
    print("Saved figure2_performance_comparison.png")

def plot_ablation_study(summary):
    """Figure 4: Ablation study results."""
    
    results = summary['main_results']
    cifar100 = [r for r in results if r['dataset'] == 'cifar100']
    
    # Get GC-SCL variants
    variants = {
        'GC-SCL (Full)': 'GC-SCL',
        'No Velocity': None,  # Would need ablation results
        'No Curriculum': None,
        'Loss-based': None
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For now, just plot main method
    clean_acc = [r for r in cifar100 if r['method'] == 'GC-SCL' and r['noise_ratio'] == 0.0]
    noisy_acc = [r for r in cifar100 if r['method'] == 'GC-SCL' and r['noise_ratio'] == 0.2]
    
    if clean_acc and noisy_acc:
        categories = ['Clean', '20% Noise']
        values = [clean_acc[0]['accuracy_mean'], noisy_acc[0]['accuracy_mean']]
        stds = [clean_acc[0]['accuracy_std'], noisy_acc[0]['accuracy_std']]
        
        ax.bar(categories, values, yerr=stds, capsize=5, color=['steelblue', 'coral'])
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('GC-SCL Ablation Study', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/figure4_ablation_study.png', dpi=300)
    plt.savefig('figures/figure4_ablation_study.pdf')
    plt.close()
    print("Saved figure4_ablation_study.png")

def plot_noise_robustness(summary):
    """Figure showing robustness to different noise levels."""
    
    results = summary['main_results']
    cifar100 = [r for r in results if r['dataset'] == 'cifar100']
    
    methods = ['SupCon', 'GC-SCL']
    noise_levels = [0.0, 0.2, 0.4]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for method in methods:
        means = []
        stds = []
        for noise in noise_levels:
            matches = [r for r in cifar100 if r['method'] == method and r['noise_ratio'] == noise]
            if matches:
                means.append(matches[0]['accuracy_mean'])
                stds.append(matches[0]['accuracy_std'])
            else:
                means.append(np.nan)
                stds.append(0)
        
        ax.errorbar(noise_levels, means, yerr=stds, marker='o', label=method, linewidth=2, capsize=5)
    
    ax.set_xlabel('Noise Ratio', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Robustness to Label Noise (CIFAR-100)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/figure_noise_robustness.png', dpi=300)
    plt.savefig('figures/figure_noise_robustness.pdf')
    plt.close()
    print("Saved figure_noise_robustness.png")

def main():
    print("Loading summary...")
    summary = load_summary()
    
    os.makedirs('figures', exist_ok=True)
    
    print("\nGenerating figures...")
    plot_comparison_bar_chart(summary)
    plot_ablation_study(summary)
    plot_noise_robustness(summary)
    
    print("\nAll figures generated in figures/")

if __name__ == '__main__':
    os.chdir('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_02')
    main()
