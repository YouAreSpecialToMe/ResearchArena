#!/usr/bin/env python3
"""Create visualization plots for the results."""
import sys
import os
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_accuracy_vs_mia():
    """Plot accuracy vs MIA AUC tradeoff."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Data for different methods
    methods = {
        'Magnitude': {
            'sparsity': [30, 50, 70],
            'acc': [94.31, 94.52, 93.47],
            'acc_err': [0, 0, 0.94],
            'mia': [0.582, 0.591, 0.584],
            'mia_err': [0.012, 0.008, 0.015]
        },
        'G3P': {
            'sparsity': [30, 50, 70],
            'acc': [87.75, 84.96, 80.79],
            'acc_err': [0.78, 0.46, 1.52],
            'mia': [0.527, 0.518, 0.516],
            'mia_err': [0.018, 0.015, 0.003]
        },
        'Hybrid': {
            'sparsity': [30, 50, 70],
            'acc': [94.27, 93.62, 94.43],
            'acc_err': [0.02, 0.87, 0.16],
            'mia': [0.537, 0.527, 0.536],
            'mia_err': [0.035, 0.037, 0.034]
        }
    }
    
    colors = {'Magnitude': 'blue', 'G3P': 'red', 'Hybrid': 'green'}
    markers = {'Magnitude': 'o', 'G3P': 's', 'Hybrid': '^'}
    
    # Plot 1: Accuracy vs Sparsity
    for method, data in methods.items():
        ax1.errorbar(data['sparsity'], data['acc'], yerr=data['acc_err'], 
                    label=method, color=colors[method], marker=markers[method],
                    markersize=8, capsize=5, linewidth=2)
    
    ax1.axhline(y=94.94, color='black', linestyle='--', label='Baseline (Unpruned)')
    ax1.set_xlabel('Sparsity (%)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy vs Sparsity', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([75, 97])
    
    # Plot 2: MIA AUC vs Sparsity
    for method, data in methods.items():
        ax2.errorbar(data['sparsity'], data['mia'], yerr=data['mia_err'],
                    label=method, color=colors[method], marker=markers[method],
                    markersize=8, capsize=5, linewidth=2)
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess (AUC=0.5)')
    ax2.set_xlabel('Sparsity (%)', fontsize=12)
    ax2.set_ylabel('MIA AUC', fontsize=12)
    ax2.set_title('Membership Inference Attack AUC vs Sparsity', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.48, 0.62])
    
    plt.tight_layout()
    plt.savefig('exp/figures/accuracy_vs_mia.png', dpi=300, bbox_inches='tight')
    print("Saved: exp/figures/accuracy_vs_mia.png")
    
    # Plot 3: Privacy-Accuracy Tradeoff
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for method, data in methods.items():
        ax.errorbar(data['mia'], data['acc'], xerr=data['mia_err'], yerr=data['acc_err'],
                   label=method, color=colors[method], marker=markers[method],
                   markersize=10, capsize=5, linewidth=2)
        # Add sparsity labels
        for i, s in enumerate(data['sparsity']):
            ax.annotate(f'{s}%', (data['mia'][i], data['acc'][i]), 
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.axhline(y=94.94, color='black', linestyle='--', alpha=0.5, label='Baseline Accuracy')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess')
    
    ax.set_xlabel('MIA AUC (Lower = Better Privacy)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Privacy-Accuracy Tradeoff', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exp/figures/privacy_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    print("Saved: exp/figures/privacy_accuracy_tradeoff.png")

def main():
    os.makedirs('exp/figures', exist_ok=True)
    plot_accuracy_vs_mia()
    print("\nAll plots created successfully!")

if __name__ == '__main__':
    main()
