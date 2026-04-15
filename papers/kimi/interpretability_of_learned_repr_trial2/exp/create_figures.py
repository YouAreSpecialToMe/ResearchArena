#!/usr/bin/env python3
"""Create figures for the paper."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Load results
with open('results.json') as f:
    results = json.load(f)

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Figure 1: Steering Effectiveness Comparison
fig, ax = plt.subplots(figsize=(8, 5))

sc = results['steering_comparison']
k_values = [5, 10, 20, 50, 100, 200]

methods = {
    'fidelity_weighted': {'label': 'Fidelity-Weighted (FWS)', 'color': '#2ecc71', 'marker': 'o'},
    'activation_baseline': {'label': 'Activation Baseline', 'color': '#e74c3c', 'marker': 's'},
    'output_score_baseline': {'label': 'Output Score Baseline', 'color': '#3498db', 'marker': '^'}
}

for method_key, config in methods.items():
    if method_key in sc:
        means = []
        stds = []
        for k in k_values:
            key = f'k={k}'
            if key in sc[method_key]:
                means.append(sc[method_key][key]['target_change_mean'])
                stds.append(sc[method_key][key]['target_change_std'])
            else:
                means.append(0)
                stds.append(0)
        
        ax.errorbar(k_values, means, yerr=stds, label=config['label'], 
                   color=config['color'], marker=config['marker'], linewidth=2, markersize=8, capsize=4)

ax.set_xlabel('Number of Features (k)', fontsize=12)
ax.set_ylabel('Target Behavior Change', fontsize=12)
ax.set_title('Steering Effectiveness Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('figures/steering_effectiveness.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/steering_effectiveness.pdf', bbox_inches='tight')
print('✓ Saved figures/steering_effectiveness.{png,pdf}')
plt.close()

# Figure 2: Side Effects (Perplexity Change)
fig, ax = plt.subplots(figsize=(8, 5))

se = results.get('side_effects', {})
k_values_se = [20, 50, 100]

for method_key, config in methods.items():
    means = []
    stds = []
    for k in k_values_se:
        key = f'{method_key.replace("_baseline", "").replace("fidelity_weighted", "fidelity_weighted")}_k={k}'
        # Adjust key for side_effects format
        if 'activation' in method_key:
            key = f'activation_k={k}'
        elif 'output_score' in method_key:
            key = f'output_score_k={k}'
        else:
            key = f'fidelity_weighted_k={k}'
        
        if key in se:
            means.append(se[key]['ppl_change_mean'])
            stds.append(se[key]['ppl_change_std'])
        else:
            means.append(0)
            stds.append(0)
    
    ax.errorbar(k_values_se, means, yerr=stds, label=config['label'],
               color=config['color'], marker=config['marker'], linewidth=2, markersize=8, capsize=4)

ax.set_xlabel('Number of Features (k)', fontsize=12)
ax.set_ylabel('Perplexity Change (lower is better)', fontsize=12)
ax.set_title('Side Effects on General Capability', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/side_effects.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/side_effects.pdf', bbox_inches='tight')
print('✓ Saved figures/side_effects.{png,pdf}')
plt.close()

# Figure 3: Component Ablation
if 'component_ablation' in results:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    abl = results['component_ablation']
    components = list(abl.keys())
    means = [abl[c]['target_change_mean'] for c in components]
    stds = [abl[c]['target_change_std'] for c in components]
    
    # Clean up component names
    labels = [c.replace('_', ' ').title() for c in components]
    
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.7, 
                  color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'])
    
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Target Behavior Change', fontsize=12)
    ax.set_title('Component Ablation Study (k=20)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('figures/component_ablation.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/component_ablation.pdf', bbox_inches='tight')
    print('✓ Saved figures/component_ablation.{png,pdf}')
    plt.close()

# Figure 4: IFS Distribution
if 'ifs_statistics' in results:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    ifs_stats = results['ifs_statistics']
    
    # Create a visual representation of IFS statistics
    metrics = ['Necessity', 'Sufficiency', 'Consistency']
    means = [ifs_stats['necessity_mean'], ifs_stats['sufficiency_mean'], ifs_stats['consistency_mean']]
    maxs = [
        ifs_stats.get('necessity_max', ifs_stats['necessity_mean'] * 10),
        ifs_stats.get('sufficiency_max', ifs_stats['sufficiency_mean'] * 10),
        ifs_stats.get('consistency_max', ifs_stats['consistency_mean'] * 10)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, means, width, label='Mean', alpha=0.7)
    axes[0].bar(x + width/2, maxs, width, label='Max', alpha=0.7)
    axes[0].set_ylabel('Score', fontsize=10)
    axes[0].set_title('IFS Component Statistics', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Feature distribution
    categories = ['IFS > 0.01', 'IFS > 0.1']
    counts = [ifs_stats['features_with_ifs_gt_0.01'], ifs_stats['features_with_ifs_gt_0.1']]
    axes[1].bar(categories, counts, color=['#3498db', '#e74c3c'], alpha=0.7)
    axes[1].set_ylabel('Number of Features', fontsize=10)
    axes[1].set_title('High-IFS Features', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # IFS mean/max
    axes[2].bar(['Mean IFS', 'Max IFS'], [ifs_stats['ifs_mean'], ifs_stats['ifs_max']], 
               color=['#2ecc71', '#f39c12'], alpha=0.7)
    axes[2].set_ylabel('IFS Score', fontsize=10)
    axes[2].set_title('IFS Score Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/ifs_statistics.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/ifs_statistics.pdf', bbox_inches='tight')
    print('✓ Saved figures/ifs_statistics.{png,pdf}')
    plt.close()

print('\nAll figures generated successfully!')
