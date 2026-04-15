#!/usr/bin/env python3
"""
Create figures from experimental results.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Load results
with open('results_final.json') as f:
    results = json.load(f)

# Figure 1: F1 Comparison by Method
fig, ax = plt.subplots(figsize=(10, 6))

methods = []
f1_means = []
f1_stds = []

# PC-FisherZ
if 'pc_fisherz' in results['baselines']:
    methods.append('PC-FisherZ')
    f1_means.append(results['baselines']['pc_fisherz']['summary']['f1']['mean'])
    f1_stds.append(results['baselines']['pc_fisherz']['summary']['f1']['std'])

# FastPC
if 'fast_pc' in results['baselines']:
    methods.append('Fast PC')
    f1_means.append(results['baselines']['fast_pc']['summary']['f1']['mean'])
    f1_stds.append(results['baselines']['fast_pc']['summary']['f1']['std'])

# MF-ACD
if 'main' in results['mf_acd']:
    methods.append('MF-ACD')
    f1_means.append(results['mf_acd']['main']['summary']['f1']['mean'])
    f1_stds.append(results['mf_acd']['main']['summary']['f1']['std'])

x = np.arange(len(methods))
ax.bar(x, f1_means, yerr=f1_stds, capsize=5, alpha=0.7, color=['blue', 'green', 'red'])
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Comparison Across Methods')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, max(f1_means) * 1.3)

# Add value labels
for i, (mean, std) in enumerate(zip(f1_means, f1_stds)):
    ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figures/f1_comparison.png', dpi=150)
plt.savefig('figures/f1_comparison.pdf')
plt.close()

print("Created figures/f1_comparison.png")

# Figure 2: Runtime Comparison
fig, ax = plt.subplots(figsize=(10, 6))

runtime_means = []
runtime_stds = []

if 'pc_fisherz' in results['baselines']:
    runtime_means.append(results['baselines']['pc_fisherz']['summary']['runtime']['mean'])
    runtime_stds.append(results['baselines']['pc_fisherz']['summary']['runtime']['std'])

if 'fast_pc' in results['baselines']:
    runtime_means.append(results['baselines']['fast_pc']['summary']['runtime']['mean'])
    runtime_stds.append(results['baselines']['fast_pc']['summary']['runtime']['std'])

if 'main' in results['mf_acd']:
    runtime_means.append(results['mf_acd']['main']['summary']['runtime']['mean'])
    runtime_stds.append(results['mf_acd']['main']['summary']['runtime']['std'])

x = np.arange(len(methods))
ax.bar(x, runtime_means, yerr=runtime_stds, capsize=5, alpha=0.7, color=['blue', 'green', 'red'])
ax.set_ylabel('Runtime (seconds)')
ax.set_title('Runtime Comparison Across Methods')
ax.set_xticks(x)
ax.set_xticklabels(methods)

# Add value labels
for i, (mean, std) in enumerate(zip(runtime_means, runtime_stds)):
    ax.text(i, mean + std + 0.1, f'{mean:.2f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figures/runtime_comparison.png', dpi=150)
plt.savefig('figures/runtime_comparison.pdf')
plt.close()

print("Created figures/runtime_comparison.png")

# Figure 3: Performance by Graph Size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# F1 by node count
node_counts = ['20 nodes', '50 nodes']
pc_f1_20 = results['baselines']['pc_fisherz']['by_nodes']['20_nodes']['f1']['mean']
pc_f1_50 = results['baselines']['pc_fisherz']['by_nodes']['50_nodes']['f1']['mean']
mf_f1_20 = results['mf_acd']['main']['by_nodes']['20_nodes']['f1']['mean']
mf_f1_50 = results['mf_acd']['main']['by_nodes']['50_nodes']['f1']['mean']

x = np.arange(len(node_counts))
width = 0.35

ax1.bar(x - width/2, [pc_f1_20, pc_f1_50], width, label='PC-FisherZ', alpha=0.7)
ax1.bar(x + width/2, [mf_f1_20, mf_f1_50], width, label='MF-ACD', alpha=0.7)
ax1.set_ylabel('F1 Score')
ax1.set_title('F1 Score by Graph Size')
ax1.set_xticks(x)
ax1.set_xticklabels(node_counts)
ax1.legend()
ax1.set_ylim(0, max(pc_f1_20, pc_f1_50, mf_f1_20, mf_f1_50) * 1.3)

# Runtime by node count
pc_time_20 = results['baselines']['pc_fisherz']['by_nodes']['20_nodes']['runtime']['mean']
pc_time_50 = results['baselines']['pc_fisherz']['by_nodes']['50_nodes']['runtime']['mean']
mf_time_20 = results['mf_acd']['main']['by_nodes']['20_nodes']['runtime']['mean']
mf_time_50 = results['mf_acd']['main']['by_nodes']['50_nodes']['runtime']['mean']

ax2.bar(x - width/2, [pc_time_20, pc_time_50], width, label='PC-FisherZ', alpha=0.7)
ax2.bar(x + width/2, [mf_time_20, mf_time_50], width, label='MF-ACD', alpha=0.7)
ax2.set_ylabel('Runtime (seconds)')
ax2.set_title('Runtime by Graph Size')
ax2.set_xticks(x)
ax2.set_xticklabels(node_counts)
ax2.legend()

plt.tight_layout()
plt.savefig('figures/performance_by_size.png', dpi=150)
plt.savefig('figures/performance_by_size.pdf')
plt.close()

print("Created figures/performance_by_size.png")

# Figure 4: Cost Savings Summary
if 'comparison' in results:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    savings = results['comparison']['cost_savings_pct']
    
    ax.bar(['Cost Savings'], [savings], color='green', alpha=0.7)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('MF-ACD Cost Savings vs PC-FisherZ')
    ax.set_ylim(0, 100)
    ax.text(0, savings + 2, f'{savings:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/cost_savings.png', dpi=150)
    plt.savefig('figures/cost_savings.pdf')
    plt.close()
    
    print("Created figures/cost_savings.png")

print("\nAll figures created in figures/")
