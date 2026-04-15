#!/usr/bin/env python3
"""
Generate figures for CDHR paper.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(pattern):
    """Load all result files matching pattern."""
    results = []
    results_dir = Path('results')
    for path in results_dir.rglob('*.json'):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                results.append(data)
        except:
            pass
    return results


def plot_accuracy_comparison(results, output_path='figures/accuracy_comparison.png'):
    """Plot accuracy comparison across methods and datasets."""
    # Group by dataset and method
    grouped = {}
    for r in results:
        dataset = os.path.basename(r['dataset']).replace('.json', '')
        method = r['method']
        key = (dataset, method)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r['metrics']['accuracy'])
    
    # Prepare data for plotting
    datasets = sorted(set(k[0] for k in grouped.keys()))
    methods = sorted(set(k[1] for k in grouped.keys()))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, method in enumerate(methods):
        accuracies = []
        errors = []
        for dataset in datasets:
            key = (dataset, method)
            if key in grouped:
                accs = grouped[key]
                accuracies.append(np.mean(accs))
                errors.append(np.std(accs) if len(accs) > 1 else 0)
            else:
                accuracies.append(0)
                errors.append(0)
        
        ax.bar(x + i*width, accuracies, width, label=method.upper(), yerr=errors)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison: CDHR vs Baseline')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_efficiency_tradeoff(results, output_path='figures/efficiency_tradeoff.png'):
    """Plot accuracy vs tokens (efficiency tradeoff)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for r in results:
        acc = r['metrics']['accuracy']
        tokens = r['metrics']['avg_tokens']
        method = r['method']
        dataset = os.path.basename(r['dataset']).replace('.json', '')
        
        marker = 'o' if method == 'cdhr' else 's'
        color = 'blue' if 'gsm8k' in dataset else 'orange' if 'math' in dataset else 'green'
        
        ax.scatter(tokens, acc, marker=marker, s=100, alpha=0.6, color=color,
                   label=f"{method.upper()}-{dataset}" if method == 'cdhr' else None)
    
    ax.set_xlabel('Average Tokens per Problem')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Efficiency Trade-off')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_strategy_distribution(results, output_path='figures/strategy_distribution.png'):
    """Plot strategy usage distribution for CDHR."""
    cdhr_results = [r for r in results if r['method'] == 'cdhr']
    
    if not cdhr_results:
        print("No CDHR results found for strategy distribution plot")
        return
    
    # Aggregate strategy distributions
    all_dist = {}
    for r in cdhr_results:
        if 'strategy_dist' in r['metrics']:
            for strategy, count in r['metrics']['strategy_dist'].items():
                all_dist[strategy] = all_dist.get(strategy, 0) + count
    
    if not all_dist:
        print("No strategy distribution data found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    strategies = list(all_dist.keys())
    counts = list(all_dist.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    ax.pie(counts, labels=strategies, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('CDHR Strategy Distribution')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")


def generate_summary_table(results, output_path='figures/summary_table.txt'):
    """Generate a text summary table."""
    lines = []
    lines.append("="*70)
    lines.append("CDHR EXPERIMENT RESULTS SUMMARY")
    lines.append("="*70)
    lines.append("")
    lines.append(f"{'Method':<15} {'Dataset':<15} {'Accuracy':<12} {'Tokens':<10} {'Latency':<10}")
    lines.append("-"*70)
    
    for r in results:
        method = r['method'].upper()
        dataset = os.path.basename(r['dataset']).replace('.json', '')[:14]
        m = r['metrics']
        acc = f"{m['accuracy']:.4f}"
        tokens = f"{m['avg_tokens']:.1f}"
        latency = f"{m['avg_latency']:.2f}s"
        lines.append(f"{method:<15} {dataset:<15} {acc:<12} {tokens:<10} {latency:<10}")
    
    lines.append("="*70)
    
    # Compute improvements
    cot_results = {r['dataset']: r for r in results if r['method'] == 'cot'}
    cdhr_results = {r['dataset']: r for r in results if r['method'] == 'cdhr'}
    
    lines.append("\nIMPROVEMENTS (CDHR vs CoT):")
    lines.append("-"*70)
    for dataset in cot_results:
        if dataset in cdhr_results:
            cot_acc = cot_results[dataset]['metrics']['accuracy']
            cdhr_acc = cdhr_results[dataset]['metrics']['accuracy']
            improvement = (cdhr_acc - cot_acc) * 100
            lines.append(f"{os.path.basename(dataset):<30} {improvement:+.2f}%")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved: {output_path}")
    print('\n'.join(lines))


def main():
    print("Loading results...")
    results = load_results('results/**/*.json')
    print(f"Found {len(results)} result files")
    
    if not results:
        print("No results found!")
        return
    
    print("\nGenerating figures...")
    plot_accuracy_comparison(results)
    plot_efficiency_tradeoff(results)
    plot_strategy_distribution(results)
    generate_summary_table(results)
    
    print("\nAll figures generated!")


if __name__ == '__main__':
    main()
