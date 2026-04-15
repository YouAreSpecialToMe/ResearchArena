#!/usr/bin/env python3
"""
Generate figures for the VAST paper.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_results(results_file: str) -> dict:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_speedup_comparison(results: dict, output_path: str):
    """Plot speedup comparison bar chart."""
    aggregated = results.get('aggregated', results)
    
    methods = []
    speedups = []
    errors = []
    colors = []
    
    method_order = [
        ('baseline_25step', 'Baseline 25-step', '#95a5a6'),
        ('baseline_17step', 'Baseline 17-step', '#7f8c8d'),
        ('vast_2x', 'VAST 2×', '#3498db'),
        ('vast_3x', 'VAST 3×', '#e74c3c'),
    ]
    
    for key, label, color in method_order:
        if key in aggregated:
            methods.append(label)
            if 'speedup_mean' in aggregated[key]:
                speedups.append(aggregated[key]['speedup_mean'])
                errors.append(aggregated[key].get('speedup_std', 0))
            elif 'speedup' in aggregated[key] and 'mean' in aggregated[key]['speedup']:
                speedups.append(aggregated[key]['speedup']['mean'])
                errors.append(aggregated[key]['speedup'].get('std', 0))
            else:
                speedups.append(0)
                errors.append(0)
            colors.append(color)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, speedups, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=14, fontweight='bold')
    ax.set_title('Speedup Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='2× Target')
    ax.axhline(y=3.0, color='gray', linestyle=':', alpha=0.5, label='3× Target')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved speedup comparison to {output_path}")


def plot_nfe_reduction(results: dict, output_path: str):
    """Plot NFE reduction bar chart."""
    aggregated = results.get('aggregated', results)
    
    methods = []
    nfe_means = []
    nfe_stds = []
    colors = []
    
    method_order = [
        ('baseline_50step', 'Baseline\n50-step', '#2c3e50'),
        ('baseline_25step', 'Baseline\n25-step', '#95a5a6'),
        ('vast_2x', 'VAST\n2×', '#3498db'),
        ('vast_3x', 'VAST\n3×', '#e74c3c'),
    ]
    
    for key, label, color in method_order:
        if key in aggregated:
            methods.append(label)
            if 'nfe' in aggregated[key] and isinstance(aggregated[key]['nfe'], dict):
                nfe_means.append(aggregated[key]['nfe']['mean'])
                nfe_stds.append(aggregated[key]['nfe'].get('std', 0))
            elif 'nfe_per_image' in aggregated[key] and isinstance(aggregated[key]['nfe_per_image'], dict):
                nfe_means.append(aggregated[key]['nfe_per_image']['mean'])
                nfe_stds.append(aggregated[key]['nfe_per_image'].get('std', 0))
            else:
                nfe_means.append(aggregated[key].get('nfe', 0))
                nfe_stds.append(0)
            colors.append(color)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, nfe_means, yerr=nfe_stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('NFE per Image', fontsize=14, fontweight='bold')
    ax.set_title('Number of Function Evaluations', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved NFE reduction plot to {output_path}")


def plot_pareto_frontier(results: dict, output_path: str):
    """Plot Pareto frontier of time vs quality."""
    aggregated = results.get('aggregated', results)
    
    methods_data = []
    
    for key, label, color, marker in [
        ('baseline_50step', 'Baseline 50-step', '#2c3e50', 'o'),
        ('baseline_25step', 'Baseline 25-step', '#95a5a6', 's'),
        ('baseline_17step', 'Baseline 17-step', '#7f8c8d', '^'),
        ('deepcache', 'DeepCache', '#9b59b6', 'D'),
        ('ras_2x', 'RAS 2×', '#f39c12', 'v'),
        ('vast_2x', 'VAST 2×', '#3498db', 'p'),
        ('vast_3x', 'VAST 3×', '#e74c3c', '*'),
    ]:
        if key not in aggregated:
            continue
        
        data = aggregated[key]
        
        # Get wall time
        if 'wall_time_per_image' in data and isinstance(data['wall_time_per_image'], dict):
            time_mean = data['wall_time_per_image']['mean']
            time_std = data['wall_time_per_image'].get('std', 0)
        else:
            time_mean = data.get('wall_time_mean', 0)
            time_std = data.get('wall_time_std', 0)
        
        # Get FID if available
        fid = None
        if 'fid' in data:
            fid = data['fid'].get('mean', data['fid']) if isinstance(data['fid'], dict) else data['fid']
        
        methods_data.append({
            'label': label,
            'time': time_mean,
            'time_std': time_std,
            'fid': fid,
            'color': color,
            'marker': marker,
        })
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for method in methods_data:
        ax.scatter(
            method['time'], 
            method.get('fid', 0) if method.get('fid') is not None else 0,
            s=200,
            c=method['color'],
            marker=method['marker'],
            edgecolors='black',
            linewidths=1.5,
            label=method['label'],
            alpha=0.8,
        )
    
    ax.set_xlabel('Time per Image (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('FID (lower is better)', fontsize=14, fontweight='bold')
    ax.set_title('Speed-Quality Trade-off', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Invert y-axis if FID values are present
    if any(m.get('fid') is not None for m in methods_data):
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Pareto frontier to {output_path}")


def plot_flops_reduction(results: dict, output_path: str):
    """Plot FLOPs reduction."""
    aggregated = results.get('aggregated', results)
    
    methods = []
    reductions = []
    errors = []
    colors = []
    
    method_order = [
        ('vast_2x', 'VAST 2×', '#3498db'),
        ('vast_3x', 'VAST 3×', '#e74c3c'),
    ]
    
    for key, label, color in method_order:
        if key in aggregated:
            methods.append(label)
            if 'flops_reduction_mean' in aggregated[key]:
                reductions.append(aggregated[key]['flops_reduction_mean'] * 100)
                errors.append(aggregated[key].get('flops_reduction_std', 0) * 100)
            elif 'flops_reduction' in aggregated[key] and isinstance(aggregated[key]['flops_reduction'], dict):
                reductions.append(aggregated[key]['flops_reduction']['mean'] * 100)
                errors.append(aggregated[key]['flops_reduction'].get('std', 0) * 100)
            else:
                reductions.append(0)
                errors.append(0)
            colors.append(color)
    
    if len(methods) == 0:
        print("No FLOPs reduction data available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, reductions, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('FLOPs Reduction (%)', fontsize=14, fontweight='bold')
    ax.set_title('Computational Savings', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Reduction')
    ax.axhline(y=67, color='gray', linestyle=':', alpha=0.5, label='67% Reduction')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved FLOPs reduction plot to {output_path}")


def generate_summary_table(results: dict, output_path: str):
    """Generate summary table image."""
    aggregated = results.get('aggregated', results)
    
    # Prepare data for table
    methods_order = [
        ('baseline_50step', 'Baseline 50-step'),
        ('baseline_25step', 'Baseline 25-step'),
        ('baseline_17step', 'Baseline 17-step'),
        ('deepcache', 'DeepCache'),
        ('ras_2x', 'RAS 2×'),
        ('vast_2x', 'VAST 2×'),
        ('vast_3x', 'VAST 3×'),
    ]
    
    rows = []
    for key, label in methods_order:
        if key not in aggregated:
            continue
        
        data = aggregated[key]
        
        # Wall time
        if 'wall_time_per_image' in data and isinstance(data['wall_time_per_image'], dict):
            time_str = f"{data['wall_time_per_image']['mean']:.2f} ± {data['wall_time_per_image'].get('std', 0):.2f}"
        elif 'wall_time_mean' in data:
            time_str = f"{data['wall_time_mean']:.2f} ± {data.get('wall_time_std', 0):.2f}"
        else:
            time_str = "N/A"
        
        # Speedup
        if 'speedup' in data and isinstance(data['speedup'], dict):
            speedup_str = f"{data['speedup']['mean']:.2f} ± {data['speedup'].get('std', 0):.2f}"
        elif 'speedup_mean' in data:
            speedup_str = f"{data['speedup_mean']:.2f} ± {data.get('speedup_std', 0):.2f}"
        else:
            speedup_str = "1.00"
        
        # NFE
        if 'nfe' in data and isinstance(data['nfe'], dict):
            nfe_str = f"{data['nfe']['mean']:.1f}"
        elif 'nfe_per_image' in data and isinstance(data['nfe_per_image'], dict):
            nfe_str = f"{data['nfe_per_image']['mean']:.1f}"
        else:
            nfe_str = str(data.get('nfe', 'N/A'))
        
        # FLOPs reduction
        if 'flops_reduction' in data and isinstance(data['flops_reduction'], dict):
            flops_str = f"{data['flops_reduction']['mean']*100:.1f}%"
        elif 'flops_reduction_mean' in data:
            flops_str = f"{data['flops_reduction_mean']*100:.1f}%"
        else:
            flops_str = "-"
        
        rows.append([label, time_str, speedup_str, nfe_str, flops_str])
    
    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=rows,
        colLabels=['Method', 'Time (s/img)', 'Speedup', 'NFE', 'FLOPs Reduction'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.2, 0.2, 0.15, 0.2],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary table to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate figures
    plot_speedup_comparison(results, os.path.join(args.output_dir, 'speedup_comparison.png'))
    plot_nfe_reduction(results, os.path.join(args.output_dir, 'nfe_reduction.png'))
    plot_pareto_frontier(results, os.path.join(args.output_dir, 'pareto_frontier.png'))
    plot_flops_reduction(results, os.path.join(args.output_dir, 'flops_reduction.png'))
    generate_summary_table(results, os.path.join(args.output_dir, 'summary_table.png'))
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
