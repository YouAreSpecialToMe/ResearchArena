#!/usr/bin/env python3
"""
Experiment 15: Visualization and Summary Report Generation
- Generate all figures for the paper
- Create summary report
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_all_results():
    """Load all experiment results."""
    results = {}
    with open("results/baseline_simple.json") as f:
        results['baseline_simple'] = json.load(f)
    with open("results/baseline_exhaustive.json") as f:
        results['baseline_exhaustive'] = json.load(f)
    with open("results/baseline_mcts.json") as f:
        results['baseline_mcts'] = json.load(f)
    with open("results/leopard_main.json") as f:
        results['leopard_main'] = json.load(f)
    with open("results/ablation_no_learning.json") as f:
        results['ablation_no_learning'] = json.load(f)
    with open("results/ablation_memory.json") as f:
        results['ablation_memory'] = json.load(f)
    with open("results/failure_analysis.json") as f:
        results['failure_analysis'] = json.load(f)
    with open("results/scorer_analysis.json") as f:
        results['scorer_analysis'] = json.load(f)
    with open("results/evaluation.json") as f:
        results['evaluation'] = json.load(f)
    with open("results/aurora_comparison.json") as f:
        results['aurora'] = json.load(f)
    return results

def figure_1_speedup_comparison(results):
    """Figure 1: Speedup comparison across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['LLVM-O3', 'Random', 'MCTS', 'LEOPARD', 'Exhaustive']
    speedups = [
        results['baseline_simple']['summary']['llvm_o3_geomean_speedup'],
        results['baseline_simple']['summary']['random_selection_geomean_speedup'],
        results['baseline_mcts']['summary']['geomean_speedup'],
        results['leopard_main']['summary']['geomean_speedup'],
        results['baseline_exhaustive']['summary']['geomean_speedup'],
    ]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
    
    bars = ax.bar(methods, speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Geomean Speedup (×)', fontsize=12)
    ax.set_title('Speedup Comparison: LEOPARD vs Baselines', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No optimization')
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}×',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, max(speedups) * 1.15)
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/figure1_speedup_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure1_speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figure1_speedup_comparison")

def figure_2_memory_quality_scatter(results):
    """Figure 2: Memory vs Quality scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect data
    leopard_mem = []
    leopard_speed = []
    exhaustive_mem = []
    exhaustive_speed = []
    
    for prog_name in results['leopard_main']['leopard'].keys():
        leopard_mem.append(results['leopard_main']['leopard'][prog_name]['peak_memory_mb_mean'])
        leopard_speed.append(results['leopard_main']['leopard'][prog_name]['speedup_mean'])
        exhaustive_mem.append(results['baseline_exhaustive']['exhaustive'][prog_name]['peak_memory_mb'])
        exhaustive_speed.append(results['baseline_exhaustive']['exhaustive'][prog_name]['speedup'])
    
    ax.scatter(exhaustive_mem, exhaustive_speed, alpha=0.6, s=100, 
               label='Exhaustive ES', color='#9b59b6', marker='o')
    ax.scatter(leopard_mem, leopard_speed, alpha=0.6, s=100,
               label='LEOPARD', color='#2ecc71', marker='s')
    
    ax.set_xlabel('Peak Memory Usage (MB)', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('Memory vs Quality Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add 80% speedup line
    exhaustive_geomean = results['baseline_exhaustive']['summary']['geomean_speedup']
    ax.axhline(y=exhaustive_geomean * 0.8, color='red', linestyle='--', 
               alpha=0.5, label='80% of exhaustive')
    
    plt.tight_layout()
    plt.savefig('figures/figure2_memory_quality_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure2_memory_quality_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figure2_memory_quality_scatter")

def figure_3_ablation_results(results):
    """Figure 3: Ablation study results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Ablation speedups
    ax1 = axes[0]
    ablations = ['Heuristic\n(No Learning)', 'LEOPARD\n(Full)']
    speedups = [
        results['ablation_no_learning']['summary']['geomean_speedup'],
        results['leopard_main']['summary']['geomean_speedup']
    ]
    colors = ['#e67e22', '#2ecc71']
    
    bars = ax1.bar(ablations, speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Geomean Speedup (×)', fontsize=12)
    ax1.set_title('Impact of Learned Scorer', fontsize=13, fontweight='bold')
    
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}×',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right: Memory budget sensitivity
    ax2 = axes[1]
    budgets = [30, 50, 70]
    budget_speedups = [
        results['ablation_memory']['tradeoff_analysis']['budget_30'],
        results['ablation_memory']['tradeoff_analysis']['budget_50'],
        results['ablation_memory']['tradeoff_analysis']['budget_70'],
    ]
    exhaustive_baseline = results['ablation_memory']['tradeoff_analysis']['exhaustive_baseline']
    
    ax2.plot(budgets, budget_speedups, 'o-', linewidth=2, markersize=10, 
             label='LEOPARD', color='#2ecc71')
    ax2.axhline(y=exhaustive_baseline, color='#9b59b6', linestyle='--', 
                linewidth=2, label='Exhaustive ES')
    ax2.set_xlabel('Memory Budget (% of Exhaustive)', fontsize=12)
    ax2.set_ylabel('Geomean Speedup (×)', fontsize=12)
    ax2.set_title('Memory Budget Sensitivity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/figure3_ablation_study.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure3_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figure3_ablation_study")

def figure_4_scorer_accuracy(results):
    """Figure 4: Scorer accuracy (predicted vs actual)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Metrics bar chart
    ax1 = axes[0]
    metrics = ['Pearson r', 'Spearman r', 'Top-1 Acc']
    values = [
        results['scorer_analysis']['accuracy']['pearson_r'],
        results['scorer_analysis']['accuracy']['spearman_r'],
        results['scorer_analysis']['accuracy']['top1_accuracy'],
    ]
    colors = ['#3498db', '#9b59b6', '#2ecc71']
    
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Scorer Prediction Quality', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right: Inference time breakdown
    ax2 = axes[1]
    times = ['Feature\nExtraction', 'Model\nInference', 'Total\nOverhead']
    time_values = [
        results['scorer_analysis']['timing']['feature_extraction_ms'],
        results['scorer_analysis']['timing']['model_inference_ms'],
        results['scorer_analysis']['timing']['total_overhead_ms'],
    ]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    bars = ax2.bar(times, time_values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Inference Overhead Breakdown', fontsize=13, fontweight='bold')
    
    for bar, val in zip(bars, time_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/figure4_scorer_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure4_scorer_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figure4_scorer_accuracy")

def figure_5_degradation_curve(results):
    """Figure 5: Degradation curve with/without fallback."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    noise_levels = [0, 10, 25, 50]
    
    with_fallback = []
    without_fallback = []
    random_baseline = []
    
    for noise in noise_levels:
        key = f"noise_{noise}"
        with_fallback.append(results['failure_analysis']['degradation_curves'][key]['with_fallback_geomean'])
        without_fallback.append(results['failure_analysis']['degradation_curves'][key]['no_fallback_geomean'])
        random_baseline.append(results['failure_analysis']['degradation_curves'][key]['random_geomean'])
    
    ax.plot(noise_levels, with_fallback, 'o-', linewidth=2.5, markersize=10,
            label='LEOPARD (with fallback)', color='#2ecc71')
    ax.plot(noise_levels, without_fallback, 's--', linewidth=2.5, markersize=10,
            label='LEOPARD (no fallback)', color='#e74c3c')
    ax.plot(noise_levels, random_baseline, '^:', linewidth=2.5, markersize=10,
            label='Random baseline', color='#95a5a6')
    
    ax.set_xlabel('Noise Level (%)', fontsize=12)
    ax.set_ylabel('Geomean Speedup (×)', fontsize=12)
    ax.set_title('Graceful Degradation Under Prediction Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/figure5_degradation_curve.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure5_degradation_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figure5_degradation_curve")

def figure_6_aurora_comparison(results):
    """Figure 6: Aurora comparison table visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create comparison table
    table_data = [
        ['Aspect', 'Aurora (GNN+RNN)', 'LEOPARD (MLP/GBDT)'],
        ['Architecture', 'GNN + RNN\n(Spatio-temporal)', 'Small MLP / GBDT\n(~1K parameters)'],
        ['Training', 'RL (PPO)\n200K+ steps, GPU', 'Supervised learning\nMinutes on CPU'],
        ['Parameters', f"{results['aurora']['aurora']['num_parameters']:,}", 
         f"{results['aurora']['leopard']['num_parameters']:,}"],
        ['Inference time', f"{results['aurora']['aurora']['measured_inference_ms']:.2f} ms",
         f"{results['aurora']['leopard']['measured_inference_ms']:.2f} ms"],
        ['Speedup vs Aurora', '1.0×', f"{results['aurora']['comparison']['speedup']:.1f}×"],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax.set_title('Aurora vs LEOPARD: Architecture Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('figures/figure6_aurora_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure6_aurora_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figure6_aurora_comparison")

def generate_summary_report(results):
    """Generate summary report markdown."""
    report = """# LEOPARD Experimental Results Summary

## Overview

This report summarizes the experimental evaluation of LEOPARD (LEarned Optimization Potential 
for Adaptive Rewrite Direction), a lightweight learned guidance system for equality saturation 
in compiler optimization.

## Key Results

### Main Performance Metrics

| Metric | Value |
|--------|-------|
| LEOPARD Geomean Speedup | {:.3f}× |
| Exhaustive ES Speedup | {:.3f}× |
| Speedup Ratio (vs Exhaustive) | {:.1f}% |
| Memory Usage (vs Exhaustive) | {:.1f}% |
| Scorer Inference Time | {:.3f} ms |
| Compilation Overhead | {:.2f}% |

### Comparison to Baselines

| Method | Geomean Speedup |
|--------|----------------|
| LLVM -O3 | {:.3f}× |
| Random Selection | {:.3f}× |
| MCTS-Guided ES | {:.3f}× |
| **LEOPARD** | **{:.3f}×** |
| Exhaustive ES | {:.3f}× |

## Success Criteria Evaluation

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Speedup vs Exhaustive | ≥80% | {:.1f}% | {} |
| Memory vs Exhaustive | ≤60% | {:.1f}% | {} |
| Scorer Accuracy | >65% | {:.1f}% | {} |
| Inference Overhead | <5% | {:.2f}% | {} |
| Graceful Degradation | Verified | {} | {} |

**Overall: {}/5 criteria passed**

## Ablation Studies

### Impact of Learned Scorer
- Heuristic-only: {:.3f}×
- LEOPARD (with learning): {:.3f}×
- **Improvement: {:.1f}%**

### Memory Budget Sensitivity
| Budget | Speedup |
|--------|---------|
| 30% | {:.3f}× |
| 50% | {:.3f}× |
| 70% | {:.3f}× |
| Exhaustive | {:.3f}× |

## Conclusion

LEOPARD achieves **{:.1f}% of exhaustive ES speedup** while using only **{:.1f}% of memory**, 
demonstrating that lightweight learned guidance can effectively guide equality saturation for 
compiler optimization. The system provides **{:.1f}× faster inference** compared to Aurora-style 
GNN+RNN approaches, making it suitable for production compiler deployment.

## Figures

All figures are saved in the `figures/` directory:
- `figure1_speedup_comparison.pdf` - Speedup comparison across methods
- `figure2_memory_quality_scatter.pdf` - Memory vs quality tradeoff
- `figure3_ablation_study.pdf` - Ablation study results
- `figure4_scorer_accuracy.pdf` - Scorer prediction quality
- `figure5_degradation_curve.pdf` - Graceful degradation analysis
- `figure6_aurora_comparison.pdf` - Aurora comparison table

---
*Generated automatically from experimental results*
""".format(
        results['leopard_main']['summary']['geomean_speedup'],
        results['baseline_exhaustive']['summary']['geomean_speedup'],
        results['evaluation']['summary_metrics']['speedup_ratio_vs_exhaustive'] * 100,
        results['evaluation']['summary_metrics']['memory_ratio_vs_exhaustive'] * 100,
        results['scorer_analysis']['timing']['model_inference_ms'],
        results['scorer_analysis']['timing']['overhead_pct_of_compilation'],
        
        results['baseline_simple']['summary']['llvm_o3_geomean_speedup'],
        results['baseline_simple']['summary']['random_selection_geomean_speedup'],
        results['baseline_mcts']['summary']['geomean_speedup'],
        results['leopard_main']['summary']['geomean_speedup'],
        results['baseline_exhaustive']['summary']['geomean_speedup'],
        
        results['evaluation']['summary_metrics']['speedup_ratio_vs_exhaustive'] * 100,
        "✓ PASS" if results['evaluation']['success_criteria']['criterion_1_speedup_memory']['passed'] else "✗ FAIL",
        results['evaluation']['summary_metrics']['memory_ratio_vs_exhaustive'] * 100,
        "✓ PASS" if results['evaluation']['success_criteria']['criterion_1_speedup_memory']['passed'] else "✗ FAIL",
        results['evaluation']['summary_metrics']['scorer_accuracy'] * 100,
        "✓ PASS" if results['evaluation']['success_criteria']['criterion_2_scorer_accuracy']['passed'] else "✗ FAIL",
        results['evaluation']['summary_metrics']['inference_overhead_pct'],
        "✓ PASS" if results['evaluation']['success_criteria']['criterion_3_inference_overhead']['passed'] else "✗ FAIL",
        results['evaluation']['success_criteria']['criterion_4_graceful_degradation']['verified'],
        "✓ PASS" if results['evaluation']['success_criteria']['criterion_4_graceful_degradation']['passed'] else "✗ FAIL",
        
        sum(1 for c in results['evaluation']['success_criteria'].values() if c['passed']),
        
        results['ablation_no_learning']['summary']['geomean_speedup'],
        results['leopard_main']['summary']['geomean_speedup'],
        results['ablation_no_learning']['comparison']['improvement'],
        
        results['ablation_memory']['tradeoff_analysis']['budget_30'],
        results['ablation_memory']['tradeoff_analysis']['budget_50'],
        results['ablation_memory']['tradeoff_analysis']['budget_70'],
        results['ablation_memory']['tradeoff_analysis']['exhaustive_baseline'],
        
        results['evaluation']['summary_metrics']['speedup_ratio_vs_exhaustive'] * 100,
        results['evaluation']['summary_metrics']['memory_ratio_vs_exhaustive'] * 100,
        results['aurora']['comparison']['speedup']
    )
    
    with open("results/summary_report.md", "w") as f:
        f.write(report)
    
    print("  Saved: summary_report.md")

def main():
    print("=" * 60)
    print("Experiment 15: Visualization and Summary Report")
    print("=" * 60)
    
    print("\nLoading results...")
    results = load_all_results()
    
    print("\nGenerating figures...")
    figure_1_speedup_comparison(results)
    figure_2_memory_quality_scatter(results)
    figure_3_ablation_results(results)
    figure_4_scorer_accuracy(results)
    figure_5_degradation_curve(results)
    figure_6_aurora_comparison(results)
    
    print("\nGenerating summary report...")
    generate_summary_report(results)
    
    with open("exp/15_visualization/results.json", "w") as f:
        json.dump({
            "experiment": "15_visualization",
            "status": "completed",
            "figures_generated": 6,
            "report_generated": "results/summary_report.md"
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("Figures saved to figures/")
    print("Report saved to results/summary_report.md")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())
