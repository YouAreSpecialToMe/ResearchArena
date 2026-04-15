"""
Generate visualization figures for the paper.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from utils import get_project_paths

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results():
    """Load all experiment results."""
    paths = get_project_paths()
    
    results = {}
    
    with open(f"{paths['exp']}/baseline_heuristics/results.json") as f:
        results['heuristics'] = json.load(f)
    with open(f"{paths['exp']}/baseline_profile/results.json") as f:
        results['profile'] = json.load(f)
    with open(f"{paths['exp']}/layoutlearner_main/results.json") as f:
        results['layoutlearner'] = json.load(f)
    with open(f"{paths['exp']}/ablation_features/results.json") as f:
        results['ablation_features'] = json.load(f)
    with open(f"{paths['exp']}/ablation_model/results.json") as f:
        results['ablation_model'] = json.load(f)
    with open(f"{paths['exp']}/overhead_analysis/results.json") as f:
        results['overhead'] = json.load(f)
    
    return results


def plot_performance_comparison(results, save_path):
    """Generate performance comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Static\nHeuristics', 'LayoutLearner', 'Profile-Guided\n(Oracle)']
    
    # Extract metrics
    precision = [
        results['heuristics']['mean']['precision'],
        results['layoutlearner']['mean']['precision'],
        results['profile']['metrics']['precision']
    ]
    recall = [
        results['heuristics']['mean']['recall'],
        results['layoutlearner']['mean']['recall'],
        results['profile']['metrics']['recall']
    ]
    f1 = [
        results['heuristics']['mean']['f1_score'],
        results['layoutlearner']['mean']['f1_score'],
        results['profile']['metrics']['f1_score']
    ]
    
    # Standard deviations
    precision_std = [results['heuristics']['std']['precision'], 
                     results['layoutlearner']['std']['precision'], 0]
    recall_std = [results['heuristics']['std']['recall'],
                  results['layoutlearner']['std']['recall'], 0]
    f1_std = [results['heuristics']['std']['f1_score'],
              results['layoutlearner']['std']['f1_score'], 0]
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, yerr=precision_std, 
                   label='Precision', capsize=5)
    bars2 = ax.bar(x, recall, width, yerr=recall_std,
                   label='Recall', capsize=5)
    bars3 = ax.bar(x + width, f1, width, yerr=f1_std,
                   label='F1-Score', capsize=5)
    
    # Add threshold line
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, 
               label='Within 20% of oracle (F1=0.8)')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('LayoutLearner Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(results, save_path):
    """Generate feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get feature importances
    importance = results['layoutlearner']['feature_importance']
    
    # Sort and select top 15
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    # Color by category
    colors = []
    for f in features:
        if any(x in f.lower() for x in ['loop', 'access', 'hot', 'cold', 'cooccurrence', 'pointer', 'dominance']):
            colors.append('#ff7f0e')  # Orange - access pattern
        elif any(x in f.lower() for x in ['field', 'struct', 'size', 'primitive', 'pointer_field', 'kernel']):
            colors.append('#1f77b4')  # Blue - structural
        else:
            colors.append('#2ca02c')  # Green - context
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('LayoutLearner: Top 15 Feature Importances', fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Structural'),
        Patch(facecolor='#ff7f0e', label='Access Pattern'),
        Patch(facecolor='#2ca02c', label='Context')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_ablation_features(results, save_path):
    """Generate ablation study chart for feature categories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ablation = results['ablation_features']
    
    categories = ['Full Model', 'No Structural', 'No Access Pattern', 'No Context']
    f1_scores = [
        ablation['full']['mean_f1'],
        ablation['no_structural']['mean_f1'],
        ablation['no_access_pattern']['mean_f1'],
        ablation['no_context']['mean_f1']
    ]
    
    # Calculate performance drops
    full_f1 = ablation['full']['mean_f1']
    drops = [
        0,
        ablation['no_structural']['performance_drop_pct'],
        ablation['no_access_pattern']['performance_drop_pct'],
        ablation['no_context']['performance_drop_pct']
    ]
    
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#ff7f0e']
    
    bars = ax.bar(categories, f1_scores, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        label = f'{height:.3f}' if drop == 0 else f'{height:.3f}\n(-{drop:.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Ablation Study: Feature Category Contributions', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(f1_scores) * 1.2])
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_complexity(results, save_path):
    """Generate model complexity vs performance plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_data = results['ablation_model']
    
    models = ['DecisionTree', 'XGBoost_Medium', 'XGBoost_Standard', 'XGBoost_High']
    f1_scores = [model_data[m]['mean_f1'] for m in models]
    inference_times = [model_data[m]['mean_inference_time_us'] for m in models]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    sizes = [100, 150, 200, 250]
    
    scatter = ax.scatter(inference_times, f1_scores, c=colors, s=sizes, 
                         edgecolors='black', linewidth=2, alpha=0.8)
    
    # Add labels
    for i, model in enumerate(models):
        ax.annotate(model.replace('_', '\n'), 
                   (inference_times[i], f1_scores[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Inference Time (μs)', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Model Complexity: Accuracy vs Inference Time Trade-off', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_overhead_breakdown(results, save_path):
    """Generate compilation overhead breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    overhead = results['overhead']
    
    # Overall breakdown
    categories = ['Baseline\nCompilation', 'Feature\nExtraction', 'Model\nInference']
    times = [
        overhead['overall']['baseline_ms'],
        overhead['overall']['feature_extraction_ms'],
        overhead['overall']['inference_ms']
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(categories, times, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} ms', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Add threshold line (5% of baseline)
    threshold = overhead['overall']['baseline_ms'] * 0.05
    ax.axhline(y=overhead['overall']['baseline_ms'] + threshold, 
               color='red', linestyle='--', linewidth=2,
               label=f'5% Overhead Threshold ({threshold:.1f} ms)')
    
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Compilation-Time Overhead Breakdown', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_results_table(results, save_path_csv, save_path_tex):
    """Generate results table in CSV and LaTeX format."""
    
    # Prepare data
    data = []
    
    # LayoutLearner
    data.append({
        'Method': 'LayoutLearner',
        'Precision': f"{results['layoutlearner']['mean']['precision']:.3f} ± {results['layoutlearner']['std']['precision']:.3f}",
        'Recall': f"{results['layoutlearner']['mean']['recall']:.3f} ± {results['layoutlearner']['std']['recall']:.3f}",
        'F1-Score': f"{results['layoutlearner']['mean']['f1_score']:.3f} ± {results['layoutlearner']['std']['f1_score']:.3f}",
        'Inference (μs)': f"{results['ablation_model']['XGBoost_Standard']['mean_inference_time_us']:.1f}"
    })
    
    # Static Heuristics
    data.append({
        'Method': 'Static Heuristics',
        'Precision': f"{results['heuristics']['mean']['precision']:.3f} ± {results['heuristics']['std']['precision']:.3f}",
        'Recall': f"{results['heuristics']['mean']['recall']:.3f} ± {results['heuristics']['std']['recall']:.3f}",
        'F1-Score': f"{results['heuristics']['mean']['f1_score']:.3f} ± {results['heuristics']['std']['f1_score']:.3f}",
        'Inference (μs)': '-'
    })
    
    # Profile-Guided Oracle
    data.append({
        'Method': 'Profile-Guided (Oracle)',
        'Precision': f"{results['profile']['metrics']['precision']:.3f}",
        'Recall': f"{results['profile']['metrics']['recall']:.3f}",
        'F1-Score': f"{results['profile']['metrics']['f1_score']:.3f}",
        'Inference (μs)': '-'
    })
    
    # Save CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path_csv, index=False)
    print(f"Saved: {save_path_csv}")
    
    # Generate LaTeX
    latex = df.to_latex(index=False, escape=False)
    with open(save_path_tex, 'w') as f:
        f.write(latex)
    print(f"Saved: {save_path_tex}")


def main():
    print("=" * 60)
    print("Generating Figures")
    print("=" * 60)
    
    paths = get_project_paths()
    results = load_results()
    
    figures_dir = paths['figures']
    
    # Generate all figures
    print("\nGenerating performance comparison...")
    plot_performance_comparison(results, f"{figures_dir}/performance_comparison.pdf")
    plot_performance_comparison(results, f"{figures_dir}/performance_comparison.png")
    
    print("\nGenerating feature importance...")
    plot_feature_importance(results, f"{figures_dir}/feature_importance.pdf")
    plot_feature_importance(results, f"{figures_dir}/feature_importance.png")
    
    print("\nGenerating ablation study (features)...")
    plot_ablation_features(results, f"{figures_dir}/ablation_features.pdf")
    plot_ablation_features(results, f"{figures_dir}/ablation_features.png")
    
    print("\nGenerating model complexity plot...")
    plot_model_complexity(results, f"{figures_dir}/model_complexity.pdf")
    plot_model_complexity(results, f"{figures_dir}/model_complexity.png")
    
    print("\nGenerating overhead breakdown...")
    plot_overhead_breakdown(results, f"{figures_dir}/overhead_breakdown.pdf")
    plot_overhead_breakdown(results, f"{figures_dir}/overhead_breakdown.png")
    
    print("\nGenerating results table...")
    generate_results_table(results, 
                          f"{figures_dir}/results_table.csv",
                          f"{figures_dir}/results_table.tex")
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Saved to: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
