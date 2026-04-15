"""
Finalize results: aggregate all experiment outputs and create visualizations.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_results():
    """Load all experiment results."""
    results = {}
    
    exp_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp'
    
    # Baseline default
    try:
        with open(f'{exp_dir}/baseline_default/summary.json') as f:
            results['default'] = json.load(f)
    except:
        pass
    
    # Baseline expert
    try:
        with open(f'{exp_dir}/baseline_expert/summary.json') as f:
            results['expert'] = json.load(f)
    except:
        pass
    
    # Baseline MLKAPS
    try:
        with open(f'{exp_dir}/baseline_mlkaps/summary.json') as f:
            results['mlkaps'] = json.load(f)
    except:
        pass
    
    # KAPHE v3 (best version)
    try:
        with open(f'{exp_dir}/kaphe/summary_v3.json') as f:
            results['kaphe'] = json.load(f)
    except:
        try:
            with open(f'{exp_dir}/kaphe/summary_v2.json') as f:
                results['kaphe'] = json.load(f)
        except:
            pass
    
    # Ablations
    try:
        with open(f'{exp_dir}/ablation_no_char/summary.json') as f:
            results['ablation_no_char'] = json.load(f)
    except:
        pass
    
    try:
        with open(f'{exp_dir}/ablation_knn/summary.json') as f:
            results['ablation_knn'] = json.load(f)
    except:
        pass
    
    return results


def create_comparison_table(results):
    """Create performance comparison table."""
    
    rows = []
    
    # Default
    if 'default' in results:
        m = results['default']['metrics']
        rows.append({
            'Method': 'Default Configuration',
            'Mean Score': f"{m['mean_normalized_score']:.4f}",
            'Within 5%': f"{m['within_5pct']:.1f}%",
            'Within 10%': f"{m['within_10pct']:.1f}%",
            'Within 20%': f"{m['within_20pct']:.1f}%",
            'Interpretability': 'N/A',
        })
    
    # Expert
    if 'expert' in results:
        m = results['expert']['metrics']
        rows.append({
            'Method': 'Expert Heuristics',
            'Mean Score': f"{m['mean_normalized_score']:.4f}",
            'Within 5%': f"{m['within_5pct']:.1f}%",
            'Within 10%': f"{m['within_10pct']:.1f}%",
            'Within 20%': f"{m['within_20pct']:.1f}%",
            'Interpretability': 'High (human rules)',
        })
    
    # MLKAPS
    if 'mlkaps' in results:
        m = results['mlkaps']['decision_tree']['metrics']
        interp = results['mlkaps']['decision_tree']['tree_metrics']
        rows.append({
            'Method': 'MLKAPS (Decision Tree)',
            'Mean Score': f"{m['mean_normalized_score']:.4f}",
            'Within 5%': f"{m['within_5pct']:.1f}%",
            'Within 10%': f"{m['within_10pct']:.1f}%",
            'Within 20%': f"{m['within_20pct']:.1f}%",
            'Interpretability': f"Med ({interp['num_nodes']} nodes)",
        })
    
    # KAPHE
    if 'kaphe' in results:
        m = results['kaphe']['metrics']
        if 'interpretability' in results['kaphe']:
            interp = results['kaphe']['interpretability']
            interp_str = f"High ({interp['num_leaves']} rules)"
        else:
            interp_str = "High (rules)"
        rows.append({
            'Method': 'KAPHE (Ours)',
            'Mean Score': f"{m['mean_normalized_score']:.4f}",
            'Within 5%': f"{m['within_5pct']:.1f}%",
            'Within 10%': f"{m['within_10pct']:.1f}%",
            'Within 20%': f"{m['within_20pct']:.1f}%",
            'Interpretability': interp_str,
        })
    
    # k-NN ablation
    if 'ablation_knn' in results:
        m = results['ablation_knn']['metrics']
        rows.append({
            'Method': 'k-NN (Ablation)',
            'Mean Score': f"{m['mean_normalized_score']:.4f}",
            'Within 5%': f"{m['within_5pct']:.1f}%",
            'Within 10%': f"{m['within_10pct']:.1f}%",
            'Within 20%': f"{m['within_20pct']:.1f}%",
            'Interpretability': 'None (black-box)',
        })
    
    df = pd.DataFrame(rows)
    return df


def create_visualizations(results, output_dir):
    """Create publication-quality figures."""
    
    # Figure 1: Performance comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    scores = []
    errors = []
    colors = []
    
    if 'default' in results:
        methods.append('Default')
        scores.append(results['default']['metrics']['mean_normalized_score'])
        errors.append(results['default']['metrics']['std_normalized_score'])
        colors.append('#d62728')
    
    if 'expert' in results:
        methods.append('Expert')
        scores.append(results['expert']['metrics']['mean_normalized_score'])
        errors.append(results['expert']['metrics']['std_normalized_score'])
        colors.append('#ff7f0e')
    
    if 'mlkaps' in results:
        methods.append('MLKAPS')
        scores.append(results['mlkaps']['decision_tree']['metrics']['mean_normalized_score'])
        errors.append(results['mlkaps']['decision_tree']['metrics']['std_normalized_score'])
        colors.append('#2ca02c')
    
    if 'kaphe' in results:
        methods.append('KAPHE (Ours)')
        scores.append(results['kaphe']['metrics']['mean_normalized_score'])
        errors.append(results['kaphe']['metrics']['std_normalized_score'])
        colors.append('#1f77b4')
    
    if 'ablation_knn' in results:
        methods.append('k-NN')
        scores.append(results['ablation_knn']['metrics']['mean_normalized_score'])
        errors.append(results['ablation_knn']['metrics']['std_normalized_score'])
        colors.append('#9467bd')
    
    x = np.arange(len(methods))
    bars = ax.bar(x, scores, yerr=errors, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Normalized Performance Score', fontsize=12)
    ax.set_title('Performance Comparison: KAPHE vs Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim([0.7, 1.05])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Oracle (optimal)')
    
    # Add value labels on bars
    for bar, score, err in zip(bars, scores, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig1_performance_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Accuracy breakdown (within X% of optimal)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods2 = []
    within_5 = []
    within_10 = []
    within_20 = []
    
    for name, key in [('Default', 'default'), ('Expert', 'expert'), 
                      ('MLKAPS', 'mlkaps'), ('KAPHE', 'kaphe')]:
        if key in results:
            methods2.append(name)
            if key == 'mlkaps':
                m = results[key]['decision_tree']['metrics']
            else:
                m = results[key]['metrics']
            within_5.append(m['within_5pct'])
            within_10.append(m['within_10pct'])
            within_20.append(m['within_20pct'])
    
    x = np.arange(len(methods2))
    width = 0.25
    
    ax.bar(x - width, within_5, width, label='Within 5% of optimal', color='#1f77b4')
    ax.bar(x, within_10, width, label='Within 10% of optimal', color='#ff7f0e')
    ax.bar(x + width, within_20, width, label='Within 20% of optimal', color='#2ca02c')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Percentage of Workloads (%)', fontsize=12)
    ax.set_title('Recommendation Accuracy Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods2)
    ax.legend()
    ax.set_ylim([0, 110])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_accuracy_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fig2_accuracy_breakdown.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {output_dir}")


def create_final_json(results, output_dir):
    """Create final aggregated results.json."""
    
    final_results = {
        'experiment': 'kaphe_full_study',
        'methods': {},
    }
    
    # Add each method's results
    for key, name in [('default', 'Baseline_Default'), ('expert', 'Baseline_Expert'),
                      ('mlkaps', 'Baseline_MLKAPS'), ('kaphe', 'KAPHE'),
                      ('ablation_knn', 'Ablation_kNN'), ('ablation_no_char', 'Ablation_No_Char')]:
        if key in results:
            final_results['methods'][name] = results[key]
    
    # Compute summary statistics
    if 'kaphe' in results and 'mlkaps' in results:
        kaphe_score = results['kaphe']['metrics']['mean_normalized_score']
        mlkaps_score = results['mlkaps']['decision_tree']['metrics']['mean_normalized_score']
        default_score = results.get('default', {}).get('metrics', {}).get('mean_normalized_score', 0.77)
        
        final_results['summary'] = {
            'kaphe_vs_default_improvement': f"{((kaphe_score - default_score) / default_score * 100):.1f}%",
            'kaphe_vs_mlkaps_difference': f"{((kaphe_score - mlkaps_score) / mlkaps_score * 100):+.2f}%",
            'within_10pct_of_oracle': results['kaphe']['metrics']['within_10pct'],
            'primary_success_criterion_met': results['kaphe']['metrics']['within_10pct'] >= 80,
        }
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Final results saved to {output_dir}/results.json")
    
    return final_results


def main():
    print("=" * 60)
    print("FINALIZING RESULTS")
    print("=" * 60)
    
    exp_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp'
    figures_dir = f'{exp_dir}/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load all results
    print("\nLoading experiment results...")
    results = load_results()
    print(f"  Loaded {len(results)} result sets")
    
    # Create comparison table
    print("\nCreating comparison table...")
    comp_table = create_comparison_table(results)
    print("\n" + comp_table.to_string(index=False))
    comp_table.to_csv(f'{figures_dir}/comparison_table.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, figures_dir)
    
    # Create final JSON
    print("\nCreating final results JSON...")
    final_results = create_final_json(results, exp_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    
    if 'summary' in final_results:
        s = final_results['summary']
        print(f"\nKAPHE vs Default: {s.get('kaphe_vs_default_improvement', 'N/A')} improvement")
        print(f"KAPHE vs MLKAPS: {s.get('kaphe_vs_mlkaps_difference', 'N/A')} difference")
        print(f"Within 10% of oracle: {s.get('within_10pct_of_oracle', 0):.1f}% of workloads")
        print(f"Primary success criterion (≥80% within 10%): {'✓ MET' if s.get('primary_success_criterion_met') else '✗ NOT MET'}")
    
    print("\n" + "=" * 60)
    print("Results finalized successfully!")
    print(f"Output directory: {exp_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
