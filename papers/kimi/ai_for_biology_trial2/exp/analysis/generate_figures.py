"""
Generate publication-quality figures for the paper.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_results():
    """Load all experimental results."""
    data_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/results')
    
    results = {}
    
    # Load baselines
    for name in ['baseline_knn', 'baseline_contrastive', 'baseline_ontology']:
        filepath = data_dir / f'{name}.json'
        if filepath.exists():
            with open(filepath) as f:
                results[name.replace('baseline_', '')] = json.load(f)
    
    # Load Tri-Con
    tricon_file = data_dir / 'tricon_v3_full.json'
    if tricon_file.exists():
        with open(tricon_file) as f:
            results['tricon'] = json.load(f)
    
    # Load ablations
    for name in ['ablation_no_cc', 'ablation_no_co', 'ablation_no_go', 
                 'ablation_uniform', 'ablation_no_evidential']:
        filepath = data_dir / f'{name}.json'
        if filepath.exists():
            with open(filepath) as f:
                ablation_name = name.replace('ablation_', '')
                results[ablation_name] = json.load(f)
    
    return results


def plot_performance_comparison(results, output_dir):
    """Figure 2: Performance comparison bar plot."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    methods = []
    test_acc_mean = []
    test_acc_std = []
    zero_shot_mean = []
    zero_shot_std = []
    ood_auroc_mean = []
    ood_auroc_std = []
    
    # Collect data
    method_order = ['knn', 'contrastive', 'ontology', 'tricon']
    method_labels = ['k-NN', 'Contrastive', 'Ontology', 'Tri-Con V3']
    
    for method in method_order:
        if method in results:
            mean = results[method].get('mean', {})
            methods.append(method_labels[method_order.index(method)])
            test_acc_mean.append(mean.get('test_accuracy', 0) * 100)
            test_acc_std.append(mean.get('test_accuracy_std', 0) * 100)
            zero_shot_mean.append(mean.get('zero_shot_accuracy', 0) * 100)
            zero_shot_std.append(mean.get('zero_shot_accuracy_std', 0) * 100)
            ood_auroc_mean.append(mean.get('ood_auroc', 0) * 100)
            ood_auroc_std.append(mean.get('ood_auroc_std', 0) * 100)
    
    x = np.arange(len(methods))
    width = 0.6
    
    # Test Accuracy
    axes[0].bar(x, test_acc_mean, width, yerr=test_acc_std, capsize=5, 
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0].set_ylim([0, 100])
    axes[0].axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('(a) Test Accuracy')
    
    # Zero-Shot Accuracy
    axes[1].bar(x, zero_shot_mean, width, yerr=zero_shot_std, capsize=5,
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].set_ylabel('Zero-Shot Accuracy (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha='right')
    axes[1].set_ylim([0, 100])
    axes[1].axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('(b) Zero-Shot Accuracy')
    
    # OOD AUROC
    axes[2].bar(x, ood_auroc_mean, width, yerr=ood_auroc_std, capsize=5,
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[2].set_ylabel('OOD AUROC (%)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15, ha='right')
    axes[2].set_ylim([0, 100])
    axes[2].axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='Target (85%)')
    axes[2].set_title('(c) OOD Detection AUROC')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_performance_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_performance_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to {output_dir / 'fig2_performance_comparison.pdf'}")


def plot_ablation_study(results, output_dir):
    """Figure 3: Ablation study bar plot."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    ablations = []
    test_acc_mean = []
    test_acc_std = []
    zero_shot_mean = []
    zero_shot_std = []
    ood_auroc_mean = []
    ood_auroc_std = []
    
    # Full model first
    if 'tricon' in results:
        mean = results['tricon'].get('mean', {})
        ablations.append('Full\nModel')
        test_acc_mean.append(mean.get('test_accuracy', 0) * 100)
        test_acc_std.append(mean.get('test_accuracy_std', 0) * 100)
        zero_shot_mean.append(mean.get('zero_shot_accuracy', 0) * 100)
        zero_shot_std.append(mean.get('zero_shot_accuracy_std', 0) * 100)
        ood_auroc_mean.append(mean.get('ood_auroc', 0) * 100)
        ood_auroc_std.append(mean.get('ood_auroc_std', 0) * 100)
    
    # Ablations
    ablation_order = ['no_cc', 'no_co', 'no_go', 'uniform', 'no_evidential']
    ablation_labels = ['-L_CC\n(No Cell-Cell)', '-L_CO\n(No Cell-Ontology)', 
                       '-L_GO\n(No Gene-Ontology)', '-Hierarchy\n(Uniform)', 
                       '-Evidential\n(Softmax)']
    
    colors = ['#2ca02c']  # Full model in green
    
    for ablation in ablation_order:
        if ablation in results:
            mean = results[ablation].get('mean', {})
            ablations.append(ablation_labels[ablation_order.index(ablation)])
            test_acc_mean.append(mean.get('test_accuracy', 0) * 100)
            test_acc_std.append(mean.get('test_accuracy_std', 0) * 100)
            zero_shot_mean.append(mean.get('zero_shot_accuracy', 0) * 100)
            zero_shot_std.append(mean.get('zero_shot_accuracy_std', 0) * 100)
            ood_auroc_mean.append(mean.get('ood_auroc', 0) * 100)
            ood_auroc_std.append(mean.get('ood_auroc_std', 0) * 100)
            # Color by impact: red for large drop, yellow for moderate, green for minimal
            acc_drop = test_acc_mean[0] - mean.get('test_accuracy', 0) * 100
            if acc_drop > 50:
                colors.append('#d62728')  # Red - catastrophic
            elif acc_drop > 5:
                colors.append('#ff7f0e')  # Orange - moderate
            else:
                colors.append('#2ca02c')  # Green - minimal
    
    x = np.arange(len(ablations))
    width = 0.6
    
    # Test Accuracy
    bars = axes[0].bar(x, test_acc_mean, width, yerr=test_acc_std, capsize=5, color=colors)
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ablations, rotation=0, ha='center', fontsize=8)
    axes[0].set_ylim([0, 100])
    axes[0].axhline(y=test_acc_mean[0], color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('(a) Test Accuracy')
    
    # Zero-Shot Accuracy
    axes[1].bar(x, zero_shot_mean, width, yerr=zero_shot_std, capsize=5, color=colors)
    axes[1].set_ylabel('Zero-Shot Accuracy (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ablations, rotation=0, ha='center', fontsize=8)
    axes[1].set_ylim([0, 100])
    axes[1].axhline(y=zero_shot_mean[0], color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('(b) Zero-Shot Accuracy')
    
    # OOD AUROC
    axes[2].bar(x, ood_auroc_mean, width, yerr=ood_auroc_std, capsize=5, color=colors)
    axes[2].set_ylabel('OOD AUROC (%)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(ablations, rotation=0, ha='center', fontsize=8)
    axes[2].set_ylim([0, 100])
    axes[2].axhline(y=85, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title('(c) OOD Detection')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_ablation_study.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_ablation_study.png', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to {output_dir / 'fig3_ablation_study.pdf'}")


def plot_generalization_gap(results, output_dir):
    """Figure: Generalization gap comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = []
    gap_mean = []
    gap_std = []
    
    method_order = ['knn', 'contrastive', 'ontology', 'tricon']
    method_labels = ['k-NN', 'Contrastive', 'Ontology', 'Tri-Con V3']
    
    for method in method_order:
        if method in results:
            mean = results[method].get('mean', {})
            methods.append(method_labels[method_order.index(method)])
            gap_mean.append(mean.get('generalization_gap', 0) * 100)
            gap_std.append(mean.get('generalization_gap_std', 0) * 100)
    
    x = np.arange(len(methods))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(x, gap_mean, 0.6, yerr=gap_std, capsize=5, color=colors)
    ax.set_ylabel('Generalization Gap (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_title('Generalization Gap: Seen vs Unseen Cell Types')
    ax.set_ylim([-20, 100])
    
    # Add value labels on bars
    for bar, val in zip(bars, gap_mean):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_generalization_gap.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig_generalization_gap.png', bbox_inches='tight')
    plt.close()
    print(f"Saved generalization gap figure to {output_dir / 'fig_generalization_gap.pdf'}")


def generate_tables(results, output_dir):
    """Generate LaTeX tables."""
    
    # Table 1: Main Results
    table1 = []
    table1.append("\\begin{table}[t]")
    table1.append("\\centering")
    table1.append("\\caption{Main Results: Comparison of Tri-Con V3 with Baselines}")
    table1.append("\\label{tab:main_results}")
    table1.append("\\begin{tabular}{lcccc}")
    table1.append("\\toprule")
    table1.append("Method & Test Acc (\\%) & Zero-Shot Acc (\\%) & Gen. Gap (\\%) & OOD AUROC \\\\")
    table1.append("\\midrule")
    
    for method, label in [('knn', 'k-NN'), ('contrastive', 'Contrastive'), 
                          ('ontology', 'Ontology'), ('tricon', 'Tri-Con V3')]:
        if method in results:
            mean = results[method].get('mean', {})
            acc = mean.get('test_accuracy', 0) * 100
            acc_std = mean.get('test_accuracy_std', 0) * 100
            zs = mean.get('zero_shot_accuracy', 0) * 100
            zs_std = mean.get('zero_shot_accuracy_std', 0) * 100
            gap = mean.get('generalization_gap', 0) * 100
            gap_std = mean.get('generalization_gap_std', 0) * 100
            ood = mean.get('ood_auroc', 0)
            ood_std = mean.get('ood_auroc_std', 0)
            
            table1.append(f"{label} & ${acc:.1f} \\pm {acc_std:.1f}$ & "
                         f"${zs:.1f} \\pm {zs_std:.1f}$ & "
                         f"${gap:.1f} \\pm {gap_std:.1f}$ & "
                         f"${ood:.3f} \\pm {ood_std:.3f}$ \\\\")
    
    table1.append("\\bottomrule")
    table1.append("\\end{tabular}")
    table1.append("\\end{table}")
    
    with open(output_dir / 'table1_main_results.tex', 'w') as f:
        f.write('\n'.join(table1))
    print(f"Saved Table 1 to {output_dir / 'table1_main_results.tex'}")
    
    # Table 2: Ablation Study
    table2 = []
    table2.append("\\begin{table}[t]")
    table2.append("\\centering")
    table2.append("\\caption{Ablation Study: Impact of Removing Each Component}")
    table2.append("\\label{tab:ablation}")
    table2.append("\\begin{tabular}{lcccc}")
    table2.append("\\toprule")
    table2.append("Configuration & Test Acc (\\%) & $\\Delta$ (\\%) & Zero-Shot (\\%) & OOD AUROC \\\\")
    table2.append("\\midrule")
    
    if 'tricon' in results:
        full_mean = results['tricon'].get('mean', {})
        full_acc = full_mean.get('test_accuracy', 0) * 100
        full_acc_std = full_mean.get('test_accuracy_std', 0) * 100
        full_zs = full_mean.get('zero_shot_accuracy', 0) * 100
        full_ood = full_mean.get('ood_auroc', 0)
        
        table2.append(f"Full Model & ${full_acc:.1f} \\pm {full_acc_std:.1f}$ & -- & "
                     f"${full_zs:.1f}$ & ${full_ood:.3f}$ \\\\")
        table2.append("\\midrule")
        
        for ablation, label in [('no_cc', '-L_CC'), ('no_co', '-L_CO'), 
                                ('no_go', '-L_GO'), ('uniform', '-Hierarchy'), 
                                ('no_evidential', '-Evidential')]:
            if ablation in results:
                mean = results[ablation].get('mean', {})
                acc = mean.get('test_accuracy', 0) * 100
                acc_std = mean.get('test_accuracy_std', 0) * 100
                delta = acc - full_acc
                zs = mean.get('zero_shot_accuracy', 0) * 100
                ood = mean.get('ood_auroc', 0)
                
                table2.append(f"{label} & ${acc:.1f} \\pm {acc_std:.1f}$ & "
                             f"${delta:+.1f}$ & ${zs:.1f}$ & ${ood:.3f}$ \\\\")
    
    table2.append("\\bottomrule")
    table2.append("\\end{tabular}")
    table2.append("\\end{table}")
    
    with open(output_dir / 'table2_ablation.tex', 'w') as f:
        f.write('\n'.join(table2))
    print(f"Saved Table 2 to {output_dir / 'table2_ablation.tex'}")


def main():
    print("Generating figures and tables...")
    
    results = load_results()
    
    figures_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    tables_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/tables')
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    plot_performance_comparison(results, figures_dir)
    plot_ablation_study(results, figures_dir)
    plot_generalization_gap(results, figures_dir)
    
    # Generate tables
    generate_tables(results, tables_dir)
    
    print("\nDone! Generated files:")
    print(f"  - {figures_dir}/fig2_performance_comparison.pdf")
    print(f"  - {figures_dir}/fig3_ablation_study.pdf")
    print(f"  - {figures_dir}/fig_generalization_gap.pdf")
    print(f"  - {tables_dir}/table1_main_results.tex")
    print(f"  - {tables_dir}/table2_ablation.tex")


if __name__ == '__main__':
    main()
