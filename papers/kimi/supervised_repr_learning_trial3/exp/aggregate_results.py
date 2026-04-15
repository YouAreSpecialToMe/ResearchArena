"""
Aggregate results from all experiments and generate visualizations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_results(pattern):
    """Load all result files matching pattern."""
    import glob
    files = glob.glob(pattern)
    results = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                results.append(json.load(fp))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return results


def compute_stats(values):
    """Compute mean, std, and confidence interval."""
    if len(values) == 0:
        return None
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0
    sem = std / np.sqrt(len(values)) if len(values) > 1 else 0
    ci = 1.96 * sem  # 95% CI
    return {
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "ci_95": float(ci),
        "values": [float(v) for v in values]
    }


def paired_t_test(values1, values2):
    """Perform paired t-test."""
    if len(values1) != len(values2) or len(values1) < 2:
        return None
    t_stat, p_value = stats.ttest_rel(values1, values2)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05
    }


def aggregate_experiment_results():
    """Aggregate results from all experiments."""
    
    aggregated = {
        "experiments": {},
        "comparisons": {},
        "success_criteria": {}
    }
    
    # Load results by experiment type
    ce_results = load_results('results/cifar100_crossentropy_seed*.json')
    supcon_results = load_results('results/cifar100_supcon_seed*.json')
    jdccl_results = load_results('results/cifar100_jdccl_fixed_seed*.json')
    caghnm_results = load_results('results/cifar100_caghnm_seed*.json')
    ablation_results = load_results('results/ablation_fixed_vs_curriculum.json')
    
    # Extract accuracies
    def get_accuracies(results, key='linear_accuracy'):
        return [r.get(key, r.get('best_accuracy', 0)) for r in results]
    
    def get_runtimes(results):
        return [r.get('runtime_minutes', 0) for r in results]
    
    # Aggregate by method
    aggregated["experiments"]["CrossEntropy"] = compute_stats(get_accuracies(ce_results))
    aggregated["experiments"]["SupCon"] = compute_stats(get_accuracies(supcon_results))
    aggregated["experiments"]["JD-CCL-Fixed"] = compute_stats(get_accuracies(jdccl_results))
    aggregated["experiments"]["CAG-HNM"] = compute_stats(get_accuracies(caghnm_results))
    
    # Runtimes
    aggregated["experiments"]["CrossEntropy"]["runtime"] = compute_stats(get_runtimes(ce_results))
    aggregated["experiments"]["SupCon"]["runtime"] = compute_stats(get_runtimes(supcon_results))
    aggregated["experiments"]["JD-CCL-Fixed"]["runtime"] = compute_stats(get_runtimes(jdccl_results))
    aggregated["experiments"]["CAG-HNM"]["runtime"] = compute_stats(get_runtimes(caghnm_results))
    
    # Statistical comparisons
    supcon_acc = get_accuracies(supcon_results)
    caghnm_acc = get_accuracies(caghnm_results)
    jdccl_acc = get_accuracies(jdccl_results)
    
    if len(supcon_acc) >= 2 and len(caghnm_acc) >= 2:
        # Compare CAG-HNM vs SupCon (use first min seeds)
        min_seeds = min(len(supcon_acc), len(caghnm_acc))
        aggregated["comparisons"]["CAG-HNM_vs_SupCon"] = paired_t_test(
            caghnm_acc[:min_seeds], supcon_acc[:min_seeds]
        )
    
    if len(jdccl_acc) >= 2 and len(caghnm_acc) >= 2:
        min_seeds = min(len(jdccl_acc), len(caghnm_acc))
        aggregated["comparisons"]["CAG-HNM_vs_JD-CCL"] = paired_t_test(
            caghnm_acc[:min_seeds], jdccl_acc[:min_seeds]
        )
    
    # Ablation analysis
    if ablation_results:
        ablation = ablation_results[0]
        aggregated["ablation"] = {
            "runs": ablation.get("runs", [])
        }
        
        # Find curriculum vs fixed comparison
        for run in ablation.get("runs", []):
            if run.get("method") == "curriculum":
                aggregated["ablation"]["curriculum_accuracy"] = run.get("linear_accuracy")
            elif run.get("method") == "fixed":
                if "fixed_results" not in aggregated["ablation"]:
                    aggregated["ablation"]["fixed_results"] = []
                aggregated["ablation"]["fixed_results"].append({
                    "lambda": run.get("lambda_weight"),
                    "accuracy": run.get("linear_accuracy")
                })
    
    # Check success criteria
    criteria = {}
    
    # Criterion 1: Significant improvement over SupCon
    if "CAG-HNM_vs_SupCon" in aggregated["comparisons"]:
        comp = aggregated["comparisons"]["CAG-HNM_vs_SupCon"]
        criteria["significant_vs_supcon"] = comp.get("significant", False) and comp.get("t_statistic", 0) > 0
    
    # Criterion 2: Outperform JD-CCL by at least 1.5%
    if aggregated["experiments"]["CAG-HNM"] and aggregated["experiments"]["JD-CCL-Fixed"]:
        cag_mean = aggregated["experiments"]["CAG-HNM"]["mean"]
        jdccl_mean = aggregated["experiments"]["JD-CCL-Fixed"]["mean"]
        criteria["outperform_jdccl_by_1.5%"] = (cag_mean - jdccl_mean) >= 1.5
    
    # Criterion 3: Ablation shows curriculum > fixed
    if "ablation" in aggregated:
        curr_acc = aggregated["ablation"].get("curriculum_accuracy", 0)
        fixed_results = aggregated["ablation"].get("fixed_results", [])
        if fixed_results:
            best_fixed = max(r["accuracy"] for r in fixed_results)
            criteria["curriculum_better_than_fixed"] = curr_acc > best_fixed
    
    aggregated["success_criteria"] = criteria
    aggregated["all_criteria_met"] = all(criteria.values())
    
    return aggregated


def generate_figures(aggregated):
    """Generate figures for the paper."""
    os.makedirs('figures', exist_ok=True)
    
    # Figure 1: Main results bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['CrossEntropy', 'SupCon', 'JD-CCL-Fixed', 'CAG-HNM']
    means = []
    stds = []
    
    for method in methods:
        stats = aggregated["experiments"].get(method)
        if stats:
            means.append(stats["mean"])
            stds.append(stats["std"])
        else:
            means.append(0)
            stds.append(0)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-100: Comparison of Methods', fontsize=14)
    ax.set_ylim([min(means) - 5, max(means) + 2])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/main_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/main_results.pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/main_results.png and .pdf")
    
    # Figure 2: Ablation comparison
    if "ablation" in aggregated:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ablation = aggregated["ablation"]
        labels = []
        values = []
        
        # Fixed methods
        for result in ablation.get("fixed_results", []):
            labels.append(f'Fixed (λ={result["lambda"]})')
            values.append(result["accuracy"])
        
        # Curriculum method
        labels.append('Curriculum\n(Ours)')
        values.append(ablation.get("curriculum_accuracy", 0))
        
        colors = ['#2ca02c', '#2ca02c', '#d62728']
        bars = ax.bar(labels, values, color=colors[:len(labels)], alpha=0.8)
        
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_title('Ablation: Fixed vs Curriculum Weighting', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{val:.2f}%',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/ablation_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('figures/ablation_comparison.pdf', bbox_inches='tight')
        plt.close()
        print("Saved figures/ablation_comparison.png and .pdf")
    
    # Figure 3: Training curves comparison
    supcon_files = load_results('results/cifar100_supcon_seed*.json')
    caghnm_files = load_results('results/cifar100_caghnm_seed*.json')
    
    if supcon_files and caghnm_files:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Average SupCon curves
        supcon_losses = [f["contrastive_losses"] for f in supcon_files if "contrastive_losses" in f]
        if supcon_losses:
            min_len = min(len(l) for l in supcon_losses)
            supcon_avg = np.mean([l[:min_len] for l in supcon_losses], axis=0)
            ax.plot(supcon_avg, label='SupCon', linewidth=2)
        
        # Average CAG-HNM curves
        caghnm_losses = [f["contrastive_losses"] for f in caghnm_files if "contrastive_losses" in f]
        if caghnm_losses:
            min_len = min(len(l) for l in caghnm_losses)
            caghnm_avg = np.mean([l[:min_len] for l in caghnm_losses], axis=0)
            ax.plot(caghnm_avg, label='CAG-HNM (Ours)', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Contrastive Loss', fontsize=12)
        ax.set_title('Training Curves: CAG-HNM vs SupCon', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig('figures/training_curves.pdf', bbox_inches='tight')
        plt.close()
        print("Saved figures/training_curves.png and .pdf")
    
    # Figure 4: Curriculum progression
    caghnm_files = load_results('results/cifar100_caghnm_seed*.json')
    if caghnm_files and "curriculum_values" in caghnm_files[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        curriculum_values = caghnm_files[0]["curriculum_values"]
        epochs = list(range(len(curriculum_values)))
        
        ax.plot(epochs, curriculum_values, linewidth=2, color='#d62728')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('λ(t) - Curriculum Parameter', fontsize=12)
        ax.set_title('Curriculum Progression: λ(t) over Training', fontsize=14)
        ax.grid(alpha=0.3)
        
        # Add annotations
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='λ_min=0.1')
        ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='λ_max=2.0')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/curriculum_progression.png', dpi=300, bbox_inches='tight')
        plt.savefig('figures/curriculum_progression.pdf', bbox_inches='tight')
        plt.close()
        print("Saved figures/curriculum_progression.png and .pdf")


def print_summary(aggregated):
    """Print summary of results."""
    print("\n" + "="*70)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*70)
    
    print("\n1. METHOD COMPARISON (CIFAR-100)")
    print("-"*70)
    print(f"{'Method':<20} {'Accuracy (%)':<20} {'Runtime (min)':<20}")
    print("-"*70)
    
    for method in ['CrossEntropy', 'SupCon', 'JD-CCL-Fixed', 'CAG-HNM']:
        exp = aggregated["experiments"].get(method)
        if exp:
            acc_str = f"{exp['mean']:.2f} ± {exp['std']:.2f}"
            runtime_str = f"{exp['runtime']['mean']:.1f} ± {exp['runtime']['std']:.1f}"
            print(f"{method:<20} {acc_str:<20} {runtime_str:<20}")
    
    print("\n2. STATISTICAL COMPARISONS")
    print("-"*70)
    for name, comp in aggregated["comparisons"].items():
        if comp:
            sig_marker = "***" if comp["significant"] else ""
            print(f"{name}:")
            print(f"  t-statistic: {comp['t_statistic']:.3f}")
            print(f"  p-value: {comp['p_value']:.4f} {sig_marker}")
    
    print("\n3. ABLATION RESULTS")
    print("-"*70)
    if "ablation" in aggregated:
        ablation = aggregated["ablation"]
        for result in ablation.get("fixed_results", []):
            print(f"Fixed (λ={result['lambda']}): {result['accuracy']:.2f}%")
        print(f"Curriculum (Ours): {ablation.get('curriculum_accuracy', 0):.2f}%")
    
    print("\n4. SUCCESS CRITERIA")
    print("-"*70)
    for criterion, met in aggregated["success_criteria"].items():
        status = "✓ MET" if met else "✗ NOT MET"
        print(f"{criterion}: {status}")
    
    overall = "✓ ALL CRITERIA MET" if aggregated["all_criteria_met"] else "✗ SOME CRITERIA NOT MET"
    print(f"\nOverall: {overall}")


def main():
    print("Aggregating results...")
    
    # Aggregate results
    aggregated = aggregate_experiment_results()
    
    # Save aggregated results
    with open('results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    print("\nSaved aggregated results to results.json")
    
    # Print summary
    print_summary(aggregated)
    
    # Generate figures
    print("\nGenerating figures...")
    generate_figures(aggregated)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
