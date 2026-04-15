"""
Analyze all experimental results and generate comprehensive report.
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def analyze_discriminative_power(vlm_results):
    """Analyze accuracy across difficulty levels."""
    levels = [1, 2, 3, 4]
    accuracies = []
    
    for level in levels:
        key = f"level{level}_existential_n500_s{100+level-1}"
        if key in vlm_results["datasets"]:
            acc = vlm_results["datasets"][key]["accuracy"]
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    gap = accuracies[0] - accuracies[-1] if len(accuracies) >= 4 else 0
    
    return {
        "levels": levels,
        "accuracies": accuracies,
        "gap_l1_l4": gap,
        "hypothesis_met": gap > 0.30
    }


def analyze_nested_quantification(vlm_results):
    """Analyze nested quantifier performance."""
    # Map dataset keys to quantifier types
    simple_key = "level2_nested_quant_n400_s401"  # Depth 2 = simpler
    nested_key = "level3_nested_quant_n400_s400"  # Depth 3 = nested
    nested_l4_key = "level4_nested_quant_n400_s402"
    
    results = {}
    
    if simple_key in vlm_results["datasets"]:
        results["simple_depth2"] = vlm_results["datasets"][simple_key]["accuracy"]
    if nested_key in vlm_results["datasets"]:
        results["nested_depth3"] = vlm_results["datasets"][nested_key]["accuracy"]
    if nested_l4_key in vlm_results["datasets"]:
        results["nested_depth4"] = vlm_results["datasets"][nested_l4_key]["accuracy"]
    
    if "simple_depth2" in results and "nested_depth3" in results:
        results["accuracy_drop"] = results["simple_depth2"] - results["nested_depth3"]
        results["hypothesis_met"] = results["accuracy_drop"] > 0.25
    
    return results


def analyze_transitive_scaling(vlm_results):
    """Analyze transitive relation accuracy vs chain length."""
    results = {}
    
    for level in [2, 3, 4]:
        key = f"level{level}_transitive_n300_s{300+level-2}"
        if key in vlm_results["datasets"]:
            chain_length = level  # Level maps to chain length for transitive
            results[f"length_{chain_length}"] = vlm_results["datasets"][key]["accuracy"]
    
    # Check if accuracy degrades with chain length
    lengths = sorted([int(k.split("_")[1]) for k in results.keys() if k.startswith("length_")])
    accs = [results[f"length_{l}"] for l in lengths]
    
    if len(accs) >= 2:
        results["degrades_with_length"] = accs[-1] < accs[0]
    
    return results


def create_discriminative_plot(disc_results, output_path):
    """Create accuracy by difficulty plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    levels = disc_results["levels"]
    accs = disc_results["accuracies"]
    
    ax.plot(levels, accs, marker='o', linewidth=2, markersize=10, label='Qwen2-VL-2B')
    
    # Add baseline
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    
    ax.set_xlabel('Difficulty Level', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Across Difficulty Levels', fontsize=14)
    ax.set_xticks(levels)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def create_quantifier_plot(nested_results, output_path):
    """Create nested quantifier performance plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    values = []
    colors = []
    
    if "simple_depth2" in nested_results:
        labels.append("Simple (Depth 2)")
        values.append(nested_results["simple_depth2"])
        colors.append('green')
    
    if "nested_depth3" in nested_results:
        labels.append("Nested (Depth 3)")
        values.append(nested_results["nested_depth3"])
        colors.append('orange')
    
    if "nested_depth4" in nested_results:
        labels.append("Nested (Depth 4)")
        values.append(nested_results["nested_depth4"])
        colors.append('red')
    
    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance on Nested Quantifiers', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def create_transitive_plot(trans_results, output_path):
    """Create transitive relation scaling plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lengths = sorted([int(k.split("_")[1]) for k in trans_results.keys() if k.startswith("length_")])
    accs = [trans_results[f"length_{l}"] for l in lengths]
    
    ax.plot(lengths, accs, marker='s', linewidth=2, markersize=10, label='Qwen2-VL-2B')
    
    ax.set_xlabel('Relation Chain Length', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Transitive Relation Accuracy vs Chain Length', fontsize=14)
    ax.set_xticks(lengths)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def create_comparison_plot(vlm_results, output_path):
    """Create query type comparison plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Aggregate results by query type
    query_types = {
        "Existential": ["level1_existential_n500_s100", "level2_existential_n500_s101", 
                       "level3_existential_n500_s102", "level4_existential_n500_s103"],
        "Universal": ["level1_universal_n500_s200"],
        "Comparative": ["level1_comparative_n500_s201"],
        "Transitive": ["level2_transitive_n300_s300", "level3_transitive_n300_s301", 
                      "level4_transitive_n300_s302"],
        "Nested Quant": ["level2_nested_quant_n400_s401", "level3_nested_quant_n400_s400",
                        "level4_nested_quant_n400_s402"]
    }
    
    type_accs = {}
    for qtype, keys in query_types.items():
        accs = []
        for key in keys:
            if key in vlm_results["datasets"]:
                accs.append(vlm_results["datasets"][key]["accuracy"])
        if accs:
            type_accs[qtype] = np.mean(accs)
    
    labels = list(type_accs.keys())
    values = list(type_accs.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Performance by Query Type', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    # Load results
    vlm_results = load_results("../../results/vlm_qwen2b_full.json")
    speed_results = load_results("../../results/speed_validation.json")
    
    # Analyze
    disc_analysis = analyze_discriminative_power(vlm_results)
    nested_analysis = analyze_nested_quantification(vlm_results)
    trans_analysis = analyze_transitive_scaling(vlm_results)
    
    # Print summary
    print("=" * 60)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 60)
    
    print("\n1. DISCRIMINATIVE POWER (RQ2)")
    print(f"   Level accuracies: {disc_analysis['accuracies']}")
    print(f"   L1-L4 Gap: {disc_analysis['gap_l1_l4']:.3f}")
    print(f"   Hypothesis (>30% gap): {'PASS' if disc_analysis['hypothesis_met'] else 'FAIL'}")
    
    print("\n2. NESTED QUANTIFICATION (RQ3)")
    for k, v in nested_analysis.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.3f}")
        else:
            print(f"   {k}: {v}")
    
    print("\n3. TRANSITIVE RELATIONS (RQ4)")
    for k, v in trans_analysis.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.3f}")
        else:
            print(f"   {k}: {v}")
    
    print("\n4. SPEED VALIDATION (RQ1)")
    print(f"   All targets met: {speed_results['overall']['all_targets_met']}")
    print(f"   Max mean time: {speed_results['overall']['max_mean_ms']:.2f}ms")
    print(f"   Max std: {speed_results['overall']['max_std_ms']:.2f}ms")
    
    # Create plots
    print("\nGenerating figures...")
    create_discriminative_plot(disc_analysis, "../../figures/accuracy_by_difficulty.png")
    create_quantifier_plot(nested_analysis, "../../figures/nested_quantifier_performance.png")
    create_transitive_plot(trans_analysis, "../../figures/transitive_accuracy.png")
    create_comparison_plot(vlm_results, "../../figures/query_type_comparison.png")
    
    # Save comprehensive results
    comprehensive = {
        "vlm_results": vlm_results,
        "speed_validation": speed_results,
        "analyses": {
            "discriminative_power": disc_analysis,
            "nested_quantification": nested_analysis,
            "transitive_scaling": trans_analysis
        }
    }
    
    with open("../../results/comprehensive_results.json", 'w') as f:
        json.dump(comprehensive, f, indent=2)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
