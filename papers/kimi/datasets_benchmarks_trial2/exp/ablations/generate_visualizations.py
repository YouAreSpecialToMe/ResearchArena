"""Generate figures for the paper."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from shared.utils import load_json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_all_results():
    """Load all experimental results."""
    baselines = load_json("outputs/baselines/random_heuristic_results.json")
    
    main_results = {}
    import glob
    main_files = glob.glob("outputs/main_results/*_summary.json")
    for f in main_files:
        key = os.path.basename(f).replace("_summary.json", "")
        main_results[key] = load_json(f)
    
    return baselines, main_results


def figure_main_results(baselines: Dict, main_results: Dict, output_dir: str):
    """Figure 1: Main results comparison across models and tasks."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    tasks = ["detection", "localization", "characterization"]
    task_names = ["Error Detection", "Error Localization", "Error Characterization"]
    metric_keys = ["accuracy", "exact_match", "accuracy"]
    
    for ax, task, task_name, metric_key in zip(axes, tasks, task_names, metric_keys):
        models = []
        means = []
        stds = []
        
        # Add baselines
        for bl_name, bl_data in baselines.items():
            if task in bl_data:
                val = bl_data[task].get(metric_key, 0)
                models.append(bl_name.replace("_", " ").title())
                means.append(val)
                stds.append(0)
        
        # Add main results
        for model_name, model_data in main_results.items():
            if task in model_data:
                mean_key = f"{task}_{metric_key}_mean" if f"{task}_{metric_key}_mean" in model_data else None
                if mean_key:
                    models.append(model_name.replace("_", " ").title())
                    means.append(model_data[mean_key])
                    stds.append(model_data.get(f"{task}_{metric_key}_std", 0))
                elif isinstance(model_data[task], dict) and "mean" in str(model_data[task]):
                    models.append(model_name.replace("_", " ").title())
                    means.append(model_data[task].get(metric_key, {}).get("mean", 0))
                    stds.append(model_data[task].get(metric_key, {}).get("std", 0))
        
        x = np.arange(len(models))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title(task_name)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_main_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure_main_results.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved figure_main_results")


def figure_domain_heatmap(main_results: Dict, output_dir: str):
    """Figure 2: Domain performance heatmap."""
    
    # Extract domain accuracies
    domains = ["math", "logic", "commonsense", "code"]
    model_names = []
    domain_matrix = []
    
    for model_name, model_data in main_results.items():
        if "domain_metrics" in model_data:
            model_names.append(model_name.replace("_", " ").title())
            row = []
            for domain in domains:
                dm = model_data["domain_metrics"].get(domain, {})
                if isinstance(dm.get("accuracy"), dict):
                    acc = dm["accuracy"].get("mean", 0)
                else:
                    acc = dm.get("accuracy", 0)
                row.append(acc)
            domain_matrix.append(row)
    
    if domain_matrix:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(domain_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(domains)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels([d.title() for d in domains])
        ax.set_yticklabels(model_names)
        
        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(domains)):
                text = ax.text(j, i, f'{domain_matrix[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title("Detection Accuracy by Domain")
        plt.colorbar(im, ax=ax, label='Accuracy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figure_domain_heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/figure_domain_heatmap.pdf", bbox_inches='tight')
        plt.close()
        print(f"Saved figure_domain_heatmap")


def figure_position_analysis(output_dir: str):
    """Figure 3: Position vs accuracy analysis."""
    
    try:
        position_data = load_json("outputs/analysis/position_ablation.json")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        positions = ["early", "middle", "late"]
        x = np.arange(len(positions))
        
        for model_name, data in position_data.items():
            loc_accs = [data["position_metrics"][p]["localization_accuracy"] for p in positions]
            ax.plot(x, loc_accs, marker='o', label=model_name.replace("_", " ").title(), linewidth=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels([p.title() for p in positions])
        ax.set_xlabel("Error Position")
        ax.set_ylabel("Localization Accuracy")
        ax.set_title("Localization Accuracy vs Error Position")
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figure_position_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/figure_position_analysis.pdf", bbox_inches='tight')
        plt.close()
        print(f"Saved figure_position_analysis")
    except FileNotFoundError:
        print("Position analysis data not found, skipping figure")


def figure_error_type_sensitivity(output_dir: str):
    """Figure 4: Error type sensitivity."""
    
    try:
        type_data = load_json("outputs/analysis/error_type_analysis.json")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        error_types = ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
        x = np.arange(len(error_types))
        width = 0.35
        
        for i, (model_name, data) in enumerate(list(type_data.items())[:2]):
            type_metrics = data.get("per_type_metrics", {})
            scores = [type_metrics.get(et, {}).get("average", 0) for et in error_types]
            offset = width * (i - 0.5)
            ax.bar(x + offset, scores, width, label=model_name.replace("_", " ").title(), alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([et.title() for et in error_types], rotation=45, ha='right')
        ax.set_ylabel("Average Score")
        ax.set_title("Performance by Error Type")
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figure_error_type_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/figure_error_type_sensitivity.pdf", bbox_inches='tight')
        plt.close()
        print(f"Saved figure_error_type_sensitivity")
    except FileNotFoundError:
        print("Error type analysis data not found, skipping figure")


def figure_introspection_scores(main_results: Dict, output_dir: str):
    """Figure 5: Introspection Score comparison."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    is_means = []
    is_stds = []
    
    for model_name, model_data in main_results.items():
        if "introspection_score" in model_data:
            models.append(model_name.replace("_", " ").title())
            if isinstance(model_data["introspection_score"], dict):
                is_means.append(model_data["introspection_score"].get("mean", 0))
                is_stds.append(model_data["introspection_score"].get("std", 0))
            else:
                is_means.append(model_data["introspection_score"])
                is_stds.append(0)
    
    if models:
        x = np.arange(len(models))
        ax.bar(x, is_means, yerr=is_stds, capsize=5, alpha=0.7, color='teal')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel("Introspection Score")
        ax.set_title("Model Comparison: Introspection Score")
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.33, color='r', linestyle='--', alpha=0.5, label='Random baseline')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figure_introspection_scores.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/figure_introspection_scores.pdf", bbox_inches='tight')
        plt.close()
        print(f"Saved figure_introspection_scores")


def main():
    os.makedirs("figures", exist_ok=True)
    
    print("Loading results...")
    baselines, main_results = load_all_results()
    
    print("\nGenerating figures...")
    figure_main_results(baselines, main_results, "figures")
    figure_domain_heatmap(main_results, "figures")
    figure_position_analysis("figures")
    figure_error_type_sensitivity("figures")
    figure_introspection_scores(main_results, "figures")
    
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
