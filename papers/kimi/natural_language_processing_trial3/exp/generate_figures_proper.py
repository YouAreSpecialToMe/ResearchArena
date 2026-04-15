"""
Generate publication-ready figures for the ESR paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10


def load_aggregate_results():
    """Load aggregate results from experiments."""
    results_dir = Path("exp/results")
    
    # Load all individual result files
    all_data = {}
    for json_file in results_dir.glob("*_gsm8k_seed*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            method = data.get("method")
            if method not in all_data:
                all_data[method] = {
                    "accuracies": [],
                    "tokens": [],
                    "seeds": []
                }
            
            all_data[method]["accuracies"].append(data.get("accuracy", 0))
            all_data[method]["tokens"].append(data.get("avg_tokens", 0))
            all_data[method]["seeds"].append(data.get("seed"))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_data


def plot_accuracy_comparison(data, output_dir="figures"):
    """Figure 1: Bar chart of accuracy by method with error bars."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    methods = []
    means = []
    stds = []
    
    for method, values in sorted(data.items()):
        if values["accuracies"]:
            methods.append(method.replace("_", " ").title())
            means.append(np.mean(values["accuracies"]))
            stds.append(np.std(values["accuracies"]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(methods))
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                   color=sns.color_palette("husl", len(methods)),
                   edgecolor='black', linewidth=1)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison on GSM8K (Mean ± Std across Seeds)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/accuracy_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/accuracy_comparison.pdf")


def plot_accuracy_vs_tokens(data, output_dir="figures"):
    """Figure 2: Accuracy vs Tokens scatter plot (efficiency frontier)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(data))
    
    for i, (method, values) in enumerate(sorted(data.items())):
        if values["accuracies"] and values["tokens"]:
            acc_mean = np.mean(values["accuracies"])
            acc_std = np.std(values["accuracies"])
            tok_mean = np.mean(values["tokens"])
            tok_std = np.std(values["tokens"])
            
            ax.scatter(tok_mean, acc_mean, s=200, c=[colors[i]], 
                      label=method.replace("_", " ").title(),
                      edgecolors='black', linewidth=1.5, zorder=5)
            
            # Add error bars
            ax.errorbar(tok_mean, acc_mean, xerr=tok_std, yerr=acc_std,
                       fmt='none', c=colors[i], capsize=4, alpha=0.5)
    
    ax.set_xlabel("Average Tokens per Problem")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Efficiency Trade-off")
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_vs_tokens.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/accuracy_vs_tokens.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/accuracy_vs_tokens.pdf")


def plot_efficiency_metric(data, output_dir="figures"):
    """Figure 3: Efficiency metric (Accuracy per 1K tokens)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    methods = []
    efficiency = []
    
    for method, values in sorted(data.items()):
        if values["accuracies"] and values["tokens"]:
            acc = np.mean(values["accuracies"])
            tok = np.mean(values["tokens"])
            methods.append(method.replace("_", " ").title())
            efficiency.append(acc / (tok / 1000))  # Acc per 1K tokens
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(methods))
    
    colors = sns.color_palette("viridis", len(methods))
    bars = ax.bar(x_pos, efficiency, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy per 1K Tokens")
    ax.set_title("Efficiency: Accuracy per 1K Tokens Generated")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_metric.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/efficiency_metric.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/efficiency_metric.pdf")


def create_results_table(data, output_dir="figures"):
    """Create a LaTeX/PDF table of results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    rows = []
    rows.append("Method & Accuracy & Tokens & Efficiency (Acc/1K) \\\\")
    rows.append("\\hline")
    
    for method, values in sorted(data.items()):
        if values["accuracies"] and values["tokens"]:
            acc_mean = np.mean(values["accuracies"])
            acc_std = np.std(values["accuracies"])
            tok_mean = np.mean(values["tokens"])
            tok_std = np.std(values["tokens"])
            efficiency = acc_mean / (tok_mean / 1000)
            
            method_name = method.replace("_", " ").title()
            row = f"{method_name} & ${acc_mean:.3f} \\pm {acc_std:.3f}$ & ${tok_mean:.0f} \\pm {tok_std:.0f}$ & ${efficiency:.3f}$ \\\\"
            rows.append(row)
    
    # Save as text
    with open(f"{output_dir}/results_table.tex", 'w') as f:
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(row + "\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    
    print(f"Saved: {output_dir}/results_table.tex")


def main():
    print("Loading experimental results...")
    data = load_aggregate_results()
    
    if not data:
        print("No results found. Run experiments first.")
        return
    
    print(f"Found results for methods: {list(data.keys())}")
    
    print("\nGenerating figures...")
    plot_accuracy_comparison(data)
    plot_accuracy_vs_tokens(data)
    plot_efficiency_metric(data)
    create_results_table(data)
    
    print("\nAll figures generated successfully!")
    print("Output directory: figures/")


if __name__ == "__main__":
    main()
