"""
Generate figures for the paper.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Color scheme
COLORS = {
    'spiced': '#1f77b4',      # Blue
    'notears': '#d62728',     # Red
    'golem': '#2ca02c',       # Green
    'pc': '#ff7f0e',          # Orange
}


def load_results():
    """Load aggregated results."""
    with open(os.path.join(PROJECT_ROOT, "results.json"), 'r') as f:
        return json.load(f)


def plot_sample_efficiency(results):
    """Figure 1: Sample Efficiency - SHD vs N."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['spiced', 'notears', 'golem', 'pc']
    sample_sizes = [50, 100, 200, 500, 1000]
    
    for ax_idx, graph_type in enumerate(['ER', 'SF']):
        ax = axes[ax_idx]
        
        for method in methods:
            if method not in results['by_configuration']:
                continue
            
            means = []
            stds = []
            valid_n = []
            
            for n in sample_sizes:
                # Aggregate across mechanisms and n_nodes
                shds = []
                for config_str, data in results['by_configuration'][method].items():
                    config = eval(config_str) if config_str.startswith('(') else config_str
                    if isinstance(config, tuple) and len(config) == 3:
                        n_nodes, mechanism, n_samples = config
                        if n_samples == n:
                            shds.append(data['shd_mean'])
                
                if shds:
                    means.append(np.mean(shds))
                    stds.append(np.std(shds))
                    valid_n.append(n)
            
            if means:
                ax.plot(valid_n, means, 'o-', label=method.upper(), 
                       color=COLORS.get(method, 'gray'), linewidth=2, markersize=8)
                ax.fill_between(valid_n, 
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.2, color=COLORS.get(method, 'gray'))
        
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Structural Hamming Distance (SHD)')
        ax.set_title(f'{graph_type} Graphs')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/sample_efficiency.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/sample_efficiency.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved sample_efficiency figure")


def plot_scalability(results):
    """Figure 2: Scalability - Runtime vs n_nodes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['spiced', 'notears', 'golem', 'pc']
    
    for method in methods:
        if method not in results['by_configuration']:
            continue
        
        # Group by n_nodes
        node_sizes = {}
        for config_str, data in results['by_configuration'][method].items():
            config = eval(config_str) if config_str.startswith('(') else config_str
            if isinstance(config, tuple) and len(config) == 3:
                n_nodes = config[0]
                if n_nodes not in node_sizes:
                    node_sizes[n_nodes] = []
                if data.get('runtime_mean'):
                    node_sizes[n_nodes].append(data['runtime_mean'])
        
        if node_sizes:
            x = sorted(node_sizes.keys())
            y = [np.mean(node_sizes[n]) for n in x]
            yerr = [np.std(node_sizes[n]) if len(node_sizes[n]) > 1 else 0 for n in x]
            
            ax.errorbar(x, y, yerr=yerr, fmt='o-', label=method.upper(),
                       color=COLORS.get(method, 'gray'), linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Scalability Analysis')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/scalability.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/scalability.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved scalability figure")


def plot_ablation_results():
    """Figure 3: Ablation study results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Ablation 1: Initialization
    init_file = os.path.join(PROJECT_ROOT, "results/ablations/initialization.json")
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            init_results = json.load(f)
        
        it_shds = [r['shd'] for r in init_results if r['init_method'] == 'IT']
        rand_shds = [r['shd'] for r in init_results if r['init_method'] == 'random']
        
        axes[0].boxplot([it_shds, rand_shds], labels=['IT Init', 'Random Init'])
        axes[0].set_ylabel('SHD')
        axes[0].set_title('Effect of IT Initialization')
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # Ablation 2: Structural Constraints
    cons_file = os.path.join(PROJECT_ROOT, "results/ablations/structural_constraints.json")
    if os.path.exists(cons_file):
        with open(cons_file, 'r') as f:
            cons_results = json.load(f)
        
        with_shds = [r['shd'] for r in cons_results if r['use_constraints']]
        without_shds = [r['shd'] for r in cons_results if not r['use_constraints']]
        
        axes[1].boxplot([with_shds, without_shds], labels=['With Constraints', 'Without'])
        axes[1].set_ylabel('SHD')
        axes[1].set_title('Effect of Structural Constraints')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/ablation_study.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/ablation_study.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ablation_study figure")


def plot_sachs_comparison(results):
    """Figure 4: Sachs dataset comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['pc', 'notears', 'golem', 'spiced']
    method_names = []
    shd_means = []
    shd_stds = []
    
    for method in methods:
        if method in results.get('sachs_results', {}):
            shds = [r['shd'] for r in results['sachs_results'][method] if r.get('shd') is not None]
            if shds:
                method_names.append(method.upper())
                shd_means.append(np.mean(shds))
                shd_stds.append(np.std(shds))
    
    if method_names:
        x = np.arange(len(method_names))
        bars = ax.bar(x, shd_means, yerr=shd_stds, capsize=5,
                     color=[COLORS.get(m.lower(), 'gray') for m in method_names],
                     alpha=0.8, edgecolor='black')
        
        # Add threshold line
        ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Target (SHD < 10)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
        ax.set_ylabel('Structural Hamming Distance (SHD)')
        ax.set_title('Sachs Dataset Results')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/sachs_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/sachs_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved sachs_comparison figure")


def main():
    """Generate all figures."""
    os.makedirs(os.path.join(PROJECT_ROOT, "figures"), exist_ok=True)
    
    try:
        results = load_results()
        
        print("Generating figures...")
        plot_sample_efficiency(results)
        plot_scalability(results)
        plot_sachs_comparison(results)
        plot_ablation_results()
        
        print("\nAll figures generated successfully!")
    except FileNotFoundError as e:
        print(f"Results file not found: {e}")
        print("Please run aggregate_results.py first.")
    except Exception as e:
        print(f"Error generating figures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
