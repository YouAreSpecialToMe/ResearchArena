"""
Generate publication-quality figures for the paper.
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results():
    """Load experiment results."""
    results = {}
    
    import glob
    result_files = glob.glob(os.path.join(PROJECT_ROOT, 'results/synthetic/*/results.json'))
    
    for result_file in result_files:
        exp_name = os.path.basename(os.path.dirname(result_file))
        with open(result_file, 'r') as f:
            results[exp_name] = json.load(f)
    
    return results


def plot_sample_efficiency(results, output_dir):
    """Figure 1: Sample Efficiency - SHD vs N for different methods."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    mechanisms = ['linear_gaussian', 'linear_nongaussian', 'nonlinear', 'anm']
    mechanism_names = ['Linear Gaussian', 'Linear Non-Gaussian', 'Nonlinear', 'ANM']
    
    for idx, (mechanism, mech_name) in enumerate(zip(mechanisms, mechanism_names)):
        ax = axes[idx]
        
        # Collect data for each method
        method_data = {'spiced': {}, 'notears': {}}
        
        for method in ['spiced', 'notears']:
            method_results = results.get(method, [])
            
            for r in method_results:
                if r.get('status') != 'success':
                    continue
                if r['mechanism'] != mechanism:
                    continue
                if r['n_nodes'] not in [10, 20]:  # Focus on smaller graphs for clarity
                    continue
                
                n = r['n_samples']
                if n not in method_data[method]:
                    method_data[method][n] = []
                method_data[method][n].append(r['shd'])
        
        # Plot lines
        for method, color, label in [('spiced', 'blue', 'SPICED'), ('notears', 'red', 'NOTEARS')]:
            if not method_data[method]:
                continue
            
            n_values = sorted(method_data[method].keys())
            means = [np.mean(method_data[method][n]) for n in n_values]
            stds = [np.std(method_data[method][n]) for n in n_values]
            
            ax.plot(n_values, means, 'o-', color=color, label=label, linewidth=2, markersize=8)
            ax.fill_between(n_values, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           color=color, alpha=0.2)
        
        ax.set_xlabel('Sample Size (N)', fontsize=12)
        ax.set_ylabel('SHD', fontsize=12)
        ax.set_title(mech_name, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure1_sample_efficiency.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'figure1_sample_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Figure 1: Sample Efficiency")


def plot_ablation(results, output_dir):
    """Figure 2: Ablation Study Results."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = ['spiced', 'spiced_no_constraints', 'spiced_random_init']
    method_names = ['Full SPICED', 'No Constraints', 'Random Init']
    colors = ['blue', 'orange', 'green']
    
    # Group by sample size
    n_values = [50, 100, 200]
    
    x = np.arange(len(n_values))
    width = 0.25
    
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        method_results = results.get(method, [])
        
        means = []
        stds = []
        
        for n in n_values:
            shds = [r['shd'] for r in method_results 
                   if r.get('status') == 'success' and r['n_samples'] == n]
            if shds:
                means.append(np.mean(shds))
                stds.append(np.std(shds))
            else:
                means.append(0)
                stds.append(0)
        
        ax.bar(x + i * width, means, width, yerr=stds, label=name, color=color, alpha=0.8, capsize=5)
    
    ax.set_xlabel('Sample Size (N)', fontsize=12)
    ax.set_ylabel('SHD', fontsize=12)
    ax.set_title('Ablation Study: Impact of Design Choices', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(n_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure2_ablation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'figure2_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Figure 2: Ablation Study")


def plot_scalability(results, output_dir):
    """Figure 3: Scalability - Runtime vs Number of Nodes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    method_data = {'spiced': {}, 'notears': {}}
    
    for method in ['spiced', 'notears']:
        method_results = results.get(method, [])
        
        for r in method_results:
            if r.get('status') != 'success':
                continue
            if r.get('runtime') is None:
                continue
            
            n_nodes = r['n_nodes']
            if n_nodes not in method_data[method]:
                method_data[method][n_nodes] = []
            method_data[method][n_nodes].append(r['runtime'])
    
    for method, color, label in [('spiced', 'blue', 'SPICED'), ('notears', 'red', 'NOTEARS')]:
        if not method_data[method]:
            continue
        
        n_values = sorted(method_data[method].keys())
        means = [np.mean(method_data[method][n]) for n in n_values]
        stds = [np.std(method_data[method][n]) for n in n_values]
        
        ax.errorbar(n_values, means, yerr=stds, fmt='o-', color=color, 
                   label=label, linewidth=2, markersize=8, capsize=5)
    
    ax.axhline(y=300, color='gray', linestyle='--', label='5 minute threshold')
    
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Scalability Analysis', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure3_scalability.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'figure3_scalability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Figure 3: Scalability")


def generate_summary_table(results, output_dir):
    """Generate LaTeX summary table."""
    
    # Group results
    table_data = []
    
    mechanisms = ['linear_gaussian', 'linear_nongaussian', 'nonlinear', 'anm']
    n_values = [50, 100, 200, 500]
    
    for mechanism in mechanisms:
        for n in n_values:
            row = {'mechanism': mechanism, 'n': n}
            
            for method in ['spiced', 'notears']:
                method_results = results.get(method, [])
                
                shds = [r['shd'] for r in method_results 
                       if r.get('status') == 'success' 
                       and r['mechanism'] == mechanism
                       and r['n_samples'] == n
                       and r['n_nodes'] == 10]  # Focus on n=10
                
                if shds:
                    row[f'{method}_shd'] = f"{np.mean(shds):.2f} ± {np.std(shds):.2f}"
                else:
                    row[f'{method}_shd'] = "N/A"
            
            table_data.append(row)
    
    # Generate LaTeX
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += "\\caption{Comparison of SPICED vs NOTEARS on synthetic data (n=10)}\n"
    latex += "\\begin{tabular}{llcc}\n"
    latex += "\\hline\n"
    latex += "Mechanism & N & SPICED SHD & NOTEARS SHD \\\\\n"
    latex += "\\hline\n"
    
    for row in table_data:
        mech = row['mechanism'].replace('_', ' ').title()
        latex += f"{mech} & {row['n']} & {row['spiced_shd']} & {row['notears_shd']} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    with open(os.path.join(output_dir, 'table1_summary.tex'), 'w') as f:
        f.write(latex)
    
    print("Saved Table 1: Summary (LaTeX)")


def main():
    """Generate all figures and tables."""
    print("="*60)
    print("Generating Figures and Tables")
    print("="*60)
    
    # Load results
    results = load_results()
    print(f"\nLoaded results from {len(results)} experiments")
    
    # Create output directory
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_sample_efficiency(results, figures_dir)
    plot_ablation(results, figures_dir)
    plot_scalability(results, figures_dir)
    
    # Generate tables
    print("\nGenerating tables...")
    generate_summary_table(results, figures_dir)
    
    print("\n" + "="*60)
    print("All figures and tables generated!")
    print(f"Output directory: {figures_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
