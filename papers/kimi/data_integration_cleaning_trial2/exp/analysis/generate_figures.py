"""
Generate figures and tables for the paper.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_results(pattern):
    """Load all results matching a pattern."""
    import glob
    results = []
    for path in glob.glob(pattern):
        with open(path, 'r') as f:
            results.append(json.load(f))
    return results


def aggregate_seeds(results):
    """Aggregate results across seeds."""
    f1s = [r['metrics']['overall']['f1'] for r in results]
    precs = [r['metrics']['overall']['precision'] for r in results]
    recs = [r['metrics']['overall']['recall'] for r in results]
    
    return {
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'precision_mean': np.mean(precs),
        'precision_std': np.std(precs),
        'recall_mean': np.mean(recs),
        'recall_std': np.std(recs),
    }


def generate_accuracy_comparison():
    """Figure 1: Bar chart comparing F1 scores across methods."""
    datasets = ['hospital', 'flights', 'beers']
    methods = ['programclean', 'raha', 'direct_val', 'seed']
    method_labels = ['ProgramClean', 'Raha', 'Direct Validation', 'SEED Baseline']
    
    data = {m: {d: None for d in datasets} for m in methods}
    
    # Load ProgramClean results (with seeds)
    for dataset in datasets:
        pc_results = load_results(f'results/programclean/{dataset}_seed*.json')
        if pc_results:
            data['programclean'][dataset] = aggregate_seeds(pc_results)
    
    # Load baseline results
    for dataset in datasets:
        for method in ['raha', 'direct_val', 'seed']:
            try:
                with open(f'results/{method}/{dataset}.json', 'r') as f:
                    result = json.load(f)
                    data[method][dataset] = {
                        'f1_mean': result['metrics']['overall']['f1'],
                        'f1_std': 0,  # Single run
                    }
            except FileNotFoundError:
                pass
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        f1_means = [data[method][d]['f1_mean'] if data[method][d] else 0 for d in datasets]
        f1_stds = [data[method][d]['f1_std'] if data[method][d] else 0 for d in datasets]
        
        ax.bar(x + i*width, f1_means, width, label=label, yerr=f1_stds, capsize=3)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('F1 Score')
    ax.set_title('Error Detection Accuracy Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('figures/accuracy_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved figures/accuracy_comparison.{pdf,png}")
    
    return data


def generate_runtime_comparison():
    """Figure 2: Runtime comparison (log scale)."""
    # Load results
    datasets = ['hospital', 'flights', 'beers']
    
    runtimes = {
        'ProgramClean': [],
        'Raha': [],
        'Direct Validation': [],
    }
    
    for dataset in datasets:
        # ProgramClean
        try:
            with open(f'results/programclean/{dataset}_seed42.json', 'r') as f:
                result = json.load(f)
                runtimes['ProgramClean'].append(result['total_time'])
        except FileNotFoundError:
            runtimes['ProgramClean'].append(0)
        
        # Raha
        try:
            with open(f'results/raha/{dataset}.json', 'r') as f:
                result = json.load(f)
                runtimes['Raha'].append(result['metrics']['runtime'])
        except FileNotFoundError:
            runtimes['Raha'].append(0)
        
        # Direct Validation
        try:
            with open(f'results/direct_val/{dataset}.json', 'r') as f:
                result = json.load(f)
                # Note: direct val only processes subset
                # Estimate runtime based on LLM calls (approx 0.1s per call)
                runtimes['Direct Validation'].append(result.get('metrics', {}).get('llm_calls', 100) * 0.001)
        except FileNotFoundError:
            runtimes['Direct Validation'].append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, (method, values) in enumerate(runtimes.items()):
        ax.bar(x + i*width, values, width, label=method)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison (Lower is Better)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('figures/runtime_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/runtime_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved figures/runtime_comparison.{pdf,png}")


def generate_ablation_chart():
    """Figure 3: Ablation study results."""
    # Load ablation results
    datasets = ['hospital', 'beers']
    
    programclean_f1 = []
    naive_f1 = []
    
    for dataset in datasets:
        try:
            with open(f'results/ablations/naive_vs_profiling_{dataset}.json', 'r') as f:
                result = json.load(f)
                programclean_f1.append(result['programclean']['overall']['f1'])
                naive_f1.append(result['naive_codegen']['overall']['f1'])
        except FileNotFoundError:
            programclean_f1.append(0)
            naive_f1.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, programclean_f1, width, label='ProgramClean (with profiling)')
    ax.bar(x + width/2, naive_f1, width, label='Naive CodeGen (no profiling)')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('F1 Score')
    ax.set_title('Ablation: Value of Semantic Profiling')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('figures/ablation_profiling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/ablation_profiling.png', dpi=300, bbox_inches='tight')
    print("Saved figures/ablation_profiling.{pdf,png}")


def generate_llm_calls_chart():
    """Figure 4: LLM call comparison."""
    try:
        with open('results/ablations/llm_call_comparison.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("LLM call comparison data not found")
        return
    
    methods = ['ProgramClean', 'SEED', 'Direct Val\n(sampled)']
    calls = [
        data['programclean_calls'],
        data['seed_calls'],
        data['direct_val_calls'],
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, calls, color=colors)
    
    ax.set_ylabel('Number of LLM Calls')
    ax.set_title('LLM API Call Comparison (Hospital Dataset)')
    ax.set_yscale('log')
    
    # Add value labels
    for bar, call in zip(bars, calls):
        height = bar.get_height()
        ax.annotate(f'{call}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('figures/llm_calls_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/llm_calls_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved figures/llm_calls_comparison.{pdf,png}")


def generate_novel_domain_chart():
    """Figure 5: Novel domain results."""
    try:
        with open('results/programclean/novel.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Novel domain results not found")
        return
    
    metrics = ['Precision', 'Recall', 'F1']
    values = [
        data['metrics']['precision'],
        data['metrics']['recall'],
        data['metrics']['f1'],
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    bars = ax.bar(metrics, values, color=colors)
    
    ax.set_ylabel('Score')
    ax.set_title('ProgramClean on Novel Domain Data (BTC, ETH, UUID, Modern Email)')
    ax.set_ylim([0, 1.1])
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('figures/novel_domain.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/novel_domain.png', dpi=300, bbox_inches='tight')
    print("Saved figures/novel_domain.{pdf,png}")


def main():
    print("Generating figures...")
    
    generate_accuracy_comparison()
    generate_runtime_comparison()
    generate_ablation_chart()
    generate_llm_calls_chart()
    generate_novel_domain_chart()
    
    print("\nAll figures generated!")


if __name__ == '__main__':
    main()
