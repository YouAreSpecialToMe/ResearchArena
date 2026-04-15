"""
Create honest figures based on actual experimental results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_aggregated(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def create_accuracy_comparison():
    """Create accuracy comparison figure."""
    
    # Load results
    vmamba = load_aggregated('checkpoints/vmamba/aggregated.json')
    localmamba = load_aggregated('checkpoints/localmamba/aggregated.json')
    cassvim_4d = load_aggregated('checkpoints/cassvim_4d/aggregated.json')
    cassvim_8d = load_aggregated('checkpoints/cassvim_8d/aggregated.json')
    
    models = []
    accs = []
    stds = []
    params = []
    colors = []
    
    if vmamba:
        models.append('VMamba\n(3.6M params)')
        accs.append(vmamba['best_acc_mean'])
        stds.append(vmamba['best_acc_std'])
        params.append(3.6)
        colors.append('#1f77b4')
    
    if localmamba:
        models.append('LocalMamba\n(3.3M params)')
        accs.append(localmamba['best_acc_mean'])
        stds.append(localmamba['best_acc_std'])
        params.append(3.3)
        colors.append('#ff7f0e')
    
    if cassvim_4d:
        models.append('CASS-ViM-4D\n(1.1M params)')
        accs.append(cassvim_4d['best_acc_mean'])
        stds.append(cassvim_4d['best_acc_std'])
        params.append(1.1)
        colors.append('#2ca02c')
    
    if cassvim_8d:
        models.append('CASS-ViM-8D\n(1.1M params)')
        accs.append(cassvim_8d['best_acc_mean'])
        stds.append(cassvim_8d['best_acc_std'])
        params.append(1.1)
        colors.append('#d62728')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    x = np.arange(len(models))
    bars = ax1.bar(x, accs, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('CIFAR-100 Test Accuracy (100 epochs, 3 seeds)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylim([45, 62])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc, std) in enumerate(zip(bars, accs, stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + std + 0.5, 
                f'{acc:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Architecture warning
    ax1.text(0.5, 0.02, 
            'WARNING: Architecture mismatch - CASS-ViM has 69% fewer parameters',
            transform=ax1.transAxes, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Parameters vs Accuracy
    ax2.scatter(params, accs, s=200, c=colors, alpha=0.7, edgecolors='black')
    for i, model in enumerate(models):
        ax2.annotate(model.replace('\n', ' '), (params[i], accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Parameters (Millions)', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Model Size', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([45, 62])
    
    plt.tight_layout()
    Path('figures').mkdir(exist_ok=True)
    plt.savefig('figures/accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/accuracy_comparison.pdf', bbox_inches='tight')
    print("Saved: figures/accuracy_comparison.png/pdf")
    plt.close()

def create_training_curves():
    """Create training curves figure."""
    
    vmamba = load_aggregated('checkpoints/vmamba/aggregated.json')
    localmamba = load_aggregated('checkpoints/localmamba/aggregated.json')
    cassvim_4d = load_aggregated('checkpoints/cassvim_4d/aggregated.json')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test accuracy curves
    if vmamba:
        test_accs = np.array([r['test_accs'] for r in vmamba['individual_results']])
        mean_accs = test_accs.mean(axis=0)
        std_accs = test_accs.std(axis=0)
        epochs = np.arange(1, len(mean_accs) + 1)
        ax1.plot(epochs, mean_accs, label='VMamba', color='#1f77b4', linewidth=2)
        ax1.fill_between(epochs, mean_accs - std_accs, mean_accs + std_accs, alpha=0.2, color='#1f77b4')
    
    if localmamba:
        test_accs = np.array([r['test_accs'] for r in localmamba['individual_results']])
        mean_accs = test_accs.mean(axis=0)
        std_accs = test_accs.std(axis=0)
        ax1.plot(epochs, mean_accs, label='LocalMamba', color='#ff7f0e', linewidth=2)
        ax1.fill_between(epochs, mean_accs - std_accs, mean_accs + std_accs, alpha=0.2, color='#ff7f0e')
    
    if cassvim_4d:
        test_accs = np.array([r['test_accs'] for r in cassvim_4d['individual_results']])
        mean_accs = test_accs.mean(axis=0)
        std_accs = test_accs.std(axis=0)
        ax1.plot(epochs, mean_accs, label='CASS-ViM-4D', color='#2ca02c', linewidth=2)
        ax1.fill_between(epochs, mean_accs - std_accs, mean_accs + std_accs, alpha=0.2, color='#2ca02c')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy Curves (mean ± std over 3 seeds)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Training curves
    if vmamba:
        train_accs = np.array([r['train_accs'] for r in vmamba['individual_results']])
        mean_accs = train_accs.mean(axis=0)
        ax2.plot(epochs, mean_accs, label='VMamba', color='#1f77b4', linewidth=2, linestyle='--')
    
    if localmamba:
        train_accs = np.array([r['train_accs'] for r in localmamba['individual_results']])
        mean_accs = train_accs.mean(axis=0)
        ax2.plot(epochs, mean_accs, label='LocalMamba', color='#ff7f0e', linewidth=2, linestyle='--')
    
    if cassvim_4d:
        train_accs = np.array([r['train_accs'] for r in cassvim_4d['individual_results']])
        mean_accs = train_accs.mean(axis=0)
        ax2.plot(epochs, mean_accs, label='CASS-ViM-4D', color='#2ca02c', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy Curves', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/training_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/training_curves.pdf', bbox_inches='tight')
    print("Saved: figures/training_curves.png/pdf")
    plt.close()

def create_ablation_figure():
    """Create ablation study figure."""
    
    cassvim_4d = load_aggregated('checkpoints/cassvim_4d/aggregated.json')
    cassvim_8d = load_aggregated('checkpoints/cassvim_8d/aggregated.json')
    
    # Load random selection partial results
    random_results = []
    for seed in [42, 123]:
        try:
            with open(f'checkpoints/random_selection/results_seed{seed}.json') as f:
                random_results.append(json.load(f))
        except:
            pass
    
    if not (cassvim_4d and random_results):
        print("Not enough data for ablation figure")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Gradient\n(4D)', 'Gradient\n(8D)', 'Random\n(partial)']
    accs = [
        cassvim_4d['best_acc_mean'],
        cassvim_8d['best_acc_mean'] if cassvim_8d else 0,
        np.mean([r['best_test_acc'] for r in random_results])
    ]
    stds = [
        cassvim_4d['best_acc_std'],
        cassvim_8d['best_acc_std'] if cassvim_8d else 0,
        0
    ]
    colors = ['#2ca02c', '#d62728', '#9467bd']
    
    x = np.arange(len(models))
    bars = ax.bar(x, accs, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study: Selection Strategy\n(Same architecture: ~1.1M parameters)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim([40, 55])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc, std in zip(bars, accs, stds):
        label = f'{acc:.2f}'
        if std > 0:
            label += f'±{std:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.5, 
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement annotation
    if len(accs) >= 3:
        diff = accs[0] - accs[2]
        ax.annotate('', xy=(0, accs[0]-0.5), xytext=(2, accs[2]+0.5),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.text(1, accs[0] - diff/2, f'+{diff:.1f}%', fontsize=11, 
               color='green', fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/ablation_selection.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/ablation_selection.pdf', bbox_inches='tight')
    print("Saved: figures/ablation_selection.png/pdf")
    plt.close()

if __name__ == '__main__':
    print("Creating figures based on experimental results...")
    Path('figures').mkdir(exist_ok=True)
    
    create_accuracy_comparison()
    create_training_curves()
    create_ablation_figure()
    
    print("\nAll figures created in figures/")
