"""
Generate visualizations for FD-SCL experiments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from shared.models import create_resnet18_encoder
from shared.data_loader import get_cifar100_loaders


def set_style():
    """Set matplotlib style."""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")


def plot_training_curves():
    """Plot training curves comparing methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Load results and plot
    seeds = [42, 123, 456]
    
    # Plot 1: Test accuracy over training
    for method, color in [('ce', 'C0'), ('scl', 'C1'), ('fdscl', 'C2')]:
        all_accs = []
        for seed in seeds:
            path = f'results/{method}_cifar100_seed{seed}.json'
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    if 'test_accuracies' in data:
                        all_accs.append(data['test_accuracies'])
        
        if all_accs:
            min_len = min(len(a) for a in all_accs)
            all_accs = [a[:min_len] for a in all_accs]
            all_accs = np.array(all_accs)
            mean_accs = all_accs.mean(axis=0)
            std_accs = all_accs.std(axis=0)
            epochs = np.arange(len(mean_accs))
            
            axes[0].plot(epochs, mean_accs, label=method.upper(), color=color)
            axes[0].fill_between(epochs, mean_accs - std_accs, mean_accs + std_accs, alpha=0.2, color=color)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy vs Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Coarse-to-Fine comparison
    scl_fine = []
    fdscl_fine = []
    for seed in seeds:
        scl_path = f'results/coarse_to_fine_scl_seed{seed}.json'
        fdscl_path = f'results/coarse_to_fine_fdscl_seed{seed}.json'
        
        if os.path.exists(scl_path):
            with open(scl_path, 'r') as f:
                scl_fine.append(json.load(f)['fine_accuracy'])
        
        if os.path.exists(fdscl_path):
            with open(fdscl_path, 'r') as f:
                fdscl_fine.append(json.load(f)['fine_accuracy'])
    
    if scl_fine and fdscl_fine:
        x = np.arange(len(seeds))
        width = 0.35
        
        axes[1].bar(x - width/2, scl_fine, width, label='SCL', color='C1')
        axes[1].bar(x + width/2, fdscl_fine, width, label='FD-SCL', color='C2')
        axes[1].set_xlabel('Seed')
        axes[1].set_ylabel('Fine-grained Accuracy (%)')
        axes[1].set_title('Coarse-to-Fine: Fine-grained Accuracy')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(s) for s in seeds])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/training_curves.png', dpi=300, bbox_inches='tight')
    print('Saved figures/training_curves.pdf')
    plt.close()


def plot_tsne():
    """Generate t-SNE visualizations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models (seed 42)
    scl_encoder = create_resnet18_encoder(projector_dim=128).to(device)
    fdscl_encoder = create_resnet18_encoder(projector_dim=128).to(device)
    
    scl_path = 'checkpoints/scl_cifar100_seed42.pth'
    fdscl_path = 'checkpoints/fdscl_cifar100_seed42.pth'
    
    if not os.path.exists(scl_path) or not os.path.exists(fdscl_path):
        print('Model checkpoints not found, skipping t-SNE')
        return
    
    scl_encoder.load_state_dict(torch.load(scl_path, map_location=device)['encoder'])
    fdscl_encoder.load_state_dict(torch.load(fdscl_path, map_location=device)['encoder'])
    
    scl_encoder.eval()
    fdscl_encoder.eval()
    
    # Get test data
    _, test_loader, _ = get_cifar100_loaders(batch_size=256, num_workers=4)
    
    # Extract embeddings
    all_labels = []
    scl_embeddings = []
    fdscl_embeddings = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            all_labels.append(labels)
            
            images = images.to(device)
            scl_emb = scl_encoder.backbone(images)
            fdscl_emb = fdscl_encoder.backbone(images)
            
            scl_embeddings.append(scl_emb.cpu())
            fdscl_embeddings.append(fdscl_emb.cpu())
    
    all_labels = torch.cat(all_labels, dim=0).numpy()
    scl_embeddings = torch.cat(scl_embeddings, dim=0).numpy()
    fdscl_embeddings = torch.cat(fdscl_embeddings, dim=0).numpy()
    
    # Sample subset for t-SNE (faster computation)
    n_samples = 2000
    indices = np.random.choice(len(all_labels), n_samples, replace=False)
    
    all_labels_sampled = all_labels[indices]
    scl_sampled = scl_embeddings[indices]
    fdscl_sampled = fdscl_embeddings[indices]
    
    # t-SNE
    print('Running t-SNE...')
    scl_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(scl_sampled)
    fdscl_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(fdscl_sampled)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = axes[0].scatter(scl_tsne[:, 0], scl_tsne[:, 1], c=all_labels_sampled, 
                               cmap='tab20', s=5, alpha=0.6)
    axes[0].set_title('SCL Embeddings (t-SNE)')
    axes[0].axis('off')
    
    scatter2 = axes[1].scatter(fdscl_tsne[:, 0], fdscl_tsne[:, 1], c=all_labels_sampled,
                               cmap='tab20', s=5, alpha=0.6)
    axes[1].set_title('FD-SCL Embeddings (t-SNE)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/tsne_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/tsne_comparison.png', dpi=300, bbox_inches='tight')
    print('Saved figures/tsne_comparison.pdf')
    plt.close()


def plot_results_table():
    """Create a results summary table figure."""
    # Load aggregated results
    if not os.path.exists('results.json'):
        print('results.json not found, skipping table')
        return
    
    with open('results.json', 'r') as f:
        results = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Create table data
    table_data = []
    
    # Main results
    if results.get('scl_baseline', {}).get('test_accuracy'):
        scl_acc = results['scl_baseline']['test_accuracy']
        table_data.append(['SCL', f"{scl_acc['mean']:.2f} ± {scl_acc['std']:.2f}"])
    
    if results.get('fdscl_main', {}).get('test_accuracy'):
        fdscl_acc = results['fdscl_main']['test_accuracy']
        table_data.append(['FD-SCL', f"{fdscl_acc['mean']:.2f} ± {fdscl_acc['std']:.2f}"])
    
    if results.get('ce_baseline', {}).get('test_accuracy'):
        ce_acc = results['ce_baseline']['test_accuracy']
        table_data.append(['Cross-Entropy', f"{ce_acc['mean']:.2f} ± {ce_acc['std']:.2f}"])
    
    # Coarse-to-Fine
    if results.get('coarse_to_fine', {}).get('scl_fine_accuracy'):
        scl_fine = results['coarse_to_fine']['scl_fine_accuracy']
        fdscl_fine = results['coarse_to_fine']['fdscl_fine_accuracy']
        table_data.append(['SCL (Fine, Coarse Train)', f"{scl_fine['mean']:.2f} ± {scl_fine['std']:.2f}"])
        table_data.append(['FD-SCL (Fine, Coarse Train)', f"{fdscl_fine['mean']:.2f} ± {fdscl_fine['std']:.2f}"])
    
    # Feature diversity
    if results.get('feature_diversity', {}).get('scl_effective_rank'):
        scl_er = results['feature_diversity']['scl_effective_rank']
        fdscl_er = results['feature_diversity']['fdscl_effective_rank']
        table_data.append(['SCL Effective Rank', f"{scl_er['mean']:.2f} ± {scl_er['std']:.2f}"])
        table_data.append(['FD-SCL Effective Rank', f"{fdscl_er['mean']:.2f} ± {fdscl_er['std']:.2f}"])
    
    if table_data:
        table = ax.table(cellText=table_data, 
                        colLabels=['Method', 'Result'],
                        loc='center',
                        cellLoc='center',
                        colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title('Summary of Results', fontsize=14, fontweight='bold', pad=20)
        plt.savefig('figures/results_table.pdf', dpi=300, bbox_inches='tight')
        print('Saved figures/results_table.pdf')
        plt.close()


def main():
    set_style()
    os.makedirs('figures', exist_ok=True)
    
    print('Generating visualizations...')
    
    try:
        plot_training_curves()
    except Exception as e:
        print(f'Error plotting training curves: {e}')
    
    try:
        plot_tsne()
    except Exception as e:
        print(f'Error plotting t-SNE: {e}')
    
    try:
        plot_results_table()
    except Exception as e:
        print(f'Error plotting results table: {e}')
    
    print('Visualization complete!')


if __name__ == '__main__':
    main()
