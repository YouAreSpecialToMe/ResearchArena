"""
Create visualizations for the paper.
- t-SNE plots
- Training curves
- Weight distribution analysis
- Effective rank comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import json
import argparse
import glob

from shared.data_loader import get_cifar100_loaders
from shared.models import create_model
from shared.metrics import extract_embeddings
from shared.utils import set_seed


def plot_tsne(embeddings, labels, title, filename, n_classes=20):
    """Create t-SNE visualization."""
    print(f"Computing t-SNE for {title}...")
    
    # Sample for faster computation
    n_samples = min(5000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    embeddings_sample = embeddings[indices]
    labels_sample = labels[indices]
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_sample.cpu().numpy())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=labels_sample.cpu().numpy(), cmap='tab20', alpha=0.6, s=10
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_training_curves(results_dict, filename):
    """Plot training curves comparing methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax = axes[0]
    for method_name, results in results_dict.items():
        if 'encoder_losses' in results:
            losses = results['encoder_losses']
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, label=method_name, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy comparison
    ax = axes[1]
    methods = []
    accuracies = []
    for method_name, results in results_dict.items():
        if 'linear_eval_acc' in results:
            methods.append(method_name)
            accuracies.append(results['linear_eval_acc'])
    
    if methods:
        bars = ax.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Linear Evaluation Accuracy', fontsize=14)
        ax.set_ylim([0, 100])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_effective_rank_comparison(metrics_dict, filename):
    """Plot effective rank comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = []
    ranks = []
    for method_name, metrics in metrics_dict.items():
        if 'avg_effective_rank' in metrics:
            methods.append(method_name)
            ranks.append(metrics['avg_effective_rank'])
    
    if methods:
        bars = ax.bar(methods, ranks, color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel('Average Effective Rank', fontsize=12)
        ax.set_title('Feature Diversity (Effective Rank)', fontsize=14)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_coarse_to_fine_comparison(scl_results, fdscl_results, filename):
    """Plot coarse-to-fine accuracy comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['SCL', 'FD-SCL']
    coarse_accs = [scl_results.get('coarse_label_accuracy', 0), fdscl_results.get('coarse_label_accuracy', 0)]
    fine_accs = [scl_results.get('fine_label_accuracy', 0), fdscl_results.get('fine_label_accuracy', 0)]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, coarse_accs, width, label='Coarse Labels (20)', color='#1f77b4')
    bars2 = ax.bar(x + width/2, fine_accs, width, label='Fine Labels (100)', color='#ff7f0e')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Feature Suppression Evaluation\n(Trained on Coarse Labels)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim([0, 100])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--output_dir', type=str, default='./figures')
    parser.add_argument('--checkpoint_scl', type=str, default='')
    parser.add_argument('--checkpoint_fdscl', type=str, default='')
    parser.add_argument('--create_tsne', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    
    # Load results files
    results = {}
    metrics = {}
    
    for result_file in glob.glob(os.path.join(args.results_dir, '*.json')):
        basename = os.path.basename(result_file)
        with open(result_file, 'r') as f:
            data = json.load(f)
            results[basename] = data
            
            if 'avg_effective_rank' in data:
                metrics[basename] = data
    
    print(f"Loaded {len(results)} result files")
    
    # Plot training curves
    if results:
        plot_training_curves(results, os.path.join(args.output_dir, 'training_curves.pdf'))
    
    # Plot effective rank comparison
    if metrics:
        plot_effective_rank_comparison(metrics, os.path.join(args.output_dir, 'effective_rank_comparison.pdf'))
    
    # Plot coarse-to-fine comparison if available
    scl_coarse = results.get('scl_coarse_to_fine_seed42.json', {})
    fdscl_coarse = results.get('fdscl_coarse_to_fine_seed42.json', {})
    
    if scl_coarse and fdscl_coarse:
        plot_coarse_to_fine_comparison(scl_coarse, fdscl_coarse, 
                                       os.path.join(args.output_dir, 'coarse_to_fine_comparison.pdf'))
    
    # Create t-SNE visualizations
    if args.create_tsne and args.checkpoint_scl and args.checkpoint_fdscl:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load test data
        _, test_loader, num_classes = get_cifar100_loaders(
            root='./data', batch_size=256, num_workers=4,
            contrastive=False, use_coarse_labels=True
        )
        
        # Extract and plot SCL embeddings
        model_scl = create_model(num_classes=20, use_projection_head=True)
        model_scl = model_scl.to(device)
        checkpoint = torch.load(args.checkpoint_scl, map_location=device)
        model_scl.load_state_dict(checkpoint['model_state_dict'])
        
        embeddings_scl, labels_scl = extract_embeddings(model_scl, test_loader, device, use_projection_head=False)
        plot_tsne(embeddings_scl, labels_scl, 'SCL Embeddings (Coarse Labels)', 
                  os.path.join(args.output_dir, 'tsne_scl.pdf'), n_classes=20)
        
        # Extract and plot FD-SCL embeddings
        model_fdscl = create_model(num_classes=20, use_projection_head=True)
        model_fdscl = model_fdscl.to(device)
        checkpoint = torch.load(args.checkpoint_fdscl, map_location=device)
        model_fdscl.load_state_dict(checkpoint['model_state_dict'])
        
        embeddings_fdscl, labels_fdscl = extract_embeddings(model_fdscl, test_loader, device, use_projection_head=False)
        plot_tsne(embeddings_fdscl, labels_fdscl, 'FD-SCL Embeddings (Coarse Labels)', 
                  os.path.join(args.output_dir, 'tsne_fdscl.pdf'), n_classes=20)
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()
