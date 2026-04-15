"""
Compute feature diversity metrics (effective rank, participation ratio, k-NN accuracy)
for all trained models.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
from collections import defaultdict

from shared.models import create_resnet18_encoder, create_resnet18_classifier
from shared.data_loader import get_cifar100_loaders
from shared.metrics import (
    compute_class_effective_ranks, compute_participation_ratio,
    knn_evaluation, extract_embeddings
)


def load_encoder(checkpoint_path, device='cuda'):
    """Load encoder from checkpoint."""
    encoder = create_resnet18_encoder(projector_dim=128).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()
    return encoder


def compute_metrics_for_model(checkpoint_path, method_name, seed, device='cuda'):
    """Compute all metrics for a trained model."""
    print(f'Computing metrics for {method_name} (seed {seed})...')
    
    # Load model
    encoder = load_encoder(checkpoint_path, device)
    
    # Get data
    train_loader, test_loader, _ = get_cifar100_loaders(
        batch_size=256, num_workers=4, use_coarse_labels=False
    )
    
    # Extract embeddings
    print(f'  Extracting train embeddings...')
    train_embeddings, train_labels = extract_embeddings(encoder, train_loader, device)
    print(f'  Extracting test embeddings...')
    test_embeddings, test_labels = extract_embeddings(encoder, test_loader, device)
    
    # Compute effective rank
    print(f'  Computing effective rank...')
    avg_erank, per_class_erank = compute_class_effective_ranks(test_embeddings, test_labels)
    
    # Compute participation ratio
    print(f'  Computing participation ratio...')
    pr = compute_participation_ratio(test_embeddings, test_labels)
    
    # Compute k-NN accuracy
    print(f'  Computing k-NN accuracy...')
    knn_acc = knn_evaluation(train_embeddings, train_labels, test_embeddings, test_labels, k=200)
    
    results = {
        'method': method_name,
        'seed': seed,
        'effective_rank': avg_erank,
        'participation_ratio': pr,
        'knn_accuracy': knn_acc * 100,  # Convert to percentage
    }
    
    print(f'  Results: ER={avg_erank:.2f}, PR={pr:.2f}, k-NN={knn_acc*100:.2f}%')
    
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []
    
    # Check which checkpoints exist
    models_to_evaluate = []
    
    for seed in [42, 123, 456]:
        # SCL
        scl_path = f'checkpoints/scl_cifar100_seed{seed}.pth'
        if os.path.exists(scl_path):
            models_to_evaluate.append((scl_path, 'SCL', seed))
        
        # FD-SCL
        fdscl_path = f'checkpoints/fdscl_cifar100_seed{seed}.pth'
        if os.path.exists(fdscl_path):
            models_to_evaluate.append((fdscl_path, 'FD-SCL', seed))
    
    print(f'Found {len(models_to_evaluate)} models to evaluate')
    
    for checkpoint_path, method_name, seed in models_to_evaluate:
        try:
            results = compute_metrics_for_model(checkpoint_path, method_name, seed, device)
            all_results.append(results)
        except Exception as e:
            print(f'Error evaluating {method_name} seed {seed}: {e}')
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/feature_diversity_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compute summary statistics
    summary = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        method = r['method']
        for metric in ['effective_rank', 'participation_ratio', 'knn_accuracy']:
            summary[method][metric].append(r[metric])
    
    print('\n=== Summary Statistics ===')
    for method in ['SCL', 'FD-SCL']:
        if method in summary:
            print(f'\n{method}:')
            for metric in ['effective_rank', 'participation_ratio', 'knn_accuracy']:
                values = summary[method][metric]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    print(f'  {metric}: {mean:.3f} ± {std:.3f}')
    
    # Save summary
    summary_dict = {}
    for method, metrics in summary.items():
        summary_dict[method] = {}
        for metric, values in metrics.items():
            if values:
                summary_dict[method][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }
    
    with open('results/metrics_summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=2)


if __name__ == '__main__':
    main()
