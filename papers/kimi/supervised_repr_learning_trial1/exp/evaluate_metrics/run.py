"""
Evaluate learned representations with metrics like effective rank and k-NN accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import json
import numpy as np

from shared.data_loader import get_cifar100_loaders, get_cifar100_fine_labels_only
from shared.models import create_model
from shared.metrics import (
    compute_class_effective_ranks, 
    compute_participation_ratio,
    knn_accuracy, 
    extract_embeddings
)
from shared.utils import set_seed, save_results, load_checkpoint


def evaluate_model(checkpoint_path, train_loader, test_loader, num_classes, device, feature_dim=512):
    """Evaluate a trained model."""
    # Load model
    model = create_model(num_classes=num_classes, use_projection_head=True, projection_dim=128)
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Extract embeddings
    print("Extracting training embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device, use_projection_head=False)
    
    print("Extracting test embeddings...")
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device, use_projection_head=False)
    
    # Normalize embeddings
    train_embeddings = torch.nn.functional.normalize(train_embeddings, dim=1)
    test_embeddings = torch.nn.functional.normalize(test_embeddings, dim=1)
    
    # Compute metrics
    print("Computing effective rank...")
    avg_erank, class_ranks = compute_class_effective_ranks(test_embeddings, test_labels, num_classes)
    
    print("Computing participation ratio...")
    pr = compute_participation_ratio(test_embeddings)
    
    print("Computing k-NN accuracy...")
    knn_acc = knn_accuracy(train_embeddings, train_labels, test_embeddings, test_labels, k=200)
    
    return {
        'avg_effective_rank': float(avg_erank),
        'class_effective_ranks': [float(r) for r in class_ranks],
        'participation_ratio': float(pr),
        'knn_accuracy': float(knn_acc),
        'num_train_samples': len(train_embeddings),
        'num_test_samples': len(test_embeddings),
        'embedding_dim': train_embeddings.shape[1]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['scl', 'fdscl', 'ce'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_coarse', action='store_true')
    parser.add_argument('--eval_fine', action='store_true', help='Evaluate on fine labels (100 classes)')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./results/metrics.json')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine number of classes
    if args.eval_fine:
        num_classes = 100
        _, test_loader, _ = get_cifar100_loaders(
            root=args.data_root, batch_size=256, num_workers=4,
            contrastive=False, use_coarse_labels=False
        )
        train_loader, _, _ = get_cifar100_loaders(
            root=args.data_root, batch_size=256, num_workers=4,
            contrastive=False, use_coarse_labels=False
        )
    else:
        num_classes = 20 if args.use_coarse else 100
        train_loader, test_loader, _ = get_cifar100_loaders(
            root=args.data_root, batch_size=256, num_workers=4,
            contrastive=False, use_coarse_labels=args.use_coarse
        )
    
    # Evaluate
    results = evaluate_model(args.checkpoint, train_loader, test_loader, num_classes, device)
    
    # Add metadata
    results['method'] = args.method
    results['checkpoint'] = args.checkpoint
    results['seed'] = args.seed
    results['use_coarse'] = args.use_coarse
    results['eval_fine'] = args.eval_fine
    
    print("\n" + "="*50)
    print(f"Results for {args.method.upper()}:")
    print(f"Average Effective Rank: {results['avg_effective_rank']:.2f}")
    print(f"Participation Ratio: {results['participation_ratio']:.2f}")
    print(f"k-NN Accuracy: {results['knn_accuracy']:.2f}%")
    print(f"{'='*50}")
    
    # Save results
    save_results(results, args.output)


if __name__ == '__main__':
    main()
