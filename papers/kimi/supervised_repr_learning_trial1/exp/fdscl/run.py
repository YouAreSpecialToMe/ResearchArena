"""
FD-SCL: Feature-Diversity-aware Supervised Contrastive Learning.
Proposed method with adaptive pair weighting.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import json
import numpy as np

from shared.data_loader import get_cifar100_loaders
from shared.models import create_model, LinearClassifier
from shared.losses import FeatureDiversitySCLLoss
from shared.metrics import linear_probe_accuracy
from shared.utils import set_seed, save_results, save_checkpoint, Timer


def train_encoder(model, train_loader, criterion, optimizer, device, epoch, return_weights=False):
    """Train encoder with FD-SCL loss."""
    model.train()
    total_loss = 0
    weight_stats_list = []
    
    for images, labels in train_loader:
        # images is a list of 2 views
        images = torch.cat([images[0], images[1]], dim=0).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - get projections
        projections = model(images)
        
        # Compute loss
        if return_weights:
            loss, weight_stats = criterion(projections, labels)
            weight_stats_list.append(weight_stats)
        else:
            loss = criterion(projections, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    if return_weights and weight_stats_list:
        # Aggregate weight statistics
        aggregated_stats = {
            'mean_weight': np.mean([s['mean_weight'] for s in weight_stats_list]),
            'std_weight': np.mean([s['std_weight'] for s in weight_stats_list]),
            'min_weight': np.min([s['min_weight'] for s in weight_stats_list]),
            'max_weight': np.max([s['max_weight'] for s in weight_stats_list]),
        }
        return avg_loss, aggregated_stats
    
    return avg_loss, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder_epochs', type=int, default=200)
    parser.add_argument('--classifier_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.1, help='Contrastive temperature')
    parser.add_argument('--weight_temperature', type=float, default=0.5, help='Weight temperature')
    parser.add_argument('--activation_threshold', type=float, default=0.0, help='Activation threshold')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--use_coarse', action='store_true', help='Use coarse (20) labels')
    parser.add_argument('--save_weights', action='store_true', help='Save weight statistics')
    parser.add_argument('--skip_linear_eval', action='store_true', help='Skip linear evaluation')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders (contrastive = True for two views)
    train_loader, test_loader, num_classes = get_cifar100_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        contrastive=True,
        use_coarse_labels=args.use_coarse
    )
    
    # For linear evaluation, we need non-contrastive loaders
    train_loader_eval, _, _ = get_cifar100_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        contrastive=False,
        use_coarse_labels=args.use_coarse
    )
    
    print(f"Number of classes: {num_classes}")
    
    # Model
    model = create_model(num_classes=num_classes, use_projection_head=True, projection_dim=128)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss
    criterion = FeatureDiversitySCLLoss(
        temperature=args.temperature,
        weight_temperature=args.weight_temperature,
        activation_threshold=args.activation_threshold,
        return_weights=args.save_weights
    )
    
    # Optimizer
    scaled_lr = args.lr * args.batch_size / 256
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Warmup + Cosine annealing
    def lr_schedule(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        else:
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - 10) / (args.encoder_epochs - 10) * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    # Training encoder
    timer = Timer()
    timer.start()
    
    print(f"\n=== Training FD-SCL Encoder for {args.encoder_epochs} epochs ===")
    print(f"Temperature: {args.temperature}, Weight Temperature: {args.weight_temperature}, "
          f"Activation Threshold: {args.activation_threshold}")
    
    encoder_losses = []
    weight_stats_history = []
    
    for epoch in range(args.encoder_epochs):
        if args.save_weights and (epoch % 20 == 0 or epoch == args.encoder_epochs - 1):
            loss, weight_stats = train_encoder(model, train_loader, criterion, optimizer, device, epoch, return_weights=True)
            weight_stats_history.append({'epoch': epoch, **weight_stats})
        else:
            loss, _ = train_encoder(model, train_loader, criterion, optimizer, device, epoch, return_weights=False)
        
        encoder_losses.append(loss)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.encoder_epochs} - Loss: {loss:.4f}")
    
    encoder_time = timer.get_elapsed()
    
    # Save encoder
    label_type = 'coarse' if args.use_coarse else 'fine'
    save_checkpoint(
        model, optimizer, args.encoder_epochs,
        f"{args.save_dir}/fdscl_cifar100_{label_type}_seed{args.seed}.pth",
    )
    
    # Save weight statistics if requested
    if args.save_weights and weight_stats_history:
        np.savez(f"./results/fdscl_weights_{label_type}_seed{args.seed}.npz", 
                 weight_stats=weight_stats_history)
    
    # Linear evaluation
    if not args.skip_linear_eval:
        print(f"\n=== Linear Evaluation for {args.classifier_epochs} epochs ===")
        linear_acc = linear_probe_accuracy(
            model, train_loader_eval, test_loader, device,
            feature_dim=512, num_classes=num_classes,
            epochs=args.classifier_epochs, lr=1.0
        )
        print(f"Linear evaluation accuracy: {linear_acc:.2f}%")
    else:
        linear_acc = 0.0
    
    total_time = timer.stop()
    
    # Save results
    results = {
        'experiment': f'fdscl_cifar100_{label_type}',
        'seed': args.seed,
        'encoder_epochs': args.encoder_epochs,
        'classifier_epochs': args.classifier_epochs,
        'linear_eval_acc': linear_acc,
        'encoder_losses': encoder_losses,
        'weight_temperature': args.weight_temperature,
        'activation_threshold': args.activation_threshold,
        'encoder_time_seconds': encoder_time,
        'total_time_seconds': total_time,
        'config': vars(args)
    }
    
    if weight_stats_history:
        results['weight_stats'] = weight_stats_history
    
    os.makedirs('./results', exist_ok=True)
    save_results(results, f'./results/fdscl_cifar100_{label_type}_seed{args.seed}.json')
    
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Encoder time: {encoder_time:.1f}s")


if __name__ == '__main__':
    main()
