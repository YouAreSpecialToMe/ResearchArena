"""
Train Gradient-Confusion Aware Supervised Contrastive Learning (GC-SCL).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import time
from tqdm import tqdm

from shared.data_loader import get_dataloader
from shared.models import ResNet18, ProjectionHead, ContrastiveModel
from shared.losses import GCSCLoss, SupConLoss
from shared.utils import set_seed, save_checkpoint, linear_evaluate, save_results
import torch.nn.functional as F


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, encoder):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Update epoch for curriculum scheduling
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    
    stats_list = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels, indices in pbar:
        # images is a list of two augmented views
        images = torch.cat([images[0], images[1]], dim=0).to(device)
        labels = labels.to(device)
        indices = indices.to(device)
        
        batch_size = labels.shape[0]
        
        # Forward pass
        projections, features = model(images, return_features=True)  # (2*batch_size, feat_dim)
        projections = F.normalize(projections, dim=1)  # Normalize for contrastive loss
        
        # Reshape projections to (batch_size, 2, feat_dim)
        p1, p2 = torch.split(projections, [batch_size, batch_size], dim=0)
        projections = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
        
        # Reshape features to (batch_size, 2, feat_dim)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features_batch = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        # Compute loss
        if isinstance(criterion, GCSCLoss):
            loss, stats = criterion(projections, labels, indices, encoder, return_stats=True)
            stats_list.append(stats)
        else:
            loss = criterion(projections, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    
    # Aggregate stats
    if stats_list:
        avg_stats = {}
        for key in stats_list[0].keys():
            avg_stats[key] = sum(s[key] for s in stats_list) / len(stats_list)
        return avg_loss, avg_stats
    
    return avg_loss, {}


def main():
    parser = argparse.ArgumentParser(description='Train GC-SCL')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    
    # GC-SCL specific parameters
    parser.add_argument('--gamma', type=float, default=2.0, help='Gradient alignment importance')
    parser.add_argument('--velocity_temp', type=float, default=0.1, help='Velocity temperature')
    parser.add_argument('--velocity_momentum', type=float, default=0.9, help='EMA momentum for velocity')
    parser.add_argument('--curriculum_epochs', type=int, default=500, help='Epochs to reach full weighting')
    
    # Ablation flags
    parser.add_argument('--no_velocity', action='store_true', help='Ablation: do not use velocity')
    parser.add_argument('--no_curriculum', action='store_true', help='Ablation: do not use curriculum')
    parser.add_argument('--use_loss_weighting', action='store_true', help='Ablation: use loss-based weighting instead of gradient')
    
    parser.add_argument('--noise_type', type=str, default='clean', choices=['clean', 'symmetric', 'asymmetric'])
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    parser.add_argument('--save_dir', type=str, default='./exp/gcscl')
    parser.add_argument('--linear_epochs', type=int, default=100)
    parser.add_argument('--linear_lr', type=float, default=0.1)
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data loaders
    print(f"Loading {args.dataset} with {args.noise_type} noise...")
    train_loader, train_dataset = get_dataloader(
        dataset=args.dataset,
        train=True,
        batch_size=args.batch_size,
        num_workers=4,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
        contrastive=True
    )
    
    test_loader, _ = get_dataloader(
        dataset=args.dataset,
        train=False,
        batch_size=args.batch_size,
        num_workers=4,
        contrastive=False
    )
    
    # For linear evaluation, we need loaders without contrastive augmentation
    eval_train_loader, _ = get_dataloader(
        dataset=args.dataset,
        train=True,
        batch_size=args.batch_size,
        num_workers=4,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
        contrastive=False
    )
    
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    # Model
    print("Creating model...")
    encoder = ResNet18(num_classes=num_classes).to(device)
    projection_head = ProjectionHead(in_dim=encoder.feature_dim, hidden_dim=128, out_dim=128).to(device)
    model = ContrastiveModel(encoder, projection_head).to(device)
    
    # Loss
    if args.use_loss_weighting:
        # Use standard SupCon for loss-based weighting (handled separately)
        print("Using loss-based weighting (ablation)")
        criterion = SupConLoss(temperature=args.temperature)
    else:
        # Use GC-SCL loss
        print(f"Using GC-SCL with gamma={args.gamma}, velocity_temp={args.velocity_temp}")
        criterion = GCSCLoss(
            temperature=args.temperature,
            gamma=args.gamma,
            velocity_temp=args.velocity_temp,
            velocity_momentum=args.velocity_momentum,
            curriculum_epochs=args.curriculum_epochs if not args.no_curriculum else 0
        )
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    print(f"Training for {args.epochs} epochs...")
    start_time = time.time()
    
    train_losses = []
    for epoch in range(args.epochs):
        loss, stats = train_epoch(model, train_loader, criterion, optimizer, device, epoch, encoder)
        scheduler.step()
        train_losses.append(loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss = {loss:.4f}")
            if stats:
                print(f"  Stats: alignment={stats.get('alignment_mean', 0):.4f}, "
                      f"utility={stats.get('utility_mean', 0):.4f}, "
                      f"weights={stats.get('weights_mean', 0):.4f}")
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, 
                          os.path.join(args.save_dir, f'checkpoint_seed{args.seed}.pt'))
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes")
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs,
                   os.path.join(args.save_dir, f'final_seed{args.seed}.pt'))
    
    # Linear evaluation
    print("Running linear evaluation...")
    # Create a clean train loader for evaluation (to measure true performance)
    clean_train_loader, _ = get_dataloader(
        dataset=args.dataset,
        train=True,
        batch_size=args.batch_size,
        num_workers=4,
        noise_type='clean',
        seed=args.seed,
        contrastive=False
    )
    
    linear_acc = linear_evaluate(
        encoder, clean_train_loader, test_loader, device, num_classes,
        epochs=args.linear_epochs, lr=args.linear_lr
    )
    
    print(f"Linear Evaluation Accuracy: {linear_acc:.2f}%")
    
    # Save results
    results = {
        'method': 'GC-SCL',
        'dataset': args.dataset,
        'noise_type': args.noise_type,
        'noise_ratio': args.noise_ratio,
        'seed': args.seed,
        'linear_accuracy': linear_acc,
        'train_time_minutes': train_time / 60,
        'config': vars(args)
    }
    
    save_results(results, os.path.join(args.save_dir, f'results_seed{args.seed}.json'))
    print(f"Results saved to {args.save_dir}/results_seed{args.seed}.json")


if __name__ == '__main__':
    main()
