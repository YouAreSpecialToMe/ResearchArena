"""
Train Supervised Contrastive Learning (SupCon) baseline.
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
from shared.losses import SupConLoss
from shared.utils import set_seed, save_checkpoint, linear_evaluate, save_results
import torch.nn.functional as F


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels, indices in pbar:
        # images is a list of two augmented views
        images = torch.cat([images[0], images[1]], dim=0).to(device)
        labels = labels.to(device)
        
        batch_size = labels.shape[0]
        
        # Forward pass
        features = model(images)  # (2*batch_size, feat_dim)
        features = F.normalize(features, dim=1)  # Normalize for contrastive loss
        
        # Reshape to (batch_size, 2, feat_dim)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        # Compute loss
        loss = criterion(features, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train SupCon baseline')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_type', type=str, default='clean', choices=['clean', 'symmetric', 'asymmetric'])
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    parser.add_argument('--save_dir', type=str, default='./exp/supcon')
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
    
    # Loss and optimizer
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    print(f"Training for {args.epochs} epochs...")
    start_time = time.time()
    
    train_losses = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        train_losses.append(loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss = {loss:.4f}")
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
        'method': 'SupCon',
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
