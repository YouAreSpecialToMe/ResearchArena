"""
CIFAR-100 Cross-Entropy Baseline Experiment.
"""

import os
import sys
import json
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from shared.data_loader import get_cifar100_dataloaders
from shared.models import create_cross_entropy_model
from shared.utils import set_seed, evaluate, save_results


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading CIFAR-100...")
    train_loader, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size, num_workers=4
    )
    
    # Model
    print("Creating model...")
    model = create_cross_entropy_model('resnet18_cifar', num_classes=100).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    print(f"Training for {args.epochs} epochs...")
    start_time = time.time()
    
    best_acc = 0.0
    train_losses = []
    test_accuracies = []
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        best_acc = max(best_acc, test_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, Test Acc={test_acc:.2f}%, Best={best_acc:.2f}%")
    
    runtime = (time.time() - start_time) / 60  # minutes
    
    # Save results
    results = {
        "experiment": "cifar100_crossentropy",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_accuracy": best_acc,
        "final_accuracy": test_accuracies[-1],
        "runtime_minutes": runtime,
        "train_losses": train_losses,
        "test_accuracies": test_accuracies
    }
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'cifar100_crossentropy_seed{args.seed}.json')
    save_results(results, save_path)
    print(f"\nResults saved to {save_path}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Runtime: {runtime:.1f} minutes")


if __name__ == '__main__':
    main()
