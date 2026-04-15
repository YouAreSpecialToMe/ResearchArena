"""
Cross-Entropy Baseline for CIFAR-100.
Standard supervised learning baseline.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from datetime import datetime

from shared.models import ResNet18Backbone, LinearClassifier
from shared.data_loader import get_cifar100_loaders
from shared.metrics import compute_accuracy


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_ce_cifar100(seed=42, epochs=150, batch_size=256, device='cuda'):
    """Train ResNet-18 with Cross-Entropy on CIFAR-100."""
    set_seed(seed)
    
    # Create model
    backbone = ResNet18Backbone().to(device)
    classifier = LinearClassifier(512, 100).to(device)
    model = nn.Sequential(backbone, classifier)
    
    # Get data loaders
    train_loader, test_loader, num_classes = get_cifar100_loaders(
        batch_size=batch_size, num_workers=4, use_coarse_labels=False
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    start_time = time.time()
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        test_acc = compute_accuracy(model, test_loader, device)
        train_losses.append(train_loss / len(train_loader))
        test_accuracies.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'[CE Seed {seed}] Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Test Acc: {test_acc:.2f}%')
    
    runtime = time.time() - start_time
    final_acc = test_accuracies[-1]
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/ce_cifar100_seed{seed}.pth'
    torch.save({
        'backbone': backbone.state_dict(),
        'classifier': classifier.state_dict(),
        'test_acc': final_acc,
        'seed': seed,
        'epochs': epochs,
    }, checkpoint_path)
    
    # Save results
    results = {
        'experiment': f'ce_baseline_seed{seed}',
        'seed': seed,
        'final_test_accuracy': final_acc,
        'runtime_minutes': runtime / 60,
        'epochs': epochs,
        'test_accuracies': test_accuracies,
        'train_losses': train_losses,
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/ce_cifar100_seed{seed}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'[CE Seed {seed}] Final Test Accuracy: {final_acc:.2f}%, Runtime: {runtime/60:.1f} min')
    
    return final_acc, runtime


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    train_ce_cifar100(seed=args.seed, epochs=args.epochs, 
                      batch_size=args.batch_size, device=args.device)
