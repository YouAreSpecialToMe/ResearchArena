"""
Train standard cross-entropy baseline.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from tqdm import tqdm

from shared.data_loader import get_dataloader
from shared.models import ResNet18
from shared.utils import set_seed, save_checkpoint, save_results


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels, _ in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train Cross-Entropy baseline')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_type', type=str, default='clean')
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    parser.add_argument('--save_dir', type=str, default='./exp/cross_entropy')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data loaders
    train_loader, _ = get_dataloader(
        dataset=args.dataset, train=True, batch_size=args.batch_size,
        num_workers=4, noise_type=args.noise_type, noise_ratio=args.noise_ratio,
        seed=args.seed, contrastive=False
    )
    test_loader, _ = get_dataloader(
        dataset=args.dataset, train=False, batch_size=args.batch_size,
        num_workers=4, contrastive=False
    )
    
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    # Model
    model = ResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        best_acc = max(best_acc, test_acc)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, Best={best_acc:.2f}%")
    
    train_time = time.time() - start_time
    
    # Save results
    results = {
        'method': 'CrossEntropy',
        'dataset': args.dataset,
        'noise_type': args.noise_type,
        'noise_ratio': args.noise_ratio,
        'seed': args.seed,
        'test_accuracy': best_acc,
        'train_time_minutes': train_time / 60,
        'config': vars(args)
    }
    
    save_results(results, os.path.join(args.save_dir, f'results_seed{args.seed}.json'))
    print(f"Final Test Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
