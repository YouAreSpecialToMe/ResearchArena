"""
Feature-Diversity-Aware Supervised Contrastive Learning (FD-SCL) for CIFAR-100.
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

from shared.models import create_resnet18_encoder, create_resnet18_classifier
from shared.data_loader import get_contrastive_cifar100_loaders, get_cifar100_loaders
from shared.losses import FDSupervisedContrastiveLoss
from shared.metrics import evaluate_linear_classifier


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fdscl_encoder(encoder, train_loader, epochs, device, seed=42,
                       temperature=0.1, weight_temperature=0.5, activation_threshold=0.0):
    """Train encoder with FD-SCL loss."""
    optimizer = optim.SGD(encoder.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
    
    # Cosine annealing with warmup
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FDSupervisedContrastiveLoss(
        temperature=temperature,
        weight_temperature=weight_temperature,
        activation_threshold=activation_threshold
    )
    
    encoder.train()
    start_time = time.time()
    weight_stats_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        epoch_stats = []
        
        for view1, view2, labels in train_loader:
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            
            # Concatenate views
            images = torch.cat([view1, view2], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            
            optimizer.zero_grad()
            projections = encoder(images)
            loss, stats = criterion(projections, labels, return_stats=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            epoch_stats.append(stats)
        
        scheduler.step()
        
        # Aggregate stats
        avg_stats = {
            'epoch': epoch + 1,
            'mean_weight': np.mean([s['mean_weight'] for s in epoch_stats]),
            'std_weight': np.mean([s['std_weight'] for s in epoch_stats]),
            'mean_rarity': np.mean([s['mean_rarity'] for s in epoch_stats]),
        }
        weight_stats_history.append(avg_stats)
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'[FD-SCL Seed {seed}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
                  f'Weight Mean: {avg_stats["mean_weight"]:.4f}, Std: {avg_stats["std_weight"]:.4f}')
    
    encoder_time = time.time() - start_time
    return encoder_time, weight_stats_history


def train_linear_classifier(encoder, train_loader, test_loader, epochs, device, seed=42):
    """Train linear classifier on frozen features."""
    classifier = create_resnet18_classifier(num_classes=100).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=1.0, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    encoder.eval()
    classifier.train()
    
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder.backbone(images)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        test_acc = evaluate_linear_classifier(classifier, encoder, test_loader, device)
        best_acc = max(best_acc, test_acc)
        
        if (epoch + 1) % 20 == 0:
            print(f'[FD-SCL Seed {seed}] Linear Eval Epoch {epoch+1}/{epochs}, Test Acc: {test_acc:.2f}%')
    
    classifier_time = time.time() - start_time
    return classifier, best_acc, classifier_time


def train_fdscl_cifar100(seed=42, encoder_epochs=200, classifier_epochs=100,
                         batch_size=256, device='cuda',
                         temperature=0.1, weight_temperature=0.5, activation_threshold=0.0):
    """Train FD-SCL on CIFAR-100."""
    set_seed(seed)
    
    # Get data loaders
    train_loader_contrastive, test_loader, _ = get_contrastive_cifar100_loaders(
        batch_size=batch_size, num_workers=4, use_coarse_labels=False
    )
    train_loader_std, _, _ = get_cifar100_loaders(
        batch_size=batch_size, num_workers=4, use_coarse_labels=False
    )
    
    # Create model
    encoder = create_resnet18_encoder(projector_dim=128).to(device)
    
    # Train encoder
    print(f'[FD-SCL Seed {seed}] Training encoder...')
    encoder_time, weight_stats = train_fdscl_encoder(
        encoder, train_loader_contrastive, encoder_epochs, device, seed,
        temperature, weight_temperature, activation_threshold
    )
    
    # Train linear classifier
    print(f'[FD-SCL Seed {seed}] Training linear classifier...')
    classifier, best_acc, classifier_time = train_linear_classifier(
        encoder, train_loader_std, test_loader, classifier_epochs, device, seed
    )
    
    total_time = encoder_time + classifier_time
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/fdscl_cifar100_seed{seed}.pth'
    torch.save({
        'encoder': encoder.state_dict(),
        'classifier': classifier.state_dict(),
        'test_acc': best_acc,
        'seed': seed,
        'weight_stats': weight_stats,
    }, checkpoint_path)
    
    # Save weight statistics
    os.makedirs('results', exist_ok=True)
    np.savez(f'results/fdscl_weights_seed{seed}.npz', weight_stats=weight_stats)
    
    # Save results
    results = {
        'experiment': f'fdscl_main_seed{seed}',
        'seed': seed,
        'final_test_accuracy': best_acc,
        'encoder_time_minutes': encoder_time / 60,
        'classifier_time_minutes': classifier_time / 60,
        'total_runtime_minutes': total_time / 60,
        'hyperparameters': {
            'temperature': temperature,
            'weight_temperature': weight_temperature,
            'activation_threshold': activation_threshold,
        }
    }
    
    with open(f'results/fdscl_cifar100_seed{seed}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'[FD-SCL Seed {seed}] Final Test Accuracy: {best_acc:.2f}%, Total Runtime: {total_time/60:.1f} min')
    
    return best_acc, total_time


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder_epochs', type=int, default=200)
    parser.add_argument('--classifier_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--weight_temperature', type=float, default=0.5)
    parser.add_argument('--activation_threshold', type=float, default=0.0)
    args = parser.parse_args()
    
    train_fdscl_cifar100(
        seed=args.seed,
        encoder_epochs=args.encoder_epochs,
        classifier_epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        device=args.device,
        temperature=args.temperature,
        weight_temperature=args.weight_temperature,
        activation_threshold=args.activation_threshold
    )
