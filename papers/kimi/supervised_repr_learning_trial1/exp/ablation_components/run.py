"""
Ablation study: Component analysis for FD-SCL.
Tests: Uniform weights (SCL), Random weights, Full FD-SCL.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

from shared.models import create_resnet18_encoder, create_resnet18_classifier
from shared.data_loader import get_contrastive_cifar100_loaders, get_cifar100_loaders
from shared.losses import FDSupervisedContrastiveLoss, FDSCLUniformAblation, FDSCLRandomAblation
from shared.metrics import evaluate_linear_classifier


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_encoder_variant(encoder, train_loader, epochs, device, loss_type='full', seed=42):
    """Train encoder with different loss variants."""
    optimizer = optim.SGD(encoder.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
    
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if loss_type == 'uniform':
        criterion = FDSCLUniformAblation(temperature=0.1)
    elif loss_type == 'random':
        criterion = FDSCLRandomAblation(temperature=0.1)
    else:  # full
        criterion = FDSupervisedContrastiveLoss(temperature=0.1, weight_temperature=0.5)
    
    encoder.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for view1, view2, labels in train_loader:
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            
            images = torch.cat([view1, view2], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            
            optimizer.zero_grad()
            projections = encoder(images)
            loss = criterion(projections, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'[{loss_type.upper()}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
    
    return encoder


def run_ablation(loss_type='full', seed=42, encoder_epochs=100, 
                 classifier_epochs=100, batch_size=256, device='cuda'):
    """Run ablation experiment."""
    set_seed(seed)
    
    train_loader_contrastive, test_loader, _ = get_contrastive_cifar100_loaders(
        batch_size=batch_size, num_workers=4, use_coarse_labels=False
    )
    train_loader_std, _, _ = get_cifar100_loaders(
        batch_size=batch_size, num_workers=4, use_coarse_labels=False
    )
    
    encoder = create_resnet18_encoder(projector_dim=128).to(device)
    
    print(f'[{loss_type.upper()}] Training encoder...')
    encoder = train_encoder_variant(encoder, train_loader_contrastive, encoder_epochs, device, loss_type, seed)
    
    print(f'[{loss_type.upper()}] Training linear classifier...')
    classifier = create_resnet18_classifier(num_classes=100).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=1.0, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=classifier_epochs)
    criterion = nn.CrossEntropyLoss()
    
    encoder.eval()
    classifier.train()
    
    for epoch in range(classifier_epochs):
        for images, labels in train_loader_std:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder.backbone(images)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    test_acc = evaluate_linear_classifier(classifier, encoder, test_loader, device)
    print(f'[{loss_type.upper()}] Test Accuracy: {test_acc:.2f}%')
    
    return test_acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder_epochs', type=int, default=100)
    parser.add_argument('--classifier_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Run all three variants
    results = {}
    for loss_type in ['uniform', 'random', 'full']:
        acc = run_ablation(
            loss_type=loss_type,
            seed=args.seed,
            encoder_epochs=args.encoder_epochs,
            classifier_epochs=args.classifier_epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        results[loss_type] = {'accuracy': acc}
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_components.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n=== Ablation Results ===')
    for loss_type, res in results.items():
        print(f'{loss_type}: {res["accuracy"]:.2f}%')
