"""
Supervised Contrastive Learning (SCL) with coarse labels on CIFAR-100.
This trains with 20 superclasses but we evaluate on 100 fine-grained classes.
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
from shared.losses import SupervisedContrastiveLoss
from shared.metrics import evaluate_linear_classifier


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_scl_encoder(encoder, train_loader, epochs, device, seed=42):
    """Train encoder with SCL loss using coarse labels."""
    optimizer = optim.SGD(encoder.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
    
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = SupervisedContrastiveLoss(temperature=0.1)
    
    encoder.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for view1, view2, labels in train_loader:
            # labels contains (coarse_label, fine_label) when use_coarse_labels=True
            # We only use coarse labels for training
            view1, view2 = view1.to(device), view2.to(device)
            coarse_labels = labels[0].to(device)
            
            images = torch.cat([view1, view2], dim=0)
            labels_cat = torch.cat([coarse_labels, coarse_labels], dim=0)
            
            optimizer.zero_grad()
            projections = encoder(images)
            loss = criterion(projections, labels_cat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'[SCL-Coarse Seed {seed}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    encoder_time = time.time() - start_time
    return encoder_time


def evaluate_on_both_labels(encoder, train_loader, test_loader, epochs, device, seed=42):
    """Train two linear classifiers: one for coarse, one for fine labels."""
    encoder.eval()
    
    # Extract features
    print(f'[SCL-Coarse Seed {seed}] Extracting features...')
    train_features, train_coarse_labels, train_fine_labels = [], [], []
    test_features, test_coarse_labels, test_fine_labels = [], [], []
    
    with torch.no_grad():
        for images, coarse_labels, fine_labels in train_loader:
            images = images.to(device)
            features = encoder.backbone(images)
            train_features.append(features.cpu())
            train_coarse_labels.append(coarse_labels)
            train_fine_labels.append(fine_labels)
        
        for images, coarse_labels, fine_labels in test_loader:
            images = images.to(device)
            features = encoder.backbone(images)
            test_features.append(features.cpu())
            test_coarse_labels.append(coarse_labels)
            test_fine_labels.append(fine_labels)
    
    train_features = torch.cat(train_features, dim=0)
    train_coarse = torch.cat(train_coarse_labels, dim=0)
    train_fine = torch.cat(train_fine_labels, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_coarse = torch.cat(test_coarse_labels, dim=0)
    test_fine = torch.cat(test_fine_labels, dim=0)
    
    # Train classifier for coarse labels (20 classes)
    print(f'[SCL-Coarse Seed {seed}] Training coarse classifier...')
    coarse_classifier = train_classifier_on_features(
        train_features, train_coarse, test_features, test_coarse, 20, epochs, device
    )
    coarse_acc = evaluate_classifier_on_features(coarse_classifier, test_features, test_coarse, device)
    
    # Train classifier for fine labels (100 classes)
    print(f'[SCL-Coarse Seed {seed}] Training fine classifier...')
    fine_classifier = train_classifier_on_features(
        train_features, train_fine, test_features, test_fine, 100, epochs, device
    )
    fine_acc = evaluate_classifier_on_features(fine_classifier, test_features, test_fine, device)
    
    return coarse_acc, fine_acc


def train_classifier_on_features(train_features, train_labels, test_features, test_labels, 
                                 num_classes, epochs, device):
    """Train a linear classifier on pre-computed features."""
    classifier = torch.nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=1.0, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        classifier.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return classifier


def evaluate_classifier_on_features(classifier, test_features, test_labels, device):
    """Evaluate classifier on features."""
    classifier.eval()
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def train_scl_coarse(seed=42, encoder_epochs=150, classifier_epochs=100,
                     batch_size=256, device='cuda'):
    """Train SCL with coarse labels on CIFAR-100."""
    set_seed(seed)
    
    # Get data loaders with coarse labels
    train_loader_contrastive, test_loader, _ = get_contrastive_cifar100_loaders(
        batch_size=batch_size, num_workers=4, use_coarse_labels=True
    )
    train_loader_std, _, _ = get_cifar100_loaders(
        batch_size=batch_size, num_workers=4, use_coarse_labels=True
    )
    
    # Create model
    encoder = create_resnet18_encoder(projector_dim=128).to(device)
    
    # Train encoder with coarse labels
    print(f'[SCL-Coarse Seed {seed}] Training encoder with coarse labels...')
    encoder_time = train_scl_encoder(encoder, train_loader_contrastive, encoder_epochs, device, seed)
    
    # Evaluate on both coarse and fine labels
    print(f'[SCL-Coarse Seed {seed}] Evaluating on coarse and fine labels...')
    coarse_acc, fine_acc = evaluate_on_both_labels(
        encoder, train_loader_std, test_loader, classifier_epochs, device, seed
    )
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/scl_coarse_seed{seed}.pth'
    torch.save({
        'encoder': encoder.state_dict(),
        'coarse_acc': coarse_acc,
        'fine_acc': fine_acc,
        'seed': seed,
    }, checkpoint_path)
    
    # Save results
    results = {
        'experiment': f'scl_coarse_seed{seed}',
        'seed': seed,
        'coarse_accuracy': coarse_acc,
        'fine_accuracy': fine_acc,
        'encoder_time_minutes': encoder_time / 60,
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/coarse_to_fine_scl_seed{seed}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'[SCL-Coarse Seed {seed}] Coarse Acc: {coarse_acc:.2f}%, Fine Acc: {fine_acc:.2f}%')
    
    return coarse_acc, fine_acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder_epochs', type=int, default=150)
    parser.add_argument('--classifier_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    train_scl_coarse(
        seed=args.seed,
        encoder_epochs=args.encoder_epochs,
        classifier_epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        device=args.device
    )
