"""
Utility functions for training, evaluation, and logging.
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def train_epoch(model, train_loader, criterion, optimizer, device, epoch=0, total_epochs=100, loss_type='supcon', attr_similarity=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    curriculum_values = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if loss_type == 'cross_entropy':
            outputs = model(images)
            loss = criterion(outputs, labels)
        else:
            # Contrastive learning
            features = model(images)
            
            if loss_type == 'caghnm':
                criterion.set_epoch(epoch, total_epochs)
                loss, lambda_t = criterion(features, labels, attr_similarity)
                curriculum_values.append(lambda_t)
            elif loss_type in ['jdccl_fixed']:
                loss = criterion(features, labels, attr_similarity)
            else:  # supcon
                loss = criterion(features, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_curriculum = np.mean(curriculum_values) if curriculum_values else None
    
    return avg_loss, avg_curriculum


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def linear_evaluation(encoder, train_loader, test_loader, device, feature_dim, num_classes, epochs=50, lr=0.1):
    """
    Linear evaluation protocol: train a linear classifier on frozen features.
    """
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Create linear classifier
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder.get_features(images)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = encoder.get_features(images)
                outputs = classifier(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        best_acc = max(best_acc, accuracy)
    
    # Unfreeze encoder
    for param in encoder.parameters():
        param.requires_grad = True
    
    return best_acc


def save_results(results, path):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path):
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_optimizer(model, lr=0.5, momentum=0.9, weight_decay=1e-4):
    """Get SGD optimizer with cosine annealing scheduler."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def get_scheduler(optimizer, epochs):
    """Get cosine annealing scheduler."""
    return CosineAnnealingLR(optimizer, T_max=epochs)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    print("Testing utilities...")
    set_seed(42)
    print("Seed set to 42")
