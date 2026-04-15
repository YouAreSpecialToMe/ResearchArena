"""
Utilities for Federated Contrastive Learning.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: [batch_size, projection_dim] normalized projections
        """
        batch_size = z_i.size(0)
        
        # Concatenate projections
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create positive pairs mask
        # For sample i in z_i, positive is i+batch_size in z_j
        # For sample i+batch_size in z_j, positive is i in z_i
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Positive pairs
        pos_sim = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ], dim=0)  # [2*batch_size]
        
        # Compute loss
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
        return loss.mean()


def fedavg_aggregate(global_model, client_models, client_weights):
    """
    FedAvg aggregation: weighted average of client models.
    
    Args:
        global_model: Global model to update
        client_models: List of client models
        client_weights: List of weights (dataset sizes) for each client
    """
    global_dict = global_model.state_dict()
    
    # Normalize weights
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    # Create new state dict with weighted average
    new_state_dict = {}
    
    for key in global_dict.keys():
        # Skip integer tensors (like batch norm num_batches_tracked)
        if global_dict[key].dtype == torch.long:
            new_state_dict[key] = global_dict[key]
            continue
            
        weighted_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
        
        for client_model, weight in zip(client_models, normalized_weights):
            client_dict = client_model.state_dict()
            weighted_sum += weight * client_dict[key].float()
        
        new_state_dict[key] = weighted_sum.to(global_dict[key].dtype)
    
    global_model.load_state_dict(new_state_dict)


def train_client_contrastive(model, dataloader, optimizer, device, epochs=1, 
                             attack_eps=None, attack_steps=None):
    """
    Train a client model with contrastive learning.
    Optionally with adversarial training.
    
    Args:
        model: Client model
        dataloader: Client's data loader
        optimizer: Optimizer
        device: Device to use
        epochs: Number of local epochs
        attack_eps: Epsilon for PGD attack (if None, no adversarial training)
        attack_steps: Number of PGD steps
    """
    model.train()
    criterion = InfoNCELoss(temperature=0.5)
    total_loss = 0
    
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Contrastive learning batch contains ((x1, x2), label)
            (x1, x2), labels = batch
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()
            
            if attack_eps is not None and attack_steps is not None:
                # Adversarial training: perturb both views
                x1_adv = pgd_attack_contrastive(model, x1, x2, attack_eps, attack_steps, device)
                x2_adv = pgd_attack_contrastive(model, x2, x1, attack_eps, attack_steps, device)
                
                # Forward pass with adversarial examples
                z1 = model(x1_adv)
                z2 = model(x2_adv)
            else:
                # Standard contrastive learning
                z1 = model(x1)
                z2 = model(x2)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / (len(dataloader) * epochs)


def pgd_attack_contrastive(model, x, x_pair, eps, steps, device):
    """
    PGD attack for contrastive learning.
    Perturbs x to maximize the contrastive loss with x_pair.
    """
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    
    step_size = eps / 4
    
    for _ in range(steps):
        x_adv.requires_grad = True
        
        # Forward pass
        z_adv = model(x_adv)
        z_pair = model(x_pair).detach()
        
        # Maximize contrastive loss (minimize negative)
        criterion = InfoNCELoss(temperature=0.5)
        loss = -criterion(z_adv, z_pair)
        
        # Gradient step
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        
        # Project to epsilon ball
        perturbation = torch.clamp(x_adv - x, -eps, eps)
        x_adv = (x + perturbation).detach()
        x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv


def linear_evaluation(encoder, train_loader, test_loader, device, num_classes=10, 
                      epochs=100, lr=0.1, feature_dim=512):
    """
    Linear evaluation protocol: freeze encoder and train linear classifier.
    
    Returns:
        best_test_acc: Best test accuracy
    """
    encoder.eval()
    
    # Create linear classifier
    from models import LinearClassifier
    classifier = LinearClassifier(feature_dim, num_classes).to(device)
    
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    
    for epoch in range(epochs):
        # Train
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Extract features
            with torch.no_grad():
                features = encoder(inputs, return_features=True)
            
            # Classify
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        scheduler.step()
        
        # Test
        classifier.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = encoder(inputs, return_features=True)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        best_acc = max(best_acc, test_acc)
    
    return best_acc


def evaluate_robust_accuracy(encoder, classifier, test_loader, device, eps=8/255, steps=20):
    """
    Evaluate robust accuracy under PGD attack.
    """
    encoder.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    
    step_size = eps / 4
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # PGD attack
        x_adv = inputs.clone().detach()
        
        for _ in range(steps):
            x_adv.requires_grad = True
            
            features = encoder(x_adv, return_features=True)
            outputs = classifier(features)
            loss = F.cross_entropy(outputs, labels)
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + step_size * grad.sign()
            perturbation = torch.clamp(x_adv - inputs, -eps, eps)
            x_adv = inputs + perturbation
            x_adv = torch.clamp(x_adv, 0, 1).detach()
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            features = encoder(x_adv, return_features=True)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
