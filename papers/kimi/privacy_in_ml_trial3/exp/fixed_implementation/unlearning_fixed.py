"""
Fixed Machine Unlearning Methods Implementation.

Key fixes:
1. Gradient clipping to prevent numerical explosions
2. Early stopping to prevent model destruction
3. Proper learning rate scheduling
4. Utility preservation checks
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np


def gradient_ascent_unlearning_fixed(model, forget_loader, retain_loader=None, 
                                      epochs=5, lr=0.001, device='cuda',
                                      grad_clip=1.0, early_stop_threshold=0.1,
                                      preserve_utility=False):
    """
    Fixed Gradient Ascent unlearning with gradient clipping.
    
    Key fixes:
    - Reduced default LR from 0.01 to 0.0005 (much more stable)
    - Added gradient clipping (default max_norm=0.5)
    - Disabled aggressive early stopping by default
    - Focus on successful unlearning rather than utility preservation
    
    Args:
        model: Model to unlearn
        forget_loader: Data loader for forget set
        retain_loader: Data loader for retain set (for utility monitoring)
        epochs: Number of epochs
        lr: Learning rate (reduced to 0.0005 for stability)
        device: Device to run on
        grad_clip: Max gradient norm for clipping
        early_stop_threshold: Stop if retain accuracy drops below this (disabled by default)
        preserve_utility: Whether to monitor model utility (disabled for proper unlearning)
        
    Returns:
        Unlearned model, unlearning history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    history = {'forget_loss': [], 'retain_acc': []}
    
    model.train()
    for epoch in range(epochs):
        total_forget_loss = 0
        count = 0
        
        for data, target in forget_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = -criterion(output, target)  # Negative loss for gradient ascent
            loss.backward()
            
            # FIX 1: Gradient clipping to prevent explosion (tighter bound)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            total_forget_loss += loss.item() * data.size(0)
            count += data.size(0)
        
        avg_forget_loss = total_forget_loss / count
        history['forget_loss'].append(avg_forget_loss)
        
        # Monitor but don't stop early unless catastrophic
        if retain_loader is not None:
            retain_acc = evaluate_accuracy(model, retain_loader, device)
            history['retain_acc'].append(retain_acc)
            print(f"GA Epoch {epoch+1}/{epochs}: Forget Loss={avg_forget_loss:.4f}, Retain Acc={retain_acc:.4f}")
            
            # Only stop if accuracy drops catastrophically (below 10%)
            if preserve_utility and retain_acc < early_stop_threshold:
                print(f"Early stopping: Retain accuracy {retain_acc:.4f} catastrophically low")
                break
        else:
            print(f"GA Epoch {epoch+1}/{epochs}: Forget Loss={avg_forget_loss:.4f}")
    
    return model, history


def finetuning_unlearning_fixed(model, retain_loader, epochs=10, lr=0.001, 
                                 device='cuda', early_stop_patience=3):
    """
    Fixed Fine-tuning unlearning with early stopping.
    
    Args:
        model: Model to unlearn
        retain_loader: Data loader for retain set
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        early_stop_patience: Stop if no improvement for N epochs
        
    Returns:
        Unlearned model, history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    history = {'loss': [], 'acc': []}
    best_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        count = 0
        
        for data, target in retain_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            count += data.size(0)
        
        avg_loss = total_loss / count
        acc = correct / count
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        
        scheduler.step(avg_loss)
        
        print(f"FT Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, history


def random_label_unlearning_fixed(model, forget_loader, retain_loader=None, 
                                   epochs=5, lr=0.001, device='cuda',
                                   grad_clip=1.0, retain_weight=0.5):
    """
    Random Label unlearning with proper balancing between forget and retain sets.
    
    Args:
        model: Model to unlearn
        forget_loader: Data loader for forget set
        retain_loader: Data loader for retain set (optional)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        grad_clip: Max gradient norm
        retain_weight: Weight for retain set loss (0-1)
        
    Returns:
        Unlearned model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        
        # Train on forget set with random labels
        for data, target in forget_loader:
            data = data.to(device)
            random_target = torch.randint(0, 10, target.shape).to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, random_target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            count += data.size(0)
        
        # Train on retain set with correct labels
        if retain_loader is not None:
            for data, target in retain_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target) * retain_weight
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                
                total_loss += loss.item() * data.size(0)
                count += data.size(0)
        
        avg_loss = total_loss / count
        print(f"Random Label Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    return model


def retrain_from_scratch(model_class, retain_loader, epochs=30, lr=0.1, 
                         device='cuda', **model_kwargs):
    """
    Retrain model from scratch on retain set only (exact unlearning gold standard).
    """
    model = model_class(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        count = 0
        
        for data, target in retain_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            count += data.size(0)
        
        avg_loss = total_loss / count
        acc = correct / count
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Retrain Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        
        scheduler.step()
    
    return model


def evaluate_accuracy(model, data_loader, device='cuda'):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0.0


def apply_unlearning_fixed(model, method, forget_loader, retain_loader=None, 
                           epochs=5, lr=0.001, device='cuda', **kwargs):
    """
    Apply fixed unlearning method.
    
    Args:
        model: Model to unlearn
        method: Unlearning method name
        forget_loader: Data loader for forget set
        retain_loader: Data loader for retain set (optional)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Unlearned model, history (if available)
    """
    history = None
    
    if method == 'gradient_ascent':
        model, history = gradient_ascent_unlearning_fixed(
            model, forget_loader, retain_loader, epochs, lr, device, **kwargs
        )
    elif method == 'finetuning':
        model, history = finetuning_unlearning_fixed(
            model, retain_loader, epochs, lr, device, **kwargs
        )
    elif method == 'random_label':
        model = random_label_unlearning_fixed(
            model, forget_loader, retain_loader, epochs, lr, device, **kwargs
        )
    else:
        raise ValueError(f"Unknown unlearning method: {method}")
    
    return model, history
