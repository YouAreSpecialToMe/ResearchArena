"""
Machine Unlearning Methods Implementation.

Includes:
- Gradient Ascent (GA)
- Fine-tuning on retain set (FT)
- Random Labeling (adversarial)
- Influence-based unlearning (simplified)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy


def gradient_ascent_unlearning(model, forget_loader, epochs=5, lr=0.01, device='cuda'):
    """
    Gradient Ascent unlearning: Maximize loss on forget set.
    
    Args:
        model: Model to unlearn
        forget_loader: Data loader for forget set
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        
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
        
        for data, target in forget_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = -criterion(output, target)  # Negative loss for gradient ascent
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            count += data.size(0)
        
        avg_loss = total_loss / count
        print(f"GA Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    return model


def finetuning_unlearning(model, retain_loader, epochs=10, lr=0.001, device='cuda'):
    """
    Fine-tuning unlearning: Train on retain set only.
    
    Args:
        model: Model to unlearn
        retain_loader: Data loader for retain set
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        
    Returns:
        Unlearned model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
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
        print(f"FT Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    
    return model


def random_label_unlearning(model, forget_loader, retain_loader=None, 
                            epochs=5, lr=0.001, device='cuda'):
    """
    Random Label unlearning (adversarial): Assign random labels to forget set.
    
    Args:
        model: Model to unlearn
        forget_loader: Data loader for forget set
        retain_loader: Data loader for retain set (optional, for balancing)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        
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
            # Random labels
            random_target = torch.randint(0, 10, target.shape).to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, random_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            count += data.size(0)
        
        # Optionally train on retain set with correct labels
        if retain_loader is not None:
            for data, target in retain_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
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
    
    Args:
        model_class: Model class to instantiate
        retain_loader: Data loader for retain set
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        model_kwargs: Additional arguments for model initialization
        
    Returns:
        Retrained model
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
        if (epoch + 1) % 5 == 0:
            print(f"Retrain Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        
        scheduler.step()
    
    return model


def salun_unlearning(model, forget_loader, retain_loader=None, epochs=5, 
                     lr=0.01, device='cuda'):
    """
    SalUn (Saliency-based Unlearning) - Simplified version.
    
    Uses gradient information to identify important weights for forgetting.
    
    Args:
        model: Model to unlearn
        forget_loader: Data loader for forget set
        retain_loader: Data loader for retain set (optional)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        
    Returns:
        Unlearned model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Compute saliency mask (which weights are important for forget set)
    model.eval()
    weight_importance = {name: torch.zeros_like(param) 
                        for name, param in model.named_parameters()}
    
    for data, target in forget_loader:
        data, target = data.to(device), target.to(device)
        
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                weight_importance[name] += param.grad.abs()
    
    # Normalize importance
    for name in weight_importance:
        weight_importance[name] = weight_importance[name] / len(forget_loader)
    
    # Create mask for top-k important weights
    all_importance = torch.cat([v.flatten() for v in weight_importance.values()])
    threshold = torch.quantile(all_importance, 0.5)  # Top 50%
    
    mask = {name: (importance > threshold).float() 
            for name, importance in weight_importance.items()}
    
    # Apply masked gradient ascent
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        
        for data, target in forget_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = -criterion(output, target)  # Negative for ascent
            loss.backward()
            
            # Apply mask
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
            
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            count += data.size(0)
        
        avg_loss = total_loss / count
        print(f"SalUn Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    return model


def apply_unlearning(model, method, forget_loader, retain_loader=None, 
                     epochs=5, lr=0.01, device='cuda'):
    """
    Apply unlearning method.
    
    Args:
        model: Model to unlearn
        method: Unlearning method name
        forget_loader: Data loader for forget set
        retain_loader: Data loader for retain set (optional)
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        
    Returns:
        Unlearned model
    """
    if method == 'gradient_ascent':
        return gradient_ascent_unlearning(model, forget_loader, epochs, lr, device)
    elif method == 'finetuning':
        return finetuning_unlearning(model, retain_loader, epochs, lr, device)
    elif method == 'random_label':
        return random_label_unlearning(model, forget_loader, retain_loader, epochs, lr, device)
    elif method == 'salun':
        return salun_unlearning(model, forget_loader, retain_loader, epochs, lr, device)
    else:
        raise ValueError(f"Unknown unlearning method: {method}")
