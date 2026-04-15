"""
Training utilities for LGSA experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time


def train_model(model, train_loader, val_loader=None, epochs=30, lr=0.1, 
                momentum=0.9, weight_decay=5e-4, device='cuda', verbose=True):
    """
    Train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of epochs
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: Weight decay
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        model, training_history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') if verbose else train_loader
        for data, target in iterator:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            val_acc = val_correct / val_total
            history['val_acc'].append(val_acc)
        else:
            val_acc = 0.0
        
        if verbose:
            print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')
        
        scheduler.step()
    
    return model, history


def train_shadow_models(train_dataset, num_models=4, subset_ratio=0.8, 
                        epochs=20, model_name='resnet18', num_classes=10, 
                        input_channels=3, device='cuda', seed=42):
    """
    Train shadow models for LiRA.
    
    Args:
        train_dataset: Full training dataset
        num_models: Number of shadow models
        subset_ratio: Ratio of data to use for each shadow model
        epochs: Training epochs per shadow model
        model_name: Model architecture
        num_classes: Number of classes
        input_channels: Input channels
        device: Device to train on
        seed: Random seed
        
    Returns:
        List of trained shadow models and their training indices
    """
    from .models import get_model
    from .data_loader import get_dataloader
    from torch.utils.data import Subset
    import numpy as np
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    shadow_models = []
    shadow_indices = []
    
    n_total = len(train_dataset)
    
    for i in range(num_models):
        print(f"Training shadow model {i+1}/{num_models}")
        
        # Random subset
        subset_size = int(n_total * subset_ratio)
        indices = np.random.choice(n_total, subset_size, replace=False)
        shadow_indices.append(indices)
        
        # Train model
        model = get_model(model_name, num_classes, input_channels)
        subset_loader = get_dataloader(train_dataset, indices=indices.tolist(), 
                                       batch_size=128, shuffle=True)
        
        model, _ = train_model(model, subset_loader, epochs=epochs, device=device, verbose=False)
        shadow_models.append(model)
    
    return shadow_models, shadow_indices


def save_model(model, path):
    """Save model checkpoint."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cuda'):
    """Load model checkpoint."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
