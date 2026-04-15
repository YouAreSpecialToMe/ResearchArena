"""
Utility functions for PRISM experiments.
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.1, 
                momentum=0.9, weight_decay=5e-4, save_path=None):
    """Train a model with standard settings."""
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        scheduler.step()
    
    return history, best_val_acc


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_probs.append(probs.cpu())
            all_labels.append(targets.cpu())
    
    accuracy = 100. * correct / total
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    return accuracy, all_probs, all_labels


def save_results(results, save_path):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(load_path):
    """Load results from JSON file."""
    with open(load_path, 'r') as f:
        return json.load(f)


def get_gradient_norms(model, data_loader, device):
    """
    Compute per-sample gradient norms for each layer.
    Returns: dict {layer_name: gradient_norms_array}
    """
    model = model.to(device)
    model.eval()
    
    all_grad_norms = {}
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        for i in range(len(inputs)):
            x = inputs[i:i+1]
            y = targets[i:i+1]
            
            model.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            
            # Collect gradient norms per layer
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in all_grad_norms:
                        all_grad_norms[name] = []
                    all_grad_norms[name].append(param.grad.norm().item())
    
    # Convert to numpy arrays
    for name in all_grad_norms:
        all_grad_norms[name] = np.array(all_grad_norms[name])
    
    return all_grad_norms


def aggregate_layer_gradients(grad_norms, model_type='resnet18'):
    """
    Aggregate gradient norms by layer.
    Returns: dict {layer_name: mean_gradient_norm}
    """
    layer_grads = {}
    
    if model_type == 'resnet18':
        # Group by layer
        layer_groups = {
            'conv1': [],
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
            'fc': []
        }
        
        for name, grads in grad_norms.items():
            if 'conv1' in name or 'bn1' in name:
                layer_groups['conv1'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
            elif 'layer1' in name:
                layer_groups['layer1'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
            elif 'layer2' in name:
                layer_groups['layer2'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
            elif 'layer3' in name:
                layer_groups['layer3'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
            elif 'layer4' in name:
                layer_groups['layer4'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
            elif 'fc' in name:
                layer_groups['fc'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
        
        for layer_name, grads in layer_groups.items():
            if grads:
                layer_grads[layer_name] = np.mean(grads)
    
    elif model_type == 'vgg16':
        # Group by conv blocks and fc layers
        layer_groups = {}
        for i in range(5):  # 5 conv blocks
            layer_groups[f'conv_block_{i}'] = []
        for i in range(3):  # 3 fc layers
            layer_groups[f'fc{i+1}'] = []
        
        for name, grads in grad_norms.items():
            if 'features' in name:
                # Determine which conv block based on parameter index
                # This is simplified; in practice we'd need more sophisticated grouping
                layer_groups['conv_block_0'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
            elif 'classifier' in name:
                if '0' in name or '1' in name:
                    layer_groups['fc1'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
                elif '3' in name or '4' in name:
                    layer_groups['fc2'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
                elif '6' in name:
                    layer_groups['fc3'].extend(grads.tolist() if hasattr(grads, 'tolist') else [grads])
        
        for layer_name, grads in layer_groups.items():
            if grads:
                layer_grads[layer_name] = np.mean(grads)
    
    return layer_grads
