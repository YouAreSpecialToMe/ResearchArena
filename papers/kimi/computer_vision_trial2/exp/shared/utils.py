"""
Utility functions for TTA experiments.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results, filepath):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_prototypes_path(dataset='cifar10'):
    """Get path for saving/loading prototypes"""
    return f'models/prototypes_{dataset}.pt'


def save_prototypes(prototypes, dataset='cifar10'):
    """Save computed prototypes"""
    os.makedirs('models', exist_ok=True)
    torch.save(prototypes, get_prototypes_path(dataset))


def load_prototypes(dataset='cifar10', device='cuda'):
    """Load pre-computed prototypes"""
    path = get_prototypes_path(dataset)
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    return None


def compute_and_save_prototypes(model, dataset='cifar10', data_dir='./data', device='cuda'):
    """Compute and save prototypes for a dataset"""
    from .data_loader import load_cifar10, load_cifar100
    from .models import compute_prototypes
    
    print(f"Computing prototypes for {dataset}...")
    
    if dataset == 'cifar10':
        loader, _ = load_cifar10(data_dir, train=True, batch_size=128)
        num_classes = 10
    elif dataset == 'cifar100':
        loader, _ = load_cifar100(data_dir, train=True, batch_size=128)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    prototypes = compute_prototypes(model, loader, device, num_classes)
    save_prototypes(prototypes, dataset)
    print(f"Saved prototypes to {get_prototypes_path(dataset)}")
    
    return prototypes


def get_meta_apn_path(dataset='cifar10'):
    """Get path for saving/loading Meta-APN"""
    return f'models/meta_apn_{dataset}.pt'


def save_meta_apn(model, dataset='cifar10'):
    """Save trained Meta-APN"""
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), get_meta_apn_path(dataset))


def load_meta_apn(feature_dim, num_classes, dataset='cifar10', device='cuda'):
    """Load trained Meta-APN"""
    from .models import MetaAPN
    
    model = MetaAPN(feature_dim, num_classes).to(device)
    path = get_meta_apn_path(dataset)
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded Meta-APN from {path}")
    else:
        print(f"Warning: No Meta-APN found at {path}")
    
    return model


class Logger:
    """Simple logger for experiments"""
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')


def configure_model(model):
    """Configure model for test-time adaptation"""
    model.train()
    # Disable dropout if present
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        elif isinstance(m, nn.Dropout2d):
            m.p = 0.0
    return model


def collect_params(model, adapt_bias=False):
    """Collect parameters for adaptation (BN stats and affine params)"""
    params = []
    names = []
    
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for np, p in m.named_parameters():
                if p.requires_grad:
                    names.append(f"{nm}.{np}")
                    params.append(p)
    
    return params, names


def copy_model(model):
    """Create a copy of the model"""
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def load_model_state(model, state_dict):
    """Load model state"""
    model.load_state_dict(state_dict)


def check_model(model):
    """Check model train/eval mode and parameters"""
    print(f"Model training mode: {model.training}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: requires_grad=True")
