"""
Train base models for LGSA experiments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import argparse
from shared.models import get_model, count_parameters
from shared.data_loader import load_dataset, create_forget_retain_splits, save_splits, get_dataloader
from shared.training import train_model, save_model
from shared.metrics import compute_accuracy


def train_base_model(dataset_name, model_name, seed=42, epochs=30, batch_size=128, 
                     lr=0.1, device='cuda', save_dir='results/models'):
    """
    Train a base model on full dataset.
    
    Args:
        dataset_name: 'cifar10' or 'fashion-mnist'
        model_name: 'resnet18' or 'simplecnn'
        seed: Random seed
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save model
        
    Returns:
        Trained model, training history
    """
    # Set seeds
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    
    # Load dataset
    print(f"Loading {dataset_name}...")
    train_dataset, test_dataset, num_classes, input_channels = load_dataset(dataset_name)
    
    # Create splits
    print("Creating forget/retain/val splits...")
    forget_indices, retain_indices, val_indices = create_forget_retain_splits(
        train_dataset, forget_ratio=0.1, seed=seed)
    
    # Save splits
    splits_path = f'data/{dataset_name}_splits_seed{seed}.pkl'
    save_splits({
        'forget': forget_indices,
        'retain': retain_indices,
        'val': val_indices,
        'seed': seed
    }, splits_path)
    print(f"Saved splits to {splits_path}")
    
    # Train on full dataset (forget + retain)
    full_train_indices = forget_indices + retain_indices
    train_loader = get_dataloader(train_dataset, indices=full_train_indices, 
                                   batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = get_dataloader(train_dataset, indices=val_indices,
                                 batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"Creating {model_name}...")
    model = get_model(model_name, num_classes=num_classes, input_channels=input_channels)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print(f"Training for {epochs} epochs...")
    model, history = train_model(model, train_loader, val_loader, epochs=epochs, 
                                  lr=lr, device=device, verbose=True)
    
    # Evaluate
    train_acc = compute_accuracy(model, train_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)
    print(f"Final Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save model
    model_path = os.path.join(save_dir, f'{dataset_name}_{model_name}_seed{seed}_base.pth')
    save_model(model, model_path)
    print(f"Saved model to {model_path}")
    
    # Save training info
    info = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'n_parameters': n_params,
        'final_train_acc': train_acc,
        'final_test_acc': test_acc,
        'history': history
    }
    
    info_path = os.path.join(save_dir, f'{dataset_name}_{model_name}_seed{seed}_base_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    return model, info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashion-mnist'])
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'simplecnn'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train_base_model(
        dataset_name=args.dataset,
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
