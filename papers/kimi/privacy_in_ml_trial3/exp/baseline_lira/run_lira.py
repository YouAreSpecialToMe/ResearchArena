"""
Run LiRA baseline with shadow model training.

This script trains shadow models and runs LiRA verification for comparison.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.models import SimpleCNN
from shared.data_loader import get_cifar10_loaders, create_forget_retain_split
from lira import LiRA


def train_shadow_model(model, train_loader, epochs=20, lr=0.01, device='cuda'):
    """Train a shadow model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%")
    
    return model


def train_shadow_models(n_shadows=8, dataset='cifar10', seed=42, device='cuda', save_dir='../../results/models'):
    """
    Train shadow models for LiRA.
    
    Each shadow model is trained on a random subset of the training data.
    We track which samples were included (IN) vs excluded (OUT) for each shadow.
    """
    print(f"\n{'='*60}")
    print(f"Training {n_shadows} Shadow Models for LiRA")
    print(f"Dataset: {dataset}, Seed: {seed}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load full training dataset
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create shadow model directory
    shadow_dir = os.path.join(save_dir, f'shadow_models_{dataset}_seed{seed}')
    os.makedirs(shadow_dir, exist_ok=True)
    
    shadow_models = []
    shadow_indices_list = []
    
    total_train_time = 0
    
    for i in range(n_shadows):
        print(f"\n--- Training Shadow Model {i+1}/{n_shadows} ---")
        
        # Randomly sample training indices (80% of training data)
        all_indices = list(range(len(train_dataset)))
        n_train = int(0.8 * len(train_dataset))
        shadow_indices = np.random.choice(all_indices, size=n_train, replace=False).tolist()
        shadow_indices_list.append(shadow_indices)
        
        # Create subset and dataloader
        shadow_subset = Subset(train_dataset, shadow_indices)
        shadow_loader = DataLoader(shadow_subset, batch_size=128, shuffle=True, num_workers=2)
        
        # Create and train model
        model = SimpleCNN(num_classes=num_classes)
        
        start_time = time.time()
        model = train_shadow_model(model, shadow_loader, epochs=20, lr=0.01, device=device)
        train_time = time.time() - start_time
        total_train_time += train_time
        
        # Save model
        model_path = os.path.join(shadow_dir, f'shadow_{i}.pth')
        torch.save(model.state_dict(), model_path)
        shadow_models.append(model)
        
        print(f"  Shadow model {i+1} trained in {train_time:.1f}s")
    
    # Save shadow indices
    indices_path = os.path.join(shadow_dir, 'shadow_indices.json')
    with open(indices_path, 'w') as f:
        json.dump(shadow_indices_list, f)
    
    print(f"\n{'='*60}")
    print(f"Shadow model training complete!")
    print(f"Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f} min)")
    print(f"Average per model: {total_train_time/n_shadows:.1f}s")
    print(f"{'='*60}\n")
    
    return shadow_models, shadow_indices_list, total_train_time


def run_lira_verification(n_shadows=8, dataset='cifar10', model='simplecnn', 
                          seed=42, device='cuda'):
    """
    Run LiRA verification on unlearned models.
    """
    print(f"\n{'='*60}")
    print(f"Running LiRA Verification")
    print(f"Dataset: {dataset}, Model: {model}, Seed: {seed}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load shadow models
    shadow_dir = os.path.join('../../results/models', f'shadow_models_{dataset}_seed{seed}')
    indices_path = os.path.join(shadow_dir, 'shadow_indices.json')
    
    if not os.path.exists(indices_path):
        print(f"Shadow models not found. Training {n_shadows} shadow models...")
        shadow_models, shadow_indices_list, train_time = train_shadow_models(
            n_shadows=n_shadows, dataset=dataset, seed=seed, device=device)
    else:
        print(f"Loading pre-trained shadow models from {shadow_dir}")
        with open(indices_path, 'r') as f:
            shadow_indices_list = json.load(f)
        
        shadow_models = []
        for i in range(n_shadows):
            model_i = SimpleCNN(num_classes=10)
            model_path = os.path.join(shadow_dir, f'shadow_{i}.pth')
            model_i.load_state_dict(torch.load(model_path, map_location=device))
            shadow_models.append(model_i)
        
        train_time = 0  # Already trained
    
    # Initialize LiRA
    lira = LiRA(shadow_models, device=device)
    
    # Load target model (unlearned)
    model_path = f'../../results/models/{dataset}_{model}_seed{seed}_unlearned.pth'
    if not os.path.exists(model_path):
        print(f"Unlearned model not found: {model_path}")
        print("Trying gold standard retrain...")
        model_path = f'../../results/models/{dataset}_{model}_seed{seed}_gold_standard.pth'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    target_model = SimpleCNN(num_classes=10)
    target_model.load_state_dict(torch.load(model_path, map_location=device))
    target_model = target_model.to(device)
    
    # Load forget/retain sets
    split_path = f'../../data/{dataset}_splits_seed{seed}.pkl'
    if not os.path.exists(split_path):
        print(f"Data splits not found: {split_path}")
        return None
    
    import pickle
    with open(split_path, 'rb') as f:
        splits = pickle.load(f)
    
    forget_data = torch.tensor(splits['forget_data']).float()
    forget_targets = torch.tensor(splits['forget_targets']).long()
    retain_data = torch.tensor(splits['retain_data']).float()
    retain_targets = torch.tensor(splits['retain_targets']).long()
    
    # Sample subset for verification (1000 samples each for speed)
    n_verify = min(1000, len(forget_data))
    forget_data = forget_data[:n_verify]
    forget_targets = forget_targets[:n_verify]
    retain_data = retain_data[:n_verify]
    retain_targets = retain_targets[:n_verify]
    
    print(f"\nRunning LiRA verification on {n_verify} forget and {n_verify} retain samples...")
    
    # Load full dataset for shadow model reference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    full_train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
    
    # Run verification
    start_time = time.time()
    results, all_scores, all_labels = lira.verify_unlearning(
        target_model, forget_data, forget_targets, 
        retain_data, retain_targets, shadow_indices_list, full_train_dataset
    )
    verify_time = time.time() - start_time
    
    results['total_time'] = train_time + verify_time if train_time > 0 else verify_time
    results['shadow_train_time'] = train_time
    results['verify_time'] = verify_time
    results['n_shadows'] = n_shadows
    results['dataset'] = dataset
    results['model'] = model
    results['seed'] = seed
    
    print(f"\n{'='*60}")
    print(f"LiRA Results:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  TPR@1%FPR: {results['tpr_at_1fpr']:.4f}")
    print(f"  Shadow training time: {train_time:.1f}s")
    print(f"  Verification time: {verify_time:.1f}s")
    print(f"  Total time: {results['total_time']:.1f}s")
    print(f"{'='*60}\n")
    
    # Save results
    results_dir = '../../results/metrics'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'lira_{dataset}_{model}_seed{seed}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run LiRA baseline')
    parser.add_argument('--n_shadows', type=int, default=8, help='Number of shadow models')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--model', type=str, default='simplecnn', help='Model architecture')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    results = run_lira_verification(
        n_shadows=args.n_shadows,
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        device=args.device
    )
    
    return results


if __name__ == '__main__':
    main()
