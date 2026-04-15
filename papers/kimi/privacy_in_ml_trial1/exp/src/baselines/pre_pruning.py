"""
Pre-pruning baseline from Adamczewski et al. (2023).
Uses "public" data to pre-prune the model before DP training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np
import json
import os
import time
from tqdm import tqdm
import copy


def pre_prune_model(model, dataloader, device, target_sparsity=0.7, pretrain_epochs=10):
    """
    Pre-train model on "public" data and prune based on magnitude.
    
    Args:
        model: PyTorch model
        dataloader: Public data loader
        device: Device
        target_sparsity: Target sparsity ratio
        pretrain_epochs: Epochs to pretrain
    
    Returns:
        pruned_model: Pre-pruned model
        mask: Binary mask for retained parameters
    """
    # Make a copy for pretraining
    pretrain_model = copy.deepcopy(model)
    pretrain_model = pretrain_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(pretrain_model.parameters(), lr=0.1, momentum=0.9)
    
    print(f"Pre-training on 'public' data for {pretrain_epochs} epochs...")
    
    # Pretrain (non-private)
    for epoch in range(pretrain_epochs):
        pretrain_model.train()
        for inputs, targets in tqdm(dataloader, desc=f'Pretrain Epoch {epoch+1}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = pretrain_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Compute magnitude-based pruning mask
    all_weights = []
    for name, param in pretrain_model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            all_weights.append(param.data.abs().flatten())
    
    all_weights = torch.cat(all_weights)
    k = int(target_sparsity * len(all_weights))
    threshold = torch.kthvalue(all_weights, k)[0] if k > 0 else 0.0
    
    # Create binary mask
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            mask[name] = (param.data.abs() >= threshold).float().to(device)
    
    print(f"Pre-pruning complete. Target sparsity: {target_sparsity:.2%}")
    
    return model, mask


def train_pre_pruning_dp(
    model,
    trainloader,
    testloader,
    device,
    target_epsilon=3.0,
    target_delta=1e-5,
    target_sparsity=0.7,
    epochs=30,
    lr=0.1,
    max_grad_norm=1.0,
    seed=42,
    public_data_ratio=0.1
):
    """
    Train model with pre-pruning (Adamczewski et al., 2023).
    
    Args:
        model: PyTorch model
        trainloader: Training data loader
        testloader: Test data loader
        device: Device
        target_epsilon: Target privacy budget
        target_delta: Target privacy delta
        target_sparsity: Target sparsity ratio
        epochs: Training epochs
        lr: Learning rate
        max_grad_norm: Max gradient norm
        seed: Random seed
        public_data_ratio: Fraction of data to use as "public" for pre-pruning
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Extract a subset for "public" pretraining
    # In practice, this would be actual public data
    all_data = []
    all_labels = []
    for inputs, targets in trainloader:
        all_data.append(inputs)
        all_labels.append(targets)
    
    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)
    
    # Shuffle
    perm = torch.randperm(len(all_data))
    all_data = all_data[perm]
    all_labels = all_labels[perm]
    
    # Split into public and private
    n_public = int(len(all_data) * public_data_ratio)
    public_data = all_data[:n_public]
    public_labels = all_labels[:n_public]
    
    public_dataset = torch.utils.data.TensorDataset(public_data, public_labels)
    public_loader = torch.utils.data.DataLoader(public_dataset, batch_size=256, shuffle=True)
    
    # Pre-prune using "public" data
    model, mask = pre_prune_model(model, public_loader, device, target_sparsity)
    
    # Prepare for DP training
    model = ModuleValidator.fix(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Setup privacy engine
    privacy_engine = PrivacyEngine()
    
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
    
    # Apply mask during training (gradient dropping)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.9)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epsilon': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply mask: zero out gradients for pruned parameters
            for name, param in model.named_parameters():
                if name in mask and param.grad is not None:
                    param.grad = param.grad * mask[name]
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        
        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epsilon'].append(epsilon)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%, ε: {epsilon:.2f}')
    
    runtime = time.time() - start_time
    
    return model, history, runtime


def main():
    import argparse
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.resnet import ResNet18
    from data_loader import get_data_loaders
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--target_epsilon', type=float, default=3.0)
    parser.add_argument('--target_sparsity', type=float, default=0.7)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./exp/baseline_prepruning')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainloader, testloader = get_data_loaders(args.dataset, batch_size=256)
    model = ResNet18(num_classes=10)
    
    print(f'Training pre-pruning baseline on {args.dataset}')
    
    model, history, runtime = train_pre_pruning_dp(
        model, trainloader, testloader, device,
        target_epsilon=args.target_epsilon,
        target_sparsity=args.target_sparsity,
        epochs=args.epochs,
        seed=args.seed
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'experiment': 'pre_pruning',
        'dataset': args.dataset,
        'target_epsilon': args.target_epsilon,
        'target_sparsity': args.target_sparsity,
        'seed': args.seed,
        'final_test_acc': history['test_acc'][-1],
        'final_epsilon': history['epsilon'][-1],
        'runtime_minutes': runtime / 60,
        'history': history
    }
    
    results_path = os.path.join(args.output_dir,
        f'results_{args.dataset}_eps{args.target_epsilon}_sparsity{args.target_sparsity}_seed{args.seed}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
