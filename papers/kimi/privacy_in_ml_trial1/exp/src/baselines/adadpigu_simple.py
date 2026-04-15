"""
Simplified AdaDPIGU-style baseline with binary importance-based masking.
Based on Zhang & Xie (2025) - simplified for comparison.
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


def train_adadpigu_simple(
    model,
    trainloader,
    testloader,
    device,
    target_epsilon=3.0,
    target_delta=1e-5,
    pretrain_epochs=5,
    retention_ratio=0.6,
    epochs=30,
    lr=0.1,
    max_grad_norm=1.0,
    seed=42
):
    """
    Train model with simplified AdaDPIGU-style binary masking.
    
    Args:
        model: PyTorch model
        trainloader: Training data loader
        testloader: Test data loader
        device: Device
        target_epsilon: Target privacy budget
        target_delta: Target privacy delta
        pretrain_epochs: Epochs for DP pretraining to estimate importance
        retention_ratio: Fraction of parameters to retain (mask=1)
        epochs: Total training epochs
        lr: Learning rate
        max_grad_norm: Max gradient norm
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ModuleValidator.fix(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: DP pretraining to estimate importance
    print(f"Phase 1: DP pretraining for {pretrain_epochs} epochs to estimate importance...")
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    privacy_engine = PrivacyEngine()
    
    model, optimizer, trainloader_pt = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        target_epsilon=target_epsilon * (pretrain_epochs / epochs),  # Proportional budget
        target_delta=target_delta,
        epochs=pretrain_epochs,
        max_grad_norm=max_grad_norm,
    )
    
    # Accumulate gradient magnitudes during pretraining
    grad_accumulator = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_accumulator[name] = torch.zeros_like(param.data)
    
    for epoch in range(pretrain_epochs):
        model.train()
        for inputs, targets in tqdm(trainloader_pt, desc=f'Pretrain Epoch {epoch+1}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Accumulate gradient magnitudes
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_accumulator[name] += param.grad.abs()
            
            optimizer.step()
    
    epsilon_used = privacy_engine.get_epsilon(delta=target_delta)
    epsilon_remaining = target_epsilon - epsilon_used
    
    # Create binary mask based on accumulated gradient magnitudes
    print("Creating binary importance mask...")
    all_grad_mags = []
    for name, grad_mag in grad_accumulator.items():
        all_grad_mags.append(grad_mag.flatten())
    
    all_grad_mags = torch.cat(all_grad_mags)
    
    # Find threshold for retention ratio
    k = int((1 - retention_ratio) * len(all_grad_mags))
    threshold = torch.kthvalue(all_grad_mags, k)[0] if k > 0 and k < len(all_grad_mags) else 0.0
    
    # Create masks
    masks = {}
    for name, grad_mag in grad_accumulator.items():
        masks[name] = (grad_mag >= threshold).float().to(device)
    
    retained = sum((m == 1).sum().item() for m in masks.values())
    total = sum(m.numel() for m in masks.values())
    print(f"Retained {retained}/{total} parameters ({retained/total:.2%})")
    
    # Phase 2: Main training with binary masking
    print(f"Phase 2: Main training with binary masking...")
    
    # Re-initialize model
    model = model._module if hasattr(model, '_module') else model
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    privacy_engine2 = PrivacyEngine()
    
    model, optimizer, trainloader_main = privacy_engine2.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        target_epsilon=epsilon_remaining,
        target_delta=target_delta,
        epochs=epochs - pretrain_epochs,
        max_grad_norm=max_grad_norm,
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.9)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epsilon': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs - pretrain_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(trainloader_main, desc=f'Epoch {epoch+1}/{epochs-pretrain_epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply binary mask
            for name, param in model.named_parameters():
                if name in masks and param.grad is not None:
                    param.grad = param.grad * masks[name]
            
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
        epsilon = epsilon_used + privacy_engine2.get_epsilon(delta=target_delta)
        
        history['train_loss'].append(train_loss / len(trainloader_main))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epsilon'].append(epsilon)
        
        print(f'Epoch {epoch+1}/{epochs-pretrain_epochs}: Train Acc: {train_acc:.2f}%, '
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
    parser.add_argument('--retention_ratio', type=float, default=0.6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./exp/baseline_adadpigu')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainloader, testloader = get_data_loaders(args.dataset, batch_size=256)
    model = ResNet18(num_classes=10)
    
    print(f'Training AdaDPIGU-style baseline on {args.dataset}')
    
    model, history, runtime = train_adadpigu_simple(
        model, trainloader, testloader, device,
        target_epsilon=args.target_epsilon,
        retention_ratio=args.retention_ratio,
        epochs=args.epochs,
        seed=args.seed
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'experiment': 'adadpigu_simple',
        'dataset': args.dataset,
        'target_epsilon': args.target_epsilon,
        'retention_ratio': args.retention_ratio,
        'seed': args.seed,
        'final_test_acc': history['test_acc'][-1],
        'final_epsilon': history['epsilon'][-1],
        'runtime_minutes': runtime / 60,
        'history': history
    }
    
    results_path = os.path.join(args.output_dir,
        f'results_{args.dataset}_eps{args.target_epsilon}_seed{args.seed}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
