"""
Standard DP-SGD training using Opacus.
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


def train_standard_dp(
    model,
    trainloader,
    testloader,
    device,
    target_epsilon=3.0,
    target_delta=1e-5,
    epochs=30,
    lr=0.1,
    max_grad_norm=1.0,
    seed=42
):
    """
    Train model with standard DP-SGD using Opacus.
    
    Args:
        model: PyTorch model
        trainloader: Training data loader
        testloader: Test data loader
        device: Device to train on
        target_epsilon: Target privacy budget
        target_delta: Target privacy delta
        epochs: Number of training epochs
        lr: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare model for DP training
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
    
    # Learning rate scheduler
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
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (total // targets.size(0) + 1),
                'acc': 100. * correct / total
            })
        
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
    from models.resnet import ResNet18
    from models.convnet import ConvNet
    from models.mlp import MLP
    from data_loader import get_data_loaders
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'purchase100'])
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'convnet', 'mlp'])
    parser.add_argument('--target_epsilon', type=float, default=3.0)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./exp/baseline_standard')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    trainloader, testloader = get_data_loaders(
        args.dataset, batch_size=args.batch_size)
    
    # Get number of classes
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 100
    
    # Create model
    if args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'convnet':
        model = ConvNet(num_classes=num_classes)
    else:
        model = MLP(input_dim=600, num_classes=num_classes)
    
    print(f'Training {args.model} on {args.dataset} with ε={args.target_epsilon}')
    
    # Train
    model, history, runtime = train_standard_dp(
        model, trainloader, testloader, device,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        epochs=args.epochs,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir, 
        f'model_{args.dataset}_{args.model}_eps{args.target_epsilon}_seed{args.seed}.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save history
    results = {
        'experiment': 'standard_dp',
        'dataset': args.dataset,
        'model': args.model,
        'target_epsilon': args.target_epsilon,
        'target_delta': args.target_delta,
        'epochs': args.epochs,
        'seed': args.seed,
        'final_test_acc': history['test_acc'][-1],
        'final_epsilon': history['epsilon'][-1],
        'runtime_minutes': runtime / 60,
        'history': history
    }
    
    results_path = os.path.join(args.output_dir,
        f'results_{args.dataset}_{args.model}_eps{args.target_epsilon}_seed{args.seed}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to {results_path}')
    print(f'Final test accuracy: {history["test_acc"][-1]:.2f}%')
    print(f'Final epsilon: {history["epsilon"][-1]:.2f}')


if __name__ == '__main__':
    main()
