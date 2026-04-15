#!/usr/bin/env python3
"""
Streamlined experiment runner for CASS-ViM.
"""

import os
import sys
import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from src.minimal_models import MinimalCASSViM, MinimalVMamba, MinimalLocalMamba


def get_data(batch_size=128):
    """Get CIFAR-100 dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, download=True, transform=transform_test
    )
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader


def train_model(model, trainloader, testloader, epochs, device):
    """Train a model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    best_acc = 0
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        # Train
        model.train()
        correct, total, running_loss = 0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        
        # Test
        model.eval()
        correct, total, test_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f'  Epoch {epoch}: Train={train_acc:.1f}%, Test={test_acc:.1f}%, Best={best_acc:.1f}%')
    
    return best_acc, train_accs, test_accs


def run_experiment(model_name, seed, epochs, device):
    """Run a single experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f'\n{"="*60}')
    print(f'Model: {model_name}, Seed: {seed}')
    print(f'{"="*60}')
    
    # Create model
    if model_name == 'cassvim_4d':
        model = MinimalCASSViM(num_directions=4, selector_type='gradient',
                               topks=[1, 1, 1, 1])
    elif model_name == 'cassvim_8d':
        model = MinimalCASSViM(num_directions=8, selector_type='gradient',
                               topks=[1, 1, 1, 1])
    elif model_name == 'vmamba':
        model = MinimalVMamba()
    elif model_name == 'localmamba':
        model = MinimalLocalMamba()
    elif model_name == 'random_selection':
        model = MinimalCASSViM(num_directions=4, selector_type='random',
                               topks=[1, 1, 1, 1])
    elif model_name == 'fixed_perlayer':
        model = MinimalCASSViM(num_directions=4, selector_type='fixed',
                               topks=[1, 1, 1, 1])
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params/1e6:.2f}M')
    
    # Get data
    trainloader, testloader = get_data()
    
    # Train
    start = time.time()
    best_acc, train_accs, test_accs = train_model(model, trainloader, testloader, epochs, device)
    elapsed = time.time() - start
    
    print(f'Result: Best Acc = {best_acc:.2f}%, Time = {elapsed/60:.1f} min')
    
    return {
        'model': model_name,
        'seed': seed,
        'best_acc': best_acc,
        'final_acc': test_accs[-1],
        'train_time': elapsed / 60,
        'n_params': n_params,
        'test_accs': test_accs
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    epochs = 30  # Fast validation
    seeds = [42, 123, 456]
    
    models = ['vmamba', 'localmamba', 'cassvim_4d', 'cassvim_8d', 
              'random_selection', 'fixed_perlayer']
    
    all_results = {}
    
    for model_name in models:
        model_results = []
        for seed in seeds:
            result = run_experiment(model_name, seed, epochs, device)
            model_results.append(result)
            
            # Save individual result
            os.makedirs('./results', exist_ok=True)
            with open(f'./results/{model_name}_seed{seed}.json', 'w') as f:
                json.dump(result, f, indent=2)
        
        # Aggregate
        accs = [r['best_acc'] for r in model_results]
        all_results[model_name] = {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'seeds': seeds,
            'accs': accs,
            'avg_time': np.mean([r['train_time'] for r in model_results]),
            'n_params': model_results[0]['n_params']
        }
    
    # Save aggregated
    with open('./results/aggregated_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    print(f'{"Model":<20} {"Accuracy":>15} {"Time (min)":>12}')
    print('-'*60)
    for model, res in all_results.items():
        print(f'{model:<20} {res["mean"]:>7.2f}±{res["std"]:<5.2f} {res["avg_time"]:>10.1f}')
    print('='*60)
    
    # Success criteria
    print('\nSuccess Criteria:')
    if 'vmamba' in all_results and 'cassvim_4d' in all_results:
        diff = all_results['cassvim_4d']['mean'] - all_results['vmamba']['mean']
        status = 'PASS' if abs(diff) <= 1.0 else 'FAIL'
        print(f'1. CASS-ViM within 1% of VMamba: {diff:+.2f}% [{status}]')
    
    if 'localmamba' in all_results and 'cassvim_4d' in all_results:
        diff = all_results['cassvim_4d']['mean'] - all_results['localmamba']['mean']
        status = 'PASS' if diff > 0 else 'FAIL'
        print(f'2. CASS-ViM > LocalMamba: {diff:+.2f}% [{status}]')
    
    if 'random_selection' in all_results and 'cassvim_4d' in all_results:
        diff = all_results['cassvim_4d']['mean'] - all_results['random_selection']['mean']
        status = 'PASS' if diff >= 0.5 else 'FAIL'
        print(f'3. Gradient > Random by 0.5%: {diff:+.2f}% [{status}]')
    
    if 'cassvim_4d' in all_results and 'cassvim_8d' in all_results:
        diff = abs(all_results['cassvim_4d']['mean'] - all_results['cassvim_8d']['mean'])
        status = 'PASS' if diff <= 1.0 else 'FAIL'
        print(f'4. 4D vs 8D similar: diff={diff:.2f}% [{status}]')


if __name__ == '__main__':
    main()
