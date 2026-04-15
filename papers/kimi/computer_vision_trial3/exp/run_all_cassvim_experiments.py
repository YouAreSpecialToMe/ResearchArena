"""
Comprehensive CASS-ViM experiment runner.
Runs all experiments: CASS-ViM-4D, CASS-ViM-8D, ablations, and baselines.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))
from src.optimized_models import (
    OptimizedCASSViM, OptimizedVMamba, OptimizedLocalMamba
)


def get_cifar100_dataloaders(batch_size=128, num_workers=4):
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
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader


def train_epoch(model, trainloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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
    
    return running_loss / len(trainloader), 100. * correct / total


def test(model, testloader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    
    return avg_loss, acc


def create_model(model_name, embed_dims=[48, 96, 192, 384], depths=[2, 2, 3, 2]):
    """Create model by name."""
    if model_name == 'cassvim_4d':
        return OptimizedCASSViM(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths,
            num_directions=4, expand=2,
            window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
            selector_type='gradient'
        )
    elif model_name == 'cassvim_8d':
        return OptimizedCASSViM(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths,
            num_directions=8, expand=2,
            window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
            selector_type='gradient'
        )
    elif model_name == 'vmamba':
        return OptimizedVMamba(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths
        )
    elif model_name == 'localmamba':
        return OptimizedLocalMamba(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths
        )
    elif model_name == 'random_selection':
        return OptimizedCASSViM(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths,
            num_directions=4, expand=2,
            window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
            selector_type='random'
        )
    elif model_name == 'fixed_perlayer':
        return OptimizedCASSViM(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths,
            num_directions=4, expand=2,
            window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
            selector_type='fixed'
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(model_name, seed, epochs, batch_size=128, lr=1e-3, 
                   weight_decay=0.05, device='cuda', save_dir='./checkpoints'):
    """Run a single experiment."""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name} | Seed: {seed} | Epochs: {epochs}")
    print(f"{'='*70}\n")
    
    # Data
    print("Loading CIFAR-100...")
    trainloader, testloader = get_cifar100_dataloaders(batch_size)
    
    # Model
    print(f"Creating model: {model_name}")
    model = create_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (epochs - 5)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    
    best_acc = 0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, epoch)
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%, Time: {epoch_time:.1f}s")
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    total_time = time.time() - start_time
    
    # Results
    results = {
        'model': model_name,
        'seed': seed,
        'epochs': epochs,
        'best_test_acc': best_acc,
        'final_test_acc': test_accs[-1],
        'train_time_seconds': total_time,
        'train_time_minutes': total_time / 60,
        'n_parameters': n_params,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/{model_name}', exist_ok=True)
    
    results_path = f'{save_dir}/{model_name}/results_seed{seed}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save checkpoint
    checkpoint_path = f'{save_dir}/{model_name}/checkpoint_seed{seed}.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'results': results
    }, checkpoint_path)
    
    print(f"\n{'='*70}")
    print(f"Completed: {model_name}, Seed: {seed}")
    print(f"Best Acc: {best_acc:.2f}%")
    print(f"Time: {total_time/60:.1f} min")
    print(f"{'='*70}")
    
    return results


def aggregate_results(model_name, save_dir='./checkpoints'):
    """Aggregate results across seeds."""
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        path = f'{save_dir}/{model_name}/results_seed{seed}.json'
        if os.path.exists(path):
            with open(path) as f:
                all_results.append(json.load(f))
    
    if not all_results:
        return None
    
    best_accs = [r['best_test_acc'] for r in all_results]
    final_accs = [r['final_test_acc'] for r in all_results]
    times = [r['train_time_minutes'] for r in all_results]
    
    aggregated = {
        'model': model_name,
        'n_seeds': len(all_results),
        'best_acc_mean': np.mean(best_accs),
        'best_acc_std': np.std(best_accs),
        'final_acc_mean': np.mean(final_accs),
        'final_acc_std': np.std(final_accs),
        'train_time_mean': np.mean(times),
        'individual_results': all_results
    }
    
    path = f'{save_dir}/{model_name}/aggregated.json'
    with open(path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['vmamba', 'localmamba', 'cassvim_4d', 'cassvim_8d', 
                                'random_selection', 'fixed_perlayer'],
                        help='Models to run')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Run all experiments
    all_results = {}
    
    for model_name in args.models:
        print(f"\n\n{'#'*70}")
        print(f"# Running {model_name}")
        print(f"{'#'*70}")
        
        for seed in args.seeds:
            result = run_experiment(
                model_name=model_name,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                save_dir=args.save_dir
            )
        
        # Aggregate
        aggregated = aggregate_results(model_name, args.save_dir)
        all_results[model_name] = aggregated
        
        if aggregated:
            print(f"\n{model_name} Aggregated:")
            print(f"  Best Acc: {aggregated['best_acc_mean']:.2f} ± {aggregated['best_acc_std']:.2f}%")
            print(f"  Time: {aggregated['train_time_mean']:.1f} min")
    
    # Save master results
    master_results = {
        'experiment_info': {
            'title': 'CASS-ViM Comprehensive Experiments',
            'dataset': 'CIFAR-100',
            'epochs': args.epochs,
            'seeds': args.seeds,
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': all_results
    }
    
    with open(f'{args.save_dir}/master_results.json', 'w') as f:
        json.dump(master_results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*70}")
    
    # Print summary table
    print("\nSummary Table:")
    print("-" * 70)
    print(f"{'Model':<20} {'Best Acc (%)':<20} {'Time (min)':<15}")
    print("-" * 70)
    for model_name, agg in all_results.items():
        if agg:
            print(f"{model_name:<20} {agg['best_acc_mean']:>6.2f} ± {agg['best_acc_std']:<6.2f}   "
                  f"{agg['train_time_mean']:>8.1f}")
    print("-" * 70)


if __name__ == '__main__':
    main()
