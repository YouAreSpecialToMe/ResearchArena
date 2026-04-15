"""
Training script for Fast CASS-ViM on CIFAR-100.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.fast_models import FastCASSViM, FastVMamba, FastLocalMamba


def get_cifar100_dataloaders(batch_size=128, num_workers=2):
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
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(trainloader)}] Loss: {running_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%')
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cassvim_4d', 
                        choices=['cassvim_4d', 'cassvim_8d', 'vmamba', 'localmamba',
                                'random_selection', 'fixed_perlayer'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default=None)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Data
    print('Loading CIFAR-100...')
    trainloader, testloader = get_cifar100_dataloaders(args.batch_size)
    
    # Create model
    print(f'Creating model: {args.model}')
    
    embed_dims = [48, 96, 192, 384]
    depths = [2, 2, 3, 2]
    
    if args.model == 'cassvim_4d':
        model = FastCASSViM(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths,
            num_directions=4, expand=2,
            window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
            selector_type='gradient'
        )
    elif args.model == 'cassvim_8d':
        model = FastCASSViM(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths,
            num_directions=8, expand=2,
            window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
            selector_type='gradient'
        )
    elif args.model == 'vmamba':
        model = FastVMamba(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths
        )
    elif args.model == 'localmamba':
        model = FastLocalMamba(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths
        )
    elif args.model == 'random_selection':
        model = FastCASSViM(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths,
            num_directions=4, expand=2,
            window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
            selector_type='random'
        )
    elif args.model == 'fixed_perlayer':
        model = FastCASSViM(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dims=embed_dims, depths=depths,
            num_directions=4, expand=2,
            window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
            selector_type='fixed'
        )
    
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params/1e6:.2f}M')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR schedule
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (args.epochs - 5)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f'\nTraining for {args.epochs} epochs...')
    start_time = time.time()
    
    best_acc = 0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, epoch)
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.1f}s')
        
        if test_acc > best_acc:
            best_acc = test_acc
            exp_name = args.exp_name or args.model
            save_path = os.path.join(args.save_dir, f'{exp_name}_seed{args.seed}_best.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'args': vars(args)
            }, save_path)
    
    total_time = time.time() - start_time
    
    results = {
        'model': args.model,
        'seed': args.seed,
        'epochs': args.epochs,
        'best_test_acc': best_acc,
        'final_test_acc': test_accs[-1],
        'train_time_seconds': total_time,
        'train_time_minutes': total_time / 60,
        'n_parameters': n_params,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }
    
    exp_name = args.exp_name or args.model
    results_path = os.path.join(args.save_dir, f'{exp_name}_seed{args.seed}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n{"="*50}')
    print(f'Completed: {args.model}, Seed: {args.seed}')
    print(f'Best Acc: {best_acc:.2f}%')
    print(f'Time: {total_time/60:.1f} min')
    print(f'{"="*50}')
    
    return results


if __name__ == '__main__':
    main()
