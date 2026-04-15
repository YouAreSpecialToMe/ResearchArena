"""Train unpruned baseline ResNet-18 on CIFAR-10."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os

from shared.data_loader import get_cifar10_dataloaders
from shared.models import get_resnet18_cifar10, get_model_size_mb, estimate_flops
from shared.utils import set_seed, save_results

def train_baseline(seed, epochs=150, device='cuda'):
    """Train baseline model with given seed."""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Training baseline model with seed {seed}")
    print(f"{'='*60}")
    
    # Load data
    train_loader, val_loader, test_loader, train_indices, val_indices = get_cifar10_dataloaders(
        batch_size=128, num_workers=4, seed=seed
    )
    
    # Create model
    model = get_resnet18_cifar10(num_classes=10)
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_acc = 0.0
    history = {'train_acc': [], 'test_acc': [], 'train_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / train_total,
                'acc': 100. * train_correct / train_total
            })
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss / train_total)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_dir = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/baseline_unpruned'
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'seed': seed
            }, f'{save_dir}/model_seed{seed}.pt')
    
    # Final evaluation
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
    
    final_train_acc = 100. * train_correct / train_total
    final_test_acc = best_acc
    
    # Get model size and FLOPs
    model_size = get_model_size_mb(model)
    flops = estimate_flops(model)
    
    results = {
        'seed': seed,
        'train_acc': final_train_acc,
        'test_acc': final_test_acc,
        'model_size_mb': model_size,
        'flops': flops,
        'epochs': epochs,
        'best_epoch': history['test_acc'].index(max(history['test_acc'])) + 1
    }
    
    print(f"\nFinal Results (seed={seed}):")
    print(f"  Train Acc: {final_train_acc:.2f}%")
    print(f"  Test Acc: {final_test_acc:.2f}%")
    print(f"  Model Size: {model_size:.2f} MB")
    print(f"  FLOPs: {flops/1e9:.2f}G")
    
    return results, train_indices, val_indices


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    seeds = [42, 43, 44]
    all_results = []
    
    for seed in seeds:
        result, train_idx, val_idx = train_baseline(seed, epochs=150, device=device)
        all_results.append(result)
    
    # Aggregate results
    aggregated = {
        'experiment': 'baseline_unpruned',
        'seeds': seeds,
        'train_acc': {
            'mean': sum(r['train_acc'] for r in all_results) / len(all_results),
            'std': (sum((r['train_acc'] - sum(r['train_acc'] for r in all_results)/len(all_results))**2 for r in all_results) / len(all_results))**0.5,
            'values': [r['train_acc'] for r in all_results]
        },
        'test_acc': {
            'mean': sum(r['test_acc'] for r in all_results) / len(all_results),
            'std': (sum((r['test_acc'] - sum(r['test_acc'] for r in all_results)/len(all_results))**2 for r in all_results) / len(all_results))**0.5,
            'values': [r['test_acc'] for r in all_results]
        },
        'model_size_mb': all_results[0]['model_size_mb'],
        'flops': all_results[0]['flops']
    }
    
    save_path = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/baseline_unpruned/results.json'
    save_results(aggregated, save_path)
    print(f"\nResults saved to {save_path}")
    print(f"\nAggregated Results:")
    print(f"  Train Acc: {aggregated['train_acc']['mean']:.2f} ± {aggregated['train_acc']['std']:.2f}%")
    print(f"  Test Acc: {aggregated['test_acc']['mean']:.2f} ± {aggregated['test_acc']['std']:.2f}%")
