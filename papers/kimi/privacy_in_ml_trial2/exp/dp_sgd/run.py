"""DP-SGD baseline using Opacus."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import torch
import torch.nn as nn
from tqdm import tqdm
import os

from opacus import PrivacyEngine
from shared.data_loader import get_cifar10_dataloaders
from shared.models import get_resnet18_cifar10, get_model_size_mb, estimate_flops
from shared.utils import set_seed, save_results


def train_dp_sgd(seed, target_epsilon=4, delta=1e-5, epochs=100, device='cuda'):
    """Train model with DP-SGD."""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"DP-SGD - Target ε={target_epsilon}, Seed: {seed}")
    print(f"{'='*60}")
    
    # Load data
    train_loader, val_loader, test_loader, train_indices, val_indices = get_cifar10_dataloaders(
        batch_size=256, num_workers=4, seed=seed
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
    
    # Setup privacy engine
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=delta,
        epochs=epochs,
        max_grad_norm=1.0
    )
    
    # Training loop
    best_acc = 0.0
    
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
        
        # Get epsilon
        epsilon = privacy_engine.get_epsilon(delta)
        
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
        
        test_acc = 100. * test_correct / test_total
        
        print(f'Epoch {epoch+1}: Train Acc: {100.*train_correct/train_total:.2f}%, Test Acc: {test_acc:.2f}%, ε={epsilon:.2f}')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_dir = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/dp_sgd'
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'epsilon': epsilon,
                'seed': seed
            }, f'{save_dir}/model_seed{seed}.pt')
    
    # Final evaluation
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
    
    final_test_acc = 100. * test_correct / test_total
    final_epsilon = privacy_engine.get_epsilon(delta)
    model_size = get_model_size_mb(model)
    flops = estimate_flops(model)
    
    results = {
        'seed': seed,
        'epsilon': final_epsilon,
        'delta': delta,
        'test_acc': final_test_acc,
        'model_size_mb': model_size,
        'flops': flops,
        'epochs': epochs
    }
    
    print(f"\nFinal Results (seed={seed}):")
    print(f"  Test Acc: {final_test_acc:.2f}%")
    print(f"  ε: {final_epsilon:.2f}")
    
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    seeds = [42]
    all_results = []
    
    for seed in seeds:
        result = train_dp_sgd(seed, target_epsilon=4, delta=1e-5, epochs=100, device=device)
        if result is not None:
            all_results.append(result)
    
    if all_results:
        aggregated = {
            'experiment': 'dp_sgd',
            'seeds': seeds,
            'test_acc': {
                'mean': sum(r['test_acc'] for r in all_results) / len(all_results),
                'std': (sum((r['test_acc'] - sum(r['test_acc'] for r in all_results)/len(all_results))**2 for r in all_results) / len(all_results))**0.5,
                'values': [r['test_acc'] for r in all_results]
            },
            'epsilon': all_results[0]['epsilon'],
            'delta': all_results[0]['delta'],
            'model_size_mb': all_results[0]['model_size_mb'],
            'flops': all_results[0]['flops']
        }
        
        save_path = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/dp_sgd/results.json'
        save_results(aggregated, save_path)
        print(f"\nResults saved to {save_path}")
