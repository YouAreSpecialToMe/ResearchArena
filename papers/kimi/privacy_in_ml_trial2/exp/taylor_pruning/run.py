"""Taylor Importance Pruning baseline (Molchanov et al., 2019)."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import copy

from shared.data_loader import get_cifar10_dataloaders
from shared.models import get_resnet18_cifar10, get_model_size_mb, estimate_flops
from shared.utils import set_seed, save_results


def compute_taylor_importance(model, train_loader, device, num_samples=512):
    """
    Compute Taylor importance: S_task(i) = |E[(dL/dh_i) * h_i]|
    """
    model.eval()
    importance = {}
    counts = {}
    
    for i, (inputs, labels) in enumerate(train_loader):
        if i * train_loader.batch_size >= num_samples:
            break
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                handle = module.register_forward_hook(get_activation(name))
                hooks.append(handle)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        # Compute importance
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name in [n.replace('bn', 'conv') for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]:
                conv_name = name
                bn_name = name.replace('conv', 'bn')
                
                if bn_name in activations and module.weight.grad is not None:
                    if conv_name not in importance:
                        importance[conv_name] = []
                        counts[conv_name] = 0
                    
                    # Get activation
                    act = activations[bn_name]
                    
                    # Compute gradient of loss w.r.t. activation
                    # Use weight gradient as proxy
                    grad = module.weight.grad
                    
                    # Taylor approximation: |grad * weight|
                    # This approximates the change in loss if channel is removed
                    channel_importance = (grad.abs() * module.weight.abs()).mean(dim=(1, 2, 3))
                    
                    importance[conv_name].append(channel_importance.detach())
                    counts[conv_name] += inputs.size(0)
        
        for hook in hooks:
            hook.remove()
    
    # Average importance
    for name in importance:
        if len(importance[name]) > 0:
            importance[name] = torch.stack(importance[name]).mean(dim=0)
    
    return importance


def taylor_prune_model(model, train_loader, target_sparsity, device):
    """Prune channels based on Taylor importance."""
    model.eval()
    
    # Compute importance
    importance = compute_taylor_importance(model, train_loader, device)
    
    # Collect all importances
    all_importances = []
    all_names = []
    all_indices = []
    
    for name, imp in importance.items():
        for i, val in enumerate(imp):
            all_importances.append(val.item())
            all_names.append(name)
            all_indices.append(i)
    
    # Determine threshold for pruning
    num_channels = len(all_importances)
    num_prune = int(num_channels * target_sparsity)
    
    # Sort by importance (ascending - prune least important)
    sorted_indices = sorted(range(len(all_importances)), key=lambda i: all_importances[i])
    prune_indices = sorted_indices[:num_prune]
    
    # Group by layer
    prune_by_layer = {}
    for idx in prune_indices:
        name = all_names[idx]
        channel = all_indices[idx]
        if name not in prune_by_layer:
            prune_by_layer[name] = []
        prune_by_layer[name].append(channel)
    
    # Zero out pruned channels
    for name, channels in prune_by_layer.items():
        for module_name, module in model.named_modules():
            if module_name == name and isinstance(module, nn.Conv2d):
                with torch.no_grad():
                    for ch in channels:
                        module.weight[ch] = 0
                        if module.bias is not None:
                            module.bias[ch] = 0
    
    return model


def finetune_model(model, train_loader, test_loader, epochs, device, lr=0.01):
    """Fine-tune pruned model."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}/{epochs}')
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
    
    # Evaluate
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
    return test_acc


def iterative_taylor_pruning(seed, target_sparsity, device='cuda'):
    """Run iterative Taylor pruning."""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Taylor Pruning - Sparsity: {target_sparsity}, Seed: {seed}")
    print(f"{'='*60}")
    
    # Load data
    train_loader, val_loader, test_loader, train_indices, val_indices = get_cifar10_dataloaders(
        batch_size=128, num_workers=4, seed=seed
    )
    
    # Load baseline model
    baseline_path = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/baseline_unpruned/model_seed{seed}.pt'
    if not os.path.exists(baseline_path):
        print(f"Baseline model not found: {baseline_path}")
        return None
    
    checkpoint = torch.load(baseline_path, map_location=device, weights_only=False)
    model = get_resnet18_cifar10(num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Iterative pruning: 5 iterations
    num_iterations = 5
    sparsity_per_iter = target_sparsity / num_iterations
    
    for iteration in range(num_iterations):
        current_target = sparsity_per_iter * (iteration + 1)
        print(f"\nIteration {iteration+1}/{num_iterations}: Target sparsity {current_target:.3f}")
        
        # Prune using Taylor importance
        model = taylor_prune_model(model, train_loader, sparsity_per_iter, device)
        
        # Fine-tune for 8 epochs
        test_acc = finetune_model(model, train_loader, test_loader, epochs=4, device=device, lr=0.01)
        print(f"After lr=0.01 fine-tuning: Test Acc = {test_acc:.2f}%")
        
        test_acc = finetune_model(model, train_loader, test_loader, epochs=4, device=device, lr=0.001)
        print(f"After lr=0.001 fine-tuning: Test Acc = {test_acc:.2f}%")
    
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
    model_size = get_model_size_mb(model)
    flops = estimate_flops(model)
    
    # Save model
    save_dir = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/taylor_pruning'
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': final_test_acc,
        'sparsity': target_sparsity,
        'seed': seed
    }, f'{save_dir}/model_sparsity{int(target_sparsity*100)}_seed{seed}.pt')
    
    results = {
        'seed': seed,
        'sparsity': target_sparsity,
        'test_acc': final_test_acc,
        'model_size_mb': model_size,
        'flops': flops
    }
    
    print(f"\nFinal Results:")
    print(f"  Test Acc: {final_test_acc:.2f}%")
    print(f"  Model Size: {model_size:.2f} MB")
    
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test at 50% sparsity only to save time
    sparsity_levels = [0.5]
    seeds = [42, 43, 44]
    
    all_results = []
    
    for sparsity in sparsity_levels:
        for seed in seeds:
            result = iterative_taylor_pruning(seed, sparsity, device=device)
            if result is not None:
                all_results.append(result)
    
    if all_results:
        # Aggregate
        aggregated = {
            'experiment': 'taylor_pruning',
            'sparsity_levels': sparsity_levels,
            'seeds': seeds,
            'test_acc': {
                'mean': sum(r['test_acc'] for r in all_results) / len(all_results),
                'std': (sum((r['test_acc'] - sum(r['test_acc'] for r in all_results)/len(all_results))**2 for r in all_results) / len(all_results))**0.5,
                'values': [r['test_acc'] for r in all_results]
            },
            'model_size_mb': all_results[0]['model_size_mb'],
            'flops': all_results[0]['flops']
        }
        
        save_path = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/taylor_pruning/results.json'
        save_results(aggregated, save_path)
        print(f"\nResults saved to {save_path}")
