"""Hybrid baseline: Magnitude Pruning + KL Regularization."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import os

from shared.data_loader import get_cifar10_dataloaders
from shared.models import get_resnet18_cifar10, get_model_size_mb, estimate_flops
from shared.utils import set_seed, save_results


def magnitude_prune_model(model, target_sparsity):
    """Prune channels based on L1-norm of weights."""
    model.eval()
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d) and 'shortcut' not in name:
            weight_norms = module.weight.abs().sum(dim=(1, 2, 3))
            num_channels = len(weight_norms)
            num_prune = int(num_channels * target_sparsity)
            
            if num_prune == 0:
                continue
            
            _, prune_indices = torch.topk(weight_norms, num_prune, largest=False)
            
            with torch.no_grad():
                for idx in prune_indices:
                    module.weight[idx] = 0
                    if module.bias is not None:
                        module.bias[idx] = 0
    
    return model


def finetune_with_kl(model, train_loader, val_loader, test_loader, epochs, device, 
                     lr=0.01, kl_weight=0.1):
    """Fine-tune with KL divergence regularization."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Compute validation distribution
    model.eval()
    val_outputs = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = F.softmax(model(inputs), dim=1)
            val_outputs.append(outputs.cpu())
    val_dist = torch.cat(val_outputs, dim=0).mean(dim=0).to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_ce_loss = 0.0
        train_kl_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)
            
            # KL divergence
            train_dist = F.softmax(outputs, dim=1).mean(dim=0)
            kl_loss = F.kl_div(train_dist.log(), val_dist, reduction='batchmean')
            
            loss = ce_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ce_loss += ce_loss.item()
            train_kl_loss += kl_loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / train_total,
                'ce': train_ce_loss / train_total,
                'kl': train_kl_loss / train_total
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


def iterative_hybrid_pruning(seed, target_sparsity, device='cuda'):
    """Run iterative hybrid pruning (Magnitude + KL)."""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Hybrid (Magnitude + KL) - Sparsity: {target_sparsity}, Seed: {seed}")
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
        
        # Prune using magnitude
        model = magnitude_prune_model(model, sparsity_per_iter)
        
        # Fine-tune with KL regularization for 8 epochs
        test_acc = finetune_with_kl(model, train_loader, val_loader, test_loader, 
                                     epochs=4, device=device, lr=0.01, kl_weight=0.1)
        print(f"After lr=0.01 fine-tuning: Test Acc = {test_acc:.2f}%")
        
        test_acc = finetune_with_kl(model, train_loader, val_loader, test_loader,
                                     epochs=4, device=device, lr=0.001, kl_weight=0.1)
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
    save_dir = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/hybrid_pruning'
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
    
    # Sparsity levels to test
    sparsity_levels = [0.3, 0.5, 0.7]
    seeds = [42, 43, 44]
    
    all_results = []
    
    for sparsity in sparsity_levels:
        for seed in seeds:
            result = iterative_hybrid_pruning(seed, sparsity, device=device)
            if result is not None:
                all_results.append(result)
    
    if all_results:
        # Aggregate by sparsity level
        aggregated = {}
        for sparsity in sparsity_levels:
            sparsity_results = [r for r in all_results if r['sparsity'] == sparsity]
            if sparsity_results:
                aggregated[f'sparsity_{int(sparsity*100)}'] = {
                    'test_acc': {
                        'mean': sum(r['test_acc'] for r in sparsity_results) / len(sparsity_results),
                        'std': (sum((r['test_acc'] - sum(r['test_acc'] for r in sparsity_results)/len(sparsity_results))**2 for r in sparsity_results) / len(sparsity_results))**0.5,
                        'values': [r['test_acc'] for r in sparsity_results]
                    },
                    'model_size_mb': sparsity_results[0]['model_size_mb'],
                    'flops': sparsity_results[0]['flops']
                }
        
        aggregated['experiment'] = 'hybrid_pruning'
        aggregated['sparsity_levels'] = sparsity_levels
        aggregated['seeds'] = seeds
        
        save_path = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/hybrid_pruning/results.json'
        save_results(aggregated, save_path)
        print(f"\nResults saved to {save_path}")
