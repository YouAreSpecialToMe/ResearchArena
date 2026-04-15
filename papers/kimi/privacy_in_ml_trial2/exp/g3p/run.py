"""G3P: Gradient-Guided Privacy-Preserving Pruning."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import numpy as np

from shared.data_loader import get_cifar10_dataloaders
from shared.models import get_resnet18_cifar10, get_model_size_mb, estimate_flops
from shared.utils import set_seed, save_results


def compute_combined_saliency(model, train_loader, val_loader, device, 
                               alpha=1.0, beta=1.0, num_samples=512):
    """
    Compute combined task-privacy saliency scores.
    
    S_combined(i) = alpha * S_task(i) - beta * S_privacy(i)
    
    where:
    - S_task(i) = |E[(dL/dh_i) * h_i]| (Taylor importance)
    - S_privacy(i) = |E_train[|grad_i|] - E_val[|grad_i|]|
    """
    model.eval()
    
    # Collect gradient statistics and activations
    task_saliency = {}
    privacy_saliency = {}
    train_grads = {}
    val_grads = {}
    
    # Compute on training data
    for i, (inputs, labels) in enumerate(train_loader):
        if i * train_loader.batch_size >= num_samples:
            break
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
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
        
        # Collect gradients and compute task saliency
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.weight.grad is not None:
                    # Train gradients
                    if name not in train_grads:
                        train_grads[name] = []
                    grad_mag = module.weight.grad.abs().mean(dim=(1, 2, 3)).detach()
                    train_grads[name].append(grad_mag)
                    
                    # Task saliency (Taylor approximation)
                    bn_name = name.replace('conv', 'bn')
                    if bn_name in activations:
                        if name not in task_saliency:
                            task_saliency[name] = []
                        
                        # Taylor: |grad * weight|
                        taylor = (module.weight.grad.abs() * module.weight.abs()).mean(dim=(1, 2, 3)).detach()
                        task_saliency[name].append(taylor)
        
        for hook in hooks:
            hook.remove()
    
    # Average training statistics
    for name in train_grads:
        if len(train_grads[name]) > 0:
            train_grads[name] = torch.stack(train_grads[name]).mean(dim=0)
    
    for name in task_saliency:
        if len(task_saliency[name]) > 0:
            task_saliency[name] = torch.stack(task_saliency[name]).mean(dim=0)
    
    # Compute on validation data
    for i, (inputs, labels) in enumerate(val_loader):
        if i * val_loader.batch_size >= num_samples:
            break
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.weight.grad is not None:
                    if name not in val_grads:
                        val_grads[name] = []
                    grad_mag = module.weight.grad.abs().mean(dim=(1, 2, 3)).detach()
                    val_grads[name].append(grad_mag)
    
    # Average validation statistics
    for name in val_grads:
        if len(val_grads[name]) > 0:
            val_grads[name] = torch.stack(val_grads[name]).mean(dim=0)
    
    # Compute privacy saliency
    for name in train_grads:
        if name in val_grads:
            privacy_saliency[name] = torch.abs(train_grads[name] - val_grads[name])
    
    # Compute combined saliency
    combined_saliency = {}
    for name in task_saliency:
        if name in privacy_saliency:
            combined_saliency[name] = alpha * task_saliency[name] - beta * privacy_saliency[name]
        else:
            combined_saliency[name] = alpha * task_saliency[name]
    
    return combined_saliency, task_saliency, privacy_saliency


def g3p_prune_model(model, train_loader, val_loader, target_sparsity, device,
                    alpha=1.0, beta=1.0):
    """Prune channels based on G3P combined saliency."""
    model.eval()
    
    # Compute combined saliency
    combined_saliency, _, _ = compute_combined_saliency(
        model, train_loader, val_loader, device, alpha, beta
    )
    
    # Collect all saliency scores
    all_saliency = []
    all_names = []
    all_indices = []
    
    for name, saliency in combined_saliency.items():
        for i, val in enumerate(saliency):
            all_saliency.append(val.item())
            all_names.append(name)
            all_indices.append(i)
    
    # Determine threshold for pruning (prune channels with lowest combined saliency)
    num_channels = len(all_saliency)
    num_prune = int(num_channels * target_sparsity)
    
    # Sort by saliency (ascending - prune lowest first)
    sorted_indices = sorted(range(len(all_saliency)), key=lambda i: all_saliency[i])
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
        
        pbar = tqdm(train_loader, desc=f'G3P Finetune Epoch {epoch+1}/{epochs}')
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


def g3p_pruning(seed, target_sparsity, device='cuda', alpha=1.0, beta=1.0):
    """Run G3P pruning."""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"G3P - Sparsity: {target_sparsity}, Seed: {seed}, α={alpha}, β={beta}")
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
        
        # Prune using G3P
        model = g3p_prune_model(model, train_loader, val_loader, sparsity_per_iter, 
                                device, alpha, beta)
        
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
    save_dir = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/g3p'
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': final_test_acc,
        'sparsity': target_sparsity,
        'seed': seed,
        'alpha': alpha,
        'beta': beta
    }, f'{save_dir}/model_sparsity{int(target_sparsity*100)}_seed{seed}.pt')
    
    results = {
        'seed': seed,
        'sparsity': target_sparsity,
        'alpha': alpha,
        'beta': beta,
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
    alpha = 1.0
    beta = 1.0
    
    all_results = []
    
    for sparsity in sparsity_levels:
        for seed in seeds:
            result = g3p_pruning(seed, sparsity, device=device, alpha=alpha, beta=beta)
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
                    'flops': sparsity_results[0]['flops'],
                    'alpha': alpha,
                    'beta': beta
                }
        
        aggregated['experiment'] = 'g3p'
        aggregated['sparsity_levels'] = sparsity_levels
        aggregated['seeds'] = seeds
        aggregated['hyperparameters'] = {'alpha': alpha, 'beta': beta}
        
        save_path = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/g3p/results.json'
        save_results(aggregated, save_path)
        print(f"\nResults saved to {save_path}")
