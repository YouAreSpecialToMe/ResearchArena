"""
TENT (Test Entropy) baseline: Minimize prediction entropy.
Wang et al., ICLR 2021
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm

from shared.models import load_pretrained_cifar_model, enable_adaptation, collect_params
from shared.data_loader import load_cifar_c
from shared.utils import set_seed


def entropy_loss(logits):
    """Compute entropy of predictions"""
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    entropy = -(p * log_p).sum(dim=1).mean()
    return entropy


def evaluate_tent_cifar10_c(seed=2022, severity=5, model_name='resnet32', 
                            lr=1e-3, batch_size=200, data_dir='./data', device='cuda'):
    """Evaluate TENT on CIFAR-10-C"""
    set_seed(seed)
    
    # Load pretrained model
    model = load_pretrained_cifar_model(model_name=model_name, dataset='cifar10', device=device)
    
    # Enable adaptation on BN parameters
    model = enable_adaptation(model, adapt_bn=True)
    
    # Collect adaptable parameters
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed, 'severity': severity}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'TENT CIFAR-10-C (seed={seed})'):
        try:
            loader, dataset = load_cifar_c(data_dir, 'cifar10', corruption, severity, batch_size=batch_size)
            
            correct = 0
            total = 0
            
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Adaptation step: minimize entropy
                loss = entropy_loss(outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Evaluate
                with torch.no_grad():
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            acc = 100.0 * correct / total
            results['corruptions'][corruption] = {'accuracy': acc, 'error': 100 - acc}
            all_accs.append(acc)
            
        except Exception as e:
            print(f"Error on {corruption}: {e}")
            results['corruptions'][corruption] = {'accuracy': None, 'error': None}
    
    if all_accs:
        results['mean_accuracy'] = np.mean(all_accs)
        results['mean_error'] = 100 - results['mean_accuracy']
    
    return results


def evaluate_tent_cifar100_c(seed=2022, severity=5, model_name='resnet20', 
                             lr=1e-3, batch_size=200, data_dir='./data', device='cuda'):
    """Evaluate TENT on CIFAR-100-C"""
    set_seed(seed)
    
    # Load pretrained model
    model = load_pretrained_cifar_model(model_name=model_name, dataset='cifar100', device=device)
    model = enable_adaptation(model, adapt_bn=True)
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed, 'severity': severity}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'TENT CIFAR-100-C (seed={seed})'):
        try:
            loader, dataset = load_cifar_c(data_dir, 'cifar100', corruption, severity, batch_size=batch_size)
            
            correct = 0
            total = 0
            
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = entropy_loss(outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            acc = 100.0 * correct / total
            results['corruptions'][corruption] = {'accuracy': acc, 'error': 100 - acc}
            all_accs.append(acc)
            
        except Exception as e:
            print(f"Error on {corruption}: {e}")
            results['corruptions'][corruption] = {'accuracy': None, 'error': None}
    
    if all_accs:
        results['mean_accuracy'] = np.mean(all_accs)
        results['mean_error'] = 100 - results['mean_accuracy']
    
    return results


def run_all_seeds(data_dir='./data', device='cuda'):
    """Run TENT for all seeds"""
    seeds = [2022, 2023, 2024, 2025, 2026]
    
    print("=" * 60)
    print("TENT Baseline Evaluation (REAL)")
    print("=" * 60)
    
    # CIFAR-10-C
    print("\nEvaluating on CIFAR-10-C...")
    cifar10_results = []
    for seed in seeds:
        result = evaluate_tent_cifar10_c(seed, severity=5, model_name='resnet32', data_dir=data_dir, device=device)
        cifar10_results.append(result)
    
    valid_results = [r for r in cifar10_results if 'mean_accuracy' in r]
    if valid_results:
        mean_acc = np.mean([r['mean_accuracy'] for r in valid_results])
        std_acc = np.std([r['mean_accuracy'] for r in valid_results])
        
        cifar10_summary = {
            'per_seed': cifar10_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'se_accuracy': std_acc / np.sqrt(len(seeds))
        }
        
        print(f"CIFAR-10-C Severity 5: {mean_acc:.2f} ± {std_acc:.2f}%")
    else:
        cifar10_summary = {'per_seed': cifar10_results, 'error': 'No valid results'}
        print("No valid results for CIFAR-10-C")
    
    with open('exp/tent/results_cifar10.json', 'w') as f:
        json.dump(cifar10_summary, f, indent=2)
    
    # CIFAR-100-C
    print("\nEvaluating on CIFAR-100-C...")
    cifar100_results = []
    for seed in seeds:
        result = evaluate_tent_cifar100_c(seed, severity=5, model_name='resnet20', data_dir=data_dir, device=device)
        cifar100_results.append(result)
    
    valid_results = [r for r in cifar100_results if 'mean_accuracy' in r]
    if valid_results:
        mean_acc = np.mean([r['mean_accuracy'] for r in valid_results])
        std_acc = np.std([r['mean_accuracy'] for r in valid_results])
        
        cifar100_summary = {
            'per_seed': cifar100_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'se_accuracy': std_acc / np.sqrt(len(seeds))
        }
        
        print(f"CIFAR-100-C Severity 5: {mean_acc:.2f} ± {std_acc:.2f}%")
    else:
        cifar100_summary = {'per_seed': cifar100_results, 'error': 'No valid results'}
        print("No valid results for CIFAR-100-C")
    
    with open('exp/tent/results_cifar100.json', 'w') as f:
        json.dump(cifar100_summary, f, indent=2)
    
    return {
        'cifar10': cifar10_summary,
        'cifar100': cifar100_summary
    }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    results = run_all_seeds(data_dir='./data', device=device)
    print("\n" + "=" * 60)
    print("TENT Baseline Complete!")
    print("=" * 60)
