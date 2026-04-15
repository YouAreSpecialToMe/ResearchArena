"""
Source baseline: Evaluate model without any adaptation.
This establishes the baseline performance before any test-time adaptation.

Uses pretrained ResNet models from pytorch-cifar-models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm

from shared.models import load_pretrained_cifar_model
from shared.data_loader import load_cifar_c, load_cifar10_1
from shared.utils import set_seed


def evaluate_source_cifar10_c(seed=2022, severity=5, model_name='resnet32', data_dir='./data', device='cuda'):
    """Evaluate source model on CIFAR-10-C"""
    set_seed(seed)
    
    # Load pretrained model
    model = load_pretrained_cifar_model(model_name=model_name, dataset='cifar10', device=device)
    model.eval()
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed, 'severity': severity}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'CIFAR-10-C (seed={seed})'):
        try:
            loader, dataset = load_cifar_c(data_dir, 'cifar10', corruption, severity, batch_size=200)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
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
    
    # Compute mean accuracy
    if all_accs:
        results['mean_accuracy'] = np.mean(all_accs)
        results['mean_error'] = 100 - results['mean_accuracy']
    
    return results


def evaluate_source_cifar100_c(seed=2022, severity=5, model_name='resnet20', data_dir='./data', device='cuda'):
    """Evaluate source model on CIFAR-100-C"""
    set_seed(seed)
    
    # Load pretrained model
    model = load_pretrained_cifar_model(model_name=model_name, dataset='cifar100', device=device)
    model.eval()
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed, 'severity': severity}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'CIFAR-100-C (seed={seed})'):
        try:
            loader, dataset = load_cifar_c(data_dir, 'cifar100', corruption, severity, batch_size=200)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
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


def evaluate_source_cifar10_1(seed=2022, model_name='resnet32', data_dir='./data', device='cuda'):
    """Evaluate source model on CIFAR-10.1 (natural distribution shift)"""
    set_seed(seed)
    
    # Load pretrained model
    model = load_pretrained_cifar_model(model_name=model_name, dataset='cifar10', device=device)
    model.eval()
    
    try:
        loader, dataset = load_cifar10_1(data_dir, batch_size=200)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        acc = 100.0 * correct / total
        
        return {
            'accuracy': acc,
            'error': 100 - acc,
            'seed': seed
        }
        
    except Exception as e:
        print(f"Error on CIFAR-10.1: {e}")
        return {'accuracy': None, 'error': None, 'seed': seed}


def run_all_seeds(data_dir='./data', device='cuda'):
    """Run source baseline for all seeds"""
    seeds = [2022, 2023, 2024, 2025, 2026]
    
    print("=" * 60)
    print("Source Baseline Evaluation (REAL)")
    print("=" * 60)
    
    # CIFAR-10-C
    print("\nEvaluating on CIFAR-10-C...")
    cifar10_results = []
    for seed in seeds:
        result = evaluate_source_cifar10_c(seed, severity=5, model_name='resnet32', data_dir=data_dir, device=device)
        cifar10_results.append(result)
    
    # Aggregate results
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
    
    # Save results
    with open('exp/source/results_cifar10.json', 'w') as f:
        json.dump(cifar10_summary, f, indent=2)
    
    # CIFAR-100-C
    print("\nEvaluating on CIFAR-100-C...")
    cifar100_results = []
    for seed in seeds:
        result = evaluate_source_cifar100_c(seed, severity=5, model_name='resnet20', data_dir=data_dir, device=device)
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
    
    with open('exp/source/results_cifar100.json', 'w') as f:
        json.dump(cifar100_summary, f, indent=2)
    
    # CIFAR-10.1
    print("\nEvaluating on CIFAR-10.1...")
    cifar10_1_results = []
    for seed in seeds:
        result = evaluate_source_cifar10_1(seed, model_name='resnet32', data_dir=data_dir, device=device)
        cifar10_1_results.append(result)
    
    valid_results = [r for r in cifar10_1_results if r['accuracy'] is not None]
    if valid_results:
        mean_acc = np.mean([r['accuracy'] for r in valid_results])
        std_acc = np.std([r['accuracy'] for r in valid_results])
        
        cifar10_1_summary = {
            'per_seed': cifar10_1_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'se_accuracy': std_acc / np.sqrt(len(seeds))
        }
        
        print(f"CIFAR-10.1: {mean_acc:.2f} ± {std_acc:.2f}%")
    else:
        cifar10_1_summary = {'per_seed': cifar10_1_results, 'error': 'No valid results'}
        print("No valid results for CIFAR-10.1")
    
    with open('exp/source/results_cifar10_1.json', 'w') as f:
        json.dump(cifar10_1_summary, f, indent=2)
    
    return {
        'cifar10': cifar10_summary,
        'cifar100': cifar100_summary,
        'cifar10_1': cifar10_1_summary
    }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    results = run_all_seeds(data_dir='./data', device=device)
    print("\n" + "=" * 60)
    print("Source Baseline Complete!")
    print("=" * 60)
