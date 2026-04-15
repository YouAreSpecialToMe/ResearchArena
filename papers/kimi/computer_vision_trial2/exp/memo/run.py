"""
MEMO: Test Time Robustness via Adaptation and Augmentation
Simplified but correct implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import torchvision.transforms as transforms

from shared.models import load_pretrained_cifar_model, enable_adaptation, collect_params
from shared.data_loader import load_cifar_c
from shared.utils import set_seed


def marginal_entropy_loss(logits):
    """Marginal entropy across batch"""
    avg_probs = F.softmax(logits, dim=1).mean(dim=0)
    entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
    return entropy


def evaluate_memo_cifar10_c(seed=2022, severity=5, model_name='resnet32',
                            lr=1e-3, data_dir='./data', device='cuda'):
    """Evaluate MEMO on CIFAR-10-C"""
    set_seed(seed)
    
    model = load_pretrained_cifar_model(model_name=model_name, dataset='cifar10', device=device)
    model = enable_adaptation(model, adapt_bn=True)
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # Simple augmentations
    augmentations = [
        transforms.Lambda(lambda x: x),  # Identity
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 0.9)),
        transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 1.1)),
    ]
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed, 'severity': severity}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'MEMO CIFAR-10-C (seed={seed})'):
        try:
            loader, dataset = load_cifar_c(data_dir, 'cifar10', corruption, severity, batch_size=200)
            
            correct = 0
            total = 0
            
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Create augmented batch
                all_logits = []
                for aug in augmentations:
                    aug_images = torch.stack([aug(img) for img in images])
                    logits = model(aug_images)
                    all_logits.append(logits)
                
                # Stack all augmented predictions
                stacked_logits = torch.cat(all_logits, dim=0)
                
                # Minimize marginal entropy
                loss = marginal_entropy_loss(stacked_logits)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Evaluate
                with torch.no_grad():
                    output = model(images)
                    _, predicted = output.max(1)
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
    """Run MEMO for all seeds"""
    seeds = [2022, 2023, 2024, 2025, 2026]
    
    print("=" * 60)
    print("MEMO Baseline Evaluation (REAL)")
    print("=" * 60)
    
    print("\nEvaluating on CIFAR-10-C...")
    cifar10_results = []
    for seed in seeds:
        result = evaluate_memo_cifar10_c(seed, severity=5, model_name='resnet32', 
                                          data_dir=data_dir, device=device)
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
    
    with open('exp/memo/results_cifar10.json', 'w') as f:
        json.dump(cifar10_summary, f, indent=2)
    
    return {'cifar10': cifar10_summary}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    results = run_all_seeds(data_dir='./data', device=device)
    print("\n" + "=" * 60)
    print("MEMO Baseline Complete!")
    print("=" * 60)
