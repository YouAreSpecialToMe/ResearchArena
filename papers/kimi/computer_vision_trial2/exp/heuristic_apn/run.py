"""
Heuristic-APN baseline: Simple distance-based severity selection.

This baseline tests whether learned policies provide benefits beyond simple heuristics.
If Meta-APN significantly outperforms Heuristic-APN, it justifies the additional complexity.

Mechanism:
- severity = max_severity * (1 - exp(-λ * d)) where d is prototype distance
- Uses fixed set of 8 augmentations
- Uses prototype consistency loss (same as APAC-TTA but without learned policy)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from shared.models import wideresnet28_10, get_prototype_distances, collect_params
from shared.data_loader import load_cifar_c, get_cifar_transforms
from shared.augmentations import AUGMENTATION_OPS, apply_augmentation
from shared.metrics import accuracy, js_divergence
from shared.utils import set_seed, save_results, load_prototypes, copy_model, load_model_state
from torchvision import transforms


def heuristic_severity(distance, max_severity=5, lambda_param=2.0):
    """
    Compute augmentation severity based on prototype distance.
    Closer prototypes -> milder augmentations
    Farther prototypes -> stronger augmentations
    """
    normalized_dist = torch.clamp(distance * lambda_param, 0, 10)
    severity = max_severity * (1 - torch.exp(-normalized_dist))
    return torch.clamp(severity, 1, max_severity).item()


def get_fixed_augmentations():
    """Get fixed set of 8 augmentations"""
    return [
        ('gaussian_noise', 3),
        ('shot_noise', 3),
        ('brightness', 3),
        ('contrast', 3),
        ('defocus_blur', 3),
        ('jpeg_compression', 3),
        ('pixelate', 3),
        ('saturate', 3),
    ]


def adapt_single_image_heuristic(model, image, prototypes, lr=1e-3, steps=1, 
                                  temperature=0.5, device='cuda'):
    """
    Adapt model using heuristic-APN.
    
    Args:
        model: Model to adapt
        image: Single test image [C, H, W]
        prototypes: Class prototypes [num_classes, feature_dim]
        lr: Learning rate
        steps: Number of adaptation steps
        temperature: Temperature for prototype target distribution
        device: Device
    
    Returns:
        adapted_model: Adapted model
    """
    # Get features and compute prototype distances
    with torch.no_grad():
        features = model.get_features(image.unsqueeze(0).to(device))
        distances = get_prototype_distances(features, prototypes, metric='cosine')
        min_dist = distances.min().item()
        nearest_proto_idx = distances.argmin().item()
    
    # Determine augmentation severity based on distance
    severity = heuristic_severity(torch.tensor(min_dist))
    
    # Get fixed augmentations with adaptive severity
    base_augmentations = get_fixed_augmentations()
    augmentations = []
    for op, base_sev in base_augmentations:
        aug_sev = min(5, max(1, int(base_sev * (severity / 3))))
        augmentations.append((op, aug_sev))
    
    # Apply augmentations
    aug_batch = []
    for op, sev in augmentations:
        aug_img = apply_augmentation(image, op, sev, dataset='cifar10')
        if not isinstance(aug_img, torch.Tensor):
            aug_img = transforms.ToTensor()(aug_img)
        aug_batch.append(aug_img)
    
    aug_batch = torch.stack(aug_batch).to(device)
    
    # Compute prototype confidence scores
    with torch.no_grad():
        aug_features = model.get_features(aug_batch)
        aug_features_norm = F.normalize(aug_features, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        
        # [num_aug, num_classes] similarity
        sim_matrix = torch.mm(aug_features_norm, prototypes_norm.t())
        
        # Confidence for each prototype
        confidences = []
        for c in range(prototypes.size(0)):
            sims = sim_matrix[:, c]
            consistency = 1.0 / (1.0 + torch.var(sims))
            mean_sim = torch.mean(sims)
            confidences.append(consistency * mean_sim)
        confidences = torch.stack(confidences)
    
    # Enable adaptation on BN parameters
    params, _ = collect_params(model)
    if len(params) == 0:
        return model
    
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # Get prototype target distribution (uniform over nearest prototype)
    with torch.no_grad():
        proto_logits = torch.zeros(1, prototypes.size(0)).to(device)
        proto_logits[0, nearest_proto_idx] = 1.0 / temperature
        target_dist = F.softmax(proto_logits / temperature, dim=1)
    
    # Adapt
    for _ in range(steps):
        outputs = model(aug_batch)
        probs = F.softmax(outputs, dim=1)
        
        # JS divergence to prototype target (same for all augmentations in this simplified version)
        target_dist_expanded = target_dist.expand(probs.size(0), -1)
        loss = js_divergence(probs, target_dist_expanded).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


def evaluate_heuristic_cifar10_c(seed=2022, severity=5, lr=1e-3, steps=1, 
                                  data_dir='./data', device='cuda'):
    """Evaluate Heuristic-APN on CIFAR-10-C"""
    set_seed(seed)
    
    # Load model
    model = wideresnet28_10(num_classes=10).to(device)
    model.eval()
    
    # Load prototypes
    prototypes = load_prototypes('cifar10', device)
    if prototypes is None:
        print("Warning: No prototypes found, using random prototypes")
        prototypes = torch.randn(10, model.feature_dim).to(device)
    
    original_state = copy_model(model)
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed, 'severity': severity}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'Heuristic-APN CIFAR-10-C (seed={seed})'):
        try:
            loader, dataset = load_cifar_c(data_dir, 'cifar10', corruption, severity, batch_size=1)
            
            correct = 0
            total = 0
            
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                # Reset model
                load_model_state(model, original_state)
                model.eval()
                
                # Adapt
                model = adapt_single_image_heuristic(
                    model, images[0], prototypes, lr, steps, device=device
                )
                
                # Predict
                with torch.no_grad():
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            acc = 100.0 * correct / total
            results['corruptions'][corruption] = {'accuracy': acc, 'error': 100 - acc}
            all_accs.append(acc)
            
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            results['corruptions'][corruption] = {'accuracy': None, 'error': None}
    
    if all_accs:
        results['mean_accuracy'] = np.mean(all_accs)
        results['mean_error'] = 100 - results['mean_accuracy']
    
    return results


def evaluate_heuristic_cifar100_c(seed=2022, severity=5, lr=1e-3, steps=1,
                                   data_dir='./data', device='cuda'):
    """Evaluate Heuristic-APN on CIFAR-100-C"""
    set_seed(seed)
    
    model = wideresnet28_10(num_classes=100).to(device)
    model.eval()
    
    prototypes = load_prototypes('cifar100', device)
    if prototypes is None:
        print("Warning: No prototypes found, using random prototypes")
        prototypes = torch.randn(100, model.feature_dim).to(device)
    
    original_state = copy_model(model)
    
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    results = {'corruptions': {}, 'seed': seed, 'severity': severity}
    all_accs = []
    
    for corruption in tqdm(corruptions, desc=f'Heuristic-APN CIFAR-100-C (seed={seed})'):
        try:
            loader, dataset = load_cifar_c(data_dir, 'cifar100', corruption, severity, batch_size=1)
            
            correct = 0
            total = 0
            
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                load_model_state(model, original_state)
                model.eval()
                
                model = adapt_single_image_heuristic(
                    model, images[0], prototypes, lr, steps, device=device
                )
                
                with torch.no_grad():
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            acc = 100.0 * correct / total
            results['corruptions'][corruption] = {'accuracy': acc, 'error': 100 - acc}
            all_accs.append(acc)
            
        except FileNotFoundError:
            results['corruptions'][corruption] = {'accuracy': None, 'error': None}
    
    if all_accs:
        results['mean_accuracy'] = np.mean(all_accs)
        results['mean_error'] = 100 - results['mean_accuracy']
    
    return results


def run_all_seeds(data_dir='./data', device='cuda'):
    """Run Heuristic-APN baseline for all seeds"""
    seeds = [2022, 2023, 2024, 2025, 2026]
    
    print("=" * 60)
    print("Heuristic-APN Baseline Evaluation")
    print("=" * 60)
    
    # CIFAR-10-C
    print("\nEvaluating Heuristic-APN on CIFAR-10-C...")
    cifar10_results = []
    for seed in seeds:
        result = evaluate_heuristic_cifar10_c(seed, severity=5, data_dir=data_dir, device=device)
        cifar10_results.append(result)
    
    mean_acc = np.mean([r['mean_accuracy'] for r in cifar10_results if 'mean_accuracy' in r])
    std_acc = np.std([r['mean_accuracy'] for r in cifar10_results if 'mean_accuracy' in r])
    
    cifar10_summary = {
        'per_seed': cifar10_results,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'se_accuracy': std_acc / np.sqrt(len(seeds))
    }
    
    print(f"CIFAR-10-C Severity 5: {mean_acc:.2f} ± {std_acc:.2f}%")
    save_results(cifar10_summary, 'exp/heuristic_apn/results_cifar10.json')
    
    # CIFAR-100-C
    print("\nEvaluating Heuristic-APN on CIFAR-100-C...")
    cifar100_results = []
    for seed in seeds:
        result = evaluate_heuristic_cifar100_c(seed, severity=5, data_dir=data_dir, device=device)
        cifar100_results.append(result)
    
    mean_acc = np.mean([r['mean_accuracy'] for r in cifar100_results if 'mean_accuracy' in r])
    std_acc = np.std([r['mean_accuracy'] for r in cifar100_results if 'mean_accuracy' in r])
    
    cifar100_summary = {
        'per_seed': cifar100_results,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'se_accuracy': std_acc / np.sqrt(len(seeds))
    }
    
    print(f"CIFAR-100-C Severity 5: {mean_acc:.2f} ± {std_acc:.2f}%")
    save_results(cifar100_summary, 'exp/heuristic_apn/results_cifar100.json')
    
    return {
        'cifar10': cifar10_summary,
        'cifar100': cifar100_summary
    }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = run_all_seeds(data_dir='./data', device=device)
    print("\n" + "=" * 60)
    print("Heuristic-APN Baseline Complete!")
    print("=" * 60)
