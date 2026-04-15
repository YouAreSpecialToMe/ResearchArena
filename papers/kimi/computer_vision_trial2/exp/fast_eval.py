"""
Fast evaluation script for TTA experiments.
Runs on a subset of corruptions for quicker results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import time

from shared.models import wideresnet28_10, MetaAPN, get_prototype_distances, collect_params
from shared.data_loader import load_cifar_c, load_cifar10_1
from shared.augmentations import AUGMENTATION_OPS, apply_augmentation
from shared.metrics import accuracy, js_divergence
from shared.utils import set_seed, load_prototypes, copy_model, load_model_state
from torchvision import transforms

# Subset of corruptions for faster evaluation
CORRUPTIONS_SUBSET = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'brightness', 'contrast']


def evaluate_source(model, num_classes, corruption, severity, seed, data_dir, device):
    """Source baseline (no adaptation)"""
    set_seed(seed)
    model.eval()
    
    loader, dataset = load_cifar_c(data_dir, f'cifar{num_classes}', corruption, severity, batch_size=200)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return 100.0 * correct / total


def entropy(outputs):
    """Compute entropy"""
    p = F.softmax(outputs, dim=1)
    log_p = F.log_softmax(outputs, dim=1)
    return -(p * log_p).sum(dim=1).mean()


def evaluate_tent(model, num_classes, corruption, severity, seed, data_dir, device):
    """TENT baseline (entropy minimization)"""
    set_seed(seed)
    model.eval()
    
    loader, dataset = load_cifar_c(data_dir, f'cifar{num_classes}', corruption, severity, batch_size=200)
    
    # Enable BN adaptation
    for module in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for param in module.parameters():
                param.requires_grad = True
            module.train()
    
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3) if params else None
    
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Adapt
        if optimizer:
            outputs = model(images)
            loss = entropy(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Predict
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return 100.0 * correct / total


def evaluate_memo(model, num_classes, corruption, severity, seed, data_dir, device):
    """MEMO baseline (entropy minimization with augmentations)"""
    set_seed(seed)
    model.eval()
    original_state = copy_model(model)
    
    loader, dataset = load_cifar_c(data_dir, f'cifar{num_classes}', corruption, severity, batch_size=1)
    
    augmentations = [('gaussian_noise', 3), ('brightness', 3), ('contrast', 3), ('defocus_blur', 3)]
    
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Reset model
        load_model_state(model, original_state)
        model.eval()
        
        # Create augmented batch
        aug_batch = [images[0]]
        for op, sev in augmentations:
            aug_img = apply_augmentation(images[0], op, sev, f'cifar{num_classes}')
            if not isinstance(aug_img, torch.Tensor):
                aug_img = transforms.ToTensor()(aug_img)
            aug_batch.append(aug_img)
        
        aug_batch = torch.stack(aug_batch).to(device)
        
        # Adapt
        params, _ = collect_params(model)
        if params:
            optimizer = torch.optim.Adam(params, lr=1e-3)
            outputs = model(aug_batch)
            p = F.softmax(outputs, dim=1)
            avg_p = p.mean(dim=0, keepdim=True)
            loss = -(avg_p * torch.log(avg_p + 1e-10)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Predict
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return 100.0 * correct / total


def evaluate_apac(model, meta_apn, prototypes, num_classes, corruption, severity, seed, data_dir, device):
    """APAC-TTA (our method)"""
    set_seed(seed)
    model.eval()
    original_state = copy_model(model)
    
    loader, dataset = load_cifar_c(data_dir, f'cifar{num_classes}', corruption, severity, batch_size=1)
    
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Reset model
        load_model_state(model, original_state)
        model.eval()
        
        # Get features and policy
        with torch.no_grad():
            features = model.get_features(images)
            distances = get_prototype_distances(features, prototypes, metric='cosine')
            policy_logits, severity_scale, _ = meta_apn(features, distances)
            op_probs = F.softmax(policy_logits / 0.5, dim=-1)
            topk_ops = torch.topk(op_probs[0], k=4)[1]
        
        # Apply augmentations
        aug_batch = []
        for op_idx in topk_ops:
            op_name = AUGMENTATION_OPS[op_idx.item()]
            sev = max(1, min(5, int(3 * severity_scale.item())))
            aug_img = apply_augmentation(images[0], op_name, sev, f'cifar{num_classes}')
            if not isinstance(aug_img, torch.Tensor):
                aug_img = transforms.ToTensor()(aug_img)
            aug_batch.append(aug_img)
        
        aug_batch = torch.stack(aug_batch).to(device)
        
        # Adapt with prototype consistency
        params, _ = collect_params(model)
        if params:
            optimizer = torch.optim.Adam(params, lr=1e-3)
            
            # Get prototype target
            with torch.no_grad():
                nearest = distances.argmin().item()
                target = torch.zeros(1, num_classes).to(device)
                target[0, nearest] = 1.0
            
            outputs = model(aug_batch)
            probs = F.softmax(outputs, dim=1)
            target_expanded = target.expand(probs.size(0), -1)
            m = 0.5 * (probs + target_expanded)
            kl1 = (probs * torch.log(probs / (m + 1e-10) + 1e-10)).sum(dim=1)
            kl2 = (target_expanded * torch.log(target_expanded / (m + 1e-10) + 1e-10)).sum(dim=1)
            loss = 0.5 * (kl1 + kl2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Predict
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return 100.0 * correct / total


def run_fast_evaluation(dataset='cifar10', data_dir='./data', device='cuda'):
    """Run fast evaluation on a subset of corruptions"""
    num_classes = 10 if dataset == 'cifar10' else 100
    seeds = [2022, 2023, 2024]
    severity = 5
    
    print(f"\n{'='*60}")
    print(f"Fast Evaluation: {dataset.upper()}-C (5 corruptions, 3 seeds)")
    print(f"{'='*60}\n")
    
    # Load model
    model = wideresnet28_10(num_classes=num_classes).to(device)
    model.eval()
    
    # Load prototypes and Meta-APN
    prototypes = load_prototypes(dataset, device)
    if prototypes is None:
        prototypes = torch.randn(num_classes, model.feature_dim).to(device)
    
    meta_apn_path = f'models/meta_apn_{dataset}.pt'
    if os.path.exists(meta_apn_path):
        meta_apn = MetaAPN(model.feature_dim, num_classes, num_operations=len(AUGMENTATION_OPS)).to(device)
        meta_apn.load_state_dict(torch.load(meta_apn_path, map_location=device))
        meta_apn.eval()
        print(f"Loaded Meta-APN from {meta_apn_path}")
    else:
        # Use untrained Meta-APN (random) - not ideal but allows testing
        meta_apn = MetaAPN(model.feature_dim, num_classes, num_operations=len(AUGMENTATION_OPS)).to(device)
        meta_apn.eval()
        print(f"Warning: Using untrained Meta-APN (file not found: {meta_apn_path})")
    
    results = {
        'source': {},
        'tent': {},
        'memo': {},
        'apac': {}
    }
    
    for corruption in CORRUPTIONS_SUBSET:
        print(f"\nEvaluating on {corruption}...")
        
        # Source
        source_accs = []
        for seed in seeds:
            acc = evaluate_source(model, num_classes, corruption, severity, seed, data_dir, device)
            source_accs.append(acc)
        results['source'][corruption] = {
            'mean': np.mean(source_accs),
            'std': np.std(source_accs),
            'se': np.std(source_accs) / np.sqrt(len(seeds))
        }
        print(f"  Source: {results['source'][corruption]['mean']:.2f}%")
        
        # TENT
        tent_accs = []
        for seed in seeds:
            acc = evaluate_tent(model, num_classes, corruption, severity, seed, data_dir, device)
            tent_accs.append(acc)
        results['tent'][corruption] = {
            'mean': np.mean(tent_accs),
            'std': np.std(tent_accs),
            'se': np.std(tent_accs) / np.sqrt(len(seeds))
        }
        print(f"  TENT:   {results['tent'][corruption]['mean']:.2f}%")
        
        # MEMO
        memo_accs = []
        for seed in seeds:
            acc = evaluate_memo(model, num_classes, corruption, severity, seed, data_dir, device)
            memo_accs.append(acc)
        results['memo'][corruption] = {
            'mean': np.mean(memo_accs),
            'std': np.std(memo_accs),
            'se': np.std(memo_accs) / np.sqrt(len(seeds))
        }
        print(f"  MEMO:   {results['memo'][corruption]['mean']:.2f}%")
        
        # APAC
        apac_accs = []
        for seed in seeds:
            acc = evaluate_apac(model, meta_apn, prototypes, num_classes, corruption, severity, seed, data_dir, device)
            apac_accs.append(acc)
        results['apac'][corruption] = {
            'mean': np.mean(apac_accs),
            'std': np.std(apac_accs),
            'se': np.std(apac_accs) / np.sqrt(len(seeds))
        }
        print(f"  APAC:   {results['apac'][corruption]['mean']:.2f}%")
    
    # Compute overall averages
    for method in results:
        mean_acc = np.mean([results[method][c]['mean'] for c in CORRUPTIONS_SUBSET])
        results[method]['average'] = mean_acc
    
    print(f"\n{'='*60}")
    print("Summary (Average over 5 corruptions):")
    print(f"{'='*60}")
    print(f"Source: {results['source']['average']:.2f}%")
    print(f"TENT:   {results['tent']['average']:.2f}%")
    print(f"MEMO:   {results['memo']['average']:.2f}%")
    print(f"APAC:   {results['apac']['average']:.2f}%")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open(f'results/fast_eval_{dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train Meta-APN quickly if needed
    if not os.path.exists('models/meta_apn_cifar10.pt'):
        print("Training Meta-APN for CIFAR-10...")
        from meta_apn.train_fast import train_meta_apn_fast
        train_meta_apn_fast('cifar10', epochs=5, device=device)
    
    # Run evaluations
    results_cifar10 = run_fast_evaluation('cifar10', './data', device)
