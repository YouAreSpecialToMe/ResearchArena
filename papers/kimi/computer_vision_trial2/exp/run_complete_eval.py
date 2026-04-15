"""
Complete evaluation for all methods.
Runs single-seed evaluation for speed.
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

from shared.models import wideresnet28_10, MetaAPN, get_prototype_distances, collect_params
from shared.data_loader import load_cifar_c
from shared.augmentations import AUGMENTATION_OPS, apply_augmentation
from shared.utils import set_seed, load_prototypes, copy_model, load_model_state
from torchvision import transforms

# 5 representative corruptions
CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'brightness', 'contrast']


def evaluate_source(model, loader, device):
    """Source baseline"""
    model.eval()
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


def evaluate_tent(model, loader, device):
    """TENT baseline"""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for param in module.parameters():
                param.requires_grad = True
            module.train()
    
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3) if params else None
    
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        if optimizer:
            outputs = model(images)
            p = F.softmax(outputs, dim=1)
            loss = -(p * torch.log(p + 1e-10)).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return 100.0 * correct / total


def evaluate_memo(model, loader, device):
    """MEMO baseline"""
    model.eval()
    original_state = copy_model(model)
    augmentations = [('gaussian_noise', 3), ('brightness', 3), ('contrast', 3)]
    
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='MEMO', leave=False):
        images, labels = images.to(device), labels.to(device)
        load_model_state(model, original_state)
        model.eval()
        
        # Create augmented batch
        aug_batch = [images[0]]
        for op, sev in augmentations:
            aug_img = apply_augmentation(images[0], op, sev, 'cifar10')
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
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return 100.0 * correct / total


def evaluate_apac(model, meta_apn, prototypes, loader, device):
    """APAC-TTA"""
    model.eval()
    original_state = copy_model(model)
    
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='APAC', leave=False):
        images, labels = images.to(device), labels.to(device)
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
            aug_img = apply_augmentation(images[0], op_name, sev, 'cifar10')
            if not isinstance(aug_img, torch.Tensor):
                aug_img = transforms.ToTensor()(aug_img)
            aug_batch.append(aug_img)
        aug_batch = torch.stack(aug_batch).to(device)
        
        # Adapt with prototype guidance
        params, _ = collect_params(model)
        if params:
            optimizer = torch.optim.Adam(params, lr=1e-3)
            
            with torch.no_grad():
                nearest = distances.argmin().item()
                target = torch.zeros(1, 10).to(device)
                target[0, nearest] = 1.0
            
            outputs = model(aug_batch)
            probs = F.softmax(outputs, dim=1)
            target_expanded = target.expand(probs.size(0), -1)
            
            # JS divergence
            m = 0.5 * (probs + target_expanded)
            kl1 = (probs * torch.log(probs / (m + 1e-10) + 1e-10)).sum(dim=1)
            kl2 = (target_expanded * torch.log(target_expanded / (m + 1e-10) + 1e-10)).sum(dim=1)
            loss = 0.5 * (kl1 + kl2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return 100.0 * correct / total


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model and prototypes
    print("Loading model and prototypes...")
    model = wideresnet28_10(num_classes=10).to(device)
    model.eval()
    
    prototypes = load_prototypes('cifar10', device)
    if prototypes is None:
        prototypes = torch.randn(10, model.feature_dim).to(device)
    
    # Load Meta-APN
    meta_apn_path = 'models/meta_apn_cifar10.pt'
    if os.path.exists(meta_apn_path):
        meta_apn = MetaAPN(model.feature_dim, 10, num_operations=len(AUGMENTATION_OPS)).to(device)
        meta_apn.load_state_dict(torch.load(meta_apn_path, map_location=device))
        meta_apn.eval()
        print("Loaded Meta-APN")
    else:
        print(f"Meta-APN not found at {meta_apn_path}")
        return
    
    # Evaluate on each corruption
    results = {}
    
    for corruption in CORRUPTIONS:
        print(f"\n{'='*60}")
        print(f"Evaluating on {corruption}")
        print('='*60)
        
        set_seed(2022)
        loader, _ = load_cifar_c('./data', 'cifar10', corruption, 5, batch_size=1)
        
        # Source
        print("Running Source...")
        set_seed(2022)
        loader_source, _ = load_cifar_c('./data', 'cifar10', corruption, 5, batch_size=200)
        source_acc = evaluate_source(model, loader_source, device)
        print(f"Source: {source_acc:.2f}%")
        
        # TENT
        print("Running TENT...")
        set_seed(2022)
        loader_tent, _ = load_cifar_c('./data', 'cifar10', corruption, 5, batch_size=200)
        tent_acc = evaluate_tent(model, loader_tent, device)
        print(f"TENT: {tent_acc:.2f}%")
        
        # MEMO
        print("Running MEMO...")
        set_seed(2022)
        loader_memo, _ = load_cifar_c('./data', 'cifar10', corruption, 5, batch_size=1)
        memo_acc = evaluate_memo(model, loader_memo, device)
        print(f"MEMO: {memo_acc:.2f}%")
        
        # APAC
        print("Running APAC-TTA...")
        set_seed(2022)
        loader_apac, _ = load_cifar_c('./data', 'cifar10', corruption, 5, batch_size=1)
        apac_acc = evaluate_apac(model, meta_apn, prototypes, loader_apac, device)
        print(f"APAC-TTA: {apac_acc:.2f}%")
        
        results[corruption] = {
            'source': source_acc,
            'tent': tent_acc,
            'memo': memo_acc,
            'apac': apac_acc
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"{'Corruption':<20} {'Source':>10} {'TENT':>10} {'MEMO':>10} {'APAC-TTA':>10}")
    print('-'*60)
    
    source_avg = []
    tent_avg = []
    memo_avg = []
    apac_avg = []
    
    for corruption in CORRUPTIONS:
        r = results[corruption]
        print(f"{corruption:<20} {r['source']:>10.1f} {r['tent']:>10.1f} {r['memo']:>10.1f} {r['apac']:>10.1f}")
        source_avg.append(r['source'])
        tent_avg.append(r['tent'])
        memo_avg.append(r['memo'])
        apac_avg.append(r['apac'])
    
    print('-'*60)
    print(f"{'Average':<20} {np.mean(source_avg):>10.1f} {np.mean(tent_avg):>10.1f} {np.mean(memo_avg):>10.1f} {np.mean(apac_avg):>10.1f}")
    print('='*60)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/complete_eval.json', 'w') as f:
        json.dump({
            'per_corruption': results,
            'average': {
                'source': float(np.mean(source_avg)),
                'tent': float(np.mean(tent_avg)),
                'memo': float(np.mean(memo_avg)),
                'apac': float(np.mean(apac_avg))
            }
        }, f, indent=2)
    
    print("\nResults saved to results/complete_eval.json")


if __name__ == '__main__':
    main()
