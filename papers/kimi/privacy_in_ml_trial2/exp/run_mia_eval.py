#!/usr/bin/env python3
"""Run MIA evaluation on all saved models."""
import sys
import os
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import json

from shared.data_loader import get_cifar10_dataloaders, get_mia_dataloaders
from shared.models import get_resnet18_cifar10

def compute_losses(model, dataloader, device):
    """Compute per-sample cross-entropy losses."""
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            losses.append(loss.cpu().numpy())
    return np.concatenate(losses)

def evaluate_mia(model_path, device, train_indices, val_indices):
    """Evaluate MIA using loss-based attack."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = get_resnet18_cifar10().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get MIA dataloaders (member = train, non-member = validation)
    member_loader, non_member_loader, _, _ = get_mia_dataloaders(
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=128,
        num_workers=4,
        num_members=1000,
        num_non_members=1000,
        seed=42
    )
    
    # Compute losses
    member_losses = compute_losses(model, member_loader, device)
    non_member_losses = compute_losses(model, non_member_loader, device)
    
    # Create labels (1 = member, 0 = non-member)
    y_true = np.concatenate([
        np.ones(len(member_losses)),
        np.zeros(len(non_member_losses))
    ])
    
    # Use negative loss as membership score (higher loss = less likely to be member)
    y_score = np.concatenate([
        -member_losses,  # Negative because lower loss = more likely member
        -non_member_losses
    ])
    
    # Compute AUC
    auc = roc_auc_score(y_true, y_score)
    
    # Compute accuracy at threshold
    threshold = np.median(y_score)
    y_pred = (y_score > threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'auc': auc,
        'accuracy': acc,
        'member_loss_mean': float(np.mean(member_losses)),
        'member_loss_std': float(np.std(member_losses)),
        'non_member_loss_mean': float(np.mean(non_member_losses)),
        'non_member_loss_std': float(np.std(non_member_losses))
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get indices for MIA evaluation (same for all models since they use same data split)
    _, _, _, train_indices, val_indices = get_cifar10_dataloaders(
        batch_size=128,
        num_workers=4,
        seed=42
    )
    
    for exp_name in ['g3p', 'hybrid_pruning']:
        exp_dir = f'exp/{exp_name}'
        if not os.path.exists(exp_dir):
            continue
        
        mia_results = {}
        
        for sparsity in [30, 50, 70]:
            results_for_sparsity = []
            
            for seed in [42, 43, 44]:
                model_path = f'{exp_dir}/model_sparsity{sparsity}_seed{seed}.pt'
                if os.path.exists(model_path):
                    print(f"\nEvaluating MIA for {exp_name} sparsity={sparsity} seed={seed}...")
                    result = evaluate_mia(model_path, device, train_indices, val_indices)
                    results_for_sparsity.append(result)
                    print(f"  MIA AUC: {result['auc']:.4f}")
                    print(f"  Member loss: {result['member_loss_mean']:.4f} ± {result['member_loss_std']:.4f}")
                    print(f"  Non-member loss: {result['non_member_loss_mean']:.4f} ± {result['non_member_loss_std']:.4f}")
            
            if results_for_sparsity:
                # Aggregate results
                aucs = [r['auc'] for r in results_for_sparsity]
                mia_results[f'sparsity_{sparsity}'] = {
                    'auc_mean': sum(aucs) / len(aucs),
                    'auc_std': np.std(aucs),
                    'auc_values': aucs,
                    'per_seed': results_for_sparsity
                }
        
        # Save MIA results
        if mia_results:
            with open(f'{exp_dir}/mia_results.json', 'w') as f:
                json.dump(mia_results, f, indent=2)
            print(f"\n{exp_name} MIA results saved to {exp_dir}/mia_results.json")

if __name__ == '__main__':
    main()
