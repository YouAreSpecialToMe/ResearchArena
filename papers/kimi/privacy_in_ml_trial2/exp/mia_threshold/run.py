"""Threshold-based Membership Inference Attack evaluation."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from tqdm import tqdm
import json
import os

from shared.data_loader import get_cifar10_dataloaders, get_mia_dataloaders
from shared.models import get_resnet18_cifar10
from shared.utils import set_seed, save_results


def compute_losses(model, dataloader, device):
    """Compute per-sample losses."""
    model.eval()
    all_losses = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            losses = F.cross_entropy(outputs, labels, reduction='none')
            all_losses.extend(losses.cpu().numpy())
    
    return np.array(all_losses)


def threshold_attack(member_losses, non_member_losses):
    """
    Threshold-based MIA using loss values.
    Lower loss -> more likely to be member.
    """
    # Create labels and scores
    # Members = 1, Non-members = 0
    # Score = -loss (higher score = more likely member)
    member_labels = np.ones(len(member_losses))
    non_member_labels = np.zeros(len(non_member_losses))
    
    all_labels = np.concatenate([member_labels, non_member_labels])
    all_scores = np.concatenate([-member_losses, -non_member_losses])
    
    # Compute AUC
    auc = roc_auc_score(all_labels, all_scores)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    # Find optimal threshold for accuracy
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute accuracy
    predictions = (all_scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    
    # Compute advantage at low FPR
    fpr_001_idx = np.where(fpr <= 0.001)[0]
    advantage_001 = tpr[fpr_001_idx[-1]] if len(fpr_001_idx) > 0 else 0.0
    
    fpr_01_idx = np.where(fpr <= 0.01)[0]
    advantage_01 = tpr[fpr_01_idx[-1]] if len(fpr_01_idx) > 0 else 0.0
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'advantage_0.1%_fpr': advantage_001,
        'advantage_1%_fpr': advantage_01
    }


def evaluate_model_mia(model_path, seed, device='cuda'):
    """Evaluate MIA on a single model."""
    set_seed(seed)
    
    # Load data
    train_loader, val_loader, test_loader, train_indices, val_indices = get_cifar10_dataloaders(
        batch_size=128, num_workers=4, seed=seed
    )
    
    # Get MIA evaluation loaders
    member_loader, non_member_loader, member_indices, non_member_indices = get_mia_dataloaders(
        train_indices, val_indices, batch_size=128, num_workers=4,
        num_members=1000, num_non_members=1000, seed=seed
    )
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = get_resnet18_cifar10(num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Compute losses
    member_losses = compute_losses(model, member_loader, device)
    non_member_losses = compute_losses(model, non_member_loader, device)
    
    # Run MIA
    results = threshold_attack(member_losses, non_member_losses)
    
    return results


def evaluate_all_models():
    """Evaluate MIA on all trained models."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    seeds = [42, 43, 44]
    all_results = {}
    
    # Evaluate unpruned baseline
    print("\n" + "="*60)
    print("Evaluating Unpruned Baseline")
    print("="*60)
    baseline_results = []
    for seed in seeds:
        model_path = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/baseline_unpruned/model_seed{seed}.pt'
        result = evaluate_model_mia(model_path, seed, device)
        if result is not None:
            baseline_results.append(result)
    
    if baseline_results:
        all_results['unpruned'] = {
            'mia_auc': {
                'mean': np.mean([r['auc'] for r in baseline_results]),
                'std': np.std([r['auc'] for r in baseline_results]),
                'values': [r['auc'] for r in baseline_results]
            },
            'mia_accuracy': {
                'mean': np.mean([r['accuracy'] for r in baseline_results]),
                'std': np.std([r['accuracy'] for r in baseline_results])
            },
            'advantage_0.1%_fpr': {
                'mean': np.mean([r['advantage_0.1%_fpr'] for r in baseline_results]),
                'std': np.std([r['advantage_0.1%_fpr'] for r in baseline_results])
            }
        }
    
    # Evaluate Magnitude Pruning at different sparsities
    for sparsity in [30, 50, 70]:
        print(f"\n{'='*60}")
        print(f"Evaluating Magnitude Pruning ({sparsity}% sparsity)")
        print("="*60)
        mag_results = []
        for seed in seeds:
            model_path = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/magnitude_pruning/model_sparsity{sparsity}_seed{seed}.pt'
            result = evaluate_model_mia(model_path, seed, device)
            if result is not None:
                mag_results.append(result)
        
        if mag_results:
            all_results[f'magnitude_{sparsity}'] = {
                'mia_auc': {
                    'mean': np.mean([r['auc'] for r in mag_results]),
                    'std': np.std([r['auc'] for r in mag_results]),
                    'values': [r['auc'] for r in mag_results]
                },
                'mia_accuracy': {
                    'mean': np.mean([r['accuracy'] for r in mag_results]),
                    'std': np.std([r['accuracy'] for r in mag_results])
                }
            }
    
    # Evaluate Hybrid (Magnitude + KL)
    for sparsity in [30, 50, 70]:
        print(f"\n{'='*60}")
        print(f"Evaluating Hybrid ({sparsity}% sparsity)")
        print("="*60)
        hybrid_results = []
        for seed in seeds:
            model_path = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/hybrid_pruning/model_sparsity{sparsity}_seed{seed}.pt'
            result = evaluate_model_mia(model_path, seed, device)
            if result is not None:
                hybrid_results.append(result)
        
        if hybrid_results:
            all_results[f'hybrid_{sparsity}'] = {
                'mia_auc': {
                    'mean': np.mean([r['auc'] for r in hybrid_results]),
                    'std': np.std([r['auc'] for r in hybrid_results]),
                    'values': [r['auc'] for r in hybrid_results]
                },
                'mia_accuracy': {
                    'mean': np.mean([r['accuracy'] for r in hybrid_results]),
                    'std': np.std([r['accuracy'] for r in hybrid_results])
                }
            }
    
    # Evaluate G3P
    for sparsity in [30, 50, 70]:
        print(f"\n{'='*60}")
        print(f"Evaluating G3P ({sparsity}% sparsity)")
        print("="*60)
        g3p_results = []
        for seed in seeds:
            model_path = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/g3p/model_sparsity{sparsity}_seed{seed}.pt'
            result = evaluate_model_mia(model_path, seed, device)
            if result is not None:
                g3p_results.append(result)
        
        if g3p_results:
            all_results[f'g3p_{sparsity}'] = {
                'mia_auc': {
                    'mean': np.mean([r['auc'] for r in g3p_results]),
                    'std': np.std([r['auc'] for r in g3p_results]),
                    'values': [r['auc'] for r in g3p_results]
                },
                'mia_accuracy': {
                    'mean': np.mean([r['accuracy'] for r in g3p_results]),
                    'std': np.std([r['accuracy'] for r in g3p_results])
                }
            }
    
    # Evaluate Taylor Pruning at 50%
    print(f"\n{'='*60}")
    print(f"Evaluating Taylor Pruning (50% sparsity)")
    print("="*60)
    taylor_results = []
    for seed in seeds:
        model_path = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/taylor_pruning/model_sparsity50_seed{seed}.pt'
        result = evaluate_model_mia(model_path, seed, device)
        if result is not None:
            taylor_results.append(result)
    
    if taylor_results:
        all_results['taylor_50'] = {
            'mia_auc': {
                'mean': np.mean([r['auc'] for r in taylor_results]),
                'std': np.std([r['auc'] for r in taylor_results]),
                'values': [r['auc'] for r in taylor_results]
            },
            'mia_accuracy': {
                'mean': np.mean([r['accuracy'] for r in taylor_results]),
                'std': np.std([r['accuracy'] for r in taylor_results])
            }
        }
    
    # Save results
    all_results['experiment'] = 'mia_threshold_attack'
    all_results['seeds'] = seeds
    
    save_path = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01/exp/mia_threshold/results.json'
    save_results(all_results, save_path)
    print(f"\nResults saved to {save_path}")
    
    return all_results


if __name__ == '__main__':
    evaluate_all_models()
