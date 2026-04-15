#!/usr/bin/env python3
"""Run a single complete experiment for validation."""
import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).parent))

from exp.shared.models import get_model
from exp.shared.data_loader import load_dataset, create_forget_retain_splits, get_dataloader
from exp.fixed_implementation.unlearning_fixed import apply_unlearning_fixed, evaluate_accuracy
from exp.fixed_implementation.lgsa_fixed import LGSAFixed
from exp.baseline_truvrf.truvrf import TruVRF
from sklearn.metrics import roc_auc_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    dataset, model_name, method, seed = 'cifar10', 'resnet18', 'gradient_ascent', 42
    
    print(f"\nExperiment: {dataset} {model_name} {method} seed={seed}")
    set_seed(seed)
    
    # Load base model
    model_path = f'results/models/{dataset}_{model_name}_seed{seed}_base.pth'
    print(f"Loading: {model_path}")
    base_model = get_model(model_name, num_classes=10, input_channels=3)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model = base_model.to(device)
    
    # Load data
    train_dataset, test_dataset, _, _ = load_dataset(dataset)
    forget_indices, retain_indices, _ = create_forget_retain_splits(train_dataset, seed=seed)
    
    forget_indices = forget_indices[:1000]
    retain_indices = retain_indices[:1000]
    
    forget_loader = get_dataloader(train_dataset, forget_indices, batch_size=128, shuffle=False, num_workers=2)
    retain_loader = get_dataloader(train_dataset, retain_indices, batch_size=128, shuffle=True, num_workers=2)
    test_loader = get_dataloader(test_dataset, None, batch_size=128, shuffle=False, num_workers=2)
    
    # Unlearn
    print("\nUnlearning...")
    unlearned_model = copy.deepcopy(base_model)
    unlearned_model, _ = apply_unlearning_fixed(
        unlearned_model, 'gradient_ascent', forget_loader, retain_loader,
        epochs=5, lr=0.001, device=device, grad_clip=1.0
    )
    
    retain_acc = evaluate_accuracy(unlearned_model, retain_loader, device)
    test_acc = evaluate_accuracy(unlearned_model, test_loader, device)
    print(f"Unlearned - Retain: {retain_acc:.4f}, Test: {test_acc:.4f}")
    
    # Get data tensors
    forget_data, forget_targets = [], []
    for data, target in forget_loader:
        forget_data.append(data)
        forget_targets.append(target)
    forget_data = torch.cat(forget_data).to(device)
    forget_targets = torch.cat(forget_targets).to(device)
    
    retain_data, retain_targets = [], []
    for data, target in retain_loader:
        retain_data.append(data)
        retain_targets.append(target)
    retain_data = torch.cat(retain_data).to(device)
    retain_targets = torch.cat(retain_targets).to(device)
    
    # LGSA
    print("\nLGSA...")
    lgsa = LGSAFixed(base_model, unlearned_model, device=device)
    weights = lgsa.learn_weights(forget_data[:300], forget_targets[:300],
                                  retain_data[:300], retain_targets[:300])
    results, _, _ = lgsa.verify_unlearning(forget_data, forget_targets, 
                                            retain_data, retain_targets)
    print(f"LGSA: AUC={results['auc']:.4f}, TPR@1%FPR={results['tpr_at_1fpr']:.4f}")
    print(f"  Forget LSS: {results['forget_lss_mean']:.4f} ± {results['forget_lss_std']:.4f}")
    print(f"  Retain LSS: {results['retain_lss_mean']:.4f} ± {results['retain_lss_std']:.4f}")
    print(f"  Weights: {results['weights']}")
    
    # TruVRF
    print("\nTruVRF...")
    truvrf = TruVRF(base_model, unlearned_model, device=device)
    
    verify_data, verify_targets = [], []
    for data, target in test_loader:
        verify_data.append(data)
        verify_targets.append(target)
        if len(verify_data) >= 8: break
    verify_data = torch.cat(verify_data).to(device)
    verify_targets = torch.cat(verify_targets).to(device)
    
    tr_results, _, _ = truvrf.verify_unlearning(
        forget_data, forget_targets, retain_data, retain_targets,
        verify_data, verify_targets, epochs=2, lr=0.001
    )
    print(f"TruVRF: AUC={tr_results['auc']:.4f}, TPR@1%FPR={tr_results['tpr_at_1fpr']:.4f}")
    
    print(f"\nComparison:")
    print(f"  LGSA AUC - TruVRF AUC = {results['auc'] - tr_results['auc']:.4f}")
    
    # Save
    os.makedirs('results/metrics', exist_ok=True)
    result = {
        'config': {'dataset': dataset, 'model': model_name, 'method': method, 'seed': seed},
        'utility': {'retain_acc': retain_acc, 'test_acc': test_acc},
        'lgsa': {'auc': results['auc'], 'tpr_at_1fpr': results['tpr_at_1fpr'], 'time': results['verify_time']},
        'truvrf': {'auc': tr_results['auc'], 'tpr_at_1fpr': tr_results['tpr_at_1fpr'], 'time': tr_results['verify_time']}
    }
    with open('results/metrics/single_experiment_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
