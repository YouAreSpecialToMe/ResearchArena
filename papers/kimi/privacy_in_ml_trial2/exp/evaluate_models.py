#!/usr/bin/env python3
"""Evaluate saved models and generate results.json."""
import sys
import os
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/privacy_in_machine_learning/idea_01')

import torch
import torch.nn as nn
import json

from shared.data_loader import get_cifar10_dataloaders
from shared.models import get_resnet18_cifar10

def evaluate_model(model_path, device):
    """Evaluate a single model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = get_resnet18_cifar10().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    _, _, test_loader, _, _ = get_cifar10_dataloaders(
        batch_size=128,
        num_workers=4,
        seed=42
    )
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for exp_name in ['g3p', 'hybrid_pruning']:
        exp_dir = f'exp/{exp_name}'
        if not os.path.exists(exp_dir):
            print(f'{exp_dir} does not exist, skipping...')
            continue
        
        results = {}
        for sparsity in [30, 50, 70]:
            accs = []
            for seed in [42, 43, 44]:
                model_path = f'{exp_dir}/model_sparsity{sparsity}_seed{seed}.pt'
                if os.path.exists(model_path):
                    print(f"Evaluating {exp_name} sparsity={sparsity} seed={seed}...")
                    acc = evaluate_model(model_path, device)
                    accs.append(acc)
                    print(f"  Accuracy: {acc:.2f}%")
                else:
                    print(f"  Model not found: {model_path}")
            
            if accs:
                mean_acc = sum(accs) / len(accs)
                std_acc = (sum((x - mean_acc)**2 for x in accs) / len(accs))**0.5 if len(accs) > 1 else 0
                results[f'sparsity_{sparsity}'] = {
                    'accuracy_mean': mean_acc,
                    'accuracy_std': std_acc,
                    'accuracies': accs
                }
                print(f"  Sparsity {sparsity}: {mean_acc:.2f} ± {std_acc:.2f}%")
        
        # Save results
        if results:
            with open(f'{exp_dir}/results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n{exp_name} results saved to {exp_dir}/results.json")
        else:
            print(f"\nNo results for {exp_name}")

if __name__ == '__main__':
    main()
