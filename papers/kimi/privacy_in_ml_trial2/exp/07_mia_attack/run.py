"""
Membership Inference Attack evaluation.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.models import get_model
from shared.data_loader import get_data_loaders
from shared.utils import set_seed


def simple_confidence_attack(model, data_loader, device):
    """Simple confidence-based MIA."""
    model.eval()
    confidences = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            conf = probs.max(dim=1)[0].cpu().numpy()
            confidences.append(conf)
    
    return np.concatenate(confidences)


def compute_mia_metrics(member_scores, non_member_scores):
    """Compute MIA metrics."""
    # Threshold attack accuracy
    all_scores = np.concatenate([member_scores, non_member_scores])
    all_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
    
    # Find best threshold
    best_acc = 0
    for threshold in np.linspace(all_scores.min(), all_scores.max(), 100):
        preds = (all_scores >= threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
    
    # AUC
    auc = roc_auc_score(all_labels, all_scores)
    
    return {
        'attack_accuracy': float(best_acc),
        'auc': float(auc),
        'member_mean_conf': float(np.mean(member_scores)),
        'non_member_mean_conf': float(np.mean(non_member_scores))
    }


def evaluate_model_mia(model_path, dataset, arch, seed, device, data_dir='./data'):
    """Evaluate a single model against MIA."""
    # Load model
    num_classes = 10 if dataset == 'cifar10' else (100 if dataset == 'cifar100' else 100)
    input_dim = 600 if dataset == 'purchase100' else None
    model = get_model(arch, num_classes, input_dim)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Load data
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        dataset, data_dir, batch_size=256, num_workers=4, seed=seed
    )
    
    # Get confidence scores
    member_scores = simple_confidence_attack(model, train_loader, device)
    non_member_scores = simple_confidence_attack(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_mia_metrics(member_scores, non_member_scores)
    
    # Get test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_acc = 100. * correct / total
    metrics['test_accuracy'] = float(test_acc)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['baseline', 'prism', 'cwrf', 'blocklevel', 'fullft'])
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = {}
    
    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Evaluating {method.upper()}")
        print('='*60)
        
        method_results = []
        
        for seed in args.seeds:
            model_path = f'./models/{args.dataset}_{args.arch}_seed{seed}_{method}.pth'
            
            if not os.path.exists(model_path):
                print(f"  Model not found: {model_path}")
                continue
            
            print(f"  Seed {seed}: {model_path}")
            set_seed(seed)
            
            metrics = evaluate_model_mia(model_path, args.dataset, args.arch, seed, device, args.data_dir)
            method_results.append(metrics)
            
            print(f"    Test Acc: {metrics['test_accuracy']:.2f}%")
            print(f"    MIA Acc: {metrics['attack_accuracy']*100:.2f}%")
            print(f"    AUC: {metrics['auc']:.4f}")
        
        if method_results:
            # Aggregate across seeds
            agg_metrics = {}
            for key in method_results[0].keys():
                values = [r[key] for r in method_results]
                agg_metrics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
            
            all_results[method] = {
                'per_seed': method_results,
                'aggregated': agg_metrics
            }
            
            print(f"\n  {method.upper()} Summary:")
            print(f"    Test Acc: {agg_metrics['test_accuracy']['mean']:.2f} ± {agg_metrics['test_accuracy']['std']:.2f}%")
            print(f"    MIA Acc: {agg_metrics['attack_accuracy']['mean']*100:.2f} ± {agg_metrics['attack_accuracy']['std']*100:.2f}%")
            print(f"    AUC: {agg_metrics['auc']['mean']:.4f} ± {agg_metrics['auc']['std']:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_path = f'results/{args.dataset}_{args.arch}_mia_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All results saved to {save_path}")
    print('='*60)


if __name__ == '__main__':
    main()
