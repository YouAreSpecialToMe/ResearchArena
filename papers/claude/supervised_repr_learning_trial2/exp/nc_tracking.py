"""Compute NC metrics at each checkpoint epoch for CE vs CCR-adaptive on CIFAR-100."""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.data_utils import get_train_eval_loader, get_dataset
from shared.models import get_model
from shared.nc_metrics import compute_nc_metrics
from shared.calibration import compute_calibration_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = './results'
    dataset = 'cifar100'
    num_classes = 100
    seed = 'seed_42'

    # Get data loaders
    train_eval_loader = get_train_eval_loader(dataset, batch_size=256, num_workers=4, data_dir='./data')
    _, test_loader, _ = get_dataset(dataset, batch_size=256, num_workers=4, data_dir='./data')

    tracking_results = {}

    # Find best CCR method (one with checkpoints)
    ccr_methods = []
    dataset_dir = os.path.join(results_dir, dataset)
    if os.path.isdir(dataset_dir):
        for m in sorted(os.listdir(dataset_dir)):
            if m.startswith('ccr_') and '_100ep' not in m:
                seed_path = os.path.join(dataset_dir, m, seed)
                if os.path.isdir(seed_path) and any(f.startswith('checkpoint_') for f in os.listdir(seed_path)):
                    ccr_methods.append(m)
    methods_to_track = ['ce'] + (ccr_methods if ccr_methods else ['ccr_soft'])
    print(f"Tracking methods: {methods_to_track}")

    for method in methods_to_track:
        tracking_results[method] = {}
        method_dir = os.path.join(results_dir, dataset, method, seed)

        # Find all checkpoints
        checkpoints = []
        for f in sorted(os.listdir(method_dir)):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
                epoch = int(f.replace('checkpoint_epoch_', '').replace('.pt', ''))
                checkpoints.append((epoch, os.path.join(method_dir, f)))
        # Add final/best model
        for name in ['final_model.pt', 'best_model.pt']:
            path = os.path.join(method_dir, name)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                epoch = ckpt.get('epoch', 100)
                checkpoints.append((epoch, path))
                break

        checkpoints = sorted(set(checkpoints), key=lambda x: x[0])

        for epoch, ckpt_path in checkpoints:
            print(f"  {method} epoch {epoch}...")
            model = get_model('resnet18', num_classes, dataset).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            # NC metrics
            nc = compute_nc_metrics(model, train_eval_loader, num_classes, device)

            # Test calibration
            all_logits, all_labels = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    logits = model(inputs)
                    all_logits.append(logits.cpu())
                    all_labels.append(targets)
            logits = torch.cat(all_logits)
            labels = torch.cat(all_labels)
            test_acc = (logits.argmax(1) == labels).float().mean().item()
            cal = compute_calibration_metrics(logits, labels)

            tracking_results[method][epoch] = {
                'nc1': nc['nc1'],
                'nc2': nc['nc2'],
                'nc3': nc['nc3'],
                'nc4': nc['nc4'],
                'mean_within_class_spread': nc['mean_within_class_spread'],
                'test_accuracy': test_acc,
                'ece': cal['ece'],
                'nll': cal['nll'],
            }
            print(f"    NC1={nc['nc1']:.4f}, ECE={cal['ece']:.4f}, Acc={test_acc:.4f}")

            del model
            torch.cuda.empty_cache()

    # Save
    os.makedirs(os.path.join(results_dir, dataset), exist_ok=True)
    with open(os.path.join(results_dir, dataset, 'nc_tracking.json'), 'w') as f:
        json.dump(tracking_results, f, indent=2)
    print(f"\nSaved to {results_dir}/{dataset}/nc_tracking.json")


if __name__ == '__main__':
    main()
