"""Evaluation script: compute calibration metrics, NC metrics, and temperature scaling."""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.data_utils import get_dataset, get_val_split, get_train_eval_loader
from shared.models import get_model
from shared.calibration import compute_calibration_metrics
from shared.nc_metrics import compute_nc_metrics
from shared.temperature_scaling import learn_temperature


@torch.no_grad()
def get_logits_and_labels(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        all_logits.append(logits.cpu())
        all_labels.append(targets)
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--skip_nc', action='store_true', help='Skip NC metrics (slow)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine num_classes
    nc_map = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}
    num_classes = nc_map[args.dataset]

    # Load model
    model = get_model(args.arch, num_classes, args.dataset).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Test data
    _, test_loader, _ = get_dataset(args.dataset, batch_size=256, num_workers=4, data_dir=args.data_dir)

    # Get logits and labels
    logits, labels = get_logits_and_labels(model, test_loader, device)
    test_acc = (logits.argmax(1) == labels).float().mean().item()

    # Calibration metrics
    cal_metrics = compute_calibration_metrics(logits, labels)

    # Temperature scaling
    val_loader = get_val_split(args.dataset, val_fraction=0.1, seed=42, batch_size=256, data_dir=args.data_dir)
    optimal_temp = learn_temperature(model, val_loader, device)

    # Calibration after TS
    scaled_logits = logits / optimal_temp
    cal_metrics_ts = compute_calibration_metrics(scaled_logits, labels)

    # NC metrics (on training set)
    nc_results = {}
    if not args.skip_nc:
        train_eval_loader = get_train_eval_loader(args.dataset, batch_size=256, num_workers=4, data_dir=args.data_dir)
        nc_results = compute_nc_metrics(model, train_eval_loader, num_classes, device)

    # Top-5 accuracy (for CIFAR-100 and TinyImageNet)
    top5_acc = None
    if num_classes > 10:
        _, top5_pred = logits.topk(5, dim=1)
        top5_acc = top5_pred.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # Compile results
    results = {
        'test_accuracy': test_acc,
        'top5_accuracy': top5_acc,
        'calibration': {
            'ece': cal_metrics['ece'],
            'mce': cal_metrics['mce'],
            'ada_ece': cal_metrics['ada_ece'],
            'nll': cal_metrics['nll'],
            'brier': cal_metrics['brier'],
        },
        'calibration_after_ts': {
            'ece': cal_metrics_ts['ece'],
            'mce': cal_metrics_ts['mce'],
            'ada_ece': cal_metrics_ts['ada_ece'],
            'nll': cal_metrics_ts['nll'],
            'brier': cal_metrics_ts['brier'],
        },
        'temperature': optimal_temp,
        'nc_metrics': nc_results,
        'reliability_bins': cal_metrics['reliability_bins'],
        'reliability_bins_ts': cal_metrics_ts['reliability_bins'],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Test accuracy: {test_acc:.4f}")
    if top5_acc is not None:
        print(f"Top-5 accuracy: {top5_acc:.4f}")
    print(f"ECE: {cal_metrics['ece']:.4f}")
    print(f"ECE (after TS, T={optimal_temp:.3f}): {cal_metrics_ts['ece']:.4f}")
    if nc_results:
        print(f"NC1: {nc_results['nc1']:.4f}")
        print(f"NC2: {nc_results['nc2']:.4f}")
    print(f"Results saved to {args.output_dir}/metrics.json")


if __name__ == '__main__':
    main()
