"""Evaluate all trained models: calibration, NC metrics, temperature scaling."""

import json
import os
import sys
import glob
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.data_utils import get_dataset, get_val_split, get_train_eval_loader
from shared.models import get_model
from shared.calibration import compute_calibration_metrics
from shared.nc_metrics import compute_nc_metrics
from shared.temperature_scaling import learn_temperature


NC_MAP = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}


def evaluate_model(checkpoint_path, dataset, arch, data_dir, output_dir, device, skip_nc=False):
    """Evaluate a single model checkpoint."""
    num_classes = NC_MAP[dataset]
    model = get_model(arch, num_classes, dataset).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Test data
    _, test_loader, _ = get_dataset(dataset, batch_size=256, num_workers=4, data_dir=data_dir)

    # Get logits and labels
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

    # Calibration
    cal = compute_calibration_metrics(logits, labels)

    # Temperature scaling
    val_loader = get_val_split(dataset, val_fraction=0.1, seed=42, batch_size=256, data_dir=data_dir)
    temp = learn_temperature(model, val_loader, device)
    scaled_logits = logits / temp
    cal_ts = compute_calibration_metrics(scaled_logits, labels)

    # Top-5
    top5 = None
    if num_classes > 10:
        _, top5_pred = logits.topk(5, dim=1)
        top5 = top5_pred.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # NC metrics
    nc = {}
    if not skip_nc:
        train_eval_loader = get_train_eval_loader(dataset, batch_size=256, num_workers=4, data_dir=data_dir)
        nc = compute_nc_metrics(model, train_eval_loader, num_classes, device)

    results = {
        'test_accuracy': test_acc,
        'top5_accuracy': top5,
        'calibration': {k: cal[k] for k in ['ece', 'mce', 'ada_ece', 'nll', 'brier']},
        'calibration_after_ts': {k: cal_ts[k] for k in ['ece', 'mce', 'ada_ece', 'nll', 'brier']},
        'temperature': temp,
        'nc_metrics': nc,
        'reliability_bins': cal['reliability_bins'],
        'reliability_bins_ts': cal_ts['reliability_bins'],
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--skip_nc', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find all trained models
    for dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
        dataset_dir = os.path.join(args.results_dir, dataset)
        if not os.path.exists(dataset_dir):
            continue
        for method in sorted(os.listdir(dataset_dir)):
            method_dir = os.path.join(dataset_dir, method)
            if not os.path.isdir(method_dir):
                continue
            for seed_dir in sorted(os.listdir(method_dir)):
                run_dir = os.path.join(method_dir, seed_dir)
                ckpt = os.path.join(run_dir, 'best_model.pt')
                if not os.path.exists(ckpt):
                    continue
                metrics_file = os.path.join(run_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    print(f"  Skipping {dataset}/{method}/{seed_dir} (already evaluated)")
                    continue

                print(f"Evaluating {dataset}/{method}/{seed_dir}...")
                try:
                    r = evaluate_model(ckpt, dataset, 'resnet18', args.data_dir, run_dir, device,
                                       skip_nc=args.skip_nc)
                    print(f"  Acc={r['test_accuracy']:.4f}, ECE={r['calibration']['ece']:.4f}, "
                          f"ECE_TS={r['calibration_after_ts']['ece']:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

    print("\nAll evaluations complete.")


if __name__ == '__main__':
    main()
