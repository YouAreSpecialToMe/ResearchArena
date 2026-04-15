"""
Source Model Baseline: No adaptation
"""

import os
import sys
import json
import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from shared.data_loader import load_imagenet_c, load_imagenet_r, load_imagenet_sketch, get_transform
from shared.models import load_pretrained_vit
from shared.metrics import accuracy, expected_calibration_error
from shared.utils import set_seed, save_results, AverageMeter, get_device


def evaluate(model, dataloader, device):
    """Evaluate model without adaptation."""
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1_meter.update(acc1, inputs.size(0))
            top5_meter.update(acc5, inputs.size(0))
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    ece = expected_calibration_error(all_outputs, all_targets)
    
    return {
        'top1_acc': top1_meter.avg,
        'top5_acc': top5_meter.avg,
        'ece': ece
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['imagenet-c', 'imagenet-r', 'imagenet-sketch'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--corruption', type=str, default=None)
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results/source')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating Source Model on {args.dataset}")
    
    # Load model
    model = load_pretrained_vit(device)
    
    # Load dataset
    if args.dataset == 'imagenet-c':
        dataloader, dataset = load_imagenet_c(
            os.path.join(args.data_root, 'imagenet-c'),
            corruption=args.corruption,
            severity=args.severity,
            batch_size=args.batch_size,
            seed=args.seed
        )
    elif args.dataset == 'imagenet-r':
        dataloader, dataset = load_imagenet_r(
            os.path.join(args.data_root, 'imagenet-r'),
            batch_size=args.batch_size,
            seed=args.seed
        )
    else:
        dataloader, dataset = load_imagenet_sketch(
            os.path.join(args.data_root, 'imagenet-sketch'),
            batch_size=args.batch_size,
            seed=args.seed
        )
    
    # Evaluate
    start_time = time.time()
    results = evaluate(model, dataloader, device)
    runtime = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5_acc']:.2f}%")
    print(f"  ECE: {results['ece']:.4f}")
    print(f"  Runtime: {runtime:.2f}s")
    
    output = {
        'experiment': 'source_model',
        'dataset': args.dataset,
        'corruption': args.corruption,
        'severity': args.severity,
        'seed': args.seed,
        'metrics': results,
        'runtime_seconds': runtime,
        'config': vars(args)
    }
    
    output_file = os.path.join(args.output_dir, f'results_{args.dataset}_seed{args.seed}.json')
    save_results(output, output_file)
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
