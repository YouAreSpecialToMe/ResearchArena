"""
Tent Baseline: Entropy Minimization with LayerNorm adaptation
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from shared.data_loader import load_imagenet_c, load_imagenet_r, load_imagenet_sketch, get_transform
from shared.models import load_pretrained_vit
from shared.metrics import accuracy, expected_calibration_error
from shared.utils import set_seed, save_results, AverageMeter, get_device, entropy_loss


def configure_model_for_tent(model):
    """Configure model for Tent adaptation (train LayerNorm only)."""
    model.train()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LayerNorm parameters
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True
    
    return model


def tent_adapt_batch(model, inputs, optimizer):
    """Adapt model on a single batch using Tent."""
    model.train()
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = entropy_loss(outputs)
    loss.backward()
    optimizer.step()


def evaluate_tent(model, dataloader, device, lr=1e-3, adapt=True):
    """Evaluate with Tent adaptation."""
    if adapt:
        model = configure_model_for_tent(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        model.eval()
    
    all_outputs = []
    all_targets = []
    
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    for inputs, targets in tqdm(dataloader, desc="Evaluating Tent"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if adapt:
            with torch.enable_grad():
                tent_adapt_batch(model, inputs, optimizer)
        
        with torch.no_grad():
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results/tent')
    parser.add_argument('--no_adapt', action='store_true')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating Tent on {args.dataset}")
    
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
    
    start_time = time.time()
    results = evaluate_tent(model, dataloader, device, lr=args.lr, adapt=not args.no_adapt)
    runtime = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5_acc']:.2f}%")
    print(f"  ECE: {results['ece']:.4f}")
    print(f"  Runtime: {runtime:.2f}s")
    
    output = {
        'experiment': 'tent',
        'dataset': args.dataset,
        'corruption': args.corruption,
        'severity': args.severity,
        'seed': args.seed,
        'adapt': not args.no_adapt,
        'metrics': results,
        'runtime_seconds': runtime,
        'config': vars(args)
    }
    
    output_file = os.path.join(args.output_dir, f'results_{args.dataset}_seed{args.seed}.json')
    save_results(output, output_file)
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
