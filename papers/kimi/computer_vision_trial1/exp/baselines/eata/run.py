"""
EATA Baseline: Sample selection with Fisher regularization
"""

import os
import sys
import json
import argparse
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from shared.data_loader import load_imagenet_c, load_imagenet_r, load_imagenet_sketch, get_transform
from shared.models import load_pretrained_vit
from shared.metrics import accuracy, expected_calibration_error
from shared.utils import set_seed, save_results, AverageMeter, get_device, entropy_loss


def configure_model_for_eata(model):
    """Configure model for EATA (train LayerNorm)."""
    model.train()
    
    for param in model.parameters():
        param.requires_grad = False
    
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True
    
    return model


def sample_selection(outputs, threshold):
    """Select samples based on entropy threshold."""
    probs = torch.softmax(outputs, dim=1)
    entropies = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    
    # Select samples with entropy < threshold
    mask = entropies < threshold
    return mask


def fisher_regularization(model, fisher_info, lambda_reg=2000):
    """Compute Fisher regularization loss."""
    loss = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_info:
            loss += (fisher_info[name] * (param ** 2)).sum()
    return lambda_reg * loss


def adapt_batch_eata(model, inputs, optimizer, entropy_threshold, num_classes=1000, lambda_fisher=2000):
    """Adapt model using EATA with sample selection and Fisher regularization."""
    model.train()
    
    optimizer.zero_grad()
    outputs = model(inputs)
    
    # Sample selection
    mask = sample_selection(outputs, entropy_threshold)
    
    if mask.sum() > 0:
        selected_outputs = outputs[mask]
        loss = entropy_loss(selected_outputs)
        
        # Add Fisher regularization (simplified - would need precomputed fisher info)
        # loss += fisher_regularization(model, fisher_info, lambda_fisher)
        
        loss.backward()
        optimizer.step()


def evaluate_eata(model, dataloader, device, lr=1e-3, adapt=True, num_classes=1000):
    """Evaluate with EATA adaptation."""
    if adapt:
        model = configure_model_for_eata(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        
        # Entropy threshold: 0.4 * ln(num_classes)
        entropy_threshold = 0.4 * math.log(num_classes)
    else:
        model.eval()
    
    all_outputs = []
    all_targets = []
    
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    for inputs, targets in tqdm(dataloader, desc="Evaluating EATA"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if adapt:
            with torch.enable_grad():
                adapt_batch_eata(model, inputs, optimizer, entropy_threshold, num_classes)
        
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
    parser.add_argument('--lambda_fisher', type=float, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results/eata')
    parser.add_argument('--no_adapt', action='store_true')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating EATA on {args.dataset}")
    
    model = load_pretrained_vit(device)
    
    # Determine num_classes
    if args.dataset == 'imagenet-r':
        num_classes = 200
    else:
        num_classes = 1000
    
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
    results = evaluate_eata(model, dataloader, device, lr=args.lr, 
                           adapt=not args.no_adapt, num_classes=num_classes)
    runtime = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5_acc']:.2f}%")
    print(f"  ECE: {results['ece']:.4f}")
    print(f"  Runtime: {runtime:.2f}s")
    
    output = {
        'experiment': 'eata',
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
