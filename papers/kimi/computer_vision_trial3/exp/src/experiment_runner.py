"""
Streamlined experiment runner for all TTA methods.
"""
import sys
import os
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/src')

import torch
import torch.nn as nn
import numpy as np
import json
import time
from torch.utils.data import DataLoader, Subset
import argparse

from data_loader import ImageNetV2Dataset, get_transform
from corruptions import CorruptedDataset, CORRUPTION_FUNCTIONS
from models import PromptedViT, SPTTTA, VPA, TENT, MEMO, load_vit_model
from utils import set_seed, evaluate_model, save_results


CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'brightness', 'contrast']


def run_experiment(method='source', seed=42, device='cuda', max_samples=500):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Method: {method.upper()} | Seed: {seed}")
    print(f"{'='*60}\n")
    
    set_seed(seed)
    
    # Configuration
    severity = 3
    batch_size = 1 if method != 'source' else 64
    num_workers = 4
    
    # Load model
    print("Loading ViT-B/16...")
    base_model = load_vit_model('vit_base_patch16_224', pretrained=True, device=device)
    
    # Load dataset
    print("Loading ImageNet-V2...")
    transform = get_transform()
    base_dataset = ImageNetV2Dataset('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/data/imagenet-v2', transform=transform)
    
    if max_samples and max_samples < len(base_dataset):
        indices = torch.randperm(len(base_dataset))[:max_samples].tolist()
        base_dataset = Subset(base_dataset, indices)
        print(f"Using {max_samples} samples")
    
    results = {
        'config': {
            'method': method,
            'seed': seed,
            'model': 'vit_base_patch16_224',
            'severity': severity,
            'max_samples': max_samples
        }
    }
    
    # Setup adaptation method
    adapt_method = None
    if method == 'spttta':
        prompted_model = PromptedViT(num_prompts=4, pretrained=False)
        prompted_model.vit = base_model
        prompted_model = prompted_model.to(device)
        adapt_method = SPTTTA(prompted_model, lr=5e-4, adapt_steps=1)
        model = prompted_model
    elif method == 'vpa':
        prompted_model = PromptedViT(num_prompts=4, pretrained=False)
        prompted_model.vit = base_model
        prompted_model = prompted_model.to(device)
        adapt_method = VPA(prompted_model, lr=5e-4, adapt_steps=1)
        model = prompted_model
    elif method == 'tent':
        # TENT adapts BatchNorm stats
        model = base_model
        adapt_method = TENT(base_model, lr=1e-3)
    elif method == 'memo':
        model = base_model
        adapt_method = MEMO(base_model, lr=5e-4)
    else:  # source
        model = base_model
    
    # Evaluate on each corruption
    all_accs = []
    for corruption in CORRUPTIONS:
        print(f"\n{corruption}...")
        corrupted_dataset = CorruptedDataset(base_dataset, corruption, severity)
        loader = DataLoader(corrupted_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
        
        # Reset prompts for methods that use them
        if method in ['spttta', 'vpa']:
            for prompt in model.prompts:
                nn.init.normal_(prompt, std=0.02)
        
        metrics = evaluate_model(model, loader, device, adapt_method)
        results[corruption] = metrics
        all_accs.append(metrics['accuracy'])
        print(f"  Acc: {metrics['accuracy']:.2f}% | Time: {metrics['time_per_image']*1000:.1f}ms")
    
    # Compute average
    results['average'] = {
        'accuracy': float(np.mean(all_accs)),
        'std': float(np.std(all_accs)),
        'min': float(np.min(all_accs)),
        'max': float(np.max(all_accs))
    }
    
    print(f"\nAverage: {results['average']['accuracy']:.2f}% (±{results['average']['std']:.2f})")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='source',
                       choices=['source', 'tent', 'memo', 'vpa', 'spttta'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Run experiment
    results = run_experiment(args.method, args.seed, device, args.max_samples)
    
    # Save results
    if args.output is None:
        args.output = f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/results/{args.method}_seed{args.seed}.json'
    save_results(results, args.output)
    print(f"\nSaved to: {args.output}")
