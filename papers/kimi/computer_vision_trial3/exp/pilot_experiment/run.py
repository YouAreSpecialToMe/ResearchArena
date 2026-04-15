"""
Pilot experiment: Validate SPT-TTA on a subset of corruptions.
Tests 3 corruption types to verify accuracy targets before full experiments.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/src')

import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from torch.utils.data import DataLoader

from data_loader import ImageNetV2Dataset, get_transform, IMAGENET_MEAN, IMAGENET_STD
from corruptions import CorruptedDataset
from models import PromptedViT, SPTTTA, VPA, TENT, MEMO, load_vit_model
from utils import set_seed, evaluate_model, save_results, print_results_table


def run_pilot_experiment(seed=42, device='cuda', max_samples=100):
    """Run pilot experiment on 3 corruption types."""
    print(f"\n{'='*60}")
    print(f"Pilot Experiment (Seed: {seed})")
    print(f"{'='*60}\n")
    
    # Set seed
    set_seed(seed)
    
    # Configuration
    corruptions = ['gaussian_noise', 'defocus_blur', 'brightness']
    severity = 3
    batch_size = 1  # TTA typically uses batch_size=1
    num_workers = 4
    
    # Load base model
    print("Loading ViT-B/16 model...")
    base_model = load_vit_model('vit_base_patch16_224', pretrained=True, device=device)
    
    # Load ImageNet-V2 dataset
    print("Loading ImageNet-V2 dataset...")
    transform = get_transform()
    base_dataset = ImageNetV2Dataset('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/data/imagenet-v2', transform=transform)
    
    # Limit dataset for pilot
    if max_samples:
        indices = torch.randperm(len(base_dataset))[:max_samples].tolist()
        base_dataset = torch.utils.data.Subset(base_dataset, indices)
        print(f"Using {max_samples} samples for pilot")
    
    results = {
        'config': {
            'seed': seed,
            'model': 'vit_base_patch16_224',
            'dataset': 'imagenet-v2-corrupted',
            'corruptions': corruptions,
            'severity': severity,
            'batch_size': batch_size,
            'max_samples': max_samples
        }
    }
    
    # Evaluate Source model (no adaptation)
    print("\n" + "-"*60)
    print("Evaluating Source Model (No Adaptation)")
    print("-"*60)
    
    for corruption in corruptions:
        print(f"\nCorruption: {corruption}")
        corrupted_dataset = CorruptedDataset(base_dataset, corruption, severity)
        loader = DataLoader(corrupted_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers, pin_memory=True)
        
        metrics = evaluate_model(base_model, loader, device, adapt_method=None)
        results[f'source_{corruption}'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Time: {metrics['time_per_image']*1000:.2f} ms/image")
    
    # Evaluate SPT-TTA
    print("\n" + "-"*60)
    print("Evaluating SPT-TTA")
    print("-"*60)
    
    # Create prompted model
    prompted_model = PromptedViT(num_prompts=4, pretrained=False)
    prompted_model.vit = base_model
    prompted_model = prompted_model.to(device)
    
    spttta = SPTTTA(
        prompted_model,
        lr=5e-4,
        layer_weights=[0.3]*4 + [0.5]*4 + [0.7]*4,
        selection_threshold=0.5,
        adapt_steps=1
    )
    
    for corruption in corruptions:
        print(f"\nCorruption: {corruption}")
        corrupted_dataset = CorruptedDataset(base_dataset, corruption, severity)
        loader = DataLoader(corrupted_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)
        
        # Reset prompts for each corruption
        for prompt in prompted_model.prompts:
            nn.init.normal_(prompt, std=0.02)
        
        metrics = evaluate_model(prompted_model, loader, device, adapt_method=spttta)
        results[f'spttta_{corruption}'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Time: {metrics['time_per_image']*1000:.2f} ms/image")
    
    # Compute averages
    source_accs = [results[f'source_{c}']['accuracy'] for c in corruptions]
    spttta_accs = [results[f'spttta_{c}']['accuracy'] for c in corruptions]
    
    results['source_average'] = {
        'accuracy': np.mean(source_accs),
        'std': np.std(source_accs)
    }
    results['spttta_average'] = {
        'accuracy': np.mean(spttta_accs),
        'std': np.std(spttta_accs)
    }
    
    print(f"\n{'='*60}")
    print("Pilot Results Summary")
    print(f"{'='*60}")
    print(f"Source Average: {results['source_average']['accuracy']:.2f}% (±{results['source_average']['std']:.2f})")
    print(f"SPT-TTA Average: {results['spttta_average']['accuracy']:.2f}% (±{results['spttta_average']['std']:.2f})")
    print(f"Improvement: {results['spttta_average']['accuracy'] - results['source_average']['accuracy']:.2f}%")
    
    # Success check
    improvement = results['spttta_average']['accuracy'] - results['source_average']['accuracy']
    if results['spttta_average']['accuracy'] >= 50:  # Adjusted target for pilot with V2
        print(f"\n✓ Pilot SUCCESS: SPT-TTA achieves {results['spttta_average']['accuracy']:.2f}%")
    else:
        print(f"\n⚠ Pilot needs adjustment: accuracy below target")
    
    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run with multiple seeds
    all_results = []
    for seed in [42, 123, 456]:
        results = run_pilot_experiment(seed=seed, device=device, max_samples=100)
        all_results.append(results)
        
        # Save individual seed results
        output_dir = '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/exp/pilot_experiment'
        save_results(results, f'{output_dir}/results_seed_{seed}.json')
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregated Results (3 seeds)")
    print(f"{'='*60}")
    
    source_accs = [r['source_average']['accuracy'] for r in all_results]
    spttta_accs = [r['spttta_average']['accuracy'] for r in all_results]
    
    print(f"Source: {np.mean(source_accs):.2f}% ± {np.std(source_accs):.2f}")
    print(f"SPT-TTA: {np.mean(spttta_accs):.2f}% ± {np.std(spttta_accs):.2f}")
    
    aggregated = {
        'source_mean': np.mean(source_accs),
        'source_std': np.std(source_accs),
        'spttta_mean': np.mean(spttta_accs),
        'spttta_std': np.std(spttta_accs),
        'improvement': np.mean(spttta_accs) - np.mean(source_accs),
        'all_results': all_results
    }
    
    save_results(aggregated, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/exp/pilot_experiment/aggregated_results.json')
