"""
Source baseline: No adaptation, frozen model.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/src')

import torch
import numpy as np
from torch.utils.data import DataLoader

from data_loader import ImageNetV2Dataset, get_transform
from corruptions import CorruptedDataset, CORRUPTION_FUNCTIONS
from models import load_vit_model
from utils import set_seed, evaluate_model, save_results


# All corruption types (synthetic)
CORRUPTIONS = list(CORRUPTION_FUNCTIONS.keys())


def run_source_baseline(seed=42, device='cuda', max_samples=None):
    """Run source baseline (no adaptation)."""
    print(f"\n{'='*60}")
    print(f"Source Baseline (Seed: {seed})")
    print(f"{'='*60}\n")
    
    set_seed(seed)
    
    # Configuration
    severity = 3
    batch_size = 64  # Can use larger batch for no adaptation
    num_workers = 4
    
    # Load model
    print("Loading ViT-B/16 model...")
    model = load_vit_model('vit_base_patch16_224', pretrained=True, device=device)
    
    # Load dataset
    print("Loading ImageNet-V2 dataset...")
    transform = get_transform()
    base_dataset = ImageNetV2Dataset('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/data/imagenet-v2', transform=transform)
    
    if max_samples:
        indices = torch.randperm(len(base_dataset))[:max_samples].tolist()
        base_dataset = torch.utils.data.Subset(base_dataset, indices)
    
    results = {
        'config': {
            'seed': seed,
            'method': 'source',
            'model': 'vit_base_patch16_224',
            'severity': severity,
            'max_samples': max_samples
        }
    }
    
    # Evaluate on clean ImageNet-V2
    print("\nEvaluating on clean ImageNet-V2...")
    loader = DataLoader(base_dataset, batch_size=batch_size, 
                       shuffle=False, num_workers=num_workers, pin_memory=True)
    clean_metrics = evaluate_model(model, loader, device, adapt_method=None)
    results['clean'] = clean_metrics
    print(f"Clean Accuracy: {clean_metrics['accuracy']:.2f}%")
    
    # Evaluate on each corruption
    all_accs = []
    for corruption in CORRUPTIONS:
        print(f"\nEvaluating on {corruption}...")
        corrupted_dataset = CorruptedDataset(base_dataset, corruption, severity)
        loader = DataLoader(corrupted_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
        
        metrics = evaluate_model(model, loader, device, adapt_method=None)
        results[corruption] = metrics
        all_accs.append(metrics['accuracy'])
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    
    # Compute average
    results['average'] = {
        'accuracy': np.mean(all_accs),
        'std': np.std(all_accs),
        'min': np.min(all_accs),
        'max': np.max(all_accs)
    }
    
    print(f"\n{'='*60}")
    print(f"Average Corruption Accuracy: {results['average']['accuracy']:.2f}%")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    all_results = []
    for seed in [42, 123, 456]:
        results = run_source_baseline(seed=seed, device=device)
        all_results.append(results)
        save_results(results, f'/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/exp/source_baseline/results_seed_{seed}.json')
    
    # Aggregate
    aggregated = {
        'method': 'source',
        'seeds': [42, 123, 456],
        'accuracy_mean': np.mean([r['average']['accuracy'] for r in all_results]),
        'accuracy_std': np.std([r['average']['accuracy'] for r in all_results]),
        'all_results': all_results
    }
    save_results(aggregated, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/computer_vision/idea_01/results/source_baseline.json')
