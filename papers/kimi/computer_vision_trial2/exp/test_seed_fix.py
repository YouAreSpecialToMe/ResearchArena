"""
Quick test to verify random seed handling produces different results.
This addresses the critical bug where all seeds produced identical results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np

from shared.models import load_pretrained_cifar_model, enable_adaptation, collect_params
from shared.data_loader import load_cifar_c
from shared.utils import set_seed


def quick_tent_test(seed=2022, corruption='gaussian_noise', device='cuda'):
    """Quick TENT test on a single corruption with limited samples"""
    set_seed(seed)
    
    # Load FRESH model for this seed
    model = load_pretrained_cifar_model(model_name='resnet32', dataset='cifar10', device=device)
    model = enable_adaptation(model, adapt_bn=True)
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    loader, _ = load_cifar_c('./data', 'cifar10', corruption, severity=5, batch_size=50)
    
    correct = 0
    total = 0
    loss_sum = 0
    n_batches = 0
    
    # Only process first 20 batches for quick test
    for i, (images, labels) in enumerate(loader):
        if i >= 20:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        p = F.softmax(outputs, dim=1)
        entropy = -(p * torch.log(p + 1e-10)).sum(dim=1).mean()
        
        optimizer.zero_grad()
        entropy.backward()
        optimizer.step()
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            loss_sum += entropy.item()
            n_batches += 1
    
    acc = 100.0 * correct / total
    avg_loss = loss_sum / n_batches if n_batches > 0 else 0
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return {'accuracy': acc, 'loss': avg_loss}


def test_seed_randomness():
    """Test that different seeds produce different results"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("\n" + "="*60)
    print("Testing Random Seed Fix")
    print("="*60)
    
    seeds = [2022, 2023, 2024]
    results = []
    
    for seed in seeds:
        print(f"\nTesting seed {seed}...")
        result = quick_tent_test(seed=seed, device=device)
        results.append(result)
        print(f"  Seed {seed}: acc={result['accuracy']:.2f}%, loss={result['loss']:.4f}")
    
    # Check if results differ
    accs = [r['accuracy'] for r in results]
    losses = [r['loss'] for r in results]
    
    acc_std = np.std(accs)
    loss_std = np.std(losses)
    
    print("\n" + "="*60)
    print("Randomness Verification Results")
    print("="*60)
    print(f"Accuracies: {accs}")
    print(f"Accuracy std: {acc_std:.4f}")
    print(f"Loss std: {loss_std:.4f}")
    
    if acc_std > 0.01 or loss_std > 0.0001:
        print("\n✓ PASS: Results vary across seeds (randomness is working)")
        return True
    else:
        print("\n✗ FAIL: Results are identical across seeds (bug still present)")
        return False


if __name__ == '__main__':
    success = test_seed_randomness()
    exit(0 if success else 1)
