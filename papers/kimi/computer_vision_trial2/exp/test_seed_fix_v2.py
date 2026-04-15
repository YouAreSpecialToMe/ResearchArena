"""
Test seed fix with data shuffling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np

from fixed_eval import get_shuffled_loader
from shared.models import load_pretrained_cifar_model, enable_adaptation, collect_params
from shared.utils import set_seed


def quick_test(seed=2022, device='cuda'):
    """Quick test with shuffled data"""
    set_seed(seed)
    
    model = load_pretrained_cifar_model(model_name='resnet32', dataset='cifar10', device=device)
    model = enable_adaptation(model, adapt_bn=True)
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    loader, _ = get_shuffled_loader('cifar10', 'gaussian_noise', 5, seed, './data', batch_size=50)
    
    correct = 0
    total = 0
    loss_sum = 0
    n = 0
    
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
            n += 1
    
    acc = 100.0 * correct / total
    avg_loss = loss_sum / n if n > 0 else 0
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return {'accuracy': acc, 'loss': avg_loss}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("\n" + "="*60)
    print("Testing Seed Fix with Data Shuffling")
    print("="*60)
    
    seeds = [2022, 2023, 2024]
    results = []
    
    for seed in seeds:
        print(f"\nTesting seed {seed}...")
        result = quick_test(seed=seed, device=device)
        results.append(result)
        print(f"  Seed {seed}: acc={result['accuracy']:.2f}%, loss={result['loss']:.4f}")
    
    accs = [r['accuracy'] for r in results]
    losses = [r['loss'] for r in results]
    
    acc_std = np.std(accs)
    loss_std = np.std(losses)
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Accuracies: {accs}")
    print(f"Accuracy std: {acc_std:.4f}")
    print(f"Loss std: {loss_std:.4f}")
    
    if acc_std > 0.01 or loss_std > 0.0001:
        print("\n✓ PASS: Results vary across seeds!")
        return True
    else:
        print("\n✗ FAIL: Results still identical")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
