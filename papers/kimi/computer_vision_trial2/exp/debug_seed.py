"""
Debug random seed handling - understand why results are identical.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import random

from shared.models import load_pretrained_cifar_model, enable_adaptation, collect_params, resnet32
from shared.data_loader import load_cifar_c


def test_randomness_sources():
    """Test different sources of randomness"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test 1: Data loading order
    print("\n" + "="*60)
    print("Test 1: Data Loading Order")
    print("="*60)
    
    for seed in [2022, 2023]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        loader, _ = load_cifar_c('./data', 'cifar10', 'gaussian_noise', severity=5, batch_size=50)
        
        # Get first batch
        for images, labels in loader:
            print(f"Seed {seed}: first batch sum = {images.sum().item():.4f}")
            break
    
    # Test 2: Model initialization
    print("\n" + "="*60)
    print("Test 2: Model Weight Randomness")
    print("="*60)
    
    for seed in [2022, 2023]:
        torch.manual_seed(seed)
        model = resnet32(num_classes=10).to(device)
        
        # Get first conv weight sum
        first_weight = model.conv1.weight
        print(f"Seed {seed}: first conv weight sum = {first_weight.sum().item():.4f}")
        del model
    
    # Test 3: Pretrained model (should be identical)
    print("\n" + "="*60)
    print("Test 3: Pretrained Model (should be identical)")
    print("="*60)
    
    for seed in [2022, 2023]:
        torch.manual_seed(seed)
        model = load_pretrained_cifar_model(model_name='resnet32', dataset='cifar10', device=device)
        
        first_weight = model.conv1.weight
        print(f"Seed {seed}: pretrained first conv weight sum = {first_weight.sum().item():.4f}")
        del model
    
    # Test 4: Optimization with random init vs pretrained
    print("\n" + "="*60)
    print("Test 4: Optimization Results")
    print("="*60)
    
    def quick_eval(use_random_init, seed):
        torch.manual_seed(seed)
        
        if use_random_init:
            model = resnet32(num_classes=10).to(device)
        else:
            model = load_pretrained_cifar_model(model_name='resnet32', dataset='cifar10', device=device)
        
        model = enable_adaptation(model, adapt_bn=True)
        params, _ = collect_params(model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        
        loader, _ = load_cifar_c('./data', 'cifar10', 'gaussian_noise', severity=5, batch_size=50)
        
        total_loss = 0
        n = 0
        for i, (images, labels) in enumerate(loader):
            if i >= 5:
                break
            images = images.to(device)
            
            outputs = model(images)
            p = F.softmax(outputs, dim=1)
            entropy = -(p * torch.log(p + 1e-10)).sum(dim=1).mean()
            
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()
            
            total_loss += entropy.item()
            n += 1
        
        avg_loss = total_loss / n
        del model, optimizer
        torch.cuda.empty_cache()
        
        return avg_loss
    
    print("\nRandom init model:")
    for seed in [2022, 2023]:
        loss = quick_eval(use_random_init=True, seed=seed)
        print(f"  Seed {seed}: loss = {loss:.4f}")
    
    print("\nPretrained model:")
    for seed in [2022, 2023]:
        loss = quick_eval(use_random_init=False, seed=seed)
        print(f"  Seed {seed}: loss = {loss:.4f}")
    
    # Test 5: Different data order via manual shuffling
    print("\n" + "="*60)
    print("Test 5: Effect of Data Shuffling")
    print("="*60)
    
    loader1, dataset1 = load_cifar_c('./data', 'cifar10', 'gaussian_noise', severity=5, batch_size=1000)
    loader2, dataset2 = load_cifar_c('./data', 'cifar10', 'gaussian_noise', severity=5, batch_size=1000)
    
    # Get first batch from each (should be identical since no shuffle)
    for (img1, _), (img2, _) in zip(loader1, loader2):
        print(f"No shuffle: batch1 sum = {img1.sum().item():.4f}, batch2 sum = {img2.sum().item():.4f}")
        break
    
    # Now manually shuffle with different seeds
    torch.manual_seed(2022)
    indices1 = torch.randperm(len(dataset1))
    torch.manual_seed(2023)
    indices2 = torch.randperm(len(dataset1))
    
    print(f"Shuffled indices (seed 2022): first 5 = {indices1[:5].tolist()}")
    print(f"Shuffled indices (seed 2023): first 5 = {indices2[:5].tolist()}")


if __name__ == '__main__':
    test_randomness_sources()
