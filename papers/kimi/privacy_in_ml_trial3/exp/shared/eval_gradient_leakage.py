"""
Gradient Leakage Attack (Deep Leakage from Gradients) evaluation.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models import create_simclr_model
from data_loader import get_cifar_transforms
from fcl_utils import InfoNCELoss, set_seed


def dlg_attack(model, original_inputs, original_labels, device, iterations=500, lr=0.1):
    """
    Deep Leakage from Gradients attack.
    Reconstructs input from gradients.
    """
    model.eval()
    
    # Get original gradients
    model.zero_grad()
    z = model(original_inputs)
    
    # For simplicity, use a dummy target
    criterion = InfoNCELoss(temperature=0.5)
    dummy_z2 = torch.randn_like(z)
    loss = criterion(z, dummy_z2)
    loss.backward()
    
    original_grads = [param.grad.clone() for param in model.parameters() if param.grad is not None]
    
    # Initialize dummy inputs
    dummy_inputs = torch.randn_like(original_inputs, requires_grad=True, device=device)
    
    optimizer = torch.optim.LBFGS([dummy_inputs], lr=lr, max_iter=1)
    
    for it in range(iterations):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            
            z_dummy = model(dummy_inputs)
            dummy_z2 = torch.randn_like(z_dummy)
            loss = criterion(z_dummy, dummy_z2)
            loss.backward()
            
            dummy_grads = [param.grad for param in model.parameters() if param.grad is not None]
            
            # Gradient matching loss
            grad_loss = 0
            for og, dg in zip(original_grads, dummy_grads):
                grad_loss += ((og - dg) ** 2).sum()
            
            grad_loss.backward()
            return grad_loss
        
        optimizer.step(closure)
        
        # Clamp to valid image range
        with torch.no_grad():
            dummy_inputs.clamp_(0, 1)
    
    return dummy_inputs.detach()


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim(img1, img2):
    """Compute simplified SSIM."""
    # Simple approximation
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.std()
    sigma2 = img2.std()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    return ssim.item()


def evaluate_gradient_leakage(model_path, dataset, data_dir, num_samples=30, device='cuda'):
    """Evaluate gradient leakage attack."""
    
    set_seed(42)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = create_simclr_model().to(device)
    
    # Handle different checkpoint formats
    if 'encoder' in checkpoint:
        state_dict = {}
        for k, v in checkpoint.items():
            if k == 'encoder':
                state_dict.update({f'encoder.{key}': val for key, val in v.items()})
            elif k == 'projection_head':
                state_dict.update({f'projection_head.{key}': val for key, val in v.items()})
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # Load test data
    from torchvision import datasets
    transform = get_cifar_transforms(train=False, contrastive=False)
    
    if dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform)
    else:
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, transform=transform)
    
    # Select random samples
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    
    psnr_scores = []
    ssim_scores = []
    
    for idx in tqdm(indices, desc="Running DLG attack"):
        img, label = test_dataset[idx]
        img = img.unsqueeze(0).to(device)
        
        # Run attack
        reconstructed = dlg_attack(model, img, label, device, iterations=300, lr=0.1)
        
        # Compute metrics
        psnr = compute_psnr(img, reconstructed)
        ssim = compute_ssim(img, reconstructed)
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
    
    return {
        'psnr_mean': float(np.mean(psnr_scores)),
        'psnr_std': float(np.std(psnr_scores)),
        'ssim_mean': float(np.mean(ssim_scores)),
        'ssim_std': float(np.std(ssim_scores)),
        'num_samples': num_samples
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = evaluate_gradient_leakage(
        args.model_path, args.dataset, args.data_dir, args.num_samples, device
    )
    
    results['model'] = args.model_path
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results: {results}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
