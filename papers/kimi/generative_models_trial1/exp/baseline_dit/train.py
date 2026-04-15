"""
Train baseline DiT-S/2 on CIFAR-10 for Flow Matching.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from tqdm import tqdm
import argparse

from src.dit import DiT_S_2
from src.data_utils import get_cifar10_loaders, flow_matching_loss


def train_baseline(args):
    """Train baseline DiT model."""
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")
    
    # Create model
    model = DiT_S_2(input_size=32, num_classes=10).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Data loaders
    train_loader, _ = get_cifar10_loaders(batch_size=args.batch_size, num_workers=4)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    losses = []
    for epoch in range(args.epochs):
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x0, x1, t, y in pbar:
            x0 = x0.to(device)
            x1 = x1.to(device)
            t = t.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            loss = flow_matching_loss(model, x0, x1, t, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            checkpoint_path = f"checkpoints/dit_baseline_seed{args.seed}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")
    
    # Save results
    results = {
        'experiment': 'baseline_dit',
        'seed': args.seed,
        'epochs': args.epochs,
        'final_loss': float(avg_loss),
        'training_time_seconds': training_time,
        'num_parameters': total_params,
        'config': vars(args)
    }
    
    results_path = f"exp/baseline_dit/results_seed{args.seed}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    train_baseline(args)
