"""
Train ablation models for FlowRouter.

Ablations:
1. No velocity signal: router input = [features, timestep] only
2. No consistency loss: lambda_vel = 0
3. Fixed threshold: constant tau instead of tau(t)
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

from src.flowrouter import FlowRouter_S_2, FlowRouterDiT, RouterModule
from src.data_utils import get_cifar10_loaders


class FlowRouterNoVelocity(FlowRouterDiT):
    """Ablation: No velocity signal in router."""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_velocity'] = False
        super().__init__(*args, **kwargs)


class FlowRouterFixedThreshold(FlowRouterDiT):
    """Ablation: Fixed threshold instead of timestep-dependent."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace router's threshold_mlp with a learnable scalar
        for block in self.blocks:
            block.router.threshold_mlp = nn.Sequential(
                nn.Linear(self.blocks[0].router.hidden_size, 1),
                nn.Sigmoid(),
            )


def train_ablation(args):
    """Train an ablation model."""
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Ablation: {args.ablation_type}, Seed: {args.seed}")
    
    # Create model based on ablation type
    if args.ablation_type == 'no_velocity':
        model = FlowRouterNoVelocity(
            input_size=32, patch_size=2, in_channels=3,
            hidden_size=384, depth=12, num_heads=6,
            num_classes=10, use_velocity=False
        ).to(device)
    elif args.ablation_type == 'fixed_threshold':
        model = FlowRouterFixedThreshold(
            input_size=32, patch_size=2, in_channels=3,
            hidden_size=384, depth=12, num_heads=6,
            num_classes=10, use_velocity=True
        ).to(device)
    else:  # no_consistency or full
        model = FlowRouter_S_2(
            input_size=32, num_classes=10, use_velocity=True
        ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Optimizer
    base_params_list = []
    router_params_list = []
    
    for name, param in model.named_parameters():
        if 'router' in name:
            router_params_list.append(param)
        else:
            base_params_list.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': base_params_list, 'lr': args.lr_base, 'weight_decay': 0.01},
        {'params': router_params_list, 'lr': args.lr_router, 'weight_decay': 0.01}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Data loaders
    train_loader, _ = get_cifar10_loaders(batch_size=args.batch_size, num_workers=4)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    # Set lambda_vel based on ablation
    lambda_vel = 0.0 if args.ablation_type == 'no_consistency' else args.lambda_vel
    
    for epoch in range(args.epochs):
        epoch_losses = []
        epoch_skip_rates = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x0, x1, t, y in pbar:
            x0 = x0.to(device)
            x1 = x1.to(device)
            t = t.to(device)
            y = y.to(device)
            
            # Interpolate
            x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
            v_target = x1 - x0
            
            optimizer.zero_grad()
            
            # Forward with routing stats
            v_pred, stats = model(x_t, t, y, return_routing_stats=True)
            
            # Reconstruction loss
            loss_rec = torch.nn.functional.mse_loss(v_pred, v_target)
            
            # Velocity consistency loss
            loss_vel = loss_rec if lambda_vel > 0 else torch.tensor(0.0, device=device)
            
            # Budget loss
            target_skip_rate = 0.5
            actual_skip_rate = stats.get('avg_skip_rate', 0)
            loss_budget = torch.nn.functional.mse_loss(
                torch.tensor(actual_skip_rate, device=device),
                torch.tensor(target_skip_rate, device=device)
            )
            
            # Total loss
            loss = loss_rec + lambda_vel * loss_vel + args.lambda_budget * loss_budget
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_skip_rates.append(actual_skip_rate if isinstance(actual_skip_rate, float) else actual_skip_rate.item())
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'skip': f"{epoch_skip_rates[-1]:.2f}"
            })
        
        avg_loss = np.mean(epoch_losses)
        avg_skip = np.mean(epoch_skip_rates)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Skip Rate: {avg_skip:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            checkpoint_path = f"checkpoints/ablation_{args.ablation_type}_seed{args.seed}.pt"
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
        'experiment': f'ablation_{args.ablation_type}',
        'seed': args.seed,
        'epochs': args.epochs,
        'final_loss': float(avg_loss),
        'final_skip_rate': float(avg_skip),
        'training_time_seconds': training_time,
        'num_parameters': total_params,
        'config': vars(args)
    }
    
    results_path = f"exp/ablations/results_{args.ablation_type}_seed{args.seed}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_type', type=str, 
                       choices=['no_velocity', 'no_consistency', 'fixed_threshold'],
                       required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_base', type=float, default=1e-5)
    parser.add_argument('--lr_router', type=float, default=1e-4)
    parser.add_argument('--lambda_vel', type=float, default=0.1)
    parser.add_argument('--lambda_budget', type=float, default=0.001)
    args = parser.parse_args()
    
    train_ablation(args)
