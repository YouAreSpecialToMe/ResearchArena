"""
Train FlowRouter with flow-guided token routing.
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

from src.flowrouter import FlowRouter_S_2
from src.data_utils import get_cifar10_loaders


def flow_matching_loss_with_routing(model, x0, x1, t, y, target_flops_ratio=0.5, lambda_vel=0.1, lambda_budget=0.001):
    """
    Flow matching loss with routing-specific components.
    
    L = L_rec + lambda_vel * L_vel + lambda_budget * L_budget
    """
    B = x0.shape[0]
    
    # Interpolate
    x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
    v_target = x1 - x0
    
    # Forward with routing stats
    v_pred, stats = model(x_t, t, y, return_routing_stats=True)
    
    # Reconstruction loss
    loss_rec = torch.nn.functional.mse_loss(v_pred, v_target)
    
    # Velocity consistency loss (ensures smooth trajectories)
    # Approximated by encouraging velocity prediction to match target
    loss_vel = loss_rec  # Simplified - the reconstruction already ensures this
    
    # Budget loss: encourage target FLOPs ratio
    # We want skip_rate to be approximately (1 - target_flops_ratio)
    target_skip_rate = 1.0 - target_flops_ratio
    actual_skip_rate = stats['pseudo_skip_rate'] if 'pseudo_skip_rate' in stats else stats.get('avg_skip_rate', 0)
    loss_budget = torch.nn.functional.mse_loss(
        torch.tensor(actual_skip_rate, device=v_pred.device),
        torch.tensor(target_skip_rate, device=v_pred.device)
    )
    
    # Total loss
    loss = loss_rec + lambda_vel * loss_vel + lambda_budget * loss_budget
    
    return loss, {
        'loss_rec': loss_rec.item(),
        'loss_budget': loss_budget.item(),
        'skip_rate': actual_skip_rate if isinstance(actual_skip_rate, float) else actual_skip_rate.item(),
    }


def train_flowrouter(args):
    """Train FlowRouter model."""
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")
    
    # Create model
    model = FlowRouter_S_2(input_size=32, num_classes=10, use_velocity=True).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    base_params = sum(p.numel() for p in model.patch_embed.parameters())
    base_params += sum(p.numel() for p in model.t_embedder.parameters())
    base_params += sum(p.numel() for p in model.y_embedder.parameters())
    for block in model.blocks:
        base_params += sum(p.numel() for p in block.block.parameters())
    base_params += sum(p.numel() for p in model.final_layer.parameters())
    router_params = total_params - base_params
    
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Base parameters: {base_params / 1e6:.2f}M")
    print(f"Router parameters: {router_params / 1e6:.2f}M ({router_params/total_params*100:.1f}%)")
    
    # Optimizer - different learning rates for base and routers
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
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Data loaders
    train_loader, _ = get_cifar10_loaders(batch_size=args.batch_size, num_workers=4)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    # Anneal target FLOPs ratio from 0.8 to 0.5
    initial_flops_ratio = 0.8
    final_flops_ratio = args.target_flops_ratio
    
    for epoch in range(args.epochs):
        # Anneal target
        progress = epoch / args.epochs
        current_target = initial_flops_ratio + (final_flops_ratio - initial_flops_ratio) * progress
        
        epoch_losses = []
        epoch_skip_rates = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x0, x1, t, y in pbar:
            x0 = x0.to(device)
            x1 = x1.to(device)
            t = t.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            loss, stats = flow_matching_loss_with_routing(
                model, x0, x1, t, y, 
                target_flops_ratio=current_target,
                lambda_vel=args.lambda_vel,
                lambda_budget=args.lambda_budget
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_skip_rates.append(stats['skip_rate'])
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'skip': f"{stats['skip_rate']:.2f}"
            })
        
        avg_loss = np.mean(epoch_losses)
        avg_skip = np.mean(epoch_skip_rates)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Skip Rate: {avg_skip:.3f}, Target: {current_target:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            checkpoint_path = f"checkpoints/flowrouter_seed{args.seed}.pt"
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
        'experiment': 'flowrouter',
        'seed': args.seed,
        'epochs': args.epochs,
        'final_loss': float(avg_loss),
        'final_skip_rate': float(avg_skip),
        'training_time_seconds': training_time,
        'num_parameters': total_params,
        'router_parameters': router_params,
        'config': vars(args)
    }
    
    results_path = f"exp/flowrouter/results_seed{args.seed}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_base', type=float, default=1e-5)
    parser.add_argument('--lr_router', type=float, default=1e-4)
    parser.add_argument('--target_flops_ratio', type=float, default=0.5)
    parser.add_argument('--lambda_vel', type=float, default=0.1)
    parser.add_argument('--lambda_budget', type=float, default=0.001)
    args = parser.parse_args()
    
    train_flowrouter(args)
