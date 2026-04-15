"""Fast training script for FlowRouter with reduced epochs."""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.flowrouter import FlowRouterDiT
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_cifar10_loaders(batch_size=128, num_workers=4):
    """Get CIFAR-10 train and test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, 
                                      download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False,
                                     download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, 
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    
    return train_loader, test_loader


def flow_matching_loss(model, x1, y, device, lambda_vel=0.1, lambda_budget=0.001, 
                       target_flops_ratio=0.5, use_velocity=True):
    """Flow matching training objective with velocity consistency and budget constraints."""
    B = x1.shape[0]
    
    # Sample timestep uniformly
    t = torch.rand(B, device=device)
    
    # Sample noise
    x0 = torch.randn_like(x1)
    
    # Linear interpolation (rectified flow)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    
    # Target velocity
    v_target = x1 - x0
    
    # Forward pass with routing stats
    v_pred, routing_stats = model(x_t, t, y, return_routing_stats=True)
    
    # 1. Reconstruction loss (standard flow matching)
    loss_rec = torch.mean((v_pred - v_target) ** 2)
    
    # 2. Velocity consistency loss
    loss_vel = torch.mean(v_pred ** 2) * 0.01
    
    # 3. Computation budget loss
    if 'pseudo_skip_rate' in routing_stats:
        skip_rate = routing_stats['pseudo_skip_rate']
    else:
        skip_rate = routing_stats['avg_skip_rate']
    flops_ratio = 1.0 - skip_rate
    target_ratio = target_flops_ratio
    
    loss_budget = torch.abs(flops_ratio - target_ratio)
    
    # Total loss
    loss = loss_rec + lambda_vel * loss_vel + lambda_budget * loss_budget
    
    return loss, {
        'loss': loss.item(),
        'loss_rec': loss_rec.item(),
        'loss_vel': loss_vel.item(),
        'loss_budget': loss_budget.item() if isinstance(loss_budget, torch.Tensor) else loss_budget,
        'skip_rate': routing_stats['avg_skip_rate'],
        'flops_ratio': 1.0 - routing_stats['avg_skip_rate'],
    }


def train_epoch(model, train_loader, optimizer, device, epoch, target_flops_ratio=0.5, 
                use_velocity=True, freeze_base=False):
    """Train one epoch."""
    model.train()
    
    if freeze_base:
        # Freeze base DiT parameters, only train routers
        for name, param in model.named_parameters():
            if 'router' not in name and 'threshold' not in name:
                param.requires_grad = False
    else:
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
    
    total_loss = 0
    total_rec = 0
    total_skip = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for x1, y in pbar:
        x1, y = x1.to(device), y.to(device)
        
        optimizer.zero_grad()
        loss, stats = flow_matching_loss(
            model, x1, y, device, 
            target_flops_ratio=target_flops_ratio,
            use_velocity=use_velocity
        )
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += stats['loss']
        total_rec += stats['loss_rec']
        total_skip += stats['skip_rate']
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{stats['loss']:.4f}",
            'rec': f"{stats['loss_rec']:.4f}",
            'skip': f"{stats['skip_rate']:.2%}",
        })
    
    return {
        'loss': total_loss / num_batches,
        'loss_rec': total_rec / num_batches,
        'skip_rate': total_skip / num_batches,
    }


def train(seed=42, epochs_warmup=2, epochs_joint=8, device='cuda', 
          lr_router=1e-3, lr_base=1e-5, target_flops=0.5):
    """Train FlowRouter with reduced epochs for faster completion."""
    set_seed(seed)
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    model = FlowRouterDiT(
        input_size=32, 
        num_classes=10, 
        use_velocity=True
    ).to(device)
    
    # Load baseline checkpoint if available
    baseline_path = f'checkpoints/dit_baseline_seed{seed}.pt'
    if os.path.exists(baseline_path):
        print(f"Loading baseline checkpoint from {baseline_path}")
        checkpoint = torch.load(baseline_path, map_location=device)
        # Load only base model parameters
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        # Filter out router parameters
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'router' not in k and 'threshold' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} parameters from baseline")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # Separate router and base parameters
    router_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'router' in name or 'threshold' in name:
            router_params.append(param)
        else:
            base_params.append(param)
    
    print(f"Router parameters: {sum(p.numel() for p in router_params) / 1e6:.2f}M")
    print(f"Base parameters: {sum(p.numel() for p in base_params) / 1e6:.2f}M")
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    total_epochs = epochs_warmup + epochs_joint
    
    for epoch in range(1, total_epochs + 1):
        # Router warmup phase
        if epoch <= epochs_warmup:
            optimizer = torch.optim.AdamW(router_params, lr=lr_router, weight_decay=0.01)
            freeze_base = True
            current_target_flops = 0.8  # Start with higher FLOPs target during warmup
            print(f"\n=== Router Warmup Epoch {epoch}/{epochs_warmup} ===")
        else:
            # Joint training
            if epoch == epochs_warmup + 1:
                optimizer = torch.optim.AdamW([
                    {'params': router_params, 'lr': lr_router * 0.5},
                    {'params': base_params, 'lr': lr_base}
                ], weight_decay=0.01)
            freeze_base = False
            # Anneal target FLOPs from 0.8 to target
            progress = (epoch - epochs_warmup) / max(epochs_joint, 1)
            current_target_flops = 0.8 - (0.8 - target_flops) * progress
            print(f"\n=== Joint Training Epoch {epoch}/{total_epochs} (target FLOPs: {current_target_flops:.2%}) ===")
        
        stats = train_epoch(
            model, train_loader, optimizer, device, epoch,
            target_flops_ratio=current_target_flops,
            use_velocity=True,
            freeze_base=freeze_base
        )
        
        print(f"Epoch {epoch}/{total_epochs}, Loss: {stats['loss']:.6f}, "
              f"Rec Loss: {stats['loss_rec']:.6f}, Skip Rate: {stats['skip_rate']:.2%}")
        
        # Save best model
        if stats['loss'] < best_loss:
            best_loss = stats['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': stats['loss'],
                'seed': seed,
                'target_flops': target_flops,
                'use_velocity': True,
            }
            save_path = f'checkpoints/flowrouter_seed{seed}.pt'
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.2f} minutes")
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs_warmup', type=int, default=2)
    parser.add_argument('--epochs_joint', type=int, default=8)
    args = parser.parse_args()
    
    train(seed=args.seed, device=args.device, 
          epochs_warmup=args.epochs_warmup, epochs_joint=args.epochs_joint)
