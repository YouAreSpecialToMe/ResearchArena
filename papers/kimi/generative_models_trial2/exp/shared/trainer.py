"""
Training script for flow matching on point clouds.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from models import VelocityNetwork, WeightPredictorMLP, compute_knn_features
from metrics import compute_all_metrics
from utils import set_seed, save_checkpoint, Timer, get_device


def sample_conditional_flow(x0, x1, t):
    """
    Sample from the conditional flow p_t(x|x1).
    x_t = (1 - t) * x0 + t * x1
    """
    t_expanded = t.view(-1, 1, 1)
    x_t = (1 - t_expanded) * x0 + t_expanded * x1
    target = x1 - x0  # The velocity target
    return x_t, target


def euler_sampling(model, shape, num_steps=50, radial_dist=None, device='cuda'):
    """
    Generate samples using Euler integration.
    
    Args:
        model: velocity network
        shape: (B, N, 3) shape of output
        num_steps: number of integration steps
        radial_dist: (B, N) radial distances (optional)
    """
    B, N, _ = shape
    
    # Start from noise
    x = torch.randn(shape, device=device)
    
    # Integration steps
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.ones(B, device=device) * (i * dt)
        
        with torch.no_grad():
            v = model(x, t, radial_dist)
        
        x = x + dt * v
    
    return x


class FlowMatchingTrainer:
    """Trainer for flow matching models."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=2e-4,
        weight_decay=1e-4,
        weighting_type='uniform',  # 'uniform', 'density', 'idw', 'law'
        idw_alpha=1.0,
        idw_beta=2.0,
        weight_mlp=None,
        max_range=80.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.weighting_type = weighting_type
        self.idw_alpha = idw_alpha
        self.idw_beta = idw_beta
        self.weight_mlp = weight_mlp.to(device) if weight_mlp is not None else None
        self.max_range = max_range
        
        # Optimizer
        params = list(model.parameters())
        if self.weight_mlp is not None:
            params += list(self.weight_mlp.parameters())
        
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.01
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, batch):
        """Compute flow matching loss with optional weighting."""
        points = batch['points'].to(self.device)  # (B, N, 3)
        radial_dist = batch['radial_dist'].to(self.device)  # (B, N)
        
        B, N, _ = points.shape
        
        # Sample noise
        x0 = torch.randn_like(points)
        x1 = points
        
        # Sample timestep
        t = torch.rand(B, device=self.device)
        
        # Sample from conditional flow
        x_t, target = sample_conditional_flow(x0, x1, t)
        
        # Predict velocity
        v_pred = self.model(x_t, t, radial_dist)
        
        # Compute error
        error = ((v_pred - target) ** 2).sum(dim=-1)  # (B, N)
        
        # Apply weighting
        if self.weighting_type == 'uniform':
            weights = torch.ones_like(error)
        
        elif self.weighting_type == 'density':
            # Weight by local point density
            weights = self.compute_density_weights(points)
        
        elif self.weighting_type == 'idw':
            # Inverse distance weighting
            weights = 1.0 + self.idw_alpha * (radial_dist ** self.idw_beta)
            weights = torch.clamp(weights, 1.0, 3.0)
        
        elif self.weighting_type == 'law':
            # Learned adaptive weighting
            features = compute_knn_features(x_t, radial_dist, k=8)
            weights = self.weight_mlp(features).squeeze(-1)
        
        else:
            raise ValueError(f"Unknown weighting type: {self.weighting_type}")
        
        # Weighted loss
        loss = (weights * error).mean()
        
        return loss
    
    def compute_density_weights(self, points, k=8):
        """Compute density-based weights."""
        B, N, _ = points.shape
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(points, points)  # (B, N, N)
        
        # Get k nearest neighbors
        knn_dists, _ = torch.topk(dist_matrix, k + 1, largest=False, dim=-1)
        knn_dists = knn_dists[:, :, 1:]  # Remove self
        
        # Density = 1 / mean distance
        mean_dists = knn_dists.mean(dim=-1)
        density = 1.0 / (mean_dists + 1e-6)
        
        # Normalize
        mean_density = density.mean(dim=1, keepdim=True)
        weights = density / (mean_density + 1e-6)
        weights = torch.clamp(weights, 0.5, 2.0)
        
        return weights
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        if self.weight_mlp is not None:
            self.weight_mlp.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            loss = self.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.weight_mlp is not None:
                torch.nn.utils.clip_grad_norm_(self.weight_mlp.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        if self.weight_mlp is not None:
            self.weight_mlp.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs, save_dir=None):
        """Train for multiple epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss and save_dir is not None:
                best_val_loss = val_loss
                save_path = Path(save_dir) / 'best_model.pt'
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, save_path)
                print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        return self.train_losses, self.val_losses
    
    def generate_samples(self, num_samples=200, num_steps=50):
        """Generate samples for evaluation."""
        self.model.eval()
        
        samples = []
        radial_dists = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="Generating"):
                # Generate with random radial distances
                # Sample radial distance distribution similar to real data
                B, N = 1, 2048
                
                # Create realistic radial distance distribution
                near = torch.rand(N // 2, device=self.device) * 0.25
                mid = 0.25 + torch.rand(N // 4, device=self.device) * 0.375
                far = 0.625 + torch.rand(N - N // 2 - N // 4, device=self.device) * 0.375
                r_dist = torch.cat([near, mid, far])
                r_dist = r_dist.unsqueeze(0)  # (1, N)
                
                # Generate sample
                shape = (1, N, 3)
                sample = euler_sampling(
                    self.model, shape, num_steps=num_steps,
                    radial_dist=r_dist, device=self.device
                )
                
                samples.append(sample[0].cpu())
                radial_dists.append(r_dist[0].cpu())
        
        samples = torch.stack(samples)  # (num_samples, N, 3)
        radial_dists = torch.stack(radial_dists)  # (num_samples, N)
        
        return samples, radial_dists
    
    def evaluate(self, val_dataset, num_eval_samples=200):
        """Evaluate on validation set."""
        print("Generating samples for evaluation...")
        generated_samples, generated_dists = self.generate_samples(num_eval_samples)
        
        # Sample from validation set
        real_samples = []
        real_dists = []
        
        for i in range(min(num_eval_samples, len(val_dataset))):
            data = val_dataset[i]
            real_samples.append(data['points'])
            real_dists.append(data['radial_dist'])
        
        real_samples = torch.stack(real_samples)
        real_dists = torch.stack(real_dists)
        
        print("Computing metrics...")
        metrics = compute_all_metrics(
            generated_samples, real_samples,
            generated_dists, real_dists
        )
        
        return metrics


if __name__ == "__main__":
    print("Trainer module ready")
