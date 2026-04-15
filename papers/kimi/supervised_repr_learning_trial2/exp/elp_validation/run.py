"""
ELP Validation Experiment: Verify that ELP distinguishes clean from noisy samples.
This is the first experiment to run for preliminary evidence.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score

from models import SupConResNet
from data_loader import get_dataloader
from losses import SupConLoss
from laser_scl import ELPTracker
from utils import set_seed, save_results


def run_elp_validation(dataset='cifar10', noise_rate=0.4, window_size=10, 
                       epochs=100, seed=42, device='cuda'):
    """Run ELP validation experiment."""
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    num_classes = 10 if dataset == 'cifar10' else 100
    
    # Get data
    train_loader, train_dataset = get_dataloader(
        dataset, train=True, batch_size=256, num_workers=4,
        noise_rate=noise_rate, noise_type='symmetric', seed=seed
    )
    
    # Get noise mask
    noise_mask = train_dataset.get_noise_mask()
    clean_mask = ~noise_mask
    
    # Initialize model
    model = SupConResNet(num_classes=num_classes, name='resnet18',
                         projection_dim=128, hidden_dim=512).to(device)
    criterion = SupConLoss(temperature=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initialize ELP tracker
    n_samples = len(train_dataset)
    elp_tracker = ELPTracker(n_samples, window_size=window_size)
    
    # Track loss history for all samples
    sample_losses = {i: [] for i in range(n_samples)}
    
    print(f"Running ELP validation on {dataset} with {noise_rate*100}% noise...")
    print(f"Clean samples: {clean_mask.sum()}, Noisy samples: {noise_mask.sum()}")
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            (view1, view2), labels, clean_labels, indices = batch
            
            # Concatenate two views along batch dimension
            images = torch.cat([view1, view2], dim=0).to(device)
            labels = labels.to(device)
            indices = indices.to(device)
            
            bsz = labels.shape[0]
            
            # Forward
            projections = model(images)
            f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            # Compute per-sample losses
            per_sample_loss = []
            for i in range(bsz):
                feat_i = features[i:i+1]
                label_i = labels[i:i+1]
                loss_i = criterion(feat_i.repeat(1, 2, 1), label_i.repeat(1))
                per_sample_loss.append(loss_i.item())
                sample_idx = indices[i].item()
                sample_losses[sample_idx].append(loss_i.item())
            
            # Update ELP tracker
            per_sample_loss_tensor = torch.tensor(per_sample_loss, device=device)
            elp_tracker.update(indices, per_sample_loss_tensor)
            
            # Standard training
            loss = criterion(features, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}: Avg Loss={np.mean(epoch_losses):.4f}')
    
    # Compute final ELP values for all samples
    all_indices = torch.arange(n_samples)
    final_elp = elp_tracker.compute_elp(all_indices)
    final_mean_loss = elp_tracker.get_mean_loss(all_indices)
    
    # Compute AUC-ROC for ELP-based discrimination
    # Clean samples should have higher ELP (positive learning trend)
    elp_scores = final_elp
    elp_auc = roc_auc_score(clean_mask.astype(int), elp_scores)
    
    # Compute AUC-ROC for loss-based discrimination (lower loss = more likely clean)
    loss_scores = -final_mean_loss  # Negative because lower loss = more likely clean
    loss_auc = roc_auc_score(clean_mask.astype(int), loss_scores)
    
    # Statistical test
    clean_elp = final_elp[clean_mask]
    noisy_elp = final_elp[noise_mask]
    
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(clean_elp, noisy_elp)
    
    # Analyze trajectories
    clean_trajectories = [sample_losses[i] for i in range(n_samples) if clean_mask[i]]
    noisy_trajectories = [sample_losses[i] for i in range(n_samples) if noise_mask[i]]
    
    # Compute average trajectory for visualization
    max_len = max(len(t) for t in clean_trajectories)
    clean_avg_traj = []
    noisy_avg_traj = []
    
    for step in range(max_len):
        clean_vals = [t[step] for t in clean_trajectories if len(t) > step]
        noisy_vals = [t[step] for t in noisy_trajectories if len(t) > step]
        clean_avg_traj.append(np.mean(clean_vals) if clean_vals else 0)
        noisy_avg_traj.append(np.mean(noisy_vals) if noisy_vals else 0)
    
    results = {
        'elp_auc': float(elp_auc),
        'loss_auc': float(loss_auc),
        'elp_vs_loss_delta': float(elp_auc - loss_auc),
        'clean_elp_mean': float(clean_elp.mean()),
        'clean_elp_std': float(clean_elp.std()),
        'noisy_elp_mean': float(noisy_elp.mean()),
        'noisy_elp_std': float(noisy_elp.std()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'clean_avg_trajectory': clean_avg_traj,
        'noisy_avg_trajectory': noisy_avg_traj,
        'dataset': dataset,
        'noise_rate': noise_rate,
        'seed': seed
    }
    
    print(f"\n=== ELP Validation Results ===")
    print(f"ELP AUC-ROC: {elp_auc:.4f}")
    print(f"Loss AUC-ROC: {loss_auc:.4f}")
    print(f"ELP advantage: {elp_auc - loss_auc:.4f}")
    print(f"Clean ELP: {clean_elp.mean():.4f} ± {clean_elp.std():.4f}")
    print(f"Noisy ELP: {noisy_elp.mean():.4f} ± {noisy_elp.std():.4f}")
    print(f"T-test p-value: {p_value:.6f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--noise_rate', type=float, default=0.4)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    results = run_elp_validation(
        dataset=args.dataset,
        noise_rate=args.noise_rate,
        window_size=args.window_size,
        epochs=args.epochs,
        seed=args.seed
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'elp_validation_{args.dataset}_n{int(args.noise_rate*100)}.json')
    save_results(results, save_path)
    print(f"\nResults saved to {save_path}")
