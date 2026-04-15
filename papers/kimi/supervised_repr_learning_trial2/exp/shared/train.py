"""
Unified training script for SupCon baselines and LASER-SCL.
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import SupConResNet
from data_loader import get_dataloader
from losses import SupConLoss, WeightedSupConLoss
from laser_scl import LASERSCL, LASERSCL_NoCurriculum, LASERSCL_NoELP, LASERSCL_Static
from utils import set_seed, save_checkpoint, save_results, train_linear_classifier, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Train SupCon or LASER-SCL')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument('--noise_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric'])
    
    # Model
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    
    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--lr_decay', type=str, default='cosine')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.1)
    
    # Method
    parser.add_argument('--method', type=str, default='supcon',
                        choices=['supcon', 'supcon_lr', 'supcon_il', 'laser_scl',
                                'ablation_no_curriculum', 'ablation_no_elp', 'ablation_static'])
    
    # LASER-SCL params
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--mu_min', type=float, default=0.3)
    parser.add_argument('--mu_max', type=float, default=0.7)
    parser.add_argument('--rho', type=float, default=2.0)
    parser.add_argument('--sigmoid_k', type=float, default=10.0)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def train_supcon(args, train_loader, test_loader, num_classes, device):
    """Train standard SupCon or with simple loss weighting."""
    set_seed(args.seed)
    
    model = SupConResNet(num_classes=num_classes, name=args.model,
                         projection_dim=args.projection_dim,
                         hidden_dim=args.hidden_dim).to(device)
    
    criterion = SupConLoss(temperature=args.temperature)
    
    # Determine weighting scheme
    weighting = 'none'
    if args.method == 'supcon_lr':
        weighting = 'loss_reweight'
    elif args.method == 'supcon_il':
        weighting = 'inverse_loss'
    
    if weighting != 'none':
        weighted_criterion = WeightedSupConLoss(temperature=args.temperature, weighting=weighting)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    
    if args.lr_decay == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 400], gamma=0.1)
    
    train_losses = []
    test_accuracies = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            (view1, view2), labels, clean_labels, indices = batch
            
            # Concatenate two views along batch dimension
            images = torch.cat([view1, view2], dim=0).to(device)
            labels = labels.to(device)
            
            bsz = labels.shape[0]
            
            # Forward pass
            projections = model(images)
            f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            # Compute loss with or without weighting
            if weighting == 'none':
                loss = criterion(features, labels)
                epoch_losses.append(loss.item())
            else:
                # First get per-sample losses
                with torch.no_grad():
                    temp_features = features.detach()
                    temp_labels = labels.detach()
                    
                    # Compute per-sample loss
                    per_sample_loss = []
                    for i in range(bsz):
                        feat_i = temp_features[i:i+1]
                        label_i = temp_labels[i:i+1]
                        loss_i = criterion(feat_i.repeat(1, 2, 1), label_i.repeat(1))
                        per_sample_loss.append(loss_i.item())
                    
                    per_sample_loss = torch.tensor(per_sample_loss, device=device)
                    weights = weighted_criterion.compute_weights(per_sample_loss, epoch, args.epochs)
                
                loss = weighted_criterion(features, labels, per_sample_loss, epoch, args.epochs)
                epoch_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # Linear evaluation every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            acc = train_linear_classifier(model, train_loader, test_loader, num_classes, 
                                         feat_dim=512, epochs=100, device=device)
            test_accuracies.append({'epoch': epoch + 1, 'accuracy': acc})
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={acc:.2f}%')
        else:
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}')
    
    elapsed = (time.time() - start_time) / 60
    
    # Final evaluation
    final_acc = train_linear_classifier(model, train_loader, test_loader, num_classes,
                                       feat_dim=512, epochs=100, device=device)
    
    return {
        'method': args.method,
        'dataset': args.dataset,
        'noise_rate': args.noise_rate,
        'seed': args.seed,
        'final_accuracy': final_acc,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'runtime_minutes': elapsed
    }


def train_laser_scl(args, train_loader, test_loader, num_classes, train_dataset, device):
    """Train LASER-SCL with ELP tracking."""
    set_seed(args.seed)
    
    model = SupConResNet(num_classes=num_classes, name=args.model,
                         projection_dim=args.projection_dim,
                         hidden_dim=args.hidden_dim).to(device)
    
    criterion = SupConLoss(temperature=args.temperature)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    
    if args.lr_decay == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 400], gamma=0.1)
    
    # Initialize LASER-SCL tracker
    n_samples = len(train_dataset)
    
    if args.method == 'laser_scl':
        laser = LASERSCL(n_samples, window_size=args.window_size, delta=args.delta,
                        mu_min=args.mu_min, mu_max=args.mu_max, rho=args.rho,
                        sigmoid_k=args.sigmoid_k, total_epochs=args.epochs)
    elif args.method == 'ablation_no_curriculum':
        laser = LASERSCL_NoCurriculum(n_samples, window_size=args.window_size, delta=args.delta,
                                     mu_min=args.mu_min, mu_max=args.mu_max, rho=args.rho,
                                     sigmoid_k=args.sigmoid_k, total_epochs=args.epochs)
    elif args.method == 'ablation_no_elp':
        laser = LASERSCL_NoELP(n_samples, window_size=args.window_size, delta=args.delta,
                              mu_min=args.mu_min, mu_max=args.mu_max, rho=args.rho,
                              sigmoid_k=args.sigmoid_k, total_epochs=args.epochs)
    elif args.method == 'ablation_static':
        laser = LASERSCL_Static(n_samples, window_size=args.window_size, delta=args.delta,
                               mu_min=args.mu_min, mu_max=args.mu_max, rho=args.rho,
                               sigmoid_k=args.sigmoid_k, total_epochs=args.epochs)
    
    train_losses = []
    test_accuracies = []
    weight_stats = []
    elp_stats = []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_weights = []
        epoch_elps = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            (view1, view2), labels, clean_labels, indices = batch
            
            # Concatenate two views along batch dimension
            images = torch.cat([view1, view2], dim=0).to(device)
            labels = labels.to(device)
            indices = indices.to(device)
            
            bsz = labels.shape[0]
            
            # Forward pass
            projections = model(images)
            f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            # Compute per-sample losses (first pass)
            per_sample_loss = []
            for i in range(bsz):
                feat_i = features[i:i+1]
                label_i = labels[i:i+1]
                loss_i = criterion(feat_i.repeat(1, 2, 1), label_i.repeat(1))
                per_sample_loss.append(loss_i)
            
            per_sample_loss = torch.stack(per_sample_loss)
            
            # Update ELP tracker
            laser.update_losses(indices, per_sample_loss.detach())
            
            # Compute weights
            if epoch >= 10:  # Warmup period
                weights, pred_loss, elp, mean_loss, mu = laser.compute_weights(indices, epoch)
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
                epoch_weights.extend(weights.tolist())
                epoch_elps.extend(elp.tolist())
            else:
                weights_tensor = torch.ones(bsz, device=device)
            
            # Compute weighted loss
            loss = criterion(features, labels, weights=weights_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if len(epoch_weights) > 0:
            weight_stats.append({
                'epoch': epoch + 1,
                'mean_weight': np.mean(epoch_weights),
                'std_weight': np.std(epoch_weights),
                'mean_elp': np.mean(epoch_elps)
            })
        
        # Linear evaluation every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            acc = train_linear_classifier(model, train_loader, test_loader, num_classes,
                                         feat_dim=512, epochs=100, device=device)
            test_accuracies.append({'epoch': epoch + 1, 'accuracy': acc})
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={acc:.2f}%, '
                  f'Avg Weight={np.mean(epoch_weights) if epoch_weights else 1.0:.3f}')
        else:
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}')
    
    elapsed = (time.time() - start_time) / 60
    
    # Final evaluation
    final_acc = train_linear_classifier(model, train_loader, test_loader, num_classes,
                                       feat_dim=512, epochs=100, device=device)
    
    return {
        'method': args.method,
        'dataset': args.dataset,
        'noise_rate': args.noise_rate,
        'seed': args.seed,
        'final_accuracy': final_acc,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'weight_stats': weight_stats,
        'elp_stats': elp_stats,
        'runtime_minutes': elapsed
    }


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data
    num_classes = 10 if args.dataset == 'cifar10' else 100
    train_loader, train_dataset = get_dataloader(
        args.dataset, train=True, batch_size=args.batch_size,
        num_workers=args.num_workers, noise_rate=args.noise_rate,
        noise_type=args.noise_type, seed=args.seed
    )
    test_loader, _ = get_dataloader(
        args.dataset, train=False, batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f'Dataset: {args.dataset}, Noise: {args.noise_rate}, Seed: {args.seed}')
    print(f'Method: {args.method}')
    
    # Train
    if args.method in ['supcon', 'supcon_lr', 'supcon_il']:
        results = train_supcon(args, train_loader, test_loader, num_classes, device)
    else:
        results = train_laser_scl(args, train_loader, test_loader, num_classes, train_dataset, device)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    result_file = f'{args.method}_{args.dataset}_n{int(args.noise_rate*100)}_s{args.seed}.json'
    result_path = os.path.join(args.save_dir, result_file)
    save_results(results, result_path)
    print(f'Results saved to {result_path}')
    print(f"Final Accuracy: {results['final_accuracy']:.2f}%")


if __name__ == '__main__':
    main()
