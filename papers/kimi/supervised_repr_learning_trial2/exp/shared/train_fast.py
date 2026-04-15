"""
Fast training script for quick verification (reduced epochs).
Use this for testing and debugging before running full experiments.
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
from utils import set_seed, save_results, train_linear_classifier


def train_supcon_fast(args, train_loader, test_loader, num_classes, device):
    """Train standard SupCon with reduced epochs for quick verification."""
    set_seed(args.seed)
    
    model = SupConResNet(num_classes=num_classes, name='resnet18',
                         projection_dim=128, hidden_dim=512).to(device)
    
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    train_losses = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            (view1, view2), labels, _, _ = batch
            images = torch.cat([view1, view2], dim=0).to(device)
            labels = labels.to(device)
            
            projections = model(images)
            bsz = labels.shape[0]
            f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            loss = criterion(features, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
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
        'epochs': args.epochs,
        'final_accuracy': final_acc,
        'train_losses': train_losses,
        'runtime_minutes': elapsed
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--noise_rate', type=float, default=0.4)
    parser.add_argument('--method', type=str, default='supcon')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    train_loader, _ = get_dataloader(args.dataset, train=True, batch_size=256,
                                     noise_rate=args.noise_rate, seed=args.seed)
    test_loader, _ = get_dataloader(args.dataset, train=False, batch_size=256)
    
    results = train_supcon_fast(args, train_loader, test_loader, num_classes, device)
    
    os.makedirs(args.save_dir, exist_ok=True)
    result_file = f'{args.method}_{args.dataset}_n{int(args.noise_rate*100)}_s{args.seed}_fast.json'
    save_results(results, os.path.join(args.save_dir, result_file))
    print(f"Final Accuracy: {results['final_accuracy']:.2f}%")


if __name__ == '__main__':
    main()
