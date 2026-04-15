"""
CIFAR-100 JD-CCL-style Fixed Selection Experiment.
Fixed attribute-guided hard negative mining.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from shared.data_loader import get_cifar100_dataloaders, load_cifar100_attributes
from shared.models import create_model
from shared.losses import JDCCLFixedLoss
from shared.utils import set_seed, linear_evaluation, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--lambda_weight', type=float, default=2.0)
    parser.add_argument('--top_k_percent', type=float, default=0.25)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading CIFAR-100...")
    train_loader, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size, num_workers=4
    )
    
    # Load attributes
    print("Loading attributes...")
    _, attr_similarity = load_cifar100_attributes()
    attr_similarity = torch.from_numpy(attr_similarity).float().to(device)
    
    # Model
    print("Creating model...")
    model = create_model('resnet18_cifar', num_classes=100, projection_dim=args.projection_dim).to(device)
    
    # Training setup
    criterion = JDCCLFixedLoss(
        temperature=args.temperature,
        lambda_weight=args.lambda_weight,
        top_k_percent=args.top_k_percent
    )
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Contrastive pre-training
    print(f"JD-CCL fixed training for {args.epochs} epochs...")
    print(f"Lambda weight: {args.lambda_weight}, Top-k: {args.top_k_percent}")
    start_time = time.time()
    
    contrastive_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = model(images)
            loss = criterion(features, labels, attr_similarity)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        contrastive_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}")
    
    # Linear evaluation
    print("\nLinear evaluation...")
    linear_acc = linear_evaluation(
        model.encoder, train_loader, test_loader, device,
        feature_dim=512, num_classes=100, epochs=args.eval_epochs
    )
    
    runtime = (time.time() - start_time) / 60
    
    # Save results
    results = {
        "experiment": "cifar100_jdccl_fixed",
        "seed": args.seed,
        "epochs": args.epochs,
        "eval_epochs": args.eval_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
        "projection_dim": args.projection_dim,
        "lambda_weight": args.lambda_weight,
        "top_k_percent": args.top_k_percent,
        "linear_accuracy": linear_acc,
        "runtime_minutes": runtime,
        "contrastive_losses": contrastive_losses
    }
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'cifar100_jdccl_fixed_seed{args.seed}.json')
    save_results(results, save_path)
    print(f"\nResults saved to {save_path}")
    print(f"Linear Accuracy: {linear_acc:.2f}%")
    print(f"Runtime: {runtime:.1f} minutes")


if __name__ == '__main__':
    main()
