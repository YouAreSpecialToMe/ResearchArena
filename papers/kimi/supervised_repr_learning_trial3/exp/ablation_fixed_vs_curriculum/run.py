"""
Ablation Study: Fixed vs Curriculum Weighting.
Compares JD-CCL-style fixed weighting with different lambda values against CAG-HNM curriculum.
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
from shared.losses import JDCCLFixedLoss, CAGHNMLoss
from shared.utils import set_seed, linear_evaluation, save_results


def run_fixed_experiment(train_loader, test_loader, attr_similarity, device, 
                         lambda_weight, epochs, eval_epochs, lr, seed):
    """Run fixed weighting experiment."""
    set_seed(seed)
    
    model = create_model('resnet18_cifar', num_classes=100, projection_dim=128).to(device)
    criterion = JDCCLFixedLoss(temperature=0.1, lambda_weight=lambda_weight)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"  Training fixed (lambda={lambda_weight})...")
    start_time = time.time()
    contrastive_losses = []
    
    for epoch in range(epochs):
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
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    # Linear evaluation
    linear_acc = linear_evaluation(
        model.encoder, train_loader, test_loader, device,
        feature_dim=512, num_classes=100, epochs=eval_epochs
    )
    
    runtime = (time.time() - start_time) / 60
    
    return {
        "method": "fixed",
        "lambda_weight": lambda_weight,
        "linear_accuracy": linear_acc,
        "runtime_minutes": runtime,
        "contrastive_losses": contrastive_losses
    }


def run_curriculum_experiment(train_loader, test_loader, attr_similarity, device,
                              lambda_min, lambda_max, gamma, epochs, eval_epochs, lr, seed):
    """Run curriculum weighting experiment."""
    set_seed(seed)
    
    model = create_model('resnet18_cifar', num_classes=100, projection_dim=128).to(device)
    criterion = CAGHNMLoss(temperature=0.1, lambda_min=lambda_min, lambda_max=lambda_max, gamma=gamma)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"  Training curriculum (λ: {lambda_min}→{lambda_max}, γ={gamma})...")
    start_time = time.time()
    contrastive_losses = []
    curriculum_values = []
    
    for epoch in range(epochs):
        model.train()
        criterion.set_epoch(epoch, epochs)
        
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features = model(images)
            loss, lambda_t = criterion(features, labels, attr_similarity)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        contrastive_losses.append(avg_loss)
        curriculum_values.append(lambda_t)
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, λ(t)={lambda_t:.4f}")
    
    # Linear evaluation
    linear_acc = linear_evaluation(
        model.encoder, train_loader, test_loader, device,
        feature_dim=512, num_classes=100, epochs=eval_epochs
    )
    
    runtime = (time.time() - start_time) / 60
    
    return {
        "method": "curriculum",
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "gamma": gamma,
        "linear_accuracy": linear_acc,
        "runtime_minutes": runtime,
        "contrastive_losses": contrastive_losses,
        "curriculum_values": curriculum_values
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    
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
    
    results = {
        "experiment": "ablation_fixed_vs_curriculum",
        "seed": args.seed,
        "epochs": args.epochs,
        "eval_epochs": args.eval_epochs,
        "runs": []
    }
    
    # Run fixed experiments with different lambda values
    print("\n=== Fixed Weighting Experiments ===")
    
    print("\n1. Fixed (lambda=2.0) - JD-CCL style")
    result = run_fixed_experiment(
        train_loader, test_loader, attr_similarity, device,
        lambda_weight=2.0, epochs=args.epochs, eval_epochs=args.eval_epochs,
        lr=args.lr, seed=args.seed
    )
    results["runs"].append(result)
    print(f"  Accuracy: {result['linear_accuracy']:.2f}%")
    
    print("\n2. Fixed-aggressive (lambda=4.0)")
    result = run_fixed_experiment(
        train_loader, test_loader, attr_similarity, device,
        lambda_weight=4.0, epochs=args.epochs, eval_epochs=args.eval_epochs,
        lr=args.lr, seed=args.seed
    )
    results["runs"].append(result)
    print(f"  Accuracy: {result['linear_accuracy']:.2f}%")
    
    # Run curriculum experiment
    print("\n=== Curriculum Weighting Experiment ===")
    print("\n3. Curriculum (0.1→2.0, γ=2.0) - CAG-HNM")
    result = run_curriculum_experiment(
        train_loader, test_loader, attr_similarity, device,
        lambda_min=0.1, lambda_max=2.0, gamma=2.0,
        epochs=args.epochs, eval_epochs=args.eval_epochs,
        lr=args.lr, seed=args.seed
    )
    results["runs"].append(result)
    print(f"  Accuracy: {result['linear_accuracy']:.2f}%")
    
    # Summary
    print("\n=== Ablation Summary ===")
    print(f"{'Method':<25} {'Accuracy':<10}")
    print("-" * 35)
    for run in results["runs"]:
        if run["method"] == "fixed":
            print(f"Fixed (λ={run['lambda_weight']}){'':<15} {run['linear_accuracy']:.2f}%")
        else:
            print(f"Curriculum ({run['lambda_min']}→{run['lambda_max']}, γ={run['gamma']}) {run['linear_accuracy']:.2f}%")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'ablation_fixed_vs_curriculum.json')
    save_results(results, save_path)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
