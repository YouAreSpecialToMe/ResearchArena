"""
Fast training script for FedSecure-CL experiments.
Optimized for quick completion with reduced parameters.
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models import create_simclr_model, LinearClassifier
from data_loader import (create_federated_datasets, get_client_dataloader, 
                         get_linear_eval_dataloaders)
from fcl_utils import InfoNCELoss, fedavg_aggregate, set_seed


def pgd_attack_fast(model, x, x_pair, eps, steps, device):
    """Fast PGD attack."""
    x_adv = x.clone().detach()
    step_size = eps / 3
    criterion = InfoNCELoss(temperature=0.5)
    
    for _ in range(steps):
        x_adv.requires_grad = True
        z_adv = model(x_adv)
        z_pair = model(x_pair).detach()
        loss = -criterion(z_adv, z_pair)
        
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        perturbation = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + perturbation, 0, 1).detach()
    
    return x_adv


def fast_linear_eval(encoder, train_loader, test_loader, device, num_classes, epochs=30):
    """Fast linear evaluation."""
    encoder.eval()
    classifier = LinearClassifier(encoder.feature_dim, num_classes).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        classifier.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder(inputs, return_features=True)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Test
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = encoder(inputs, return_features=True)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def train_fast(args):
    """Fast training with reduced complexity."""
    print(f"DEBUG: Starting train_fast for {args.experiment_name}")
    sys.stdout.flush()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Fast Training: {args.experiment_name}, Seed: {args.seed}")
    print(f"Rounds: {args.global_rounds}, Clients: {args.num_clients}, Local Epochs: {args.local_epochs}")
    sys.stdout.flush()
    
    # Create datasets
    print("Creating datasets...")
    sys.stdout.flush()
    client_datasets, test_dataset, client_indices = create_federated_datasets(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        data_dir=args.data_dir,
        seed=args.seed
    )
    print(f"Created {len(client_datasets)} client datasets")
    sys.stdout.flush()
    
    # Create dataloaders
    print("Creating dataloaders...")
    sys.stdout.flush()
    client_loaders = [get_client_dataloader(ds, batch_size=args.batch_size, num_workers=0) 
                      for ds in client_datasets]
    
    # Get eval loaders
    num_classes = 10 if args.dataset == 'cifar10' else 100
    train_loader_eval, test_loader_eval = get_linear_eval_dataloaders(
        dataset_name=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size, num_workers=0
    )
    print("Data loaders ready")
    sys.stdout.flush()
    
    # Create model
    print("Creating model...")
    sys.stdout.flush()
    global_model = create_simclr_model().to(device)
    
    history = {'train_loss': [], 'linear_acc': [], 'round_times': []}
    criterion = InfoNCELoss(temperature=0.5)
    
    start_time = time.time()
    
    print(f"Starting training for {args.global_rounds} rounds...")
    sys.stdout.flush()
    
    for round_idx in range(args.global_rounds):
        round_start = time.time()
        
        client_models = []
        client_weights = []
        
        for client_id in range(args.num_clients):
            local_model = create_simclr_model().to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9)
            local_model.train()
            
            epoch_loss = 0
            num_batches = 0
            
            for epoch in range(args.local_epochs):
                for batch in client_loaders[client_id]:
                    (x1, x2), _ = batch
                    x1, x2 = x1.to(device), x2.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Handle different experiment types
                    if args.experiment_type in ['fcl_at', 'fedsecure'] and args.use_adversarial:
                        # Adversarial training with PGD
                        x1_adv = pgd_attack_fast(local_model, x1, x2, args.attack_eps, 3, device)
                        x2_adv = pgd_attack_fast(local_model, x2, x1, args.attack_eps, 3, device)
                        z1 = local_model(x1_adv)
                        z2 = local_model(x2_adv)
                    else:
                        z1 = local_model(x1)
                        z2 = local_model(x2)
                    
                    loss = criterion(z1, z2)
                    
                    # Privacy regularization for FedSecure
                    if args.experiment_type == 'fedsecure' and args.use_privacy_reg:
                        # Simple entropy regularization
                        z = torch.cat([z1, z2], dim=0)
                        z_norm = F.normalize(z, dim=1)
                        similarity = torch.mm(z_norm, z_norm.t())
                        entropy_reg = -torch.mean(torch.log(similarity.abs() + 1e-10))
                        loss = loss + args.beta_privacy * entropy_reg
                    
                    loss.backward()
                    
                    # Gradient noise for DP
                    if args.experiment_type in ['fcl_dp', 'fcl_dp_at'] or (args.experiment_type == 'fedsecure' and args.use_grad_noise):
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.max_grad_norm)
                        for param in local_model.parameters():
                            if param.grad is not None:
                                noise = torch.randn_like(param.grad) * args.noise_multiplier * args.max_grad_norm
                                param.grad += noise
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            client_models.append(local_model)
            client_weights.append(len(client_datasets[client_id]))
        
        # Aggregate
        fedavg_aggregate(global_model, client_models, client_weights)
        
        history['round_times'].append(time.time() - round_start)
        
        # Evaluate every 5 rounds
        if (round_idx + 1) % 5 == 0 or round_idx == args.global_rounds - 1:
            linear_acc = fast_linear_eval(global_model.encoder, train_loader_eval, test_loader_eval, 
                                         device, num_classes, epochs=20)
            history['linear_acc'].append(linear_acc)
            print(f"Round {round_idx + 1}/{args.global_rounds}, Linear Acc: {linear_acc:.2f}%")
            sys.stdout.flush()
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("Final evaluation...")
    sys.stdout.flush()
    final_acc = fast_linear_eval(global_model.encoder, train_loader_eval, test_loader_eval,
                                 device, num_classes, epochs=50)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'models', f'{args.experiment_name}_seed{args.seed}.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'encoder': global_model.encoder.state_dict(),
        'projection_head': global_model.projection_head.state_dict(),
        'args': vars(args),
        'history': history
    }, model_path)
    
    # Save results
    results = {
        'experiment': args.experiment_name,
        'seed': args.seed,
        'dataset': args.dataset,
        'linear_accuracy': final_acc,
        'total_time_seconds': total_time,
        'history': history,
        'config': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, f'{args.experiment_name}_seed{args.seed}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed: {final_acc:.2f}% accuracy in {total_time/60:.1f} minutes")
    sys.stdout.flush()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--experiment_type', type=str, default='standard',
                       choices=['standard', 'fcl_at', 'fcl_dp', 'fcl_dp_at', 'fedsecure'])
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--global_rounds', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_adversarial', action='store_true')
    parser.add_argument('--use_privacy_reg', action='store_true')
    parser.add_argument('--use_grad_noise', action='store_true')
    parser.add_argument('--alpha_at', type=float, default=1.0)
    parser.add_argument('--beta_privacy', type=float, default=0.5)
    parser.add_argument('--attack_eps', type=float, default=8/255)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--noise_multiplier', type=float, default=0.5)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    
    args = parser.parse_args()
    train_fast(args)


if __name__ == '__main__':
    main()
