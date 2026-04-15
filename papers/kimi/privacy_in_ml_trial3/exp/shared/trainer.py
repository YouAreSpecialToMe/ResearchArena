"""
Main training script for all FCL experiments.
Supports: Standard FCL, FCL-AT, FCL-DP, FCL-DP-AT, FedSecure-CL, and ablations.
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add shared to path
sys.path.insert(0, os.path.dirname(__file__))

from models import create_simclr_model, LinearClassifier, MembershipClassifier
from data_loader import (create_federated_datasets, get_client_dataloader, 
                         get_linear_eval_dataloaders, create_membership_split,
                         FederatedDataset)
from fcl_utils import (InfoNCELoss, fedavg_aggregate, linear_evaluation, 
                       evaluate_robust_accuracy, set_seed, save_results)


def train_client_fedsecure(model, dataloader, optimizer, device, epochs=1,
                           use_adversarial=False, use_privacy_reg=False, 
                           use_grad_noise=False, membership_classifier=None,
                           alpha=1.0, beta=0.5, attack_eps=8/255, attack_steps=7):
    """
    Train client with FedSecure-CL components.
    
    Args:
        model: Client model
        dataloader: Client's data loader
        optimizer: Optimizer
        device: Device
        epochs: Local epochs
        use_adversarial: Whether to use adversarial training
        use_privacy_reg: Whether to use privacy regularization
        use_grad_noise: Whether to use gradient noise
        membership_classifier: Membership classifier for privacy regularization
        alpha: Weight for adversarial loss
        beta: Weight for privacy loss
        attack_eps: Epsilon for PGD
        attack_steps: Steps for PGD
    """
    model.train()
    if membership_classifier is not None:
        membership_classifier.train()
    
    contrastive_criterion = InfoNCELoss(temperature=0.5)
    total_loss = 0
    total_batches = 0
    
    for epoch in range(epochs):
        for batch in dataloader:
            (x1, x2), labels = batch
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()
            
            # Standard contrastive loss
            z1 = model(x1)
            z2 = model(x2)
            loss_contrastive = contrastive_criterion(z1, z2)
            
            loss = loss_contrastive
            
            # Adversarial training component
            if use_adversarial:
                # Generate adversarial examples
                x1_adv = pgd_attack_contrastive(model, x1, x2, attack_eps, attack_steps, device)
                x2_adv = pgd_attack_contrastive(model, x2, x1, attack_eps, attack_steps, device)
                
                z1_adv = model(x1_adv)
                z2_adv = model(x2_adv)
                loss_robustness = contrastive_criterion(z1_adv, z2_adv)
                loss = loss + alpha * loss_robustness
            
            # Privacy regularization component
            if use_privacy_reg and membership_classifier is not None:
                # Get projections for membership classification
                z = torch.cat([z1, z2], dim=0)
                
                # Create synthetic membership labels (1 for member, 0 for non-member)
                # In training, all are members
                membership_labels = torch.ones(z.size(0), dtype=torch.long, device=device)
                
                # Membership classifier predictions
                membership_logits = membership_classifier(z.detach())
                
                # Privacy loss: maximize entropy of membership predictions
                # (make membership hard to predict)
                membership_probs = torch.softmax(membership_logits, dim=1)
                entropy = -torch.sum(membership_probs * torch.log(membership_probs + 1e-10), dim=1)
                loss_privacy = -entropy.mean()  # Negative because we maximize entropy
                
                loss = loss + beta * loss_privacy
                
                # Train membership classifier (adversarial)
                membership_optimizer = optim.Adam(membership_classifier.parameters(), lr=0.001)
                membership_optimizer.zero_grad()
                membership_loss = nn.CrossEntropyLoss()(membership_logits, membership_labels)
                membership_loss.backward()
                membership_optimizer.step()
            
            loss.backward()
            
            # Gradient noise
            if use_grad_noise:
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * 0.001
                        param.grad += noise
            
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches if total_batches > 0 else 0


def pgd_attack_contrastive(model, x, x_pair, eps, steps, device):
    """PGD attack for contrastive learning."""
    x_adv = x.clone().detach()
    step_size = eps / 4
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


def train_federated(args):
    """Main federated training loop."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on {device}")
    print(f"Config: {args.experiment_name}, Dataset: {args.dataset}, Seed: {args.seed}")
    
    # Create federated datasets
    print("Creating federated datasets...")
    client_datasets, test_dataset, client_indices = create_federated_datasets(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        data_dir=args.data_dir,
        seed=args.seed
    )
    
    # Create dataloaders
    client_loaders = [get_client_dataloader(ds, batch_size=args.batch_size, num_workers=2) 
                      for ds in client_datasets]
    
    # Get linear eval loaders
    num_classes = 10 if args.dataset == 'cifar10' else 100
    train_loader_eval, test_loader_eval = get_linear_eval_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Create global model
    global_model = create_simclr_model().to(device)
    
    # Create membership classifier if needed
    membership_classifier = None
    if args.use_privacy_reg:
        membership_classifier = MembershipClassifier(feature_dim=128, hidden_dim=64).to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'linear_acc': [],
        'round_times': []
    }
    
    start_time = time.time()
    
    # Federated training
    for round_idx in range(args.global_rounds):
        round_start = time.time()
        
        # Select clients (full participation for simplicity)
        selected_clients = list(range(args.num_clients))
        
        # Local training
        client_models = []
        client_weights = []
        
        for client_id in selected_clients:
            # Create local model
            local_model = create_simclr_model().to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            # Optimizer
            optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            
            # Local training
            if args.experiment_type == 'fedsecure':
                loss = train_client_fedsecure(
                    local_model, client_loaders[client_id], optimizer, device,
                    epochs=args.local_epochs,
                    use_adversarial=args.use_adversarial,
                    use_privacy_reg=args.use_privacy_reg,
                    use_grad_noise=args.use_grad_noise,
                    membership_classifier=membership_classifier,
                    alpha=args.alpha_at,
                    beta=args.beta_privacy,
                    attack_eps=args.attack_eps,
                    attack_steps=args.attack_steps
                )
            elif args.experiment_type == 'fcl_at':
                loss = train_client_fedsecure(
                    local_model, client_loaders[client_id], optimizer, device,
                    epochs=args.local_epochs,
                    use_adversarial=True,
                    use_privacy_reg=False,
                    use_grad_noise=False,
                    membership_classifier=None,
                    alpha=1.0,
                    beta=0.0,
                    attack_eps=args.attack_eps,
                    attack_steps=args.attack_steps
                )
            elif args.experiment_type == 'fcl_dp':
                # DP training
                loss = train_client_dp(
                    local_model, client_loaders[client_id], optimizer, device,
                    epochs=args.local_epochs,
                    max_grad_norm=args.max_grad_norm,
                    noise_multiplier=args.noise_multiplier
                )
            elif args.experiment_type == 'fcl_dp_at':
                # DP + AT
                loss = train_client_dp_at(
                    local_model, client_loaders[client_id], optimizer, device,
                    epochs=args.local_epochs,
                    max_grad_norm=args.max_grad_norm,
                    noise_multiplier=args.noise_multiplier,
                    attack_eps=args.attack_eps,
                    attack_steps=args.attack_steps
                )
            else:  # standard fcl
                loss = train_client_fedsecure(
                    local_model, client_loaders[client_id], optimizer, device,
                    epochs=args.local_epochs,
                    use_adversarial=False,
                    use_privacy_reg=False,
                    use_grad_noise=False
                )
            
            client_models.append(local_model)
            client_weights.append(len(client_datasets[client_id]))
        
        # Aggregate
        fedavg_aggregate(global_model, client_models, client_weights)
        
        # Linear evaluation every 10 rounds
        if (round_idx + 1) % 10 == 0 or round_idx == args.global_rounds - 1:
            linear_acc = linear_evaluation(
                global_model.encoder, train_loader_eval, test_loader_eval, 
                device, num_classes=num_classes, epochs=50, lr=0.1
            )
            history['linear_acc'].append(linear_acc)
            print(f"Round {round_idx + 1}/{args.global_rounds}, Linear Acc: {linear_acc:.2f}%")
        
        round_time = time.time() - round_start
        history['round_times'].append(round_time)
        
        if (round_idx + 1) % 10 == 0:
            print(f"Round {round_idx + 1}/{args.global_rounds} completed in {round_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("Final linear evaluation...")
    final_linear_acc = linear_evaluation(
        global_model.encoder, train_loader_eval, test_loader_eval,
        device, num_classes=num_classes, epochs=100, lr=0.1
    )
    
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
        'linear_accuracy': final_linear_acc,
        'total_time_seconds': total_time,
        'history': history,
        'config': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, f'{args.experiment_name}_seed{args.seed}_results.json')
    save_results(results, results_path)
    
    print(f"Training completed. Linear Accuracy: {final_linear_acc:.2f}%")
    print(f"Model saved to {model_path}")
    print(f"Results saved to {results_path}")
    
    return results


def train_client_dp(model, dataloader, optimizer, device, epochs=1, 
                    max_grad_norm=1.0, noise_multiplier=1.0):
    """Train client with DP-SGD."""
    model.train()
    criterion = InfoNCELoss(temperature=0.5)
    total_loss = 0
    
    for epoch in range(epochs):
        for batch in dataloader:
            (x1, x2), labels = batch
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()
            
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Add noise
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_multiplier * max_grad_norm
                    param.grad += noise
            
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / (len(dataloader) * epochs)


def train_client_dp_at(model, dataloader, optimizer, device, epochs=1,
                       max_grad_norm=1.0, noise_multiplier=1.0,
                       attack_eps=8/255, attack_steps=7):
    """Train client with DP + Adversarial Training."""
    model.train()
    criterion = InfoNCELoss(temperature=0.5)
    total_loss = 0
    
    for epoch in range(epochs):
        for batch in dataloader:
            (x1, x2), labels = batch
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()
            
            # Adversarial examples
            x1_adv = pgd_attack_contrastive(model, x1, x2, attack_eps, attack_steps, device)
            x2_adv = pgd_attack_contrastive(model, x2, x1, attack_eps, attack_steps, device)
            
            z1 = model(x1_adv)
            z2 = model(x2_adv)
            loss = criterion(z1, z2)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Add noise
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_multiplier * max_grad_norm
                    param.grad += noise
            
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / (len(dataloader) * epochs)


def main():
    parser = argparse.ArgumentParser()
    
    # Experiment config
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--experiment_type', type=str, default='standard',
                       choices=['standard', 'fcl_at', 'fcl_dp', 'fcl_dp_at', 'fedsecure'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--seed', type=int, default=42)
    
    # Federated learning
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--global_rounds', type=int, default=50)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    
    # Adversarial training
    parser.add_argument('--use_adversarial', action='store_true')
    parser.add_argument('--alpha_at', type=float, default=1.0, help='Weight for AT loss')
    parser.add_argument('--attack_eps', type=float, default=8/255)
    parser.add_argument('--attack_steps', type=int, default=7)
    
    # Privacy regularization
    parser.add_argument('--use_privacy_reg', action='store_true')
    parser.add_argument('--beta_privacy', type=float, default=0.5, help='Weight for privacy loss')
    
    # Gradient noise
    parser.add_argument('--use_grad_noise', action='store_true')
    
    # DP
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--noise_multiplier', type=float, default=1.0)
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    train_federated(args)


if __name__ == '__main__':
    main()
