"""
PHCA-DP-SGD: Post-Hoc Compression-Aware DP-SGD
Implements continuous per-parameter clipping weights based on survival probability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np
import json
import os
import time
from tqdm import tqdm
from collections import defaultdict


class PHCAOptimizer:
    """
    PHCA-DP-SGD Optimizer with per-parameter clipping weights
    and compression-aware noise.
    """
    
    def __init__(self, optimizer, noise_multiplier, max_grad_norm, expected_batch_size,
                 target_sparsity=0.7, alpha=0.1, beta=0.5, ema_gamma=0.9):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.expected_batch_size = expected_batch_size
        self.target_sparsity = target_sparsity
        self.alpha = alpha
        self.beta = beta
        self.ema_gamma = ema_gamma
        
        self.grad_ema = defaultdict(lambda: None)
        self.survival_probs = defaultdict(lambda: torch.tensor(1.0))
        self.clipping_weights = defaultdict(lambda: torch.tensor(1.0))
        self.current_epoch = 0
        self.params = list(optimizer.param_groups[0]['params'])
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def update_grad_ema(self):
        for p in self.params:
            if p.grad is not None:
                grad_mag = p.grad.abs()
                if self.grad_ema[p] is None:
                    self.grad_ema[p] = grad_mag.clone().detach()
                else:
                    self.grad_ema[p] = self.ema_gamma * self.grad_ema[p] + (1 - self.ema_gamma) * grad_mag
    
    def compute_survival_probabilities(self):
        if self.current_epoch < 5:
            for p in self.params:
                self.survival_probs[p] = torch.ones_like(p.data)
                self.clipping_weights[p] = torch.ones_like(p.data)
            return
        
        all_ema_values = []
        for p in self.params:
            if self.grad_ema[p] is not None:
                all_ema_values.append(self.grad_ema[p].flatten())
        
        if len(all_ema_values) == 0:
            return
        
        all_ema_values = torch.cat(all_ema_values)
        k = int(self.target_sparsity * len(all_ema_values))
        threshold = torch.kthvalue(all_ema_values, k)[0].item() if k > 0 and k < len(all_ema_values) else 0.0
        
        for p in self.params:
            if self.grad_ema[p] is not None:
                ema_normalized = self.grad_ema[p] / (threshold + 1e-8)
                p_surv = torch.sigmoid(2 * (ema_normalized - 1))
                self.survival_probs[p] = p_surv
                self.clipping_weights[p] = p_surv + self.alpha * (1 - p_surv)
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.update_grad_ema()
        self.compute_survival_probabilities()
        
        for p in self.params:
            if p.grad is not None:
                w = self.clipping_weights[p]
                per_param_clip = self.max_grad_norm * w
                
                grad_norm = p.grad.norm()
                clip_factor = torch.clamp_max(per_param_clip / (grad_norm + 1e-8), 1.0)
                clipped_grad = p.grad * clip_factor
                
                survival_prob = self.survival_probs[p]
                noise_scale = (1 - survival_prob) ** self.beta
                noise_std = self.noise_multiplier * self.max_grad_norm * noise_scale
                noise = torch.randn_like(p.grad) * noise_std / self.expected_batch_size ** 0.5
                
                p.grad = clipped_grad + noise
        
        self.optimizer.step()


def train_phca_dp_sgd(
    model,
    trainloader,
    testloader,
    device,
    target_epsilon=3.0,
    target_delta=1e-5,
    target_sparsity=0.7,
    alpha=0.1,
    beta=0.5,
    ema_gamma=0.9,
    epochs=30,
    lr=0.1,
    max_grad_norm=1.0,
    seed=42
):
    """
    Train model with PHCA-DP-SGD.
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare model for DP training
    model = ModuleValidator.fix(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Setup privacy engine with custom optimizer
    privacy_engine = PrivacyEngine()
    
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
    
    # Wrap optimizer with PHCA optimizer
    phca_optimizer = PHCAOptimizer(
        optimizer=optimizer.original_optimizer,
        noise_multiplier=optimizer.noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=optimizer.expected_batch_size,
        target_sparsity=target_sparsity,
        alpha=alpha,
        beta=beta,
        ema_gamma=ema_gamma,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        phca_optimizer.optimizer, milestones=[10, 20], gamma=0.9)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epsilon': [],
        'mean_clipping_weight': [],
        'mean_survival_prob': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        phca_optimizer.set_epoch(epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            phca_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            phca_optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (total // targets.size(0) + 1),
                'acc': 100. * correct / total
            })
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        
        # Track clipping weight statistics
        mean_clip_weight = np.mean([w.mean().item() for w in phca_optimizer.clipping_weights.values()])
        mean_surv_prob = np.mean([p.mean().item() for p in phca_optimizer.survival_probs.values()])
        
        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epsilon'].append(epsilon)
        history['mean_clipping_weight'].append(mean_clip_weight)
        history['mean_survival_prob'].append(mean_surv_prob)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%, ε: {epsilon:.2f}, '
              f'Mean Clip W: {mean_clip_weight:.3f}')
    
    runtime = time.time() - start_time
    
    return model, history, runtime


def main():
    import argparse
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.resnet import ResNet18
    from models.convnet import ConvNet
    from models.mlp import MLP
    from data_loader import get_data_loaders
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--target_epsilon', type=float, default=3.0)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--target_sparsity', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--ema_gamma', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./exp/phca_full')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    trainloader, testloader = get_data_loaders(args.dataset, batch_size=args.batch_size)
    
    # Get number of classes
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 100
    
    # Create model
    if args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'convnet':
        model = ConvNet(num_classes=num_classes)
    else:
        model = MLP(input_dim=600, num_classes=num_classes)
    
    print(f'Training PHCA-DP-SGD on {args.dataset} with ε={args.target_epsilon}, '
          f'sparsity={args.target_sparsity}')
    
    # Train
    model, history, runtime = train_phca_dp_sgd(
        model, trainloader, testloader, device,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        target_sparsity=args.target_sparsity,
        alpha=args.alpha,
        beta=args.beta,
        ema_gamma=args.ema_gamma,
        epochs=args.epochs,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir,
        f'model_{args.dataset}_{args.model}_eps{args.target_epsilon}_sparsity{args.target_sparsity}_seed{args.seed}.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save history
    results = {
        'experiment': 'phca_dp_sgd',
        'dataset': args.dataset,
        'model': args.model,
        'target_epsilon': args.target_epsilon,
        'target_delta': args.target_delta,
        'target_sparsity': args.target_sparsity,
        'alpha': args.alpha,
        'beta': args.beta,
        'epochs': args.epochs,
        'seed': args.seed,
        'final_test_acc': history['test_acc'][-1],
        'final_epsilon': history['epsilon'][-1],
        'runtime_minutes': runtime / 60,
        'history': history
    }
    
    results_path = os.path.join(args.output_dir,
        f'results_{args.dataset}_{args.model}_eps{args.target_epsilon}_sparsity{args.target_sparsity}_seed{args.seed}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to {results_path}')
    print(f'Final test accuracy: {history["test_acc"][-1]:.2f}%')


if __name__ == '__main__':
    main()
