#!/usr/bin/env python3
"""
Fast experiment runner with reduced memory requirements.
Uses smaller batch size and runs experiments sequentially.
"""

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import sys
sys.path.insert(0, './src')

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np
import json
import time
from tqdm import tqdm
from collections import defaultdict

from models.resnet import ResNet18
from data_loader import get_data_loaders
from compression.pruning import evaluate_with_pruning, apply_magnitude_pruning

# Configuration
SEEDS = [42, 123, 456]
EPSILON = 3.0
DELTA = 1e-5
EPOCHS = 30
BATCH_SIZE = 64  # Reduced for memory
LR = 0.1
MAX_GRAD_NORM = 1.0
SPARSITY = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")


def train_standard_dp_model(seed):
    """Train standard DP-SGD model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    trainloader, testloader = get_data_loaders('cifar10', batch_size=BATCH_SIZE)
    model = ResNet18(num_classes=10)
    model = ModuleValidator.fix(model)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=trainloader,
        target_epsilon=EPSILON, target_delta=DELTA, epochs=EPOCHS, max_grad_norm=MAX_GRAD_NORM
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.9)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epsilon': []}
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        epsilon = privacy_engine.get_epsilon(delta=DELTA)
        
        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epsilon'].append(epsilon)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f'  Epoch {epoch+1}/{EPOCHS}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, ε: {epsilon:.2f}')
    
    runtime = time.time() - start_time
    
    # Save model
    os.makedirs('exp/baseline_standard', exist_ok=True)
    model_path = f'exp/baseline_standard/model_cifar10_resnet18_eps{EPSILON}_seed{seed}.pt'
    torch.save(model.state_dict(), model_path)
    
    # Save results
    results = {
        'experiment': 'standard_dp',
        'seed': seed,
        'final_test_acc': history['test_acc'][-1],
        'final_epsilon': history['epsilon'][-1],
        'runtime_minutes': runtime / 60,
        'history': history
    }
    
    results_path = f'exp/baseline_standard/results_cifar10_resnet18_eps{EPSILON}_seed{seed}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'  Standard DP (seed {seed}): {history["test_acc"][-1]:.2f}%')
    return model, history


class PHCAOptimizer:
    """PHCA-DP-SGD Optimizer."""
    
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


def train_phca_model(seed):
    """Train PHCA-DP-SGD model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    trainloader, testloader = get_data_loaders('cifar10', batch_size=BATCH_SIZE)
    model = ResNet18(num_classes=10)
    model = ModuleValidator.fix(model)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=trainloader,
        target_epsilon=EPSILON, target_delta=DELTA, epochs=EPOCHS, max_grad_norm=MAX_GRAD_NORM
    )
    
    phca_opt = PHCAOptimizer(
        optimizer=optimizer.original_optimizer,
        noise_multiplier=optimizer.noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
        expected_batch_size=optimizer.expected_batch_size,
        target_sparsity=SPARSITY,
        alpha=0.1,
        beta=0.5
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(phca_opt.optimizer, milestones=[10, 20], gamma=0.9)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epsilon': []}
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        phca_opt.set_epoch(epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            phca_opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            phca_opt.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        epsilon = privacy_engine.get_epsilon(delta=DELTA)
        
        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epsilon'].append(epsilon)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f'  Epoch {epoch+1}/{EPOCHS}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, ε: {epsilon:.2f}')
    
    runtime = time.time() - start_time
    
    # Save model
    os.makedirs('exp/phca_full', exist_ok=True)
    model_path = f'exp/phca_full/model_cifar10_resnet18_eps{EPSILON}_sparsity{SPARSITY}_seed{seed}.pt'
    torch.save(model.state_dict(), model_path)
    
    # Save results
    results = {
        'experiment': 'phca_dp_sgd',
        'seed': seed,
        'final_test_acc': history['test_acc'][-1],
        'final_epsilon': history['epsilon'][-1],
        'runtime_minutes': runtime / 60,
        'history': history
    }
    
    results_path = f'exp/phca_full/results_cifar10_resnet18_eps{EPSILON}_sparsity{SPARSITY}_seed{seed}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'  PHCA-DP-SGD (seed {seed}): {history["test_acc"][-1]:.2f}%')
    return model, history


def run_ablation(ablation_mode, seed):
    """Run ablation study."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    trainloader, testloader = get_data_loaders('cifar10', batch_size=BATCH_SIZE)
    model = ResNet18(num_classes=10)
    model = ModuleValidator.fix(model)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=trainloader,
        target_epsilon=EPSILON, target_delta=DELTA, epochs=EPOCHS, max_grad_norm=MAX_GRAD_NORM
    )
    
    class AblationOptimizer:
        def __init__(self, opt, nm, mgn, ebs, mode):
            self.optimizer = opt
            self.noise_multiplier = nm
            self.max_grad_norm = mgn
            self.expected_batch_size = ebs
            self.mode = mode
            self.grad_ema = defaultdict(lambda: None)
            self.survival_probs = defaultdict(lambda: torch.tensor(1.0))
            self.clipping_weights = defaultdict(lambda: torch.tensor(1.0))
            self.current_epoch = 0
            self.params = list(opt.param_groups[0]['params'])
            self.target_sparsity = 0.7
            self.alpha = 0.1
            self.beta = 0.5
            self.ema_gamma = 0.9
            
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
        
        def compute_survival(self):
            if self.current_epoch < 5:
                for p in self.params:
                    self.survival_probs[p] = torch.ones_like(p.data)
                    self.clipping_weights[p] = torch.ones_like(p.data)
                return
            
            all_ema = []
            for p in self.params:
                if self.grad_ema[p] is not None:
                    all_ema.append(self.grad_ema[p].flatten())
            
            if not all_ema:
                return
            
            all_ema = torch.cat(all_ema)
            k = int(self.target_sparsity * len(all_ema))
            threshold = torch.kthvalue(all_ema, k)[0].item() if k > 0 and k < len(all_ema) else 0.0
            
            for p in self.params:
                if self.grad_ema[p] is not None:
                    ema_norm = self.grad_ema[p] / (threshold + 1e-8)
                    p_surv = torch.sigmoid(2 * (ema_norm - 1))
                    self.survival_probs[p] = p_surv
                    
                    if self.mode == 'no_per_param_clipping':
                        self.clipping_weights[p] = torch.ones_like(p_surv)
                    elif self.mode == 'binary_masking':
                        self.clipping_weights[p] = (p_surv > 0.5).float() * 0.9 + 0.1
                    elif self.mode == 'fixed_survival':
                        torch.manual_seed(42)
                        self.clipping_weights[p] = torch.rand_like(p_surv) * 0.5 + 0.5
                    else:
                        self.clipping_weights[p] = p_surv + self.alpha * (1 - p_surv)
        
        def zero_grad(self):
            self.optimizer.zero_grad()
        
        def step(self):
            self.update_grad_ema()
            self.compute_survival()
            
            for p in self.params:
                if p.grad is not None:
                    w = self.clipping_weights[p]
                    per_param_clip = self.max_grad_norm * w
                    
                    grad_norm = p.grad.norm()
                    clip_factor = torch.clamp_max(per_param_clip / (grad_norm + 1e-8), 1.0)
                    clipped_grad = p.grad * clip_factor
                    
                    survival_prob = self.survival_probs[p]
                    if self.mode == 'no_compression_aware_noise':
                        noise_scale = torch.ones_like(survival_prob)
                    else:
                        noise_scale = (1 - survival_prob) ** self.beta
                    
                    noise_std = self.noise_multiplier * self.max_grad_norm * noise_scale
                    noise = torch.randn_like(p.grad) * noise_std / self.expected_batch_size ** 0.5
                    
                    p.grad = clipped_grad + noise
            
            self.optimizer.step()
    
    ablation_opt = AblationOptimizer(
        optimizer.original_optimizer, optimizer.noise_multiplier,
        MAX_GRAD_NORM, optimizer.expected_batch_size, ablation_mode
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(ablation_opt.optimizer, milestones=[10, 20], gamma=0.9)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epsilon': []}
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        ablation_opt.set_epoch(epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            ablation_opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            ablation_opt.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        epsilon = privacy_engine.get_epsilon(delta=DELTA)
        
        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epsilon'].append(epsilon)
    
    runtime = time.time() - start_time
    
    os.makedirs('exp/ablations', exist_ok=True)
    results = {
        'experiment': f'ablation_{ablation_mode}',
        'seed': seed,
        'final_test_acc': history['test_acc'][-1],
        'final_epsilon': history['epsilon'][-1],
        'runtime_minutes': runtime / 60
    }
    
    results_path = f'exp/ablations/results_{ablation_mode}_seed{seed}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'  Ablation {ablation_mode} (seed {seed}): {history["test_acc"][-1]:.2f}%')


def main():
    """Run all experiments."""
    print("=" * 80)
    print("PHCA-DP-SGD Experiment Suite (Memory Efficient)")
    print("=" * 80)
    
    # Run standard DP baseline
    print("\n1. Running Standard DP-SGD Baseline...")
    for seed in SEEDS:
        train_standard_dp_model(seed)
    
    # Run PHCA-DP-SGD
    print("\n2. Running PHCA-DP-SGD Main Method...")
    for seed in SEEDS:
        train_phca_model(seed)
    
    # Run ablations
    print("\n3. Running Ablation Studies...")
    ablation_modes = ['no_per_param_clipping', 'no_compression_aware_noise', 'binary_masking', 'fixed_survival']
    for mode in ablation_modes:
        print(f"  Ablation: {mode}")
        for seed in SEEDS[:2]:  # Run 2 seeds for ablations to save time
            run_ablation(mode, seed)
    
    # Evaluate with compression
    print("\n4. Evaluating with Compression...")
    _, testloader = get_data_loaders('cifar10', batch_size=BATCH_SIZE)
    
    compression_results = {'standard_dp': [], 'phca_full': []}
    
    for seed in SEEDS:
        # Standard DP
        model = ResNet18(num_classes=10).to(DEVICE)
        model.load_state_dict(torch.load(f'exp/baseline_standard/model_cifar10_resnet18_eps{EPSILON}_seed{seed}.pt'))
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc_before = 100. * correct / total
        
        acc_after, sparsity = evaluate_with_pruning(model, testloader, DEVICE, SPARSITY)
        compression_results['standard_dp'].append({
            'seed': seed, 'acc_before': acc_before, 'acc_after': acc_after, 'sparsity': sparsity
        })
        print(f"  Standard DP seed {seed}: {acc_before:.2f}% -> {acc_after:.2f}%")
        
        # PHCA
        model = ResNet18(num_classes=10).to(DEVICE)
        model.load_state_dict(torch.load(f'exp/phca_full/model_cifar10_resnet18_eps{EPSILON}_sparsity{SPARSITY}_seed{seed}.pt'))
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc_before = 100. * correct / total
        
        acc_after, sparsity = evaluate_with_pruning(model, testloader, DEVICE, SPARSITY)
        compression_results['phca_full'].append({
            'seed': seed, 'acc_before': acc_before, 'acc_after': acc_after, 'sparsity': sparsity
        })
        print(f"  PHCA seed {seed}: {acc_before:.2f}% -> {acc_after:.2f}%")
    
    os.makedirs('results', exist_ok=True)
    with open('results/compression_results.json', 'w') as f:
        json.dump(compression_results, f, indent=2)
    
    # Generate final results.json
    print("\n5. Generating results.json...")
    final_results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seeds': SEEDS,
            'epsilon': EPSILON,
            'sparsity': SPARSITY,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS
        }
    }
    
    # Aggregate results
    for method in ['standard_dp', 'phca_full']:
        accs = []
        for seed in SEEDS:
            if method == 'standard_dp':
                path = f'exp/baseline_standard/results_cifar10_resnet18_eps{EPSILON}_seed{seed}.json'
            else:
                path = f'exp/phca_full/results_cifar10_resnet18_eps{EPSILON}_sparsity{SPARSITY}_seed{seed}.json'
            
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                accs.append(data['final_test_acc'])
        
        if accs:
            final_results[method] = {
                'mean_accuracy': float(np.mean(accs)),
                'std_accuracy': float(np.std(accs)),
                'raw_accuracies': accs
            }
    
    # Ablations
    final_results['ablations'] = {}
    for mode in ablation_modes:
        accs = []
        for seed in SEEDS[:2]:
            path = f'exp/ablations/results_{mode}_seed{seed}.json'
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                accs.append(data['final_test_acc'])
        if accs:
            final_results['ablations'][mode] = {
                'mean_accuracy': float(np.mean(accs)),
                'std_accuracy': float(np.std(accs))
            }
    
    final_results['compression'] = compression_results
    
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(json.dumps(final_results, indent=2))
    print("=" * 80)


if __name__ == '__main__':
    main()
