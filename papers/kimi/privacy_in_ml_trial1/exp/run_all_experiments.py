#!/usr/bin/env python3
"""
Master experiment runner for PHCA-DP-SGD research.
Runs all experiments with proper random seeds and saves real results.
"""

import subprocess
import sys
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, './src')

from models.resnet import ResNet18
from models.convnet import ConvNet
from data_loader import get_data_loaders
from compression.pruning import evaluate_with_pruning, apply_magnitude_pruning

# Experiment configuration
SEEDS = [42, 123, 456]
EPSILON = 3.0
DELTA = 1e-5
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.1
MAX_GRAD_NORM = 1.0
SPARSITY = 0.7

def run_standard_dp_baseline():
    """Run standard DP-SGD baseline experiments."""
    print("=" * 80)
    print("Running Standard DP-SGD Baseline")
    print("=" * 80)
    
    results = []
    for seed in SEEDS:
        print(f"\n--- Running with seed {seed} ---")
        cmd = [
            'python', 'src/train_standard_dp.py',
            '--dataset', 'cifar10',
            '--model', 'resnet18',
            '--target_epsilon', str(EPSILON),
            '--target_delta', str(DELTA),
            '--epochs', str(EPOCHS),
            '--batch_size', str(BATCH_SIZE),
            '--lr', str(LR),
            '--max_grad_norm', str(MAX_GRAD_NORM),
            '--seed', str(seed),
            '--output_dir', 'exp/baseline_standard'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    
    print("\nStandard DP baseline complete!")


def run_prepruning_baseline():
    """Run pre-pruning baseline (Adamczewski et al. 2023)."""
    print("=" * 80)
    print("Running Pre-Pruning Baseline (Adamczewski et al. 2023)")
    print("=" * 80)
    
    for seed in SEEDS:
        print(f"\n--- Running with seed {seed} ---")
        cmd = [
            'python', 'src/baselines/pre_pruning.py',
            '--dataset', 'cifar10',
            '--target_epsilon', str(EPSILON),
            '--target_sparsity', str(SPARSITY),
            '--epochs', str(EPOCHS),
            '--seed', str(seed),
            '--output_dir', 'exp/baseline_prepruning'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    
    print("\nPre-pruning baseline complete!")


def run_adadpigu_baseline():
    """Run AdaDPIGU-style baseline."""
    print("=" * 80)
    print("Running AdaDPIGU-style Baseline")
    print("=" * 80)
    
    for seed in SEEDS:
        print(f"\n--- Running with seed {seed} ---")
        cmd = [
            'python', 'src/baselines/adadpigu_simple.py',
            '--dataset', 'cifar10',
            '--target_epsilon', str(EPSILON),
            '--retention_ratio', '0.6',
            '--epochs', str(EPOCHS),
            '--seed', str(seed),
            '--output_dir', 'exp/baseline_adadpigu'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    
    print("\nAdaDPIGU baseline complete!")


def run_phca_full():
    """Run PHCA-DP-SGD main experiments."""
    print("=" * 80)
    print("Running PHCA-DP-SGD Main Method")
    print("=" * 80)
    
    for seed in SEEDS:
        print(f"\n--- Running with seed {seed} ---")
        cmd = [
            'python', 'src/methods/phca_dp_sgd.py',
            '--dataset', 'cifar10',
            '--model', 'resnet18',
            '--target_epsilon', str(EPSILON),
            '--target_delta', str(DELTA),
            '--target_sparsity', str(SPARSITY),
            '--alpha', '0.1',
            '--beta', '0.5',
            '--epochs', str(EPOCHS),
            '--batch_size', str(BATCH_SIZE),
            '--lr', str(LR),
            '--max_grad_norm', str(MAX_GRAD_NORM),
            '--seed', str(seed),
            '--output_dir', 'exp/phca_full'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    
    print("\nPHCA-DP-SGD complete!")


def run_ablations():
    """Run ablation studies."""
    print("=" * 80)
    print("Running Ablation Studies")
    print("=" * 80)
    
    # Create ablation runner script
    ablation_script = '''
import sys
sys.path.insert(0, './src')
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

from models.resnet import ResNet18
from data_loader import get_data_loaders

class AblationOptimizer:
    """Ablation variants of PHCA-DP-SGD."""
    
    def __init__(self, optimizer, noise_multiplier, max_grad_norm, expected_batch_size,
                 target_sparsity=0.7, alpha=0.1, beta=0.5, ema_gamma=0.9, 
                 ablation_mode='full', survival_estimation_start_epoch=5):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.expected_batch_size = expected_batch_size
        self.target_sparsity = target_sparsity
        self.alpha = alpha
        self.beta = beta
        self.ema_gamma = ema_gamma
        self.ablation_mode = ablation_mode
        self.survival_estimation_start_epoch = survival_estimation_start_epoch
        
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
        if self.current_epoch < self.survival_estimation_start_epoch:
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
                
                # Ablation: different clipping weight strategies
                if self.ablation_mode == 'full':
                    self.clipping_weights[p] = p_surv + self.alpha * (1 - p_surv)
                elif self.ablation_mode == 'no_per_param_clipping':
                    self.clipping_weights[p] = torch.ones_like(p_surv)
                elif self.ablation_mode == 'binary_masking':
                    self.clipping_weights[p] = (p_surv > 0.5).float() * 0.9 + 0.1
                elif self.ablation_mode == 'fixed_survival':
                    # Use random fixed survival probabilities
                    torch.manual_seed(42)
                    self.clipping_weights[p] = torch.rand_like(p_surv) * 0.5 + 0.5
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.update_grad_ema()
        self.compute_survival_probabilities()
        
        for p in self.params:
            if p.grad is not None:
                # Clip gradients
                w = self.clipping_weights[p]
                per_param_clip = self.max_grad_norm * w
                
                grad_norm = p.grad.norm()
                clip_factor = torch.clamp_max(per_param_clip / (grad_norm + 1e-8), 1.0)
                clipped_grad = p.grad * clip_factor
                
                # Add noise
                survival_prob = self.survival_probs[p]
                if self.ablation_mode == 'no_compression_aware_noise':
                    noise_scale = torch.ones_like(survival_prob)
                else:
                    noise_scale = (1 - survival_prob) ** self.beta
                
                noise_std = self.noise_multiplier * self.max_grad_norm * noise_scale
                noise = torch.randn_like(p.grad) * noise_std / self.expected_batch_size ** 0.5
                
                p.grad = clipped_grad + noise
        
        self.optimizer.step()


def train_ablation(model, trainloader, testloader, device, ablation_mode, target_epsilon=3.0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ModuleValidator.fix(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=trainloader,
        target_epsilon=target_epsilon, target_delta=1e-5, epochs=30, max_grad_norm=1.0
    )
    
    ablation_opt = AblationOptimizer(
        optimizer=optimizer.original_optimizer,
        noise_multiplier=optimizer.noise_multiplier,
        max_grad_norm=1.0,
        expected_batch_size=optimizer.expected_batch_size,
        target_sparsity=0.7,
        alpha=0.1,
        beta=0.5,
        ablation_mode=ablation_mode
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(ablation_opt.optimizer, milestones=[10, 20], gamma=0.9)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epsilon': []}
    start_time = time.time()
    
    for epoch in range(30):
        ablation_opt.set_epoch(epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(trainloader, desc=f'Epoch {epoch+1}/30'):
            inputs, targets = inputs.to(device), targets.to(device)
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
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        
        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epsilon'].append(epsilon)
        
        print(f'Epoch {epoch+1}/30: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, ε: {epsilon:.2f}')
    
    runtime = time.time() - start_time
    return model, history, runtime


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_mode', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, testloader = get_data_loaders('cifar10', batch_size=256)
    model = ResNet18(num_classes=10)
    
    model, history, runtime = train_ablation(
        model, trainloader, testloader, device, 
        ablation_mode=args.ablation_mode, seed=args.seed
    )
    
    os.makedirs('exp/ablations', exist_ok=True)
    results = {
        'experiment': f'ablation_{args.ablation_mode}',
        'ablation_mode': args.ablation_mode,
        'seed': args.seed,
        'final_test_acc': history['test_acc'][-1],
        'final_epsilon': history['epsilon'][-1],
        'runtime_minutes': runtime / 60,
        'history': history
    }
    
    results_path = f'exp/ablations/results_{args.ablation_mode}_seed{args.seed}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')
'''
    
    with open('run_ablation.py', 'w') as f:
        f.write(ablation_script)
    
    # Run ablations
    ablation_modes = ['no_per_param_clipping', 'no_compression_aware_noise', 'binary_masking', 'fixed_survival']
    
    for mode in ablation_modes:
        for seed in SEEDS:
            print(f"\n--- Running ablation: {mode}, seed {seed} ---")
            cmd = ['python', 'run_ablation.py', '--ablation_mode', mode, '--seed', str(seed)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error: {result.stderr[-500:]}")
            else:
                print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)
    
    print("\nAblations complete!")


def evaluate_all_with_compression():
    """Evaluate all trained models with compression."""
    print("=" * 80)
    print("Evaluating All Models with Compression")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, testloader = get_data_loaders('cifar10', batch_size=256)
    
    results = {
        'standard_dp': [],
        'pre_pruning': [],
        'adadpigu': [],
        'phca_full': [],
        'ablations': {}
    }
    
    # Evaluate standard DP models
    print("\nEvaluating Standard DP models...")
    for seed in SEEDS:
        model_path = f'exp/baseline_standard/model_cifar10_resnet18_eps{EPSILON}_seed{seed}.pt'
        if os.path.exists(model_path):
            model = ResNet18(num_classes=10).to(device)
            model.load_state_dict(torch.load(model_path))
            
            # Before compression
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            acc_before = 100. * correct / total
            
            # After compression
            acc_after, actual_sparsity = evaluate_with_pruning(model, testloader, device, SPARSITY)
            
            results['standard_dp'].append({
                'seed': seed,
                'accuracy_before': acc_before,
                'accuracy_after': acc_after,
                'sparsity': actual_sparsity
            })
            print(f"  Seed {seed}: {acc_before:.2f}% -> {acc_after:.2f}% (sparsity: {actual_sparsity:.2%})")
    
    # Evaluate PHCA models
    print("\nEvaluating PHCA-DP-SGD models...")
    for seed in SEEDS:
        model_path = f'exp/phca_full/model_cifar10_resnet18_eps{EPSILON}_sparsity{SPARSITY}_seed{seed}.pt'
        if os.path.exists(model_path):
            model = ResNet18(num_classes=10).to(device)
            model.load_state_dict(torch.load(model_path))
            
            # Before compression
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            acc_before = 100. * correct / total
            
            # After compression
            acc_after, actual_sparsity = evaluate_with_pruning(model, testloader, device, SPARSITY)
            
            results['phca_full'].append({
                'seed': seed,
                'accuracy_before': acc_before,
                'accuracy_after': acc_after,
                'sparsity': actual_sparsity
            })
            print(f"  Seed {seed}: {acc_before:.2f}% -> {acc_after:.2f}% (sparsity: {actual_sparsity:.2%})")
    
    # Evaluate pre-pruning models
    print("\nEvaluating Pre-Pruning models...")
    for seed in SEEDS:
        result_path = f'exp/baseline_prepruning/results_cifar10_eps{EPSILON}_sparsity{SPARSITY}_seed{seed}.json'
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
            results['pre_pruning'].append({
                'seed': seed,
                'accuracy': data['final_test_acc'],
                'epsilon': data['final_epsilon']
            })
            print(f"  Seed {seed}: {data['final_test_acc']:.2f}%")
    
    # Evaluate AdaDPIGU models
    print("\nEvaluating AdaDPIGU models...")
    for seed in SEEDS:
        result_path = f'exp/baseline_adadpigu/results_cifar10_eps{EPSILON}_seed{seed}.json'
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
            results['adadpigu'].append({
                'seed': seed,
                'accuracy': data['final_test_acc'],
                'epsilon': data['final_epsilon']
            })
            print(f"  Seed {seed}: {data['final_test_acc']:.2f}%")
    
    # Evaluate ablations
    print("\nEvaluating Ablation models...")
    ablation_modes = ['no_per_param_clipping', 'no_compression_aware_noise', 'binary_masking', 'fixed_survival']
    for mode in ablation_modes:
        results['ablations'][mode] = []
        for seed in SEEDS:
            result_path = f'exp/ablations/results_{mode}_seed{seed}.json'
            if os.path.exists(result_path):
                with open(result_path) as f:
                    data = json.load(f)
                results['ablations'][mode].append({
                    'seed': seed,
                    'accuracy': data['final_test_acc'],
                    'epsilon': data['final_epsilon']
                })
                print(f"  {mode} seed {seed}: {data['final_test_acc']:.2f}%")
    
    # Save aggregated results
    os.makedirs('results', exist_ok=True)
    with open('results/compression_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nCompression evaluation complete!")
    return results


def generate_results_json():
    """Generate final results.json from all experiments."""
    print("=" * 80)
    print("Generating Final results.json")
    print("=" * 80)
    
    results = {
        'experiment_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seeds': SEEDS,
            'epsilon': EPSILON,
            'delta': DELTA,
            'sparsity': SPARSITY,
            'dataset': 'cifar10',
            'model': 'resnet18'
        },
        'baselines': {},
        'phca_full': {},
        'ablations': {},
        'compression_comparison': {}
    }
    
    # Load baseline_standard results
    baseline_accs = []
    for seed in SEEDS:
        path = f'exp/baseline_standard/results_cifar10_resnet18_eps{EPSILON}_seed{seed}.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            baseline_accs.append(data['final_test_acc'])
    
    if baseline_accs:
        results['baselines']['standard_dp'] = {
            'mean_accuracy': float(np.mean(baseline_accs)),
            'std_accuracy': float(np.std(baseline_accs)),
            'raw_accuracies': baseline_accs
        }
    
    # Load PHCA results
    phca_accs = []
    for seed in SEEDS:
        path = f'exp/phca_full/results_cifar10_resnet18_eps{EPSILON}_sparsity{SPARSITY}_seed{seed}.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            phca_accs.append(data['final_test_acc'])
    
    if phca_accs:
        results['phca_full'] = {
            'mean_accuracy': float(np.mean(phca_accs)),
            'std_accuracy': float(np.std(phca_accs)),
            'raw_accuracies': phca_accs
        }
    
    # Load ablation results
    ablation_modes = ['no_per_param_clipping', 'no_compression_aware_noise', 'binary_masking', 'fixed_survival']
    for mode in ablation_modes:
        accs = []
        for seed in SEEDS:
            path = f'exp/ablations/results_{mode}_seed{seed}.json'
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                accs.append(data['final_test_acc'])
        
        if accs:
            results['ablations'][mode] = {
                'mean_accuracy': float(np.mean(accs)),
                'std_accuracy': float(np.std(accs)),
                'raw_accuracies': accs
            }
    
    # Load compression results
    if os.path.exists('results/compression_results.json'):
        with open('results/compression_results.json') as f:
            comp_results = json.load(f)
        results['compression_comparison'] = comp_results
    
    # Save final results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nFinal results.json generated!")
    print(json.dumps(results, indent=2))
    return results


def main():
    """Main entry point."""
    start_time = time.time()
    
    print("=" * 80)
    print("PHCA-DP-SGD Experiment Suite")
    print(f"Running with seeds: {SEEDS}")
    print(f"Target epsilon: {EPSILON}, Target sparsity: {SPARSITY}")
    print("=" * 80)
    
    # Run all experiments
    run_standard_dp_baseline()
    run_prepruning_baseline()
    run_adadpigu_baseline()
    run_phca_full()
    run_ablations()
    
    # Evaluate and aggregate results
    evaluate_all_with_compression()
    generate_results_json()
    
    elapsed = time.time() - start_time
    print(f"\n\nAll experiments completed in {elapsed/3600:.2f} hours")


if __name__ == '__main__':
    main()
