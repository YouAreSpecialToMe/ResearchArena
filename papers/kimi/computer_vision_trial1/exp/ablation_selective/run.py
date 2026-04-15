"""
Ablation 2: Selective vs Uniform Prompt Application
Compare uncertainty-guided layer selection vs other strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from shared.models import ViTWithPrompts, UncertaintyEstimator
from shared.data_loader import get_corruption_dataloaders, get_domain_dataloaders, get_calibration_dataloader
from shared.metrics import MetricsTracker
from shared.utils import set_seed, save_results, compute_calibration_stats
import numpy as np


def evaluate_with_strategy(seed=42, strategy='selective', num_samples=300, batch_size=32, lr=5e-3):
    """
    Evaluate with different layer selection strategies.
    
    Strategies:
    - 'selective': DU-VPT uncertainty-guided selection
    - 'uniform': All 12 layers
    - 'random': Random selection (same avg number of layers as selective)
    - 'early': Fixed early layers (1-4)
    - 'deep': Fixed deep layers (9-12)
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Strategy: {strategy} (seed={seed})...")
    
    model = ViTWithPrompts(num_prompts=10, prompt_dim=768, num_layers=12)
    model = model.to(device)
    model.eval()
    
    # Setup uncertainty estimator for selective strategy
    uncertainty_est = None
    if strategy == 'selective':
        uncertainty_est = UncertaintyEstimator(num_layers=12, feature_dim=768)
        calib_loader = get_calibration_dataloader(num_samples=300, batch_size=batch_size)
        calib_stats = compute_calibration_stats(model, calib_loader, num_layers=12, device=device)
        uncertainty_est.set_calibration_stats(calib_stats)
    
    results = {}
    corruption_types = ['gaussian_noise', 'defocus_blur', 'brightness']
    dataloaders = get_corruption_dataloaders(corruption_types, severity=5, 
                                             batch_size=batch_size, num_samples=num_samples)
    
    total_layers_used = []
    
    for corruption, loader in dataloaders.items():
        tracker = MetricsTracker()
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Determine target layers based on strategy
            if strategy == 'selective':
                with torch.no_grad():
                    _ = model(x)
                    layer_features = model.get_layer_features()
                uncertainty = uncertainty_est.decompose_uncertainty(layer_features)
                shift_type, target_layers, prompt_type = uncertainty_est.diagnose_shift(uncertainty)
                
            elif strategy == 'uniform':
                target_layers = list(range(12))
                prompt_type = 'semantic'
                
            elif strategy == 'random':
                # Randomly select ~6 layers on average
                target_layers = [i for i in range(12) if np.random.random() > 0.5]
                if not target_layers:
                    target_layers = [0]
                prompt_type = 'semantic'
                
            elif strategy == 'early':
                target_layers = list(range(4))  # Layers 0-3
                prompt_type = 'semantic'
                
            elif strategy == 'deep':
                target_layers = list(range(8, 12))  # Layers 8-11
                prompt_type = 'semantic'
            
            else:
                target_layers = []
                prompt_type = 'semantic'
            
            total_layers_used.append(len(target_layers))
            
            if target_layers:
                prompt_params = model.get_prompt_params(target_layers)
                optimizer = torch.optim.Adam(prompt_params, lr=lr)
                
                model.train()
                optimizer.zero_grad()
                outputs = model(x, active_layers=target_layers, prompt_type=prompt_type)
                
                probs = F.softmax(outputs, dim=1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                
                entropy.backward()
                optimizer.step()
                model.eval()
            else:
                with torch.no_grad():
                    outputs = model(x, active_layers=[])
            
            tracker.update(outputs.detach(), y)
        
        metrics = tracker.compute()
        results[f'imagenet_c_{corruption}'] = metrics['top1_acc']
    
    results['imagenet_c_avg'] = np.mean([results[f'imagenet_c_{c}'] for c in corruption_types])
    results['avg_layers_used'] = np.mean(total_layers_used)
    results['adaptation_efficiency'] = results['avg_layers_used'] / 12 * 100
    results['seed'] = seed
    results['strategy'] = strategy
    
    return results


def main():
    seeds = [42, 123, 456]
    strategies = ['selective', 'uniform', 'random', 'early', 'deep']
    
    all_results = {s: [] for s in strategies}
    
    for strategy in strategies:
        for seed in seeds:
            result = evaluate_with_strategy(seed=seed, strategy=strategy, num_samples=300, batch_size=32)
            all_results[strategy].append(result)
    
    # Aggregate
    aggregated = {}
    for strategy in strategies:
        accs = [r['imagenet_c_avg'] for r in all_results[strategy]]
        efficiencies = [r['adaptation_efficiency'] for r in all_results[strategy]]
        aggregated[strategy] = {
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'efficiency_mean': np.mean(efficiencies),
            'efficiency_std': np.std(efficiencies)
        }
    
    aggregated['raw_results'] = all_results
    
    save_results(aggregated, 'exp/ablation_selective/results.json')
    print("\n=== Ablation: Selective vs Uniform Prompt Application ===")
    for strategy in strategies:
        acc = aggregated[strategy]['accuracy_mean']
        std = aggregated[strategy]['accuracy_std']
        eff = aggregated[strategy]['efficiency_mean']
        print(f"{strategy.capitalize():12s}: {acc:.2f} ± {std:.2f}% (efficiency: {eff:.1f}%)")
    
    # Key comparison
    selective_acc = aggregated['selective']['accuracy_mean']
    uniform_acc = aggregated['uniform']['accuracy_mean']
    improvement = selective_acc - uniform_acc
    print(f"\nKey finding: Selective vs Uniform improvement: {improvement:+.2f}%")


if __name__ == '__main__':
    main()
