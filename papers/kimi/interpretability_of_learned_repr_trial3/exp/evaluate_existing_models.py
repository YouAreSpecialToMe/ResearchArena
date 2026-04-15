"""
Evaluate existing trained models and generate results files.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import torch
import json
import os
import argparse

from exp.shared.models import TopKSAE, JumpReLUSAE, DenoisingSAE, RobustSAE
from exp.shared.metrics import evaluate_sae_model
from exp.shared.utils import set_seed, save_checkpoint

def load_model(model_type, checkpoint_path, device='cuda'):
    """Load a trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    d_model = checkpoint['model_state_dict']['W_enc'].shape[0]
    d_sae = checkpoint['model_state_dict']['W_enc'].shape[1]
    
    if model_type == 'topk':
        model = TopKSAE(d_model, d_sae, topk=32)
    elif model_type == 'jumprelu':
        model = JumpReLUSAE(d_model, d_sae)
    elif model_type == 'denoising':
        model = DenoisingSAE(d_model, d_sae, topk=32)
    elif model_type == 'robust':
        model = RobustSAE(d_model, d_sae, topk=32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def evaluate_model(model_type, seed, device='cuda'):
    """Evaluate a trained model and save results."""
    set_seed(seed)
    
    # Load data
    data = torch.load('data/activations_pythia70m_layer3.pt', weights_only=False)
    val_acts = data['val']
    
    # Map model type to folder name
    folder_map = {
        'topk': 'topk_baseline',
        'jumprelu': 'jumprelu_baseline',
        'denoising': 'denoising_baseline',
        'robust': 'robustsae_full',
        'robust_no_proxy': 'robustsae_no_proxy'
    }
    
    folder_name = folder_map[model_type]
    checkpoint_path = f'models/{folder_name}_seed{seed}_best.pt'
    
    print(f"Evaluating {model_type} seed {seed}...")
    
    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load model
    model, checkpoint = load_model(
        'robust' if model_type in ['robust', 'robust_no_proxy'] else model_type,
        checkpoint_path, device
    )
    
    # Evaluate
    metrics = evaluate_sae_model(model, val_acts, batch_size=2048, device=device)
    
    # Add metadata
    metrics['seed'] = seed
    metrics['config'] = {
        'model_type': model_type,
        'd_model': model.d_model,
        'd_sae': model.d_sae,
    }
    
    # Compute proxy scores for RobustSAE
    if model_type in ['robust', 'robust_no_proxy']:
        model.eval()
        with torch.no_grad():
            sample_size = min(1000, len(val_acts))
            sample = val_acts[:sample_size].to(device)
            proxy_scores = model.compute_proxy_scores(sample)
        
        metrics['proxy_scores'] = {
            'mean': proxy_scores.mean().item(),
            'std': proxy_scores.std().item(),
            'max': proxy_scores.max().item(),
            'min': proxy_scores.min().item(),
        }
    
    # Save results
    os.makedirs(f'exp/{folder_name}', exist_ok=True)
    results_path = f'exp/{folder_name}/results_seed{seed}.json'
    
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  FVU: {metrics['fvu']:.6f}, L0: {metrics['l0_sparsity']:.2f}, Dead: {metrics['dead_features_pct']:.2f}%")
    print(f"  Saved to {results_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['topk', 'jumprelu', 'denoising', 'robust', 'robust_no_proxy'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--all_seeds', action='store_true', help='Run all seeds 42,43,44')
    args = parser.parse_args()
    
    if args.all_seeds:
        for seed in [42, 43, 44]:
            evaluate_model(args.model_type, seed, args.device)
    else:
        evaluate_model(args.model_type, args.seed, args.device)

if __name__ == '__main__':
    main()
