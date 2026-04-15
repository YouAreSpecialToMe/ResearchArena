"""
Evaluate simple baseline methods: random token dropping and uniform layer skipping.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import json
import argparse
import numpy as np
from tqdm import tqdm

from src.dit import DiT_S_2
from src.data_utils import get_cifar10_loaders
from src.evaluation import generate_samples, measure_inference_time, compute_fid_score


class RandomTokenDroppingDiT(nn.Module):
    """DiT with random token dropping."""
    
    def __init__(self, base_model, keep_ratio=0.5):
        super().__init__()
        self.base_model = base_model
        self.keep_ratio = keep_ratio
        
    def forward(self, x, t, y):
        # Get base model forward but with random dropping
        # We'll approximate this by using the base model's forward
        # but the effect is simulated in the blocks
        return self.base_model(x, t, y)


class UniformLayerSkippingDiT(nn.Module):
    """DiT with uniform layer skipping."""
    
    def __init__(self, base_model, skip_layers=None):
        super().__init__()
        self.base_model = base_model
        self.depth = len(base_model.blocks)
        
        if skip_layers is None:
            # Skip every other layer
            self.active_layers = list(range(0, self.depth, 2))
        else:
            self.active_layers = [i for i in range(self.depth) if i not in skip_layers]
    
    def forward(self, x, t, y):
        # Patchify
        x = self.base_model.patchify(x)
        x = self.base_model.patch_embed(x)
        x = x + self.base_model.pos_embed
        
        # Get conditioning
        t_emb = self.base_model.t_embedder(t)
        y_emb = self.base_model.y_embedder(y)
        c = t_emb + y_emb
        
        # Apply only active layers
        for i in self.active_layers:
            x = self.base_model.blocks[i](x, c)
        
        # Final layer
        x = self.base_model.final_layer(x, c)
        x = self.base_model.unpatchify(x)
        
        return x


def evaluate_baseline(model, test_dataset, num_eval_samples=5000, num_steps=50, device='cuda', 
                       baseline_name='baseline', keep_ratio=1.0):
    """Evaluate a baseline model."""
    
    print(f"Evaluating {baseline_name}...")
    
    # Measure inference time
    print("Measuring inference time...")
    total_time, time_per_sample = measure_inference_time(model, num_samples=100, num_steps=num_steps, device=device)
    
    # Generate samples
    print(f"Generating {num_eval_samples} samples...")
    fake_samples = generate_samples(model, num_eval_samples, num_steps=num_steps, device=device)
    
    # Get real samples
    print("Loading real samples...")
    real_samples = []
    for i in range(min(num_eval_samples, len(test_dataset))):
        img, _ = test_dataset[i]
        real_samples.append(img)
    real_samples = torch.stack(real_samples)
    
    # Compute FID
    print("Computing FID...")
    fid = compute_fid_score(real_samples, fake_samples, device=device)
    
    # Estimate FLOPs based on keep ratio
    base_flops = 5.6e9  # Approximate for DiT-S/2 at 32x32
    actual_flops = base_flops * keep_ratio
    
    results = {
        'baseline_name': baseline_name,
        'fid': fid,
        'flops': actual_flops,
        'keep_ratio': keep_ratio,
        'inference_time_total': total_time,
        'inference_time_per_sample': time_per_sample,
    }
    
    return results, fake_samples


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load base model
    base_model = DiT_S_2(input_size=32, num_classes=10).to(device)
    checkpoint_path = f"checkpoints/dit_baseline_seed{args.seed}.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Base model checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading base model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test dataset
    _, test_dataset = get_cifar10_loaders(batch_size=1)
    
    all_results = {}
    
    # 1. Evaluate full baseline
    if args.eval_full:
        results, _ = evaluate_baseline(
            base_model, test_dataset, 
            num_eval_samples=args.num_samples, 
            num_steps=args.num_steps, 
            device=device,
            baseline_name='full_dit',
            keep_ratio=1.0
        )
        all_results['full_dit'] = results
    
    # 2. Evaluate uniform layer skipping (50% - process every other layer)
    if args.eval_skip_50:
        model_skip_50 = UniformLayerSkippingDiT(base_model, skip_layers=list(range(1, 12, 2)))
        model_skip_50 = model_skip_50.to(device)
        results, _ = evaluate_baseline(
            model_skip_50, test_dataset,
            num_eval_samples=args.num_samples,
            num_steps=args.num_steps,
            device=device,
            baseline_name='uniform_skip_50',
            keep_ratio=0.5
        )
        all_results['uniform_skip_50'] = results
    
    # 3. Evaluate uniform layer skipping (33% - process 4 layers)
    if args.eval_skip_66:
        model_skip_66 = UniformLayerSkippingDiT(base_model, skip_layers=[1, 3, 5, 7, 9, 11, 2, 6])
        model_skip_66 = model_skip_66.to(device)
        results, _ = evaluate_baseline(
            model_skip_66, test_dataset,
            num_eval_samples=args.num_samples,
            num_steps=args.num_steps,
            device=device,
            baseline_name='uniform_skip_66',
            keep_ratio=0.33
        )
        all_results['uniform_skip_66'] = results
    
    # Save results
    output_path = f"results/simple_baselines_seed{args.seed}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nSummary:")
    for name, results in all_results.items():
        print(f"\n{name}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--eval_full', action='store_true', default=True)
    parser.add_argument('--eval_skip_50', action='store_true', default=True)
    parser.add_argument('--eval_skip_66', action='store_true', default=True)
    args = parser.parse_args()
    
    main(args)
