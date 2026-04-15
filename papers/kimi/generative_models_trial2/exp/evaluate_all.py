#!/usr/bin/env python3
"""
Evaluate all trained models and generate results.json with actual computed metrics.
"""
import json
import sys
import torch
from pathlib import Path
import numpy as np

sys.path.insert(0, 'exp')
from shared.models import VelocityNetwork, WeightPredictorMLP
from shared.data_loader import get_dataloader, KITTI360Dataset
from shared.trainer import FlowMatchingTrainer, euler_sampling
from shared.metrics import compute_all_metrics
from shared.utils import set_seed, get_device


def evaluate_model(model_path, config, val_dataset, num_samples=200):
    """Evaluate a trained model."""
    device = get_device()
    
    # Load model
    model = VelocityNetwork(
        point_dim=3,
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        use_distance_conditioning=config.get('use_distance_conditioning', False),
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Generate samples
    print(f"  Generating {num_samples} samples...")
    generated_samples = []
    generated_dists = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Create realistic radial distance distribution
            N = 2048
            near = torch.rand(N // 2, device=device) * 0.25
            mid = 0.25 + torch.rand(N // 4, device=device) * 0.375
            far = 0.625 + torch.rand(N - N // 2 - N // 4, device=device) * 0.375
            r_dist = torch.cat([near, mid, far]).unsqueeze(0)
            
            shape = (1, N, 3)
            sample = euler_sampling(
                model, shape, num_steps=50,
                radial_dist=r_dist, device=device
            )
            
            generated_samples.append(sample[0].cpu())
            generated_dists.append(r_dist[0].cpu())
    
    generated_samples = torch.stack(generated_samples)
    generated_dists = torch.stack(generated_dists)
    
    # Get real samples
    real_samples = []
    real_dists = []
    
    for i in range(min(num_samples, len(val_dataset))):
        data = val_dataset[i]
        real_samples.append(data['points'])
        real_dists.append(data['radial_dist'])
    
    real_samples = torch.stack(real_samples)
    real_dists = torch.stack(real_dists)
    
    # Compute metrics
    print(f"  Computing metrics...")
    metrics = compute_all_metrics(
        generated_samples, real_samples,
        generated_dists, real_dists
    )
    
    return metrics


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Evaluating All Trained Models")
    print("=" * 60)
    
    # Load validation dataset
    data_dir = "data/kitti360"
    val_dataset = KITTI360Dataset(data_dir, split='val')
    
    print(f"Validation dataset: {len(val_dataset)} scans")
    
    # Find all trained models
    outputs_dir = Path("outputs")
    results = {}
    
    for model_dir in outputs_dir.glob("*_seed*"):
        model_path = model_dir / "best_model.pt"
        result_path = model_dir / "results.json"
        
        if not model_path.exists():
            continue
        
        exp_name = model_dir.name
        print(f"\nEvaluating: {exp_name}")
        
        try:
            # Load config if results exist
            config = {}
            if result_path.exists():
                with open(result_path) as f:
                    existing = json.load(f)
                    config = existing.get('config', {})
            
            # Evaluate
            metrics = evaluate_model(model_path, config, val_dataset, num_samples=200)
            
            print(f"  Results:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.6f}")
            
            results[exp_name] = metrics
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    with open('evaluated_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluated {len(results)} models")
    print("Results saved to evaluated_results.json")


if __name__ == "__main__":
    main()
