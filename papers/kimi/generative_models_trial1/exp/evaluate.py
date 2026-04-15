"""
Evaluate trained models and compute metrics.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import argparse
import numpy as np

from src.dit import DiT_S_2
from src.flowrouter import FlowRouter_S_2
from src.data_utils import get_cifar10_loaders
from src.evaluation import evaluate_model


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    if args.model_type == 'baseline':
        model = DiT_S_2(input_size=32, num_classes=10).to(device)
        checkpoint_path = f"checkpoints/dit_baseline_seed{args.seed}.pt"
        is_flowrouter = False
    else:
        model = FlowRouter_S_2(input_size=32, num_classes=10, use_velocity=True).to(device)
        checkpoint_path = f"checkpoints/flowrouter_seed{args.seed}.pt"
        is_flowrouter = True
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test dataset
    _, test_dataset = get_cifar10_loaders(batch_size=1)
    
    # Evaluate
    results, samples = evaluate_model(
        model, test_dataset, 
        num_eval_samples=args.num_samples, 
        num_steps=args.num_steps, 
        device=device,
        is_flowrouter=is_flowrouter
    )
    
    # Save results
    results['model_type'] = args.model_type
    results['seed'] = args.seed
    
    output_path = f"results/{args.model_type}_eval_seed{args.seed}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"\nSummary:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['baseline', 'flowrouter'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--num_steps', type=int, default=50)
    args = parser.parse_args()
    
    main(args)
