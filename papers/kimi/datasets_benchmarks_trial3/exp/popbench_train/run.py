#!/usr/bin/env python3
"""
Phase 1: Train Hierarchical Population Model.
Learn family distributions and covariances from historical data.
FIXED: Proper seed handling and metadata network training.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch

from exp.shared.data_loader import MMLUDataset
from exp.shared.models import HierarchicalPopulationModel


def train_with_seed(seed: int):
    """Train hierarchical model with a specific random seed."""
    print(f"\n{'='*60}")
    print(f"Training with seed {seed}")
    print(f"{'='*60}")
    
    # CRITICAL: Set seeds for reproducibility AND variance across seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic for this seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load dataset
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    train_models = split['train_models']
    
    print(f"\nTraining on {len(train_models)} models")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize and train hierarchical model
    pop_model = HierarchicalPopulationModel(
        n_dimensions=3,
        n_families=8,
        use_metadata_network=True
    )
    
    print("\nTraining hierarchical model with metadata network...")
    history = pop_model.fit(
        dataset=dataset,
        train_models=train_models,
        n_steps=3000,
        lr=0.01,
        batch_size=10,
        seed=seed
    )
    
    # Save trained model with seed in filename
    model_path = f"models/population_model_seed{seed}.npy"
    pop_model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Print learned family statistics
    print("\nLearned Family Means:")
    family_names = ['llama2', 'llama3', 'qwen2', 'qwen3', 'gemma', 'mistral', 'phi', 'other']
    for i, name in enumerate(family_names):
        mean = pop_model.family_means[i]
        std = pop_model.family_stds[i]
        print(f"  {name}: mean=[{mean[0]:.3f},{mean[1]:.3f},{mean[2]:.3f}], "
              f"std=[{std[0]:.3f},{std[1]:.3f},{std[2]:.3f}]")
    
    # Verify metadata network is working
    if pop_model.metadata_net is not None:
        print("\nMetadata network predictions (sample):")
        with torch.no_grad():
            pop_model.metadata_net.eval()
            # Test with a sample input
            sample_input = torch.randn(1, 11)
            sample_output = pop_model.metadata_net(sample_input)
            print(f"  Sample output: {sample_output.numpy()[0]}")
    
    return pop_model, history


def main():
    print("=" * 60)
    print("Phase 1: Train Hierarchical Population Model (FIXED)")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    start_time = time.time()
    
    # Train with multiple seeds
    all_results = []
    for seed in seeds:
        seed_start = time.time()
        pop_model, history = train_with_seed(seed)
        seed_time = (time.time() - seed_start) / 60
        
        results = {
            'seed': seed,
            'final_loss': float(history['losses'][-1]) if history['losses'] else None,
            'final_nll': float(history['nlls'][-1]) if 'nlls' in history else None,
            'runtime_minutes': seed_time,
            'family_means': pop_model.family_means.tolist(),
            'family_stds': pop_model.family_stds.tolist()
        }
        all_results.append(results)
    
    total_time = (time.time() - start_time) / 60
    
    # Verify losses are different across seeds
    losses = [r['final_loss'] for r in all_results]
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"  Seed 42 final loss: {losses[0]:.2f}")
    print(f"  Seed 123 final loss: {losses[1]:.2f}")
    print(f"  Seed 456 final loss: {losses[2]:.2f}")
    print(f"  Loss variance: {np.std(losses):.4f}")
    if np.std(losses) < 0.01:
        print("  WARNING: Losses are nearly identical across seeds!")
    else:
        print("  ✓ Losses show proper variance across seeds")
    print(f"{'='*60}")
    
    # Save aggregate results
    final_results = {
        'experiment': 'popbench_train',
        'description': 'Train hierarchical population model with variational inference',
        'config': {
            'seeds': seeds,
            'n_train_models': 60,
            'n_dimensions': 3,
            'n_families': 8,
            'n_steps': 3000,
            'learning_rate': 0.01,
            'batch_size': 10,
            'use_metadata_network': True
        },
        'training_results': all_results,
        'loss_variance_across_seeds': float(np.std(losses)),
        'total_runtime_minutes': total_time
    }
    
    with open('exp/popbench_train/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete across {len(seeds)} seeds in {total_time:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
