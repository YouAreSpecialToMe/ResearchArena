#!/usr/bin/env python3
"""
Phase 1: Train Hierarchical Population Model (FIXED V2).

Key fixes:
- Proper loss weighting (KL terms scaled to match NLL)
- Better initialization
- More training steps
- Verify gradients are flowing
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch

from exp.shared.data_loader import MMLUDataset
from exp.shared.models_v2 import HierarchicalPopulationModelV2


def train_with_seed(seed: int):
    """Train with fixed model."""
    print(f"\n{'='*60}")
    print(f"Training with seed {seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load dataset
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    train_models = split['train_models']
    
    print(f"\nTraining on {len(train_models)} models")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize model
    pop_model = HierarchicalPopulationModelV2(
        n_dimensions=3,
        n_families=8,
        use_metadata_network=True
    )
    
    print("\nTraining hierarchical model...")
    start_train = time.time()
    
    history = pop_model.fit(
        dataset=dataset,
        train_models=train_models,
        n_steps=5000,
        lr=0.01,
        batch_size=15,
        seed=seed,
        verbose=True
    )
    
    train_time = time.time() - start_train
    
    # Save model
    model_path = f"models/population_model_v2_seed{seed}.npy"
    pop_model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Report learned parameters
    print("\nLearned Family Statistics:")
    for i in range(8):
        name = pop_model.idx_to_family.get(i, f'family_{i}')
        mean = pop_model.family_means[i]
        std = pop_model.family_stds[i]
        print(f"  {name:10s}: mean=[{mean[0]:6.3f},{mean[1]:6.3f},{mean[2]:6.3f}], "
              f"std=[{std[0]:5.3f},{std[1]:5.3f},{std[2]:5.3f}]")
    
    # Check if family means are diverse (indicator of learning)
    family_mean_std = pop_model.family_means.std(axis=0).mean()
    print(f"\n  Family mean diversity (std across families): {family_mean_std:.4f}")
    if family_mean_std < 0.1:
        print("  WARNING: Family means are too similar - model may not be learning!")
    else:
        print("  ✓ Family means show good diversity")
    
    # Test metadata network
    if pop_model.metadata_net is not None:
        print("\n  Testing metadata network:")
        with torch.no_grad():
            pop_model.metadata_net.eval()
            # Test different family inputs
            families = ['llama2', 'llama3', 'qwen2', 'gemma', 'mistral', 'phi', 'other']
            for fam in families[:4]:
                from exp.shared.data_loader import ModelMetadata
                meta = ModelMetadata(name=f"{fam}-test", family=fam, params=7, is_instruct=False, architecture='base')
                feat = torch.tensor(meta.to_features()).unsqueeze(0)
                pred = pop_model.metadata_net(feat).numpy()[0]
                print(f"    {fam}: offset=[{pred[0]:6.3f},{pred[1]:6.3f},{pred[2]:6.3f}]")
    
    return pop_model, history, train_time


def main():
    print("=" * 60)
    print("Phase 1: Train Hierarchical Population Model (V2 - FIXED)")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_results = []
    
    total_start = time.time()
    
    for seed in seeds:
        pop_model, history, train_time = train_with_seed(seed)
        
        results = {
            'seed': seed,
            'final_loss': float(history['losses'][-1]),
            'final_nll': float(history['nlls'][-1]),
            'train_time_seconds': train_time,
            'family_means': pop_model.family_means.tolist(),
            'family_stds': pop_model.family_stds.tolist(),
            'family_mean_diversity': float(pop_model.family_means.std(axis=0).mean())
        }
        all_results.append(results)
    
    total_time = (time.time() - total_start) / 60
    
    # Verify diversity across seeds
    print(f"\n{'='*60}")
    print("Training Summary Across Seeds:")
    print(f"{'='*60}")
    
    for r in all_results:
        print(f"  Seed {r['seed']}: diversity={r['family_mean_diversity']:.4f}, "
              f"NLL={r['final_nll']:.1f}, time={r['train_time_seconds']:.1f}s")
    
    diversities = [r['family_mean_diversity'] for r in all_results]
    print(f"\n  Diversity across seeds: mean={np.mean(diversities):.4f}, std={np.std(diversities):.4f}")
    
    if np.mean(diversities) < 0.1:
        print("  WARNING: Low diversity - model may not be learning population structure!")
    else:
        print("  ✓ Good population structure learned")
    
    print(f"\n  Total training time: {total_time:.2f} minutes")
    
    # Save results
    final_results = {
        'experiment': 'popbench_train_v2',
        'description': 'Fixed hierarchical population model training',
        'config': {
            'seeds': seeds,
            'n_steps': 5000,
            'learning_rate': 0.01,
            'batch_size': 15,
            'n_train_models': 60
        },
        'results': all_results,
        'total_runtime_minutes': total_time
    }
    
    with open('exp/popbench_train/results_v2.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
