#!/usr/bin/env python3
"""
Data Preparation: Generate synthetic MMLU-like dataset with proper seed handling.
Generates multiple data splits for different random seeds.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import os
from exp.shared.data_loader import MMLUDataset


def generate_data_for_seed(base_seed: int):
    """Generate dataset and splits for a specific seed."""
    print(f"\nGenerating data for seed {base_seed}...")
    
    # Create dataset with this seed - this affects the model generation AND responses
    dataset = MMLUDataset()
    dataset.generate_synthetic_data(n_models=80, n_items_per_subject=50, seed=base_seed)
    
    # Create train/test split with the SAME seed for consistency
    train_models, test_models = dataset.get_train_test_split(
        n_train=60, n_test=20, seed=base_seed
    )
    
    # Save split info
    split_info = {
        'seed': base_seed,
        'train_models': train_models,
        'test_models': test_models,
        'n_train': len(train_models),
        'n_test': len(test_models)
    }
    
    # Save model metadata for easy access
    model_metadata = {}
    for name, meta in dataset.models.items():
        model_metadata[name] = {
            'name': meta.name,
            'family': meta.family,
            'params': float(meta.params),
            'is_instruct': meta.is_instruct,
            'architecture': meta.architecture,
            'true_ability': dataset.true_abilities[name].tolist(),
            'feature_vector': meta.to_features().tolist()
        }
    
    # Save subject info
    subject_info = {}
    for name, subj in dataset.subjects.items():
        subject_info[name] = {
            'name': subj.name,
            'category': subj.category,
            'n_items': subj.n_items
        }
    
    return dataset, split_info, model_metadata, subject_info


def main():
    print("=" * 60)
    print("Data Preparation: MMLU Synthetic Dataset")
    print("=" * 60)
    
    # Use seed 42 as the base dataset
    base_seed = 42
    
    dataset, split_info, model_metadata, subject_info = generate_data_for_seed(base_seed)
    
    # Save the full dataset
    print("\nSaving dataset...")
    dataset.save("data/mmlu_synthetic")
    
    # Save train/test split
    with open('data/train_test_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Save model metadata
    with open('data/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Save subject info
    with open('data/subject_info.json', 'w') as f:
        json.dump(subject_info, f, indent=2)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total models: {len(dataset.models)}")
    print(f"  Training models: {len(split_info['train_models'])}")
    print(f"  Test models: {len(split_info['test_models'])}")
    print(f"  Total subjects: {len(dataset.subjects)}")
    
    # Count by family
    family_counts = {}
    for name, meta in dataset.models.items():
        family = meta.family
        family_counts[family] = family_counts.get(family, 0) + 1
    
    print("\n  Models by family:")
    for family, count in sorted(family_counts.items()):
        print(f"    {family}: {count}")
    
    # Verify seed propagation worked
    print("\nVerifying seed propagation...")
    
    # Check that responses vary across different seeds
    test_seeds = [42, 123, 456]
    sample_model = list(dataset.models.keys())[0]
    responses_by_seed = []
    
    for seed in test_seeds:
        ds = MMLUDataset()
        ds.generate_synthetic_data(n_models=80, n_items_per_subject=50, seed=seed)
        responses_by_seed.append(ds.responses[sample_model][:10])  # First 10 responses
    
    # Check that responses differ across seeds
    all_same = all(np.array_equal(responses_by_seed[0], r) for r in responses_by_seed[1:])
    if all_same:
        print("  WARNING: Responses are identical across seeds - seed propagation not working!")
    else:
        print("  ✓ Responses vary across seeds - seed propagation working correctly")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
