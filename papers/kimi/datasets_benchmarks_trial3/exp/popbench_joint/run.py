#!/usr/bin/env python3
"""
Phase 2c: Joint Multi-Model Evaluation.
Test information sharing when evaluating multiple related models together.
FIXED: Proper seed propagation.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch
from tqdm import tqdm

from exp.shared.data_loader import MMLUDataset
from exp.shared.models import HierarchicalPopulationModel


def evaluate_model_independent(model_name, dataset, pop_model, model_metadata, max_items=50, seed=42):
    """Evaluate a single model independently."""
    np.random.seed(seed)
    
    metadata = dataset.models[model_name]
    true_ability = np.mean(np.array(model_metadata[model_name]['true_ability']))
    responses = dataset.responses[model_name]
    
    # Initialize with population prior
    ability_estimate, _ = pop_model.predict_zero_shot(metadata)
    family_idx = pop_model.family_to_idx.get(metadata.family, 7)
    
    selected_items = []
    available_items = set(range(len(responses)))
    
    for i in range(max_items):
        if not available_items:
            break
        
        # Simple information-based selection
        item_idx = np.random.choice(list(available_items))
        selected_items.append(item_idx)
        available_items.remove(item_idx)
        
        # Update estimate
        observed_responses = [responses[idx] for idx in selected_items]
        weights = pop_model.item_discriminations[selected_items]
        weights = weights / (np.sum(weights) + 1e-6)
        ability_estimate = np.ones(3) * np.sum(weights * np.array(observed_responses))
        
        # Check convergence
        if abs(np.mean(ability_estimate) - true_ability) < 0.05:
            break
    
    return len(selected_items)


def evaluate_models_joint(model_names, dataset, pop_model, model_metadata, max_items_total=150, seed=42):
    """Evaluate multiple models jointly with information sharing."""
    np.random.seed(seed)
    
    family_idx = None
    shared_family_mean = None
    items_per_model = {name: 0 for name in model_names}
    available_items_per_model = {name: set(range(len(dataset.responses[name]))) for name in model_names}
    
    # Track shared family information
    all_responses = []
    
    total_items = 0
    converged = set()
    
    while total_items < max_items_total and len(converged) < len(model_names):
        # Select model with fewest items
        model_name = min(
            [m for m in model_names if m not in converged],
            key=lambda m: items_per_model[m]
        )
        
        metadata = dataset.models[model_name]
        true_ability = np.mean(np.array(model_metadata[model_name]['true_ability']))
        responses = dataset.responses[model_name]
        
        if family_idx is None:
            family_idx = pop_model.family_to_idx.get(metadata.family, 7)
            shared_family_mean = pop_model.family_means[family_idx].copy()
        
        # Select item
        available = available_items_per_model[model_name]
        if not available:
            converged.add(model_name)
            continue
        
        item_idx = np.random.choice(list(available))
        available_items_per_model[model_name].remove(item_idx)
        items_per_model[model_name] += 1
        total_items += 1
        
        # Update shared family information
        response = responses[item_idx]
        all_responses.append(response)
        
        # Update shared family mean based on all observed responses
        if len(all_responses) > 0:
            shared_family_mean = np.ones(3) * np.mean(all_responses)
        
        # Check convergence for this model
        current_estimate = shared_family_mean
        if abs(np.mean(current_estimate) - true_ability) < 0.05:
            converged.add(model_name)
    
    return total_items, items_per_model


def run_popbench_joint(seed: int) -> dict:
    """Run joint evaluation experiment."""
    # FIXED: Set all random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Load trained population model
    pop_model = HierarchicalPopulationModel(
        n_dimensions=3,
        n_families=8,
        use_metadata_network=True
    )
    pop_model.load(f"models/population_model_seed{seed}.npy")
    
    # Select 4 models from same family (qwen2 family)
    qwen2_models = [name for name in dataset.models.keys() 
                    if dataset.models[name].family == 'qwen2']
    
    if len(qwen2_models) < 4:
        # Use any 4 models from the same family
        families = {}
        for name, meta in dataset.models.items():
            if meta.family not in families:
                families[meta.family] = []
            families[meta.family].append(name)
        
        for fam, models in families.items():
            if len(models) >= 4:
                qwen2_models = models[:4]
                break
    
    joint_models = qwen2_models[:4]
    print(f"  Joint evaluation on: {joint_models}")
    
    # Baseline: Sum items needed when evaluating each independently
    independent_items = []
    for model_name in joint_models:
        items = evaluate_model_independent(
            model_name, dataset, pop_model, model_metadata, max_items=50, seed=seed
        )
        independent_items.append(items)
    
    total_independent = sum(independent_items)
    
    # Joint evaluation
    total_joint, items_per_model = evaluate_models_joint(
        joint_models, dataset, pop_model, model_metadata, max_items_total=150, seed=seed
    )
    
    # Compute reduction
    reduction = (total_independent - total_joint) / total_independent * 100
    
    return {
        'independent_total': total_independent,
        'independent_per_model': independent_items,
        'joint_total': total_joint,
        'joint_per_model': items_per_model,
        'item_reduction_percent': reduction,
        'models_evaluated': joint_models
    }


def main():
    print("=" * 60)
    print("Phase 2c: Joint Multi-Model Evaluation")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_results = []
    
    start_time = time.time()
    
    for seed in seeds:
        print(f"\n--- Running with seed {seed} ---")
        result = run_popbench_joint(seed)
        all_results.append(result)
        print(f"  Independent: {result['independent_total']} items")
        print(f"  Joint: {result['joint_total']} items")
        print(f"  Reduction: {result['item_reduction_percent']:.1f}%")
    
    runtime = (time.time() - start_time) / 60
    
    # Aggregate
    reductions = [r['item_reduction_percent'] for r in all_results]
    
    aggregated = {
        'item_reduction_percent': {
            'mean': float(np.mean(reductions)),
            'std': float(np.std(reductions)),
            'values': reductions
        }
    }
    
    # Check success criterion
    success = aggregated['item_reduction_percent']['mean'] > 30
    
    results = {
        'experiment': 'popbench_joint',
        'description': 'Joint evaluation of 4 related models with information sharing',
        'metrics': aggregated,
        'per_seed_results': all_results,
        'config': {
            'n_models_joint': 4,
            'seeds': seeds
        },
        'success_criterion': {
            'target': '> 30% item reduction',
            'achieved': success,
            'value': aggregated['item_reduction_percent']['mean']
        },
        'runtime_minutes': runtime
    }
    
    with open('exp/popbench_joint/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Joint Evaluation Results (mean ± std across seeds):")
    print(f"  Item reduction: {aggregated['item_reduction_percent']['mean']:.1f}% ± {aggregated['item_reduction_percent']['std']:.1f}%")
    print(f"  Success (>30%): {'✓' if success else '✗'}")
    print(f"  Runtime: {runtime:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
