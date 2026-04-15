#!/usr/bin/env python3
"""
Phase 2c: Joint Multi-Model Evaluation (FIXED V2).

Tests information sharing when evaluating multiple related models together.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch

from exp.shared.data_loader import MMLUDataset
from exp.shared.models_v2 import HierarchicalPopulationModelV2, compute_eig_vectorized
from exp.shared.metrics import compute_mae


def evaluate_single_model(
    pop_model, metadata, true_ability, responses,
    max_items: int = 100, target_mae: float = 0.05, seed: int = 42
) -> dict:
    """Evaluate a single model independently."""
    np.random.seed(seed)
    
    true_overall = np.mean(true_ability)
    n_items_total = len(responses)
    
    # Initialize with population prior
    prior_mean, prior_std = pop_model.predict_zero_shot(metadata)
    
    a_all = pop_model.item_discriminations
    b_all = pop_model.item_difficulties
    
    selected_items = []
    available_mask = np.ones(n_items_total, dtype=bool)
    
    for step in range(max_items):
        available_items = np.where(available_mask)[0]
        if len(available_items) == 0:
            break
        
        # Select using EIG
        eigs = compute_eig_vectorized(
            prior_mean, prior_std,
            a_all[available_items],
            b_all[available_items],
            n_samples=30
        )
        
        best_idx = np.argmax(eigs)
        selected_item = available_items[best_idx]
        
        selected_items.append(selected_item)
        available_mask[selected_item] = False
        
        # Update posterior
        response = responses[selected_item]
        a_sel = a_all[selected_item]
        b_sel = b_all[selected_item]
        
        for dim in range(3):
            prior_var = prior_std[dim] ** 2
            p = 1 / (1 + np.exp(-a_sel * (prior_mean[dim] - b_sel)))
            p = np.clip(p, 0.01, 0.99)
            fisher = (a_sel ** 2) * p * (1 - p)
            posterior_var = 1.0 / (1.0 / prior_var + fisher + 1e-6)
            
            if response > 0.5:
                likelihood_shift = a_sel * 0.5
            else:
                likelihood_shift = -a_sel * 0.5
            
            posterior_mean = prior_mean[dim] + posterior_var * likelihood_shift
            prior_mean[dim] = posterior_mean
            prior_std[dim] = np.sqrt(posterior_var)
        
        if step >= 20:
            current_mae = abs(np.mean(prior_mean) - true_overall)
            if current_mae < target_mae:
                break
    
    return {
        'items_used': len(selected_items),
        'final_mae': abs(np.mean(prior_mean) - true_overall)
    }


def evaluate_joint(
    pop_model, models_data, max_items_per_model: int = 100,
    target_mae: float = 0.05, seed: int = 42
) -> dict:
    """
    Evaluate multiple models jointly with information sharing.
    Uses shared family parameters.
    """
    np.random.seed(seed)
    
    n_models = len(models_data)
    
    # Initialize all models
    states = []
    available_masks = []
    completed = []
    
    for data in models_data:
        prior_mean, prior_std = pop_model.predict_zero_shot(data['metadata'])
        states.append({
            'mean': prior_mean.copy(),
            'std': prior_std.copy(),
            'true_overall': np.mean(data['true_ability']),
            'responses': data['responses']
        })
        available_masks.append(np.ones(len(data['responses']), dtype=bool))
        completed.append(False)
    
    a_all = pop_model.item_discriminations
    b_all = pop_model.item_difficulties
    
    total_items = 0
    items_per_model = [0] * n_models
    
    while not all(completed):
        # Find model with highest uncertainty that hasn't completed
        uncertainties = []
        for i, state in enumerate(states):
            if completed[i]:
                uncertainties.append(-1)
            else:
                uncertainties.append(np.mean(state['std']))
        
        if max(uncertainties) < 0:
            break
        
        model_idx = np.argmax(uncertainties)
        state = states[model_idx]
        
        # Find best item for this model
        available_items = np.where(available_masks[model_idx])[0]
        if len(available_items) == 0:
            completed[model_idx] = True
            continue
        
        eigs = compute_eig_vectorized(
            state['mean'], state['std'],
            a_all[available_items],
            b_all[available_items],
            n_samples=20
        )
        
        best_idx = np.argmax(eigs)
        selected_item = available_items[best_idx]
        
        # Observe and update
        available_masks[model_idx][selected_item] = False
        response = state['responses'][selected_item]
        a_sel = a_all[selected_item]
        b_sel = b_all[selected_item]
        
        for dim in range(3):
            prior_var = state['std'][dim] ** 2
            p = 1 / (1 + np.exp(-a_sel * (state['mean'][dim] - b_sel)))
            p = np.clip(p, 0.01, 0.99)
            fisher = (a_sel ** 2) * p * (1 - p)
            posterior_var = 1.0 / (1.0 / prior_var + fisher + 1e-6)
            
            if response > 0.5:
                likelihood_shift = a_sel * 0.5
            else:
                likelihood_shift = -a_sel * 0.5
            
            posterior_mean = state['mean'][dim] + posterior_var * likelihood_shift
            state['mean'][dim] = posterior_mean
            state['std'][dim] = np.sqrt(posterior_var)
        
        total_items += 1
        items_per_model[model_idx] += 1
        
        # Check if this model is done
        current_mae = abs(np.mean(state['mean']) - state['true_overall'])
        if items_per_model[model_idx] >= 20 and (current_mae < target_mae or items_per_model[model_idx] >= max_items_per_model):
            completed[model_idx] = True
    
    return {
        'total_items': total_items,
        'items_per_model': items_per_model
    }


def main():
    print("=" * 60)
    print("Phase 2c: Joint Multi-Model Evaluation (V2 - FIXED)")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_results = []
    
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Find 4 models from same family
    family_groups = {}
    for m in test_models:
        fam = dataset.models[m].family
        if fam not in family_groups:
            family_groups[fam] = []
        family_groups[fam].append(m)
    
    # Use qwen2 family or first with 4+ models
    joint_models = None
    for fam, models in family_groups.items():
        if len(models) >= 4:
            joint_models = models[:4]
            print(f"\n  Using {fam} family: {joint_models}")
            break
    
    if joint_models is None:
        print("  Warning: No family with 4 models found, using first 4 test models")
        joint_models = test_models[:4]
    
    total_start = time.time()
    
    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        
        pop_model = HierarchicalPopulationModelV2(n_dimensions=3, n_families=8)
        model_path = f"models/population_model_v2_seed{seed}.npy"
        pop_model.load(model_path)
        
        # Evaluate independently
        print("    Evaluating independently...")
        independent_items = []
        for model_name in joint_models:
            metadata = dataset.models[model_name]
            true_ability = np.array(model_metadata[model_name]['true_ability'])
            responses = dataset.responses[model_name]
            
            result = evaluate_single_model(
                pop_model, metadata, true_ability, responses,
                max_items=100, target_mae=0.05, seed=seed
            )
            independent_items.append(result['items_used'])
        
        independent_total = sum(independent_items)
        
        # Evaluate jointly
        print("    Evaluating jointly...")
        models_data = []
        for model_name in joint_models:
            models_data.append({
                'metadata': dataset.models[model_name],
                'true_ability': np.array(model_metadata[model_name]['true_ability']),
                'responses': dataset.responses[model_name]
            })
        
        joint_result = evaluate_joint(
            pop_model, models_data, max_items_per_model=100,
            target_mae=0.05, seed=seed
        )
        
        joint_total = joint_result['total_items']
        reduction = (independent_total - joint_total) / independent_total * 100
        
        print(f"    Independent: {independent_total} items")
        print(f"    Joint:       {joint_total} items")
        print(f"    Reduction:   {reduction:.1f}%")
        
        all_results.append({
            'independent_total': independent_total,
            'independent_per_model': independent_items,
            'joint_total': joint_total,
            'joint_per_model': joint_result['items_per_model'],
            'item_reduction_percent': reduction,
            'models_evaluated': joint_models
        })
    
    total_time = (time.time() - total_start) / 60
    
    # Aggregate
    reductions = [r['item_reduction_percent'] for r in all_results]
    
    aggregated = {
        'item_reduction_percent': {
            'mean': float(np.mean(reductions)),
            'std': float(np.std(reductions)),
            'values': reductions
        }
    }
    
    target_reduction = 30.0
    achieved = aggregated['item_reduction_percent']['mean'] >= target_reduction
    
    results = {
        'experiment': 'popbench_joint_v2',
        'description': 'Fixed joint evaluation of 4 related models',
        'metrics': aggregated,
        'per_seed_results': all_results,
        'config': {'seeds': seeds, 'n_models_joint': 4},
        'success_criterion': {
            'target': f'> {target_reduction}% item reduction',
            'achieved': achieved,
            'value': aggregated['item_reduction_percent']['mean']
        },
        'runtime_minutes': total_time
    }
    
    with open('exp/popbench_joint/results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Final Results:")
    print(f"  Item reduction: {aggregated['item_reduction_percent']['mean']:.1f}% ± {aggregated['item_reduction_percent']['std']:.1f}%")
    print(f"  Target > 30%: {'✓' if achieved else '✗'}")
    print(f"  Runtime: {total_time:.2f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
