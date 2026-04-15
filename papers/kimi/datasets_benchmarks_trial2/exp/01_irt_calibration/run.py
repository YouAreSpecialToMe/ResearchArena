#!/usr/bin/env python3
"""
Initial IRT calibration on base question pool.
Generates synthetic responses and calibrates item parameters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import matplotlib.pyplot as plt
from shared.irt_utils import (
    IRT2PL, generate_synthetic_responses, 
    save_item_parameters, load_item_parameters
)
from shared.data_loader import load_questions


def main():
    print("=" * 60)
    print("Initial IRT Calibration")
    print("=" * 60)
    
    # Load questions
    mmlu_questions = load_questions('data/mmlu_test.json')
    gsm8k_questions = load_questions('data/gsm8k_test.json')
    all_questions = mmlu_questions + gsm8k_questions
    
    n_items = len(all_questions)
    print(f"Total questions: {n_items} (MMLU: {len(mmlu_questions)}, GSM8K: {len(gsm8k_questions)})")
    
    # Generate synthetic responses from 20 "models" with varying abilities
    n_persons = 20
    print(f"\nGenerating synthetic responses from {n_persons} simulated models...")
    
    responses, true_theta, initial_params = generate_synthetic_responses(
        n_persons=n_persons,
        n_items=n_items,
        theta_range=(-2.5, 2.5),
        seed=42
    )
    
    print(f"Response matrix shape: {responses.shape}")
    print(f"Average accuracy: {np.mean(responses):.3f}")
    
    # Calibrate IRT parameters
    print("\nCalibrating IRT parameters using MML...")
    irt = IRT2PL(n_items, n_persons)
    calibration_result = irt.calibrate_items_mml(responses, max_iter=100, tolerance=1e-4)
    
    print(f"Calibration completed in {calibration_result['iterations']} iterations")
    
    # Compute fit statistics
    print("\nComputing item fit statistics...")
    infit, outfit = irt.compute_infit_outfit(responses)
    
    # Filter items based on quality criteria
    print("\nFiltering items based on quality criteria...")
    print("  - Discrimination a > 0.3")
    print("  - Infit between 0.5-1.5")
    print("  - Outfit between 0.5-1.5")
    
    valid_mask = (
        (irt.a > 0.3) & 
        (infit > 0.5) & (infit < 1.5) & 
        (outfit > 0.5) & (outfit < 1.5)
    )
    
    n_valid = np.sum(valid_mask)
    print(f"\nValid items: {n_valid}/{n_items} ({100*n_valid/n_items:.1f}%)")
    
    # Save calibrated parameters
    item_params = {
        'a': irt.a,
        'b': irt.b,
        'c': irt.c,
        'infit': infit,
        'outfit': outfit,
        'valid_mask': valid_mask
    }
    
    save_item_parameters(item_params, 'data/item_parameters_initial.json')
    print("Saved item parameters to data/item_parameters_initial.json")
    
    # Create difficulty band distribution
    print("\nCreating difficulty distribution...")
    difficulty_bands = np.arange(-3, 3.5, 0.5)
    band_counts, _ = np.histogram(irt.b[valid_mask], bins=difficulty_bands)
    
    print("Difficulty band distribution (valid items only):")
    for i in range(len(difficulty_bands)-1):
        print(f"  [{difficulty_bands[i]:.1f}, {difficulty_bands[i+1]:.1f}): {band_counts[i]} items")
    
    # Create visualization
    print("\nGenerating parameter distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Discrimination distribution
    axes[0, 0].hist(irt.a[valid_mask], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Discrimination (a)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Discrimination Parameters')
    axes[0, 0].axvline(x=1.0, color='red', linestyle='--', label='Typical value')
    axes[0, 0].legend()
    
    # Difficulty distribution
    axes[0, 1].hist(irt.b[valid_mask], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Difficulty (b)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Difficulty Parameters')
    axes[0, 1].axvline(x=0.0, color='red', linestyle='--', label='Average ability')
    axes[0, 1].legend()
    
    # Infit vs Outfit scatter
    axes[1, 0].scatter(infit[valid_mask], outfit[valid_mask], alpha=0.6)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Infit')
    axes[1, 0].set_ylabel('Outfit')
    axes[1, 0].set_title('Item Fit Statistics')
    axes[1, 0].set_xlim(0, 2)
    axes[1, 0].set_ylim(0, 2)
    
    # ICC for a few items
    theta_range = np.linspace(-4, 4, 100)
    for i in range(min(5, n_items)):
        if valid_mask[i]:
            probs = [irt.probability(t, irt.a[i], irt.b[i], irt.c[i]) for t in theta_range]
            axes[1, 1].plot(theta_range, probs, label=f'Item {i} (a={irt.a[i]:.2f})')
    axes[1, 1].set_xlabel('Ability (θ)')
    axes[1, 1].set_ylabel('P(correct)')
    axes[1, 1].set_title('Item Characteristic Curves (Sample Items)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/irt_calibration.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/irt_calibration.pdf', bbox_inches='tight')
    print("Saved figures/irt_calibration.png")
    
    # Save results summary
    results = {
        'experiment': 'initial_irt_calibration',
        'n_items': int(n_items),
        'n_persons': n_persons,
        'n_valid_items': int(n_valid),
        'parameter_statistics': {
            'a_mean': float(np.mean(irt.a[valid_mask])),
            'a_std': float(np.std(irt.a[valid_mask])),
            'b_mean': float(np.mean(irt.b[valid_mask])),
            'b_std': float(np.std(irt.b[valid_mask])),
            'infit_mean': float(np.mean(infit[valid_mask])),
            'outfit_mean': float(np.mean(outfit[valid_mask]))
        },
        'difficulty_bands': {
            'bands': difficulty_bands.tolist(),
            'counts': band_counts.tolist()
        }
    }
    
    with open('exp/01_irt_calibration/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Calibration Summary:")
    print(f"  Valid items: {n_valid}/{n_items}")
    print(f"  Avg discrimination: {np.mean(irt.a[valid_mask]):.3f} ± {np.std(irt.a[valid_mask]):.3f}")
    print(f"  Avg difficulty: {np.mean(irt.b[valid_mask]):.3f} ± {np.std(irt.b[valid_mask]):.3f}")
    print(f"  Avg infit: {np.mean(infit[valid_mask]):.3f}")
    print(f"  Avg outfit: {np.mean(outfit[valid_mask]):.3f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
