"""
Data preparation script for DynaScale experiments.
Generates item pools and simulated model population.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import random
from shared.data_generation import (
    generate_math_problems, generate_code_problems, generate_science_problems,
    create_model_population, simulate_temporal_evolution, simulate_responses
)


def prepare_all_data(seed=42):
    """Prepare all data needed for DynaScale experiments."""
    np.random.seed(seed)
    
    print("=" * 60)
    print("DynaScale Data Preparation")
    print("=" * 60)
    
    # 1. Generate item pools for each domain
    print("\n[1/4] Generating math problem pool...")
    math_problems = generate_math_problems(3500, difficulty_range=(1, 6), seed=seed)
    print(f"      Generated {len(math_problems)} math problems")
    
    print("\n[2/4] Generating code problem pool...")
    code_problems = generate_code_problems(3500, difficulty_range=(1, 5), seed=seed+1)
    print(f"      Generated {len(code_problems)} code problems")
    
    print("\n[3/4] Generating science problem pool...")
    science_problems = generate_science_problems(3000, difficulty_range=(1, 4), seed=seed+2)
    print(f"      Generated {len(science_problems)} science problems")
    
    # Combine all problems
    all_problems = math_problems + code_problems + science_problems
    print(f"\n      Total items: {len(all_problems)}")
    
    # Add domain labels
    for p in all_problems:
        if 'domain' not in p:
            if p['id'].startswith('math'):
                p['domain'] = 'math'
            elif p['id'].startswith('code'):
                p['domain'] = 'code'
            elif p['id'].startswith('science'):
                p['domain'] = 'science'
    
    # Save item pool
    os.makedirs('data/pools', exist_ok=True)
    with open('data/pools/item_pool.jsonl', 'w') as f:
        for problem in all_problems:
            f.write(json.dumps(problem) + '\n')
    print("      Saved: data/pools/item_pool.jsonl")
    
    # 2. Create model population
    print("\n[4/4] Creating simulated model population...")
    population = create_model_population(n_models=28, ability_range=(-2, 2.5), 
                                         frontier_progression=True, seed=seed)
    print(f"      Created {len(population)} models")
    
    # Simulate temporal evolution
    evolution = simulate_temporal_evolution(population, n_timepoints=5, 
                                           ability_growth=0.5, seed=seed)
    print("      Simulated evolution over 5 time periods (t=0,3,6,9,12)")
    
    # Save population data
    os.makedirs('data/population', exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    population_serializable = convert_to_serializable(population)
    evolution_serializable = {str(t): convert_to_serializable([m for m in models]) 
                             for t, models in evolution.items()}
    
    with open('data/population/model_population.json', 'w') as f:
        json.dump({
            'base_population': population_serializable,
            'evolution': evolution_serializable
        }, f, indent=2)
    print("      Saved: data/population/model_population.json")
    
    # Save abilities as simple array for easy loading
    for t in range(5):
        abilities = np.array([m['ability'] for m in evolution[t]])
        np.save(f'data/population/abilities_t{t}.npy', abilities)
    print("      Saved: ability arrays for each time period")
    
    # 3. Generate IRT item parameters
    print("\n[Bonus] Generating IRT item parameters...")
    n_items = len(all_problems)
    
    # Generate realistic IRT parameters
    # Difficulties: centered around 0, with some spread
    difficulties = np.random.normal(0, 1.2, n_items)
    
    # Add correlation with difficulty_proxy
    for i, p in enumerate(all_problems):
        proxy = p.get('difficulty_proxy', 2)
        # Map proxy (1-6) to difficulty adjustment
        difficulties[i] += (proxy - 3) * 0.3
    
    # Discriminations: mostly around 1.0, some higher
    discriminations = np.random.lognormal(0, 0.3, n_items)
    discriminations = np.clip(discriminations, 0.5, 2.5)
    
    # Save item parameters
    np.save('data/pools/difficulties.npy', difficulties)
    np.save('data/pools/discriminations.npy', discriminations)
    
    item_params = {
        'difficulties': difficulties.tolist(),
        'discriminations': discriminations.tolist(),
        'n_items': n_items,
        'domain_breakdown': {
            'math': sum(1 for p in all_problems if p['domain'] == 'math'),
            'code': sum(1 for p in all_problems if p['domain'] == 'code'),
            'science': sum(1 for p in all_problems if p['domain'] == 'science')
        }
    }
    with open('data/pools/item_params.json', 'w') as f:
        json.dump(item_params, f, indent=2)
    print("      Saved: item parameters")
    
    # 4. Generate ground truth responses for all time periods
    print("\n[Bonus] Generating ground truth responses...")
    for t in range(5):
        abilities = np.load(f'data/population/abilities_t{t}.npy')
        responses = simulate_responses(abilities, difficulties, discriminations, 
                                       seed=seed+t, noise_rate=0.05)
        np.save(f'data/population/responses_t{t}.npy', responses)
    print("      Saved: response matrices for each time period")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    
    return {
        'n_items': n_items,
        'n_models': len(population),
        'n_timepoints': 5
    }


if __name__ == '__main__':
    prepare_all_data(seed=42)
