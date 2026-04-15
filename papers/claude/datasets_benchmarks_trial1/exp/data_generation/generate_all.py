"""Main script to generate the full FlipBench dataset for all seeds."""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from generate_logic import generate_logic_dataset
from generate_arithmetic import generate_arithmetic_dataset
from generate_relational import generate_relational_dataset
from generate_functions import generate_function_dataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEEDS = [42, 123, 456]


def verify_dataset(dataset):
    """Verify dataset integrity."""
    # Check counts
    domains = Counter(d['domain'] for d in dataset)
    directions = Counter(d['direction'] for d in dataset)
    difficulties = Counter(d['difficulty'] for d in dataset)

    assert directions['forward'] == directions['backward'], \
        f"Unequal directions: {directions}"

    # Check matched pairs
    pair_ids = Counter(d['matched_pair_id'] for d in dataset)
    for pid, count in pair_ids.items():
        assert count == 2, f"Pair {pid} has {count} instances (expected 2)"

    # Check all answers are non-empty
    for d in dataset:
        assert d['answer'] is not None and str(d['answer']).strip() != '', \
            f"Empty answer for {d['id']}"

    return domains, directions, difficulties


def generate_and_save(seed):
    """Generate full dataset for one seed."""
    print(f"\n=== Generating dataset for seed {seed} ===")

    logic = generate_logic_dataset(seed)
    print(f"  Logic: {len(logic)} instances")

    arithmetic = generate_arithmetic_dataset(seed)
    print(f"  Arithmetic: {len(arithmetic)} instances")

    relational = generate_relational_dataset(seed)
    print(f"  Relational: {len(relational)} instances")

    functions = generate_function_dataset(seed)
    print(f"  Functions: {len(functions)} instances")

    dataset = logic + arithmetic + relational + functions
    print(f"  Total: {len(dataset)} instances")

    # Verify
    domains, directions, difficulties = verify_dataset(dataset)
    print(f"  Domains: {dict(domains)}")
    print(f"  Directions: {dict(directions)}")
    print(f"  Difficulties: {dict(difficulties)}")

    # Save
    seed_dir = os.path.join(DATA_DIR, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)
    outpath = os.path.join(seed_dir, 'flipbench.json')
    with open(outpath, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"  Saved to {outpath}")

    return dataset


def compute_dataset_stats(all_datasets):
    """Compute and save dataset statistics."""
    stats = {}
    # Use seed 42 as reference
    dataset = all_datasets[42]

    for domain in ['propositional_logic', 'arithmetic_reasoning',
                   'relational_reasoning', 'function_computation']:
        domain_data = [d for d in dataset if d['domain'] == domain]
        stats[domain] = {
            'total': len(domain_data),
            'forward': len([d for d in domain_data if d['direction'] == 'forward']),
            'backward': len([d for d in domain_data if d['direction'] == 'backward']),
            'by_difficulty': {}
        }
        for diff in [1, 2, 3]:
            diff_data = [d for d in domain_data if d['difficulty'] == diff]
            answers = [d['answer'] for d in diff_data]
            stats[domain]['by_difficulty'][str(diff)] = {
                'count': len(diff_data),
                'unique_answers': len(set(answers)),
                'answer_distribution': dict(Counter(answers).most_common(10))
            }

    outpath = os.path.join(DATA_DIR, 'dataset_stats.json')
    with open(outpath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nDataset stats saved to {outpath}")
    return stats


def compute_random_baseline():
    """Compute expected random guessing accuracy."""
    baseline = {
        'propositional_logic': {
            'forward': {'method': 'random True/False', 'expected_accuracy': 0.5},
            'backward': {'method': 'random True/False', 'expected_accuracy': 0.5}
        },
        'arithmetic_reasoning': {
            'forward': {'method': 'random int [1,1000]', 'expected_accuracy': 0.001},
            'backward': {'method': 'random int [1,100]', 'expected_accuracy': 0.01}
        },
        'relational_reasoning': {
            'forward': {'method': 'random relationship from 7', 'expected_accuracy': 0.143},
            'backward': {'method': 'random relationship from 7', 'expected_accuracy': 0.143}
        },
        'function_computation': {
            'forward': {'method': 'random int [-100,200]', 'expected_accuracy': 0.003},
            'backward': {'method': 'random int [1,50]', 'expected_accuracy': 0.02}
        }
    }

    outpath = os.path.join(BASE_DIR, 'results', 'random_baseline.json')
    with open(outpath, 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f"Random baseline expectations saved to {outpath}")
    return baseline


if __name__ == '__main__':
    all_datasets = {}
    for seed in SEEDS:
        all_datasets[seed] = generate_and_save(seed)

    compute_dataset_stats(all_datasets)
    compute_random_baseline()
    print("\nDone! All datasets generated and verified.")
