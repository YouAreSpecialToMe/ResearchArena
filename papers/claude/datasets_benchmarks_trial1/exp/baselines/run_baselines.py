"""Run simple baselines: random and heuristic."""

import json
import os
import random
import re
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'exp', 'shared'))
from parse_answers import check_answer
from metrics import compute_metrics

DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
NAMES_IN_DATASET = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace",
                    "Henry", "Iris", "Jack", "Kate", "Leo", "Mia", "Noah",
                    "Olivia", "Paul", "Quinn", "Rose", "Sam", "Tina"]

RELATIONSHIPS = ['parent', 'child', 'sibling', 'grandparent', 'grandchild',
                 'great-grandparent', 'great-grandchild']


def run_random_baseline(dataset, seed):
    """Random guessing baseline."""
    rng = random.Random(seed)
    results = []

    for inst in dataset:
        domain = inst['domain']
        direction = inst['direction']

        if domain == 'propositional_logic':
            # Both directions are now True/False
            guess = rng.choice(['True', 'False'])
        elif domain == 'arithmetic_reasoning':
            guess = str(rng.randint(1, 100))
        elif domain == 'relational_reasoning':
            # Both directions now return relationship types
            guess = rng.choice(RELATIONSHIPS)
        elif domain == 'function_computation':
            guess = str(rng.randint(-50, 150))
        else:
            guess = "unknown"

        correct = check_answer(guess, inst['answer'], domain, direction)
        results.append({
            'id': inst['id'],
            'domain': domain,
            'difficulty': inst['difficulty'],
            'direction': direction,
            'gold_answer': inst['answer'],
            'parsed_answer': guess,
            'correct': correct,
            'parse_success': True,
            'matched_pair_id': inst['matched_pair_id']
        })

    return results


def run_heuristic_baseline(dataset):
    """Heuristic pattern-matching baseline."""
    results = []

    for inst in dataset:
        domain = inst['domain']
        direction = inst['direction']
        text = inst['problem_text']

        if domain == 'propositional_logic':
            # Heuristic: always answer True (majority-class bias check)
            guess = 'True'

        elif domain == 'arithmetic_reasoning':
            if direction == 'forward':
                numbers = re.findall(r'\b(\d+)\b', text)
                if len(numbers) >= 2:
                    nums = [int(n) for n in numbers[:3]]
                    guess = str(sum(nums))
                else:
                    guess = str(numbers[0]) if numbers else '50'
            else:
                numbers = re.findall(r'\b(\d+)\b', text)
                guess = str(numbers[0]) if numbers else '25'

        elif domain == 'relational_reasoning':
            # Both directions: guess based on keyword patterns
            text_lower = text.lower()
            if 'great-grandparent' in text_lower or 'great-grandchild' in text_lower:
                guess = 'parent'
            elif 'grandparent' in text_lower or 'grandchild' in text_lower:
                guess = 'parent'
            elif 'parent' in text_lower:
                if direction == 'forward':
                    guess = 'parent'
                else:
                    guess = 'child'
            else:
                guess = 'sibling'

        elif domain == 'function_computation':
            if direction == 'forward':
                numbers = re.findall(r'(-?\d+)', text)
                if len(numbers) >= 2:
                    guess = str(int(numbers[0]) * int(numbers[1]))
                else:
                    guess = '0'
            else:
                guess = '10'

        else:
            guess = 'unknown'

        correct = check_answer(guess, inst['answer'], domain, direction)
        results.append({
            'id': inst['id'],
            'domain': domain,
            'difficulty': inst['difficulty'],
            'direction': direction,
            'gold_answer': inst['answer'],
            'parsed_answer': guess,
            'correct': correct,
            'parse_success': True,
            'matched_pair_id': inst['matched_pair_id']
        })

    return results


def main():
    # Load seed_42 dataset
    with open(os.path.join(DATA_DIR, 'seed_42', 'flipbench.json')) as f:
        dataset = json.load(f)

    print("Running baselines on seed_42 dataset...")
    print(f"Dataset size: {len(dataset)}")

    # Random baseline (3 seeds for variance)
    all_random_metrics = []
    for s in [42, 123, 456]:
        random_results = run_random_baseline(dataset, s)
        random_metrics = compute_metrics(random_results, dataset)
        all_random_metrics.append(random_metrics)

    # Average random baseline metrics
    avg_random = {}
    for domain in ['propositional_logic', 'arithmetic_reasoning',
                   'relational_reasoning', 'function_computation', 'overall']:
        avg_random[domain] = {}
        for key in ['forward_accuracy', 'backward_accuracy', 'drg', 'consistency_rate']:
            if key in all_random_metrics[0].get(domain, {}):
                vals = [m[domain][key] for m in all_random_metrics]
                import numpy as np
                avg_random[domain][key] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'std': round(float(np.std(vals)), 4)
                }

    outpath = os.path.join(RESULTS_DIR, 'random_baseline_results.json')
    with open(outpath, 'w') as f:
        json.dump(avg_random, f, indent=2)
    print(f"\nRandom baseline results saved to {outpath}")

    # Heuristic baseline
    heuristic_results = run_heuristic_baseline(dataset)
    heuristic_metrics = compute_metrics(heuristic_results, dataset)
    heuristic_metrics['meta'] = {'method': 'heuristic_pattern_matching'}

    outpath = os.path.join(RESULTS_DIR, 'heuristic_baseline_results.json')
    with open(outpath, 'w') as f:
        json.dump(heuristic_metrics, f, indent=2)
    print(f"Heuristic baseline results saved to {outpath}")

    # Print summary
    print("\n=== Random Baseline (mean +/- std) ===")
    for domain in ['propositional_logic', 'arithmetic_reasoning',
                   'relational_reasoning', 'function_computation']:
        d = avg_random[domain]
        print(f"  {domain:25s}: FA={d['forward_accuracy']['mean']:.3f}+/-{d['forward_accuracy']['std']:.3f} "
              f"BA={d['backward_accuracy']['mean']:.3f}+/-{d['backward_accuracy']['std']:.3f} "
              f"DRG={d['drg']['mean']:.3f}")

    print("\n=== Heuristic Baseline ===")
    for domain in ['propositional_logic', 'arithmetic_reasoning',
                   'relational_reasoning', 'function_computation']:
        d = heuristic_metrics[domain]
        print(f"  {domain:25s}: FA={d['forward_accuracy']:.3f} BA={d['backward_accuracy']:.3f} "
              f"DRG={d['drg']:.3f} CR={d['consistency_rate']:.3f}")


if __name__ == '__main__':
    main()
