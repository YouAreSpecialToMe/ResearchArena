"""Error analysis: categorize backward reasoning errors per domain."""

import json
import os
import re
import sys
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, os.path.join(BASE_DIR, 'exp', 'shared'))
from parse_answers import parse_answer, check_answer

DOMAINS = ['propositional_logic', 'arithmetic_reasoning',
           'relational_reasoning', 'function_computation']


def load_raw_results(model_short, seed='seed_42'):
    path = os.path.join(RESULTS_DIR, 'raw', f'{model_short}_{seed}.jsonl')
    results = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # Re-check correctness with updated parser
            parsed, success = parse_answer(r.get('raw_output', ''), r['domain'], r['direction'])
            r['correct'] = check_answer(parsed, r['gold_answer'], r['domain'], r['direction'])
            r['parsed_answer'] = str(parsed) if parsed is not None else None
            results.append(r)
    return results


def categorize_logic_error(raw_output, gold_answer, parsed_answer, direction):
    """Categorize logic reasoning errors."""
    text = raw_output.lower()
    if direction == 'forward':
        if 'cannot' in text or 'not enough' in text or 'insufficient' in text:
            return 'uncertainty_hedging'
        if gold_answer.lower() == 'true' and parsed_answer and 'false' in str(parsed_answer).lower():
            return 'incorrect_chain_traversal'
        if gold_answer.lower() == 'false' and parsed_answer and 'true' in str(parsed_answer).lower():
            return 'false_positive_derivation'
        return 'other_forward'
    else:
        if 'ANSWER:' not in raw_output.upper() and len(raw_output) > 800:
            return 'output_truncation'
        if 'cannot' in text or 'not necessarily' in text or 'not enough' in text:
            return 'rejects_closed_world'
        if gold_answer.lower() == 'true' and parsed_answer and 'false' in str(parsed_answer).lower():
            return 'incorrect_backward_chain'
        if gold_answer.lower() == 'false' and parsed_answer and 'true' in str(parsed_answer).lower():
            return 'false_positive_backward'
        return 'other_backward'


def categorize_arithmetic_error(raw_output, gold_answer, parsed_answer, direction):
    """Categorize arithmetic reasoning errors."""
    if direction == 'forward':
        try:
            parsed_num = int(parsed_answer) if parsed_answer else None
            gold_num = int(gold_answer)
            if parsed_num is not None:
                diff = abs(parsed_num - gold_num)
                if diff < gold_num * 0.1:
                    return 'rounding_close'
                return 'computation_error'
        except (ValueError, TypeError):
            return 'parse_failure'
        return 'computation_error'
    else:
        try:
            parsed_num = int(parsed_answer) if parsed_answer else None
            gold_num = int(gold_answer)
            if parsed_num is not None and abs(parsed_num - gold_num) <= 1:
                return 'off_by_one'
            return 'incorrect_inverse'
        except (ValueError, TypeError):
            return 'parse_failure'


def categorize_relational_error(raw_output, gold_answer, parsed_answer, direction):
    """Categorize relational reasoning errors."""
    if direction == 'forward':
        if parsed_answer and gold_answer.lower() != str(parsed_answer).lower():
            return 'wrong_relationship_type'
        return 'failed_inference'
    else:
        if parsed_answer and parsed_answer[0].isupper() and len(parsed_answer) > 1:
            return 'wrong_entity'
        return 'failed_identification'


def categorize_function_error(raw_output, gold_answer, parsed_answer, direction):
    """Categorize function computation errors."""
    if direction == 'forward':
        return 'computation_error'
    else:
        return 'incorrect_inversion'


def run_error_analysis():
    # Use the model with most errors for informative analysis
    error_analysis = {}

    for model in ['llama31_8b', 'deepseek_r1_7b', 'phi35']:
        raw = load_raw_results(model)
        model_errors = {}

        for domain in DOMAINS:
            domain_errors = [r for r in raw if r['domain'] == domain and not r['correct']]

            # Sample up to 50 errors
            sample = domain_errors[:50]

            categorizers = {
                'propositional_logic': categorize_logic_error,
                'arithmetic_reasoning': categorize_arithmetic_error,
                'relational_reasoning': categorize_relational_error,
                'function_computation': categorize_function_error,
            }

            categories = Counter()
            for_bwd = Counter()
            for err in sample:
                cat = categorizers[domain](
                    err.get('raw_output', ''),
                    err['gold_answer'],
                    err['parsed_answer'],
                    err['direction']
                )
                categories[cat] += 1
                for_bwd[err['direction']] += 1

            model_errors[domain] = {
                'total_errors': len(domain_errors),
                'forward_errors': sum(1 for e in domain_errors if e['direction'] == 'forward'),
                'backward_errors': sum(1 for e in domain_errors if e['direction'] == 'backward'),
                'error_categories': dict(categories),
                'sampled': len(sample)
            }

        error_analysis[model] = model_errors

    outpath = os.path.join(RESULTS_DIR, 'aggregated', 'error_analysis.json')
    with open(outpath, 'w') as f:
        json.dump(error_analysis, f, indent=2)
    print(f"Error analysis saved to {outpath}")

    # Print summary
    print("\n=== Error Analysis Summary ===")
    for model in error_analysis:
        print(f"\n{model}:")
        for domain in DOMAINS:
            d = error_analysis[model][domain]
            print(f"  {domain}: {d['total_errors']} total "
                  f"(fwd={d['forward_errors']}, bwd={d['backward_errors']})")
            for cat, count in sorted(d['error_categories'].items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")


if __name__ == '__main__':
    run_error_analysis()
