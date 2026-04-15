"""Re-parse all raw results with updated parser."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from parse_answers import parse_answer, check_answer
from metrics import compute_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')


def reparse_file(raw_path, parsed_path, dataset):
    """Re-parse a raw results file and recompute metrics."""
    results = []
    parse_failures = 0

    with open(raw_path) as f:
        for line in f:
            r = json.loads(line)
            # Re-parse the raw output
            parsed, success = parse_answer(r['raw_output'], r['domain'], r['direction'])
            if not success:
                parse_failures += 1
            correct = check_answer(parsed, r['gold_answer'], r['domain'], r['direction'])

            result = {
                'id': r['id'],
                'domain': r['domain'],
                'difficulty': r['difficulty'],
                'direction': r['direction'],
                'gold_answer': r['gold_answer'],
                'parsed_answer': str(parsed) if parsed is not None else None,
                'correct': correct,
                'parse_success': success,
                'matched_pair_id': r['matched_pair_id']
            }
            results.append(result)

    metrics = compute_metrics(results, dataset)

    # Preserve meta from existing parsed file if available
    if os.path.exists(parsed_path):
        with open(parsed_path) as f:
            old = json.load(f)
            if 'meta' in old:
                metrics['meta'] = old['meta']
                metrics['meta']['parse_failures'] = parse_failures

    with open(parsed_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    # Load datasets
    datasets = {}
    for seed in ['seed_42', 'seed_123', 'seed_456']:
        path = os.path.join(DATA_DIR, seed, 'flipbench.json')
        with open(path) as f:
            datasets[seed] = json.load(f)

    # Re-parse all raw files
    raw_dir = os.path.join(RESULTS_DIR, 'raw')
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith('.jsonl'):
            continue

        raw_path = os.path.join(raw_dir, filename)
        # Determine seed from filename
        for seed in ['seed_42', 'seed_123', 'seed_456']:
            if seed in filename:
                dataset = datasets[seed]
                break
        else:
            print(f"Unknown seed for {filename}, skipping")
            continue

        parsed_name = filename.replace('.jsonl', '.json')
        parsed_path = os.path.join(RESULTS_DIR, 'parsed', parsed_name)

        print(f"Re-parsing {filename}...")
        metrics = reparse_file(raw_path, parsed_path, dataset)

        # Print summary
        print(f"  Overall: FA={metrics['overall']['forward_accuracy']:.3f}, "
              f"BA={metrics['overall']['backward_accuracy']:.3f}, "
              f"DRG={metrics['overall']['drg']:.3f}")
        for domain in ['propositional_logic', 'arithmetic_reasoning',
                       'relational_reasoning', 'function_computation']:
            d = metrics[domain]
            print(f"    {domain:25s}: FA={d['forward_accuracy']:.3f} BA={d['backward_accuracy']:.3f} "
                  f"DRG={d['drg']:.3f}")

    print("\nDone re-parsing all files.")


if __name__ == '__main__':
    main()
