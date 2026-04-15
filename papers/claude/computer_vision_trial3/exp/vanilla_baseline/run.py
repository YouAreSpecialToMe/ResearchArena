"""Evaluate vanilla pretrained ViTs on clean ImageNet and all ImageNet-C corruptions."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
os.environ.setdefault('TMPDIR', '/var/tmp')

import torch
from data_loader import (ImageNetValDataset, CorruptedImageNetDataset, get_transform,
                         get_dataloader, CORRUPTION_TYPES, compute_mce)
from models import load_model, MODEL_CONFIGS
from metrics import evaluate_accuracy, save_results, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
SEVERITIES = [1, 2, 3, 4, 5]


def evaluate_model(model_key):
    set_seed(42)
    model, config = load_model(model_key)
    batch_size = config['batch_size']

    results = {'model': model_key, 'config': config['name']}

    # Clean accuracy
    print(f"\n{'='*60}")
    print(f"Evaluating {model_key} - Clean ImageNet")
    dataset = ImageNetValDataset(DATA_DIR, transform=get_transform())
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)
    clean_result = evaluate_accuracy(model, loader)
    results['clean'] = clean_result
    print(f"  Clean top-1: {clean_result['top1']*100:.2f}%, top-5: {clean_result['top5']*100:.2f}%")

    # ImageNet-C corruptions
    corruption_results = {}
    model_errors = {}
    t0 = time.time()

    for corruption in CORRUPTION_TYPES:
        corruption_results[corruption] = {}
        errors = []
        for severity in SEVERITIES:
            dataset = CorruptedImageNetDataset(DATA_DIR, corruption, severity,
                                               indices=None)
            loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)
            acc = evaluate_accuracy(model, loader)
            corruption_results[corruption][str(severity)] = acc
            errors.append(1.0 - acc['top1'])
            print(f"  {model_key} | {corruption} sev={severity}: {acc['top1']*100:.2f}%")
        model_errors[corruption] = errors

    elapsed = time.time() - t0
    mce = compute_mce(model_errors)
    results['corruptions'] = corruption_results
    results['model_errors'] = {k: [float(e) for e in v] for k, v in model_errors.items()}
    results['mCE'] = float(mce)
    results['eval_time_minutes'] = elapsed / 60

    print(f"\n{model_key} mCE: {mce*100:.2f}%  (eval time: {elapsed/60:.1f} min)")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deit_small',
                        choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    results = evaluate_model(args.model)
    save_results(results, os.path.join(RESULTS_DIR, f'vanilla_{args.model}.json'))
    print(f"\nResults saved to results/vanilla_{args.model}.json")
