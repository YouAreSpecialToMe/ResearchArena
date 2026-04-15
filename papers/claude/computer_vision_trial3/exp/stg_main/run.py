"""Calibrate and evaluate STG on clean ImageNet and all ImageNet-C corruptions."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
os.environ.setdefault('TMPDIR', '/var/tmp')

import torch
from data_loader import (ImageNetValDataset, CorruptedImageNetDataset, get_transform,
                         get_dataloader, get_calibration_indices, CORRUPTION_TYPES, compute_mce)
from models import load_model, MODEL_CONFIGS
from stg import SpectralTokenGating
from metrics import evaluate_accuracy, save_results, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
SEVERITIES = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 456]


def run_stg_experiment(model_key, seed):
    set_seed(seed)
    model, config = load_model(model_key)
    batch_size = config['batch_size']

    results = {'model': model_key, 'seed': seed, 'method': 'stg',
               'config': {'K': 3, 'alpha': 5.0, 'tau_percentile': 95,
                          'layers': config['default_stg_layers']}}

    # Calibration
    print(f"\n{'='*60}")
    print(f"STG Calibration: {model_key}, seed={seed}")
    cal_indices = get_calibration_indices(seed, n_calibration=1000)
    cal_dataset = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=cal_indices)
    cal_loader = get_dataloader(cal_dataset, batch_size=batch_size, num_workers=4)

    stg = SpectralTokenGating(model, model_key, K=3, alpha=5.0, tau_percentile=95)
    t0 = time.time()
    stg.calibrate(cal_loader, n_images=1000)
    cal_time = time.time() - t0
    print(f"  Calibration time: {cal_time:.1f}s")

    # Save calibration
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cal_path = os.path.join(CHECKPOINT_DIR, f'calibration_{model_key}_seed{seed}.pt')
    stg.save_calibration(cal_path)
    results['calibration_time_s'] = cal_time

    # Enable gating
    stg.enable()

    # Clean accuracy
    print(f"Evaluating STG on {model_key} - Clean ImageNet (seed={seed})")
    dataset = ImageNetValDataset(DATA_DIR, transform=get_transform())
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)
    clean_result = evaluate_accuracy(model, loader)
    results['clean'] = clean_result
    print(f"  Clean top-1: {clean_result['top1']*100:.2f}%")

    # ImageNet-C corruptions
    corruption_results = {}
    model_errors = {}
    t0 = time.time()

    for corruption in CORRUPTION_TYPES:
        corruption_results[corruption] = {}
        errors = []
        for severity in SEVERITIES:
            dataset = CorruptedImageNetDataset(DATA_DIR, corruption, severity)
            loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)
            acc = evaluate_accuracy(model, loader)
            corruption_results[corruption][str(severity)] = acc
            errors.append(1.0 - acc['top1'])
            print(f"  STG {model_key} seed={seed} | {corruption} sev={severity}: {acc['top1']*100:.2f}%")
        model_errors[corruption] = errors

    stg.disable()

    elapsed = time.time() - t0
    mce = compute_mce(model_errors)
    results['corruptions'] = corruption_results
    results['model_errors'] = {k: [float(e) for e in v] for k, v in model_errors.items()}
    results['mCE'] = float(mce)
    results['eval_time_minutes'] = elapsed / 60

    print(f"\nSTG {model_key} seed={seed} mCE: {mce*100:.2f}%")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deit_small',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--seed', type=int, default=42, choices=SEEDS)
    args = parser.parse_args()

    results = run_stg_experiment(args.model, args.seed)
    save_results(results, os.path.join(RESULTS_DIR, f'stg_{args.model}_seed{args.seed}.json'))
    print(f"Results saved to results/stg_{args.model}_seed{args.seed}.json")
