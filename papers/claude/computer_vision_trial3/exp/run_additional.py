"""Run additional experiments to address reviewer feedback:
1. Full 50K ImageNet validation for DeiT-S (vanilla + STG + Tent)
2. Tent baseline on DeiT-B and Swin-T (severity 5 only for speed)
3. Oracle upper-bound experiment
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
os.environ.setdefault('TMPDIR', '/var/tmp')

import torch
import numpy as np
from data_loader import (ImageNetValDataset, CorruptedImageNetDataset, get_transform,
                         get_dataloader, get_calibration_indices, CORRUPTION_TYPES, compute_mce,
                         REPRESENTATIVE_CORRUPTIONS)
from models import load_model, MODEL_CONFIGS
from stg import SpectralTokenGating
from metrics import evaluate_accuracy, save_results, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
SEVERITIES = [1, 2, 3, 4, 5]


def run_full_50k_clean(model_key, method='vanilla'):
    """Evaluate on full 50K ImageNet validation."""
    model, config = load_model(model_key)
    batch_size = config['batch_size']
    dataset = ImageNetValDataset(DATA_DIR, transform=get_transform())
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)

    stg = None
    if method == 'stg':
        stg = SpectralTokenGating(model, model_key, K=3, alpha=5.0, tau_percentile=95)
        cal_path = os.path.join(CHECKPOINT_DIR, f'calibration_{model_key}_seed42.pt')
        stg.load_calibration(cal_path)
        stg.enable()

    print(f"\nEvaluating {method} {model_key} on full 50K clean ImageNet...")
    result = evaluate_accuracy(model, loader)
    print(f"  Clean top-1: {result['top1']*100:.2f}% ({result['total']} images)")

    if stg:
        stg.disable()
    return result


def run_full_50k_corruptions(model_key, method='vanilla', seed=42):
    """Evaluate on full 50K for all corruptions."""
    model, config = load_model(model_key)
    batch_size = config['batch_size']

    stg = None
    if method == 'stg':
        stg = SpectralTokenGating(model, model_key, K=3, alpha=5.0, tau_percentile=95)
        cal_path = os.path.join(CHECKPOINT_DIR, f'calibration_{model_key}_seed{seed}.pt')
        stg.load_calibration(cal_path)
        stg.enable()

    results = {'model': model_key, 'method': method, 'seed': seed if method == 'stg' else None}
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
            print(f"  {method} {model_key} | {corruption} sev={severity}: {acc['top1']*100:.2f}% ({acc['total']})")
        model_errors[corruption] = errors

    if stg:
        stg.disable()

    elapsed = time.time() - t0
    mce = compute_mce(model_errors)
    results['corruptions'] = corruption_results
    results['model_errors'] = {k: [float(e) for e in v] for k, v in model_errors.items()}
    results['mCE'] = float(mce)
    results['eval_time_minutes'] = elapsed / 60

    print(f"\n{method} {model_key} mCE: {mce*100:.2f}%")
    return results


def run_tent_experiment(model_key):
    """Run Tent baseline on a model - severity 5 only for speed."""
    set_seed(42)
    model, config = load_model(model_key)
    batch_size = min(64, config['batch_size'])  # smaller batch for Tent (needs gradients)

    results = {'model': model_key, 'method': 'tent'}

    # Clean accuracy first
    print(f"\nEvaluating Tent on {model_key} - Clean ImageNet...")
    dataset = ImageNetValDataset(DATA_DIR, transform=get_transform())
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)

    # Setup Tent: optimize LayerNorm parameters
    def setup_tent(model):
        model.train()
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze LayerNorm affine parameters
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                if module.elementwise_affine:
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
        # Collect trainable params
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=1e-3)
        return optimizer, params

    def tent_forward(model, images, optimizer):
        """Tent: forward pass with entropy minimization on LN params."""
        outputs = model(images)
        # Entropy minimization
        probs = torch.softmax(outputs, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        optimizer.zero_grad()
        entropy.backward()
        optimizer.step()
        return outputs

    # Save original state for reset between corruptions
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Evaluate Tent on clean
    optimizer, params = setup_tent(model)
    correct, total = 0, 0
    with torch.enable_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = tent_forward(model, images, optimizer)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    clean_acc = correct / total
    results['clean'] = {'top1': clean_acc, 'total': total}
    print(f"  Clean top-1: {clean_acc*100:.2f}%")

    # Reset model
    model.load_state_dict(original_state)

    # Evaluate on all corruptions at severity 5
    corruption_results = {}
    model_errors = {}
    t0 = time.time()

    for corruption in CORRUPTION_TYPES:
        # Reset model state for each corruption
        model.load_state_dict(original_state)
        optimizer, params = setup_tent(model)

        corruption_results[corruption] = {}
        sev5_errors = []

        for severity in SEVERITIES:
            dataset = CorruptedImageNetDataset(DATA_DIR, corruption, severity)
            loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)

            correct, total_imgs = 0, 0
            with torch.enable_grad():
                for images, labels in loader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = tent_forward(model, images, optimizer)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total_imgs += labels.size(0)

            acc = correct / total_imgs
            corruption_results[corruption][str(severity)] = {'top1': acc, 'total': total_imgs}
            sev5_errors.append(1.0 - acc)
            print(f"  Tent {model_key} | {corruption} sev={severity}: {acc*100:.2f}%")

        model_errors[corruption] = sev5_errors

    elapsed = time.time() - t0
    mce = compute_mce(model_errors)
    results['corruptions'] = corruption_results
    results['model_errors'] = {k: [float(e) for e in v] for k, v in model_errors.items()}
    results['mCE'] = float(mce)
    results['eval_time_minutes'] = elapsed / 60

    # Reset
    model.load_state_dict(original_state)

    print(f"\nTent {model_key} mCE: {mce*100:.2f}%")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['full50k_clean', 'full50k_corrupt', 'tent_deitb', 'tent_swint',
                                 'stg_deitb_seed123', 'stg_deitb_seed456',
                                 'stg_swint_seed123', 'stg_swint_seed456'])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.experiment == 'full50k_clean':
        # Run vanilla and STG clean accuracy on full 50K
        van_result = run_full_50k_clean('deit_small', 'vanilla')
        stg_result = run_full_50k_clean('deit_small', 'stg')
        results = {
            'vanilla_clean_50k': van_result,
            'stg_clean_50k': stg_result,
        }
        save_results(results, os.path.join(RESULTS_DIR, 'full50k_clean_deit_small.json'))

    elif args.experiment == 'full50k_corrupt':
        # Full 50K corruption eval for DeiT-S vanilla
        results = run_full_50k_corruptions('deit_small', 'vanilla')
        save_results(results, os.path.join(RESULTS_DIR, 'full50k_vanilla_deit_small.json'))

    elif args.experiment == 'tent_deitb':
        results = run_tent_experiment('deit_base')
        save_results(results, os.path.join(RESULTS_DIR, 'tent_deit_base.json'))

    elif args.experiment == 'tent_swint':
        results = run_tent_experiment('swin_tiny')
        save_results(results, os.path.join(RESULTS_DIR, 'tent_swin_tiny.json'))

    elif args.experiment == 'stg_deitb_seed123':
        # Additional seeds for DeiT-B
        set_seed(123)
        from stg_main.run import run_stg_experiment  # reuse existing
        # Manual implementation to avoid import issues
        model, config = load_model('deit_base')
        stg = SpectralTokenGating(model, 'deit_base', K=3, alpha=5.0, tau_percentile=95)
        cal_indices = get_calibration_indices(123, n_calibration=1000)
        cal_dataset = ImageNetValDataset(DATA_DIR, transform=get_transform(), indices=cal_indices)
        cal_loader = get_dataloader(cal_dataset, batch_size=config['batch_size'], num_workers=4)
        stg.calibrate(cal_loader, n_images=1000)
        stg.enable()

        dataset = ImageNetValDataset(DATA_DIR, transform=get_transform())
        loader = get_dataloader(dataset, batch_size=config['batch_size'], num_workers=4)
        clean = evaluate_accuracy(model, loader)

        corruption_results = {}
        model_errors = {}
        for corruption in CORRUPTION_TYPES:
            corruption_results[corruption] = {}
            errors = []
            for severity in SEVERITIES:
                ds = CorruptedImageNetDataset(DATA_DIR, corruption, severity)
                dl = get_dataloader(ds, batch_size=config['batch_size'], num_workers=4)
                acc = evaluate_accuracy(model, dl)
                corruption_results[corruption][str(severity)] = acc
                errors.append(1.0 - acc['top1'])
                print(f"  STG deit_base seed=123 | {corruption} sev={severity}: {acc['top1']*100:.2f}%")
            model_errors[corruption] = errors

        stg.disable()
        mce = compute_mce(model_errors)
        results = {'model': 'deit_base', 'seed': 123, 'method': 'stg',
                   'clean': clean, 'corruptions': corruption_results,
                   'model_errors': {k: [float(e) for e in v] for k, v in model_errors.items()},
                   'mCE': float(mce)}
        save_results(results, os.path.join(RESULTS_DIR, 'stg_deit_base_seed123.json'))

    print("Done!")
