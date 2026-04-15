"""Evaluate Tent (test-time adaptation) baseline on ImageNet-C.

Tent adapts LayerNorm affine parameters at test time by minimizing
prediction entropy. For ViTs with LayerNorm instead of BatchNorm.
"""
import sys, os, json, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
os.environ.setdefault('TMPDIR', '/var/tmp')

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import (ImageNetValDataset, CorruptedImageNetDataset, get_transform,
                         get_dataloader, CORRUPTION_TYPES, compute_mce)
from models import load_model, MODEL_CONFIGS
from metrics import evaluate_accuracy, save_results, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
SEVERITIES = [1, 2, 3, 4, 5]


def setup_tent(model, lr=1e-3):
    """Configure model for Tent: only LayerNorm params are trainable."""
    model.train()  # Enable training mode for LayerNorm
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient for LayerNorm affine parameters
    trainable_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            module.weight.requires_grad = True
            module.bias.requires_grad = True
            trainable_params.extend([module.weight, module.bias])

    optimizer = optim.SGD(trainable_params, lr=lr)
    return optimizer


def tent_forward(model, images, optimizer):
    """Single Tent step: forward, compute entropy, update, return predictions."""
    outputs = model(images)
    # Entropy minimization
    probs = torch.softmax(outputs, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
    entropy.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs.detach()


def evaluate_tent(model, dataloader, device='cuda', lr=1e-3):
    """Evaluate with Tent adaptation."""
    # Save original state for reset
    original_state = copy.deepcopy({name: p.data.clone() for name, p in model.named_parameters()})

    optimizer = setup_tent(model, lr=lr)
    correct_1 = 0
    correct_5 = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = tent_forward(model, images, optimizer)

        _, pred_1 = outputs.topk(1, dim=1)
        _, pred_5 = outputs.topk(5, dim=1)
        correct_1 += (pred_1.squeeze() == labels).sum().item()
        correct_5 += (pred_5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    # Restore original parameters
    for name, param in model.named_parameters():
        param.data.copy_(original_state[name])
    model.eval()

    return {'top1': correct_1 / total, 'top5': correct_5 / total, 'total': total}


def evaluate_model_tent(model_key):
    set_seed(42)
    model, config = load_model(model_key)
    # Use smaller batch for Tent (needs gradients)
    batch_size = min(config['batch_size'], 64)

    results = {'model': model_key, 'config': config['name'], 'method': 'tent'}

    # Clean accuracy with Tent
    print(f"\n{'='*60}")
    print(f"Evaluating Tent on {model_key} - Clean ImageNet")
    dataset = ImageNetValDataset(DATA_DIR, transform=get_transform())
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)
    clean_result = evaluate_tent(model, loader)
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
            acc = evaluate_tent(model, loader)
            corruption_results[corruption][str(severity)] = acc
            errors.append(1.0 - acc['top1'])
            print(f"  Tent {model_key} | {corruption} sev={severity}: {acc['top1']*100:.2f}%")
        model_errors[corruption] = errors

    elapsed = time.time() - t0
    mce = compute_mce(model_errors)
    results['corruptions'] = corruption_results
    results['model_errors'] = {k: [float(e) for e in v] for k, v in model_errors.items()}
    results['mCE'] = float(mce)
    results['eval_time_minutes'] = elapsed / 60

    print(f"\nTent {model_key} mCE: {mce*100:.2f}%  (eval time: {elapsed/60:.1f} min)")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deit_small',
                        choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    results = evaluate_model_tent(args.model)
    save_results(results, os.path.join(RESULTS_DIR, f'tent_{args.model}.json'))
    print(f"\nResults saved to results/tent_{args.model}.json")
