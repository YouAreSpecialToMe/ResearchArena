"""Run Tent on DeiT-B and Swin-T (severity 5 only) to complete baseline comparison."""
import sys, os, json, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
os.environ.setdefault('TMPDIR', '/var/tmp')

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import (ImageNetValDataset, CorruptedImageNetDataset, get_transform,
                         get_dataloader, CORRUPTION_TYPES)
from models import load_model, MODEL_CONFIGS
from metrics import save_results, set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def setup_tent(model, lr=1e-3):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    trainable_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            module.weight.requires_grad = True
            module.bias.requires_grad = True
            trainable_params.extend([module.weight, module.bias])
    optimizer = optim.SGD(trainable_params, lr=lr)
    return optimizer


def tent_forward(model, images, optimizer):
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
    entropy.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs.detach()


def evaluate_tent(model, dataloader, device='cuda', lr=1e-3):
    original_state = {name: p.data.clone() for name, p in model.named_parameters()}
    optimizer = setup_tent(model, lr=lr)
    correct_1 = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = tent_forward(model, images, optimizer)
        _, pred_1 = outputs.topk(1, dim=1)
        correct_1 += (pred_1.squeeze() == labels).sum().item()
        total += labels.size(0)
    for name, param in model.named_parameters():
        param.data.copy_(original_state[name])
    model.eval()
    return correct_1 / total


def run_tent_sev5(model_key):
    set_seed(42)
    model, config = load_model(model_key)
    batch_size = min(config['batch_size'], 64)

    results = {'model': model_key, 'method': 'tent'}

    # Clean accuracy
    print(f"Evaluating Tent on {model_key} - Clean")
    dataset = ImageNetValDataset(DATA_DIR, transform=get_transform())
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)
    clean_acc = evaluate_tent(model, loader)
    results['clean'] = {'top1': clean_acc}
    print(f"  Clean top-1: {clean_acc*100:.2f}%")

    # Severity 5 only
    sev5_accs = {}
    t0 = time.time()
    for corruption in CORRUPTION_TYPES:
        dataset = CorruptedImageNetDataset(DATA_DIR, corruption, severity=5)
        loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4)
        acc = evaluate_tent(model, loader)
        sev5_accs[corruption] = acc
        print(f"  {corruption} sev5: {acc*100:.2f}%")

    results['sev5_accs'] = sev5_accs
    results['mean_sev5_acc'] = sum(sev5_accs.values()) / len(sev5_accs)
    results['time_min'] = (time.time() - t0) / 60
    print(f"\n{model_key} mean sev5 acc: {results['mean_sev5_acc']*100:.2f}%")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deit_base', 'swin_tiny'])
    args = parser.parse_args()

    results = run_tent_sev5(args.model)
    save_results(results, os.path.join(RESULTS_DIR, f'tent_{args.model}.json'))
    print(f"Saved to results/tent_{args.model}.json")
