"""Evaluation metrics and utilities."""
import torch
import time
import json
import os
import numpy as np


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device='cuda', desc='Evaluating'):
    """Evaluate top-1 and top-5 accuracy."""
    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        _, pred_1 = outputs.topk(1, dim=1)
        _, pred_5 = outputs.topk(5, dim=1)
        correct_1 += (pred_1.squeeze() == labels).sum().item()
        correct_5 += (pred_5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    top1 = correct_1 / total
    top5 = correct_5 / total
    return {'top1': top1, 'top5': top5, 'total': total}


def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def measure_latency(model, input_tensor, n_warmup=10, n_measure=100, device='cuda'):
    """Measure model inference latency."""
    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_tensor)
    torch.cuda.synchronize()

    # Measure
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]

    for i in range(n_measure):
        start_events[i].record()
        with torch.no_grad():
            model(input_tensor)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'per_image_ms': np.mean(times) / input_tensor.shape[0],
    }
