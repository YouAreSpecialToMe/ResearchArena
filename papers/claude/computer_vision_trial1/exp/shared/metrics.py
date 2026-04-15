import torch
import time
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def evaluate_accuracy(model, loader, device=None, desc="Evaluating"):
    """Evaluate top-1 accuracy."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def compute_relative_accuracy_drop(clean_acc, corrupt_acc):
    """Compute relative accuracy drop: (clean - corrupt) / clean."""
    if clean_acc == 0:
        return 0.0
    return (clean_acc - corrupt_acc) / clean_acc


def compute_mce(per_corruption_errors, reference_errors=None):
    """Compute mean Corruption Error.
    per_corruption_errors: dict of {corruption_name: error_rate}
    reference_errors: optional dict of reference model errors (e.g., AlexNet)
    If no reference, mCE = mean of error rates.
    """
    if reference_errors:
        ces = []
        for corr, err in per_corruption_errors.items():
            ref = reference_errors.get(corr, 1.0)
            ces.append(err / ref if ref > 0 else err)
        return float(np.mean(ces))
    else:
        return float(np.mean(list(per_corruption_errors.values())))


def measure_throughput(model, input_shape=(1, 3, 224, 224), batch_size=128,
                       warmup=20, timed=100, device=None):
    """Measure throughput in images/sec."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    dummy = torch.randn(batch_size, *input_shape[1:], device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()

    # Timed
    start = time.time()
    for _ in range(timed):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = (timed * batch_size) / elapsed
    return throughput


def measure_throughput_with_loader(model, loader, warmup_batches=10, timed_batches=50,
                                    device=None):
    """Measure throughput using actual data loader."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    batch_count = 0
    total_images = 0

    for images, _ in loader:
        images = images.to(device)
        with torch.no_grad():
            _ = model(images)
        batch_count += 1
        if batch_count >= warmup_batches:
            break

    torch.cuda.synchronize()
    start = time.time()
    batch_count = 0
    for images, _ in loader:
        images = images.to(device)
        with torch.no_grad():
            _ = model(images)
        total_images += images.shape[0]
        batch_count += 1
        if batch_count >= timed_batches:
            break
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return total_images / elapsed if elapsed > 0 else 0.0
