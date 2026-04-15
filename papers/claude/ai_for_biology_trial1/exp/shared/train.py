"""Training loop for contrastive learning experiments."""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from exp.shared.models import ProjectionHead
from exp.shared.losses import SupConLoss, CurriculumLoss
from exp.shared.data_loader import EmbeddingDataset, BalancedBatchSampler
from exp.shared.eval import evaluate_model
from exp.shared.utils import set_seed, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader


def train_phase(model, train_dataset, level, num_epochs, lr, temperature,
                consistency_levels=None, consistency_weight=0.5,
                batch_size=512, device="cuda", verbose=True):
    """Train one phase of the curriculum.

    Args:
        model: ProjectionHead
        train_dataset: EmbeddingDataset
        level: str, EC level for primary loss (e.g., "ec_l1")
        num_epochs: int
        lr: float
        temperature: float
        consistency_levels: list of str, coarser EC levels for regularization
        consistency_weight: float (lambda)
        batch_size: int
        device: str
        verbose: bool
    Returns:
        model, loss_history (list of dicts per epoch)
    """
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Get labels
    primary_labels = train_dataset.get_labels(level).to(device)
    consistency_labels_list = []
    if consistency_levels:
        for cl in consistency_levels:
            consistency_labels_list.append(train_dataset.get_labels(cl).to(device))

    loss_fn = CurriculumLoss(temperature=temperature, consistency_weight=consistency_weight)

    # Build balanced sampler
    n_classes = train_dataset.n_classes(level)
    if n_classes <= 32:
        classes_per_batch = n_classes
        samples_per_class = min(batch_size // max(n_classes, 1), 64)
    else:
        classes_per_batch = min(32, n_classes)
        samples_per_class = batch_size // classes_per_batch

    sampler = BalancedBatchSampler(
        train_dataset.get_labels(level), batch_size,
        classes_per_batch=classes_per_batch,
        samples_per_class=max(samples_per_class, 2)
    )

    all_embeddings = train_dataset.embeddings.to(device)
    loss_history = []

    for epoch in range(num_epochs):
        epoch_losses = {"primary": 0.0, "consistency": 0.0, "total": 0.0}
        n_batches = 0

        for batch_indices in sampler:
            batch_emb = all_embeddings[batch_indices]
            batch_primary = primary_labels[batch_indices]
            batch_consistency = [cl[batch_indices] for cl in consistency_labels_list]

            projected = model(batch_emb)
            total_loss, primary_val, consistency_val = loss_fn(
                projected, batch_primary, batch_consistency
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses["primary"] += primary_val
            epoch_losses["consistency"] += consistency_val
            epoch_losses["total"] += total_loss.item()
            n_batches += 1

        scheduler.step()

        avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        loss_history.append(avg_losses)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"loss={avg_losses['total']:.4f} "
                  f"(primary={avg_losses['primary']:.4f}, "
                  f"consistency={avg_losses['consistency']:.4f})")

    return model, loss_history


def run_flat_supcon(seed, train_dataset, test_datasets, rare_classes_str,
                    num_epochs=90, lr=5e-4, temperature=0.1,
                    batch_size=512, device="cuda"):
    """Run flat SupCon baseline (no hierarchy, no curriculum)."""
    set_seed(seed)
    model = ProjectionHead().to(device)

    model, loss_history = train_phase(
        model, train_dataset, level="ec_l4",
        num_epochs=num_epochs, lr=lr, temperature=temperature,
        consistency_levels=None, consistency_weight=0.0,
        batch_size=batch_size, device=device
    )

    # Evaluate
    results = {"seed": seed, "method": "flat_supcon", "loss_history": loss_history}
    for test_name, test_ds in test_datasets.items():
        metrics = evaluate_model(model, train_dataset, test_ds,
                                level="ec_l4", rare_classes_str=rare_classes_str,
                                device=device)
        results[test_name] = metrics

    return results, model


def run_joint_hierarchical(seed, train_dataset, test_datasets, rare_classes_str,
                           num_epochs=90, lr=5e-4, temperature=0.1,
                           weights=(0.1, 0.2, 0.3, 0.4), batch_size=512, device="cuda"):
    """Run joint hierarchical SupCon baseline (all levels simultaneously)."""
    set_seed(seed)
    model = ProjectionHead().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    all_emb = train_dataset.embeddings.to(device)
    levels = ["ec_l1", "ec_l2", "ec_l3", "ec_l4"]
    all_labels = {lv: train_dataset.get_labels(lv).to(device) for lv in levels}
    loss_fns = {lv: SupConLoss(temperature=temperature) for lv in levels}

    # Use level-4 balanced sampler
    n_classes = train_dataset.n_classes("ec_l4")
    classes_per_batch = min(32, n_classes)
    samples_per_class = max(batch_size // classes_per_batch, 2)
    sampler = BalancedBatchSampler(
        train_dataset.get_labels("ec_l4"), batch_size,
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class
    )

    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_indices in sampler:
            batch_emb = all_emb[batch_indices]
            projected = model(batch_emb)

            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for lv, w in zip(levels, weights):
                batch_labels = all_labels[lv][batch_indices]
                lv_loss = loss_fns[lv](projected, batch_labels)
                total_loss = total_loss + w * lv_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append({"total": avg_loss})

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

    # Evaluate
    results = {"seed": seed, "method": "joint_hierarchical", "loss_history": loss_history}
    for test_name, test_ds in test_datasets.items():
        metrics = evaluate_model(model, train_dataset, test_ds,
                                level="ec_l4", rare_classes_str=rare_classes_str,
                                device=device)
        results[test_name] = metrics

    return results, model


def run_currec(seed, train_dataset, test_datasets, rare_classes_str,
               phase_config=None, consistency_weight=0.5, temp_schedule=True,
               device="cuda"):
    """Run CurrEC: curriculum contrastive learning.

    Args:
        phase_config: list of dicts with keys: level, epochs, lr, temperature
                      If None, uses default 4-phase config.
        consistency_weight: lambda for consistency regularization
        temp_schedule: whether to use progressive temperature scheduling
    """
    set_seed(seed)
    model = ProjectionHead().to(device)

    tau_base = 0.1
    gamma = 0.85

    if phase_config is None:
        phase_config = [
            {"level": "ec_l1", "epochs": 20, "lr": 5e-4, "temp": tau_base * gamma**0 if temp_schedule else tau_base},
            {"level": "ec_l2", "epochs": 15, "lr": 3e-4, "temp": tau_base * gamma**1 if temp_schedule else tau_base},
            {"level": "ec_l3", "epochs": 15, "lr": 1e-4, "temp": tau_base * gamma**2 if temp_schedule else tau_base},
            {"level": "ec_l4", "epochs": 40, "lr": 5e-5, "temp": tau_base * gamma**3 if temp_schedule else tau_base},
        ]

    all_loss_history = []
    phase_results = []

    # Determine consistency levels for each phase
    completed_levels = []

    for phase_idx, config in enumerate(phase_config):
        level = config["level"]
        epochs = config["epochs"]
        lr = config["lr"]
        temp = config["temp"]

        print(f"\n  Phase {phase_idx+1}: {level}, {epochs} epochs, lr={lr}, temp={temp:.4f}")

        # Consistency levels: all previously completed levels
        consistency_levels = list(completed_levels) if consistency_weight > 0 else None

        model, loss_history = train_phase(
            model, train_dataset, level=level,
            num_epochs=epochs, lr=lr, temperature=temp,
            consistency_levels=consistency_levels,
            consistency_weight=consistency_weight,
            device=device
        )

        all_loss_history.append({
            "phase": phase_idx + 1,
            "level": level,
            "loss_history": loss_history
        })

        # Evaluate after this phase
        phase_metrics = {}
        for test_name, test_ds in test_datasets.items():
            metrics = evaluate_model(model, train_dataset, test_ds,
                                    level="ec_l4", rare_classes_str=rare_classes_str,
                                    device=device)
            phase_metrics[test_name] = metrics

        phase_results.append({
            "phase": phase_idx + 1,
            "level": level,
            "metrics": phase_metrics
        })
        print(f"    After Phase {phase_idx+1}: "
              f"New-392 F1={phase_metrics.get('new392', {}).get('macro_f1', 0):.4f}, "
              f"Price-149 F1={phase_metrics.get('price149', {}).get('macro_f1', 0):.4f}")

        completed_levels.append(level)

    # Final results
    results = {
        "seed": seed,
        "method": "currec",
        "consistency_weight": consistency_weight,
        "temp_schedule": temp_schedule,
        "phase_config": [{k: v for k, v in c.items()} for c in phase_config],
        "all_loss_history": all_loss_history,
        "phase_results": phase_results,
    }

    # Final metrics (from last phase)
    for test_name, test_ds in test_datasets.items():
        results[test_name] = phase_results[-1]["metrics"].get(test_name, {})

    return results, model
