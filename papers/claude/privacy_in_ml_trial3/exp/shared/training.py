"""Training loops for standard and DP-SGD training."""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune_utils
import numpy as np
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from collections import defaultdict


def train_standard(model, train_loader, val_loader, config, device):
    """Standard training without differential privacy.

    config: dict with lr, momentum, weight_decay, epochs, patience
    Returns: trained model, training log
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                 momentum=config.get("momentum", 0.9),
                                 weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    patience = config.get("patience", 5)
    log = []

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            images, labels, subgroups = batch
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += len(images)

        scheduler.step()

        val_loss, val_acc = _validate(model, val_loader, criterion, device)
        train_loss /= train_total
        train_acc = train_correct / train_total

        log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, log


def train_dp(model, train_loader, val_loader, dp_config, device, log_grad_norms=False,
             val_subgroup_loader=None):
    """DP-SGD training using Opacus.

    dp_config: dict with target_epsilon, target_delta, max_grad_norm,
               epochs, lr, max_physical_batch_size
    log_grad_norms: if True, log per-subgroup gradient norms EVERY epoch
                    using a dedicated calibration pass with 200+ samples
    val_subgroup_loader: separate loader for per-subgroup val accuracy logging
    Returns: trained model, training log, final epsilon, grad_norm_log (or None)
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=dp_config["lr"])
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader_dp = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=dp_config["epochs"],
        target_epsilon=dp_config["target_epsilon"],
        target_delta=dp_config["target_delta"],
        max_grad_norm=dp_config["max_grad_norm"],
    )

    log = []
    grad_norm_log = [] if log_grad_norms else None
    max_physical_batch_size = dp_config.get("max_physical_batch_size", 128)

    for epoch in range(dp_config["epochs"]):
        model.train()
        train_loss = 0
        train_total = 0
        train_correct = 0

        with BatchMemoryManager(
            data_loader=train_loader_dp,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for batch in memory_safe_data_loader:
                images, labels, subgroups = batch
                images = images.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(images)
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += len(images)

        eps = privacy_engine.get_epsilon(dp_config["target_delta"])
        val_loss, val_acc = _validate(model, val_loader, criterion, device)
        train_loss_avg = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Per-subgroup val accuracy
        per_sg_acc = None
        if val_subgroup_loader is not None:
            per_sg_acc = _compute_subgroup_accuracy(model, val_subgroup_loader, device)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epsilon_spent": eps,
        }
        if per_sg_acc is not None:
            epoch_log["per_subgroup_val_acc"] = per_sg_acc
        log.append(epoch_log)

        # Gradient norm logging: every 3 epochs, dedicated calibration pass
        if log_grad_norms and (epoch % 3 == 0 or epoch == dp_config["epochs"] - 1):
            epoch_grad_log = _compute_epoch_grad_norms(
                model, train_loader, criterion, dp_config["max_grad_norm"],
                device, n_samples=150, epoch=epoch
            )
            if epoch_grad_log:
                grad_norm_log.append(epoch_grad_log)

        if (epoch + 1) % 5 == 0 or epoch == dp_config["epochs"] - 1:
            print(f"  Epoch {epoch+1}/{dp_config['epochs']}: "
                  f"train_loss={train_loss_avg:.4f}, val_acc={val_acc:.4f}, eps={eps:.2f}")

    final_epsilon = privacy_engine.get_epsilon(dp_config["target_delta"])

    if hasattr(model, '_module'):
        model = model._module

    return model, log, final_epsilon, grad_norm_log


def _compute_epoch_grad_norms(model, data_loader, criterion, max_grad_norm, device,
                               n_samples=150, epoch=0, model_constructor=None,
                               num_classes=None):
    """Compute per-subgroup gradient norms using a dedicated calibration pass.

    Creates a temporary clean model (no Opacus hooks) to avoid conflicts.
    Uses up to n_samples from the data loader, computing per-sample gradient norms
    grouped by subgroup.
    """
    # Extract clean state dict from Opacus-wrapped model
    base_model = model._module if hasattr(model, '_module') else model
    state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    # Create a clean temporary model without Opacus hooks
    from shared.models import get_model as _get_model
    if num_classes is None:
        # Infer from the fc layer
        num_classes = base_model.fc.out_features
    temp_model = _get_model("resnet18", num_classes).to(device)
    temp_model.load_state_dict(state_dict)
    temp_model.eval()

    subgroup_norms = defaultdict(list)
    count = 0
    crit = nn.CrossEntropyLoss()

    for batch in data_loader:
        if count >= n_samples:
            break
        images, labels, subgroups = batch
        if isinstance(subgroups, torch.Tensor):
            sg_np = subgroups.numpy()
        else:
            sg_np = np.array(subgroups)
        if isinstance(labels, torch.Tensor):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)

        # Process individual samples
        for i in range(min(len(images), n_samples - count)):
            img = images[i:i+1].to(device)
            lab = torch.tensor([int(labels_np[i])], dtype=torch.long, device=device)
            sg = int(sg_np[i])

            temp_model.zero_grad()
            out = temp_model(img)
            loss = crit(out, lab)
            loss.backward()

            total_norm = 0.0
            for p in temp_model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            subgroup_norms[sg].append(total_norm)
            count += 1

    del temp_model
    torch.cuda.empty_cache()

    if not subgroup_norms:
        return None

    epoch_grad_log = {"epoch": epoch}
    for sg, norms in subgroup_norms.items():
        norms_arr = np.array(norms)
        epoch_grad_log[f"subgroup_{sg}_mean_norm"] = float(norms_arr.mean())
        epoch_grad_log[f"subgroup_{sg}_std_norm"] = float(norms_arr.std())
        epoch_grad_log[f"subgroup_{sg}_median_norm"] = float(np.median(norms_arr))
        epoch_grad_log[f"subgroup_{sg}_n_samples"] = len(norms)
        clip_frac = float((norms_arr > max_grad_norm).mean())
        epoch_grad_log[f"subgroup_{sg}_clip_fraction"] = clip_frac
    return epoch_grad_log


def _compute_subgroup_accuracy(model, data_loader, device):
    """Compute per-subgroup accuracy on a data loader.
    Uses the underlying module to avoid Opacus hook interference.
    """
    base_model = model._module if hasattr(model, '_module') else model
    # Temporarily disable Opacus grad sample hooks during evaluation
    was_training = base_model.training
    base_model.eval()
    sg_correct = defaultdict(int)
    sg_total = defaultdict(int)
    with torch.no_grad():
        for batch in data_loader:
            images, labels, subgroups = batch
            images = images.to(device)
            labels_t = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
            # Use base_model directly (no_grad avoids Opacus hook issues)
            outputs = base_model(images)
            preds = outputs.argmax(1)
            correct = (preds == labels_t)
            if isinstance(subgroups, torch.Tensor):
                sg_np = subgroups.numpy()
            else:
                sg_np = np.array(subgroups)
            for i in range(len(sg_np)):
                sg = int(sg_np[i])
                sg_correct[sg] += int(correct[i].item())
                sg_total[sg] += 1
    if was_training:
        base_model.train()
    return {sg: sg_correct[sg] / max(sg_total[sg], 1) for sg in sorted(sg_total.keys())}


def finetune_with_masks(model, train_loader, config, device):
    """Fine-tune a pruned model. Pruning hooks maintain sparsity automatically.

    The model should have active pruning masks (from prune.custom_from_mask or
    prune.global_unstructured without prune.remove). The forward hooks ensure
    that pruned weights stay zero during training.

    After fine-tuning, call compression.finalize_pruning(model) to make masks permanent.
    """
    model = model.to(device)

    # Only optimize non-pruned parameters (all params, hooks handle masking)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.get("ft_lr", 0.001),
                                 momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ft_epochs = config.get("ft_epochs", 5)
    for epoch in range(ft_epochs):
        model.train()
        for batch in train_loader:
            images, labels, subgroups = batch
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Zero out gradients for pruned weights (belt-and-suspenders with hooks)
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight_mask'):
                        if module.weight_orig.grad is not None:
                            module.weight_orig.grad *= module.weight_mask

            optimizer.step()

    return model


# Keep old name as alias for backward compatibility
def finetune_standard(model, train_loader, config, device):
    """Alias for finetune_with_masks."""
    return finetune_with_masks(model, train_loader, config, device)


def _validate(model, val_loader, criterion, device):
    """Compute validation loss and accuracy."""
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels, subgroups = batch
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * len(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += len(images)
    return val_loss / max(val_total, 1), val_correct / max(val_total, 1)
