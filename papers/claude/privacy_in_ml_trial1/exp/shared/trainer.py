"""Training and fine-tuning utilities."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm


def train_model(model, train_loader, test_loader, epochs=100, lr=0.1,
                weight_decay=5e-4, device='cuda', mask_dict=None,
                verbose=True, scheduler_type='cosine'):
    """
    Train or fine-tune a model.

    Args:
        model: model to train
        train_loader: training DataLoader
        test_loader: test DataLoader
        epochs: number of training epochs
        lr: initial learning rate
        weight_decay: L2 regularization
        device: cuda or cpu
        mask_dict: optional binary mask for pruned weights
        verbose: print progress
        scheduler_type: 'cosine' or 'step'

    Returns:
        dict with training history and final metrics
    """
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply mask to gradients if pruning
            if mask_dict is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in mask_dict and param.grad is not None:
                            param.grad.mul_(mask_dict[name].to(device))

            optimizer.step()

            # Re-apply mask to weights
            if mask_dict is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in mask_dict:
                            param.mul_(mask_dict[name].to(device))

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%")

    return {
        'best_test_acc': best_acc,
        'final_test_acc': history['test_acc'][-1],
        'final_train_acc': history['train_acc'][-1],
        'best_state_dict': best_state,
        'history': history,
    }


def evaluate(model, data_loader, device='cuda'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def evaluate_per_class(model, data_loader, num_classes, device='cuda'):
    """Evaluate per-class accuracy."""
    model.eval()
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for c in range(num_classes):
                mask = targets == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (predicted[mask] == targets[mask]).sum().item()

    per_class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc[c] = 100.0 * class_correct[c] / class_total[c]
    return per_class_acc


def train_with_dp_sgd(model, train_loader, test_loader, epochs=20, lr=0.01,
                      target_epsilon=8.0, target_delta=1e-5, max_grad_norm=1.0,
                      device='cuda', mask_dict=None, verbose=True):
    """
    Fine-tune model with DP-SGD using Opacus.
    """
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    model = model.to(device)

    # Make model compatible with Opacus
    model = ModuleValidator.fix(model)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader_dp = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader_dp:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            # Re-apply mask to weights
            if mask_dict is not None:
                with torch.no_grad():
                    for name, param in model._module.named_parameters():
                        full_name = name
                        if full_name in mask_dict:
                            param.mul_(mask_dict[full_name].to(device))

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 5 == 0:
            eps = privacy_engine.get_epsilon(target_delta)
            print(f"  DP Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%, eps={eps:.2f}")

    final_epsilon = privacy_engine.get_epsilon(target_delta)

    # Unwrap model
    unwrapped_model = model._module if hasattr(model, '_module') else model

    return {
        'best_test_acc': best_acc,
        'final_test_acc': test_acc,
        'final_train_acc': train_acc,
        'final_epsilon': final_epsilon,
        'best_state_dict': best_state,
        'model': unwrapped_model,
    }
