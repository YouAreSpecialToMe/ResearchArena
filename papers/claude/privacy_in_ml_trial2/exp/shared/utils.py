import os
import json
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_classification

from .config import *
from .models import get_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar_transforms(dataset):
    if dataset == 'cifar10':
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, test_transform


def load_dataset(dataset, root='exp/data'):
    os.makedirs(root, exist_ok=True)
    if dataset == 'cifar10':
        train_transform, test_transform = get_cifar_transforms('cifar10')
        train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
        # Also load without augmentation for eval
        eval_transform = test_transform
        train_eval_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=eval_transform)
        return train_ds, test_ds, train_eval_ds

    elif dataset == 'cifar100':
        train_transform, test_transform = get_cifar_transforms('cifar100')
        train_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        test_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
        eval_transform = test_transform
        train_eval_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=eval_transform)
        return train_ds, test_ds, train_eval_ds

    elif dataset == 'purchase100':
        npz_path = os.path.join(root, 'purchase100_v2.npz')
        if not os.path.exists(npz_path):
            print("Generating Purchase-100 proxy dataset (v2 - fixed)...")
            X, y = make_classification(
                n_samples=50000, n_features=600, n_informative=400,
                n_redundant=100, n_classes=100, n_clusters_per_class=1,
                class_sep=2.0, flip_y=0.01, shuffle=True, random_state=0
            )
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X).astype(np.float32)
            # Shuffle with fixed seed to ensure train/test have same distribution
            perm = np.random.RandomState(42).permutation(len(X))
            X, y = X[perm], y[perm]
            np.savez(npz_path, X=X, y=y)
        else:
            data = np.load(npz_path)
            X, y = data['X'].astype(np.float32), data['y']

        X_train, y_train = X[:40000], y[:40000]
        X_test, y_test = X[40000:], y[40000:]

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
        train_eval_ds = train_ds  # No augmentation difference for tabular
        return train_ds, test_ds, train_eval_ds
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_splits(dataset, train_ds, seed, forget_size=FORGET_SIZE, ref_pool_size=REF_POOL_SIZE):
    """Create forget/retain/ref_pool splits."""
    n_train = len(train_ds)
    ref_pool_indices = list(range(ref_pool_size))
    non_ref_indices = list(range(ref_pool_size, n_train))

    rng = np.random.RandomState(seed)
    forget_indices = sorted(rng.choice(non_ref_indices, size=forget_size, replace=False).tolist())
    retain_indices = sorted(set(non_ref_indices) - set(forget_indices))

    return {
        'forget_indices': forget_indices,
        'retain_indices': retain_indices,
        'ref_pool_indices': ref_pool_indices,
        'non_ref_indices': non_ref_indices,
    }


def train_model(model, train_loader, dataset, epochs=None, lr=None, device=DEVICE, verbose=True):
    """Train a model with dataset-appropriate hyperparams."""
    if dataset in ('cifar10', 'cifar100'):
        epochs = epochs or CIFAR_EPOCHS
        lr = lr or 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:  # purchase100
        epochs = epochs or PURCHASE_EPOCHS
        lr = lr or 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        if scheduler:
            scheduler.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")

    return model


def evaluate_model(model, data_loader, device=DEVICE):
    """Return accuracy and per-sample losses."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses, all_correct = [], []

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            loss = criterion(out, y)
            all_losses.append(loss.cpu())
            all_correct.append((out.argmax(1) == y).cpu())

    losses = torch.cat(all_losses).numpy()
    correct = torch.cat(all_correct).numpy()
    acc = correct.mean()
    return acc, losses


def get_loader(dataset_obj, indices=None, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS):
    if indices is not None:
        subset = Subset(dataset_obj, indices)
    else:
        subset = dataset_obj
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=(num_workers > 0))
