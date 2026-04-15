#!/usr/bin/env python3
"""
Clean experiment pipeline for Difficulty-Calibrated Unlearning Auditing (DCUA).
Addresses all reviewer feedback from attempt 1:
  - Fix unlearning hyperparams so methods ACTUALLY unlearn
  - Fix Purchase-100 reference model quality
  - Report DAU results honestly (including negative results)
  - Pivot focus to DCUA evaluation framework as main contribution
"""
import os
import sys
import json
import time
import copy
import random
import warnings
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 456]
REF_SEEDS = [100, 101, 102, 103]
BATCH_SIZE = 512
FORGET_SIZE = 1000
REF_POOL_SIZE = 10000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models_v3')
RESULT_DIR = os.path.join(BASE_DIR, 'results_v3')

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# Dataset-specific training configs
TRAIN_CONFIG = {
    'cifar10': {
        'epochs': 100, 'lr': 0.1, 'optimizer': 'sgd',
        'momentum': 0.9, 'weight_decay': 5e-4, 'scheduler': 'cosine',
        'num_classes': 10,
    },
    'cifar100': {
        'epochs': 100, 'lr': 0.1, 'optimizer': 'sgd',
        'momentum': 0.9, 'weight_decay': 5e-4, 'scheduler': 'cosine',
        'num_classes': 100,
    },
    'purchase100': {
        'epochs': 100, 'lr': 0.001, 'optimizer': 'adam',
        'weight_decay': 1e-4, 'scheduler': None,
        'num_classes': 100,
    },
}

# FIXED unlearning configs - stronger than attempt 1
# Key insight: methods must actually modify the model to unlearn
UNLEARN_CONFIG = {
    'ft': {
        'cifar10': {'epochs': 10, 'lr': 0.01},
        'cifar100': {'epochs': 10, 'lr': 0.01},
        'purchase100': {'epochs': 10, 'lr': 0.005},
    },
    'ga': {
        'cifar10': {'epochs': 10, 'lr': 0.001},
        'cifar100': {'epochs': 10, 'lr': 0.001},
        'purchase100': {'epochs': 10, 'lr': 0.001},
    },
    'rl': {
        'cifar10': {'epochs': 10, 'lr': 0.005},
        'cifar100': {'epochs': 10, 'lr': 0.005},
        'purchase100': {'epochs': 10, 'lr': 0.005},
    },
    'scrub': {
        'cifar10': {'passes': 10, 'forget_lr': 0.001, 'retain_lr': 0.001},
        'cifar100': {'passes': 10, 'forget_lr': 0.001, 'retain_lr': 0.001},
        'purchase100': {'passes': 10, 'forget_lr': 0.001, 'retain_lr': 0.001},
    },
    'neggrad': {
        'cifar10': {'epochs': 10, 'lr': 0.001},
        'cifar100': {'epochs': 10, 'lr': 0.001},
        'purchase100': {'epochs': 10, 'lr': 0.001},
    },
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# MODELS
# ============================================================================
def get_resnet18_cifar(num_classes=10):
    import torchvision.models as models
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

class PurchaseMLP(nn.Module):
    def __init__(self, n_features=600, n_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.net(x)

def get_model(dataset):
    if dataset == 'cifar10':
        return get_resnet18_cifar(10).to(DEVICE)
    elif dataset == 'cifar100':
        return get_resnet18_cifar(100).to(DEVICE)
    elif dataset == 'purchase100':
        return PurchaseMLP(600, 100).to(DEVICE)

# ============================================================================
# DATA
# ============================================================================
def get_cifar_transforms(dataset):
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_t, test_t

def load_dataset(dataset):
    os.makedirs(DATA_DIR, exist_ok=True)
    if dataset in ('cifar10', 'cifar100'):
        train_t, test_t = get_cifar_transforms(dataset)
        cls = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
        train_ds = cls(root=DATA_DIR, train=True, download=True, transform=train_t)
        test_ds = cls(root=DATA_DIR, train=False, download=True, transform=test_t)
        train_eval_ds = cls(root=DATA_DIR, train=True, download=False, transform=test_t)
        return train_ds, test_ds, train_eval_ds
    elif dataset == 'purchase100':
        npz_path = os.path.join(DATA_DIR, 'purchase100_v3.npz')
        if not os.path.exists(npz_path):
            log("Generating Purchase-100 proxy dataset v3...")
            X, y = make_classification(
                n_samples=50000, n_features=600, n_informative=400,
                n_redundant=100, n_classes=100, n_clusters_per_class=1,
                class_sep=2.0, flip_y=0.01, shuffle=True, random_state=0
            )
            scaler = StandardScaler()
            X = scaler.fit_transform(X).astype(np.float32)
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
        return train_ds, test_ds, train_ds  # No augmentation for tabular

def create_splits(n_train, seed):
    ref_pool = list(range(REF_POOL_SIZE))
    non_ref = list(range(REF_POOL_SIZE, n_train))
    rng = np.random.RandomState(seed)
    forget = sorted(rng.choice(non_ref, size=FORGET_SIZE, replace=False).tolist())
    retain = sorted(set(non_ref) - set(forget))
    return {'forget': forget, 'retain': retain, 'ref_pool': ref_pool, 'non_ref': non_ref}

def get_loader(ds, indices=None, batch_size=BATCH_SIZE, shuffle=True):
    subset = Subset(ds, indices) if indices is not None else ds
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

# ============================================================================
# TRAINING
# ============================================================================
def train_model(model, train_loader, dataset, epochs=None, lr=None, verbose=True):
    cfg = TRAIN_CONFIG[dataset]
    epochs = epochs or cfg['epochs']
    lr = lr or cfg['lr']

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg['momentum'],
                             weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cfg.get('scheduler') == 'cosine' else None
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
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
        if verbose and (epoch + 1) % 25 == 0:
            log(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")
    return model

def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            all_losses.append(criterion(out, y).cpu())
            all_preds.append(out.argmax(1).cpu())
            all_labels.append(y.cpu())
    losses = torch.cat(all_losses).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    return acc, losses

# ============================================================================
# UNLEARNING METHODS (with stronger hyperparams)
# ============================================================================
def unlearn_ft(model, forget_loader, retain_loader, dataset, sample_weights=None):
    """Fine-tune on retain set only."""
    cfg = UNLEARN_CONFIG['ft'][dataset]
    model = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(cfg['epochs']):
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model

def unlearn_ga(model, forget_loader, retain_loader, dataset, sample_weights=None):
    """Gradient ascent on forget set — STRONGER than attempt 1."""
    cfg = UNLEARN_CONFIG['ga'][dataset]
    model = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.train()
    for epoch in range(cfg['epochs']):
        for batch_idx, (x, y) in enumerate(forget_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            losses = criterion(model(x), y)
            if sample_weights is not None:
                start = batch_idx * BATCH_SIZE
                end = start + x.size(0)
                w = sample_weights[start:end].to(DEVICE)
                losses = losses * w
            loss = -losses.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model

def unlearn_rl(model, forget_loader, retain_loader, dataset, num_classes=10, sample_weights=None):
    """Random labels on forget set + true labels on retain set."""
    cfg = UNLEARN_CONFIG['rl'][dataset]
    model = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(cfg['epochs']):
        for batch_idx, (x, y) in enumerate(forget_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            fake_y = torch.randint(0, num_classes, (x.size(0),), device=DEVICE)
            optimizer.zero_grad()
            losses = criterion_none(model(x), fake_y)
            if sample_weights is not None:
                start = batch_idx * BATCH_SIZE
                end = start + x.size(0)
                w = sample_weights[start:end].to(DEVICE)
                losses = losses * w
            loss = losses.mean()
            loss.backward()
            optimizer.step()
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model

def unlearn_scrub(model, forget_loader, retain_loader, dataset, sample_weights=None):
    """SCRUB with MORE passes and higher LR than attempt 1."""
    cfg = UNLEARN_CONFIG['scrub'][dataset]
    teacher = copy.deepcopy(model)
    teacher.eval()
    student = copy.deepcopy(model)

    if dataset in ('cifar10', 'cifar100'):
        opt_forget = optim.SGD(student.parameters(), lr=cfg['forget_lr'], momentum=0.9, weight_decay=5e-4)
        opt_retain = optim.SGD(student.parameters(), lr=cfg['retain_lr'], momentum=0.9, weight_decay=5e-4)
    else:
        opt_forget = optim.Adam(student.parameters(), lr=cfg['forget_lr'], weight_decay=1e-4)
        opt_retain = optim.Adam(student.parameters(), lr=cfg['retain_lr'], weight_decay=1e-4)

    student.train()
    for pass_idx in range(cfg['passes']):
        # Maximize divergence from teacher on forget set (multiple batches now)
        for batch_idx, (x, y) in enumerate(forget_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt_forget.zero_grad()
            logits = student(x)
            n_classes = logits.shape[1]
            uniform = torch.full_like(logits, 1.0 / n_classes)
            per_sample = F.kl_div(F.log_softmax(logits, dim=1), uniform, reduction='none').sum(dim=1)
            if sample_weights is not None:
                start = batch_idx * BATCH_SIZE
                end = start + x.size(0)
                w = sample_weights[start:end].to(DEVICE)
                per_sample = per_sample * w
            loss = -per_sample.mean()
            loss.backward()
            opt_forget.step()

        # Minimize divergence from teacher on retain set (3 batches)
        retain_iter = iter(retain_loader)
        for _ in range(3):
            try:
                x, y = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                x, y = next(retain_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt_retain.zero_grad()
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = F.kl_div(F.log_softmax(s_logits, dim=1), F.softmax(t_logits, dim=1), reduction='batchmean')
            loss.backward()
            opt_retain.step()
    return student

def unlearn_neggrad(model, forget_loader, retain_loader, dataset, sample_weights=None):
    """NegGrad + KD."""
    cfg = UNLEARN_CONFIG['neggrad'][dataset]
    teacher = copy.deepcopy(model)
    teacher.eval()
    student = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        optimizer = optim.SGD(student.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(student.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    student.train()
    for epoch in range(cfg['epochs']):
        retain_iter = iter(retain_loader)
        for batch_idx, (x, y) in enumerate(forget_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            losses = criterion(student(x), y)
            if sample_weights is not None:
                start = batch_idx * BATCH_SIZE
                end = start + x.size(0)
                w = sample_weights[start:end].to(DEVICE)
                losses = losses * w
            loss_forget = -0.5 * losses.mean()
            try:
                rx, ry = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rx, ry = next(retain_iter)
            rx, ry = rx.to(DEVICE), ry.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(rx)
            s_logits = student(rx)
            loss_retain = 0.5 * F.kl_div(F.log_softmax(s_logits, dim=1), F.softmax(t_logits, dim=1), reduction='batchmean')
            loss = loss_forget + loss_retain
            loss.backward()
            optimizer.step()
    return student

UNLEARN_FNS = {
    'ft': unlearn_ft,
    'ga': unlearn_ga,
    'rl': unlearn_rl,
    'scrub': unlearn_scrub,
    'neggrad': unlearn_neggrad,
}

# ============================================================================
# MIA
# ============================================================================
def loss_mia(member_losses, nonmember_losses):
    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5

def calibrated_mia(member_losses, nonmember_losses, member_ref, nonmember_ref):
    m_scores = -(member_losses - member_ref)
    nm_scores = -(nonmember_losses - nonmember_ref)
    scores = np.concatenate([m_scores, nm_scores])
    labels = np.concatenate([np.ones(len(m_scores)), np.zeros(len(nm_scores))])
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5

def lira_mia(member_losses, nonmember_losses, member_ref_list, nonmember_ref_list):
    ref_m = np.stack(member_ref_list, axis=0)
    ref_nm = np.stack(nonmember_ref_list, axis=0)
    m_mean = ref_m.mean(axis=0)
    m_std = ref_m.std(axis=0) + 1e-8
    nm_mean = ref_nm.mean(axis=0)
    nm_std = ref_nm.std(axis=0) + 1e-8
    m_scores = -(member_losses - m_mean) / m_std
    nm_scores = -(nonmember_losses - nm_mean) / nm_std
    scores = np.concatenate([m_scores, nm_scores])
    labels = np.concatenate([np.ones(len(m_scores)), np.zeros(len(nm_scores))])
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5

def run_mia(member_losses, nonmember_losses, member_ref_mean, nonmember_ref_mean,
            member_ref_list, nonmember_ref_list):
    la = loss_mia(member_losses, nonmember_losses)
    ca = calibrated_mia(member_losses, nonmember_losses, member_ref_mean, nonmember_ref_mean)
    li = lira_mia(member_losses, nonmember_losses, member_ref_list, nonmember_ref_list)
    return {'loss_auc': la, 'calibrated_auc': ca, 'lira_auc': li, 'best_auc': max(la, ca, li)}

def stratified_mia(member_losses, nonmember_losses,
                   member_ref_mean, nonmember_ref_mean,
                   member_ref_list, nonmember_ref_list,
                   member_quintiles, nonmember_quintiles, n_strata=5):
    results = {}
    for q in range(n_strata):
        mm = member_quintiles == q
        nm = nonmember_quintiles == q
        if mm.sum() < 10 or nm.sum() < 10:
            results[f'q{q+1}'] = {'best_auc': 0.5, 'n_members': int(mm.sum()), 'n_nonmembers': int(nm.sum())}
            continue
        ml, nml = member_losses[mm], nonmember_losses[nm]
        mrm, nmrm = member_ref_mean[mm], nonmember_ref_mean[nm]
        mrl = [r[mm] for r in member_ref_list]
        nmrl = [r[nm] for r in nonmember_ref_list]
        aucs = run_mia(ml, nml, mrm, nmrm, mrl, nmrl)
        aucs['n_members'] = int(mm.sum())
        aucs['n_nonmembers'] = int(nm.sum())
        results[f'q{q+1}'] = aucs

    agg = run_mia(member_losses, nonmember_losses, member_ref_mean, nonmember_ref_mean,
                  member_ref_list, nonmember_ref_list)
    results['aggregate'] = agg
    best_aucs = [results[f'q{q+1}']['best_auc'] for q in range(n_strata)]
    results['wq_auc'] = best_aucs[-1]
    results['dg'] = best_aucs[-1] - best_aucs[0]
    results['max_spread'] = max(best_aucs) - min(best_aucs)
    results['per_quintile_aucs'] = best_aucs
    return results

# ============================================================================
# DAU WEIGHTS
# ============================================================================
def compute_dau_weights(difficulty_scores, alpha=1.0):
    d_mean = difficulty_scores.mean()
    d_std = difficulty_scores.std() + 1e-8
    weights = 1.0 + alpha * (difficulty_scores - d_mean) / d_std
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

def main():
    start_time = time.time()
    all_results = {}

    # ========================================================================
    # PHASE 1: Data preparation + Reference models + Difficulty scores
    # ========================================================================
    log("=" * 70)
    log("PHASE 1: Data preparation + Reference models")
    log("=" * 70)

    datasets_info = {}
    ref_models = {}
    difficulty_scores = {}
    difficulty_quintiles = {}

    for ds_name in ['cifar10', 'cifar100', 'purchase100']:
        log(f"\n--- Dataset: {ds_name} ---")
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)
        n_train = len(train_ds)
        n_test = len(test_ds)
        num_classes = TRAIN_CONFIG[ds_name]['num_classes']
        log(f"  Train: {n_train}, Test: {n_test}, Classes: {num_classes}")

        datasets_info[ds_name] = {
            'train_ds': train_ds, 'test_ds': test_ds, 'train_eval_ds': train_eval_ds,
            'n_train': n_train, 'n_test': n_test, 'num_classes': num_classes,
        }

        # Train K=4 reference models
        ref_models[ds_name] = []
        ref_quality = []
        non_ref_indices = list(range(REF_POOL_SIZE, n_train))

        for k, rseed in enumerate(REF_SEEDS):
            model_path = os.path.join(MODEL_DIR, f'ref_{ds_name}_{rseed}.pt')
            if os.path.exists(model_path):
                log(f"  Loading reference model {k+1}/4 (seed {rseed})")
                model = get_model(ds_name)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            else:
                log(f"  Training reference model {k+1}/4 (seed {rseed})")
                set_seed(rseed)
                rng = np.random.RandomState(rseed)
                ref_train_idx = sorted(rng.choice(non_ref_indices,
                                                   size=int(0.8 * len(non_ref_indices)),
                                                   replace=False).tolist())
                model = get_model(ds_name)
                loader = get_loader(train_ds, ref_train_idx)
                model = train_model(model, loader, ds_name, verbose=True)
                torch.save(model.state_dict(), model_path)

            # Evaluate
            test_loader = get_loader(test_ds, shuffle=False)
            test_acc, _ = evaluate(model, test_loader)
            ref_quality.append(test_acc)
            ref_models[ds_name].append(model)
            log(f"    Test acc: {test_acc:.4f}")

        log(f"  Reference model quality: {[f'{a:.4f}' for a in ref_quality]}")
        save_json({'dataset': ds_name, 'ref_quality': ref_quality},
                  os.path.join(RESULT_DIR, f'ref_quality_{ds_name}.json'))

        # Compute difficulty scores for ALL training samples
        log(f"  Computing difficulty scores...")
        all_train_losses = []
        train_eval_loader = get_loader(train_eval_ds, shuffle=False)
        for ref_model in ref_models[ds_name]:
            _, losses = evaluate(ref_model, train_eval_loader)
            all_train_losses.append(losses)
        # Average across ref models
        diff_scores = np.mean(all_train_losses, axis=0)
        difficulty_scores[ds_name] = diff_scores

        # Also compute for test set
        test_loader = get_loader(test_ds, shuffle=False)
        all_test_losses = []
        for ref_model in ref_models[ds_name]:
            _, losses = evaluate(ref_model, test_loader)
            all_test_losses.append(losses)
        test_diff_scores = np.mean(all_test_losses, axis=0)
        difficulty_scores[f'{ds_name}_test'] = test_diff_scores

        # Store per-ref-model losses for LiRA
        difficulty_scores[f'{ds_name}_train_per_ref'] = all_train_losses
        difficulty_scores[f'{ds_name}_test_per_ref'] = all_test_losses

        # Assign quintiles
        percentiles = np.percentile(diff_scores, [20, 40, 60, 80])
        quints = np.digitize(diff_scores, percentiles)  # 0-4
        difficulty_quintiles[ds_name] = quints

        # Test set quintiles (matched to training distribution)
        test_quints = np.digitize(test_diff_scores, percentiles)
        difficulty_quintiles[f'{ds_name}_test'] = test_quints

        # Validate: Q5 should have lower ref accuracy than Q1
        for q in range(5):
            mask = quints == q
            q_mean = diff_scores[mask].mean()
            log(f"    Q{q+1}: n={mask.sum()}, mean_difficulty={q_mean:.4f}")

    log(f"\nPhase 1 complete. Time: {(time.time()-start_time)/60:.1f} min")

    # ========================================================================
    # PHASE 2: Train original + retrain models
    # ========================================================================
    log("\n" + "=" * 70)
    log("PHASE 2: Train original + retrain models")
    log("=" * 70)

    original_models = {}
    retrain_models = {}
    training_log = []

    for ds_name in ['cifar10', 'cifar100', 'purchase100']:
        info = datasets_info[ds_name]
        for seed in SEEDS:
            splits = create_splits(info['n_train'], seed)

            # Original model
            orig_path = os.path.join(MODEL_DIR, f'orig_{ds_name}_s{seed}.pt')
            if os.path.exists(orig_path):
                log(f"  Loading original {ds_name} seed={seed}")
                model = get_model(ds_name)
                model.load_state_dict(torch.load(orig_path, map_location=DEVICE, weights_only=True))
            else:
                log(f"  Training original {ds_name} seed={seed}")
                set_seed(seed)
                model = get_model(ds_name)
                loader = get_loader(info['train_ds'], splits['non_ref'])
                t0 = time.time()
                model = train_model(model, loader, ds_name, verbose=True)
                torch.save(model.state_dict(), orig_path)
                training_log.append({'type': 'original', 'dataset': ds_name, 'seed': seed, 'time': time.time()-t0})

            test_loader = get_loader(info['test_ds'], shuffle=False)
            test_acc, _ = evaluate(model, test_loader)
            log(f"    Test acc: {test_acc:.4f}")
            original_models[(ds_name, seed)] = model

            # Retrain model (on retain set only)
            retrain_path = os.path.join(MODEL_DIR, f'retrain_{ds_name}_s{seed}.pt')
            if os.path.exists(retrain_path):
                log(f"  Loading retrain {ds_name} seed={seed}")
                model_r = get_model(ds_name)
                model_r.load_state_dict(torch.load(retrain_path, map_location=DEVICE, weights_only=True))
            else:
                log(f"  Training retrain {ds_name} seed={seed}")
                set_seed(seed + 1000)
                model_r = get_model(ds_name)
                loader_r = get_loader(info['train_ds'], splits['retain'])
                t0 = time.time()
                model_r = train_model(model_r, loader_r, ds_name, verbose=True)
                torch.save(model_r.state_dict(), retrain_path)
                training_log.append({'type': 'retrain', 'dataset': ds_name, 'seed': seed, 'time': time.time()-t0})

            test_acc_r, _ = evaluate(model_r, test_loader)
            log(f"    Retrain test acc: {test_acc_r:.4f}")
            retrain_models[(ds_name, seed)] = model_r

    save_json(training_log, os.path.join(RESULT_DIR, 'training_log.json'))
    log(f"\nPhase 2 complete. Time: {(time.time()-start_time)/60:.1f} min")

    # ========================================================================
    # PHASE 3: Run unlearning baselines (with FIXED hyperparams)
    # ========================================================================
    log("\n" + "=" * 70)
    log("PHASE 3: Unlearning baselines (tuned hyperparams)")
    log("=" * 70)

    unlearned_models = {}
    unlearning_results = []

    for ds_name in ['cifar10', 'cifar100', 'purchase100']:
        info = datasets_info[ds_name]
        num_classes = info['num_classes']
        for seed in SEEDS:
            splits = create_splits(info['n_train'], seed)
            orig_model = original_models[(ds_name, seed)]
            forget_loader = get_loader(info['train_ds'], splits['forget'], shuffle=True)
            retain_loader = get_loader(info['train_ds'], splits['retain'], shuffle=True)

            for method in ['ft', 'ga', 'rl', 'scrub', 'neggrad']:
                key = (ds_name, seed, method)
                model_path = os.path.join(MODEL_DIR, f'unlearn_{method}_{ds_name}_s{seed}.pt')

                if os.path.exists(model_path):
                    log(f"  Loading {method} {ds_name} seed={seed}")
                    umodel = get_model(ds_name)
                    umodel.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                else:
                    log(f"  Running {method} on {ds_name} seed={seed}")
                    set_seed(seed)
                    t0 = time.time()
                    fn = UNLEARN_FNS[method]
                    if method == 'rl':
                        umodel = fn(orig_model, forget_loader, retain_loader, ds_name, num_classes=num_classes)
                    else:
                        umodel = fn(orig_model, forget_loader, retain_loader, ds_name)
                    torch.save(umodel.state_dict(), model_path)
                    log(f"    Time: {time.time()-t0:.1f}s")

                # Evaluate utility
                test_loader = get_loader(info['test_ds'], shuffle=False)
                test_acc, _ = evaluate(umodel, test_loader)

                forget_eval_loader = get_loader(info['train_eval_ds'], splits['forget'], shuffle=False)
                forget_acc, _ = evaluate(umodel, forget_eval_loader)

                retain_eval_loader = get_loader(info['train_eval_ds'], splits['retain'], shuffle=False)
                retain_acc, _ = evaluate(umodel, retain_eval_loader)

                random_chance = 1.0 / num_classes
                log(f"    TA={test_acc:.4f}, RA={retain_acc:.4f}, FA={forget_acc:.4f} (random={random_chance:.2f})")

                unlearned_models[key] = umodel
                unlearning_results.append({
                    'method': method, 'dataset': ds_name, 'seed': seed,
                    'test_acc': float(test_acc), 'retain_acc': float(retain_acc),
                    'forget_acc': float(forget_acc), 'random_chance': random_chance,
                })

    save_json(unlearning_results, os.path.join(RESULT_DIR, 'unlearning_baselines.json'))
    log(f"\nPhase 3 complete. Time: {(time.time()-start_time)/60:.1f} min")

    # ========================================================================
    # PHASE 4: DCUA Stratified MIA Evaluation
    # ========================================================================
    log("\n" + "=" * 70)
    log("PHASE 4: DCUA Stratified MIA Evaluation")
    log("=" * 70)

    dcua_results = []

    for ds_name in ['cifar10', 'cifar100', 'purchase100']:
        info = datasets_info[ds_name]
        train_diff = difficulty_scores[ds_name]
        test_diff = difficulty_scores[f'{ds_name}_test']
        train_per_ref = difficulty_scores[f'{ds_name}_train_per_ref']
        test_per_ref = difficulty_scores[f'{ds_name}_test_per_ref']
        train_quints = difficulty_quintiles[ds_name]
        test_quints = difficulty_quintiles[f'{ds_name}_test']

        for seed in SEEDS:
            splits = create_splits(info['n_train'], seed)
            forget_idx = np.array(splits['forget'])

            # Get difficulty info for forget set
            forget_diff = train_diff[forget_idx]
            forget_quints = train_quints[forget_idx]

            # Select matched non-members from test set
            n_test = info['n_test']
            rng = np.random.RandomState(seed)
            nonmember_idx = rng.choice(n_test, size=min(FORGET_SIZE, n_test), replace=False)
            nonmember_diff = test_diff[nonmember_idx]
            nonmember_quints = test_quints[nonmember_idx]

            # Get reference model losses for members and non-members
            member_ref_mean = np.mean([ref_losses[forget_idx] for ref_losses in train_per_ref], axis=0)
            nonmember_ref_mean = np.mean([ref_losses[nonmember_idx] for ref_losses in test_per_ref], axis=0)
            member_ref_list = [ref_losses[forget_idx] for ref_losses in train_per_ref]
            nonmember_ref_list = [ref_losses[nonmember_idx] for ref_losses in test_per_ref]

            # Evaluate all models
            models_to_eval = {}
            models_to_eval['retrain'] = retrain_models[(ds_name, seed)]
            for method in ['ft', 'ga', 'rl', 'scrub', 'neggrad']:
                models_to_eval[method] = unlearned_models[(ds_name, seed, method)]

            forget_eval_loader = get_loader(info['train_eval_ds'], splits['forget'], shuffle=False, batch_size=BATCH_SIZE)
            nonmember_loader = get_loader(info['test_ds'], nonmember_idx.tolist(), shuffle=False, batch_size=BATCH_SIZE)

            for model_name, model in models_to_eval.items():
                _, member_losses = evaluate(model, forget_eval_loader)
                _, nonmember_losses = evaluate(model, nonmember_loader)

                strat = stratified_mia(
                    member_losses, nonmember_losses,
                    member_ref_mean, nonmember_ref_mean,
                    member_ref_list, nonmember_ref_list,
                    forget_quints, nonmember_quints,
                )

                result = {
                    'method': model_name, 'dataset': ds_name, 'seed': seed,
                    'aggregate_auc': strat['aggregate']['best_auc'],
                    'wq_auc': strat['wq_auc'],
                    'dg': strat['dg'],
                    'max_spread': strat['max_spread'],
                    'per_quintile': strat['per_quintile_aucs'],
                }
                dcua_results.append(result)
                log(f"  {ds_name} s{seed} {model_name}: Agg={strat['aggregate']['best_auc']:.4f}, WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")

    save_json(dcua_results, os.path.join(RESULT_DIR, 'dcua_results.json'))
    log(f"\nPhase 4 complete. Time: {(time.time()-start_time)/60:.1f} min")

    # ========================================================================
    # PHASE 5: DAU Defense (honest attempt)
    # ========================================================================
    log("\n" + "=" * 70)
    log("PHASE 5: DAU Defense (honest reporting)")
    log("=" * 70)

    dau_results = []

    for ds_name in ['cifar10', 'cifar100', 'purchase100']:
        info = datasets_info[ds_name]
        train_diff = difficulty_scores[ds_name]
        test_diff = difficulty_scores[f'{ds_name}_test']
        train_per_ref = difficulty_scores[f'{ds_name}_train_per_ref']
        test_per_ref = difficulty_scores[f'{ds_name}_test_per_ref']
        train_quints = difficulty_quintiles[ds_name]
        test_quints = difficulty_quintiles[f'{ds_name}_test']
        num_classes = info['num_classes']

        for seed in SEEDS:
            splits = create_splits(info['n_train'], seed)
            forget_idx = np.array(splits['forget'])
            orig_model = original_models[(ds_name, seed)]
            forget_loader = get_loader(info['train_ds'], splits['forget'], shuffle=False)  # No shuffle for weight alignment
            retain_loader = get_loader(info['train_ds'], splits['retain'], shuffle=True)

            forget_diff = train_diff[forget_idx]
            dau_weights = compute_dau_weights(forget_diff, alpha=1.0)

            # Run DAU for GA and SCRUB (the two most promising candidates)
            for method in ['ga', 'scrub']:
                key = f'{method}_dau'
                model_path = os.path.join(MODEL_DIR, f'unlearn_{key}_{ds_name}_s{seed}.pt')

                if os.path.exists(model_path):
                    log(f"  Loading {key} {ds_name} seed={seed}")
                    umodel = get_model(ds_name)
                    umodel.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                else:
                    log(f"  Running {key} on {ds_name} seed={seed}")
                    set_seed(seed)
                    t0 = time.time()
                    fn = UNLEARN_FNS[method]
                    umodel = fn(orig_model, forget_loader, retain_loader, ds_name, sample_weights=dau_weights)
                    torch.save(umodel.state_dict(), model_path)
                    log(f"    Time: {time.time()-t0:.1f}s")

                # Evaluate
                test_loader = get_loader(info['test_ds'], shuffle=False)
                test_acc, _ = evaluate(umodel, test_loader)
                forget_eval_loader = get_loader(info['train_eval_ds'], splits['forget'], shuffle=False)
                forget_acc, _ = evaluate(umodel, forget_eval_loader)
                retain_eval_loader = get_loader(info['train_eval_ds'], splits['retain'], shuffle=False)
                retain_acc, _ = evaluate(umodel, retain_eval_loader)

                # MIA evaluation
                nonmember_idx = np.random.RandomState(seed).choice(info['n_test'], size=min(FORGET_SIZE, info['n_test']), replace=False)
                member_ref_mean = np.mean([ref_losses[forget_idx] for ref_losses in train_per_ref], axis=0)
                nonmember_ref_mean = np.mean([ref_losses[nonmember_idx] for ref_losses in test_per_ref], axis=0)
                member_ref_list = [ref_losses[forget_idx] for ref_losses in train_per_ref]
                nonmember_ref_list = [ref_losses[nonmember_idx] for ref_losses in test_per_ref]
                forget_quints = train_quints[forget_idx]
                nonmember_quints = test_quints[nonmember_idx]

                _, member_losses = evaluate(umodel, forget_eval_loader)
                nonmember_loader = get_loader(info['test_ds'], nonmember_idx.tolist(), shuffle=False)
                _, nonmember_losses = evaluate(umodel, nonmember_loader)

                strat = stratified_mia(
                    member_losses, nonmember_losses,
                    member_ref_mean, nonmember_ref_mean,
                    member_ref_list, nonmember_ref_list,
                    forget_quints, nonmember_quints,
                )

                result = {
                    'method': key, 'dataset': ds_name, 'seed': seed,
                    'test_acc': float(test_acc), 'retain_acc': float(retain_acc),
                    'forget_acc': float(forget_acc),
                    'aggregate_auc': strat['aggregate']['best_auc'],
                    'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                    'per_quintile': strat['per_quintile_aucs'],
                }
                dau_results.append(result)
                log(f"    TA={test_acc:.4f}, RA={retain_acc:.4f}, FA={forget_acc:.4f}")
                log(f"    Agg={strat['aggregate']['best_auc']:.4f}, WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")

    save_json(dau_results, os.path.join(RESULT_DIR, 'dau_results.json'))
    log(f"\nPhase 5 complete. Time: {(time.time()-start_time)/60:.1f} min")

    # ========================================================================
    # PHASE 6: Ablations
    # ========================================================================
    log("\n" + "=" * 70)
    log("PHASE 6: Ablations")
    log("=" * 70)

    ablation_results = {}

    # --- Ablation 1: Alpha sensitivity (GA-DAU on cifar10) ---
    log("\n--- Ablation: Alpha sensitivity ---")
    alpha_results = []
    ds_name = 'cifar10'
    info = datasets_info[ds_name]
    train_diff = difficulty_scores[ds_name]
    test_diff = difficulty_scores[f'{ds_name}_test']
    train_per_ref = difficulty_scores[f'{ds_name}_train_per_ref']
    test_per_ref = difficulty_scores[f'{ds_name}_test_per_ref']
    train_quints = difficulty_quintiles[ds_name]
    test_quints = difficulty_quintiles[f'{ds_name}_test']

    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        for seed in SEEDS:
            splits = create_splits(info['n_train'], seed)
            forget_idx = np.array(splits['forget'])
            orig_model = original_models[(ds_name, seed)]
            forget_loader = get_loader(info['train_ds'], splits['forget'], shuffle=False)
            retain_loader = get_loader(info['train_ds'], splits['retain'], shuffle=True)

            forget_diff = train_diff[forget_idx]
            weights = compute_dau_weights(forget_diff, alpha=alpha) if alpha > 0 else None

            model_path = os.path.join(MODEL_DIR, f'ablation_alpha_{alpha}_{ds_name}_s{seed}.pt')
            if os.path.exists(model_path):
                umodel = get_model(ds_name)
                umodel.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            else:
                set_seed(seed)
                umodel = unlearn_ga(orig_model, forget_loader, retain_loader, ds_name, sample_weights=weights)
                torch.save(umodel.state_dict(), model_path)

            test_loader = get_loader(info['test_ds'], shuffle=False)
            test_acc, _ = evaluate(umodel, test_loader)
            forget_eval_loader = get_loader(info['train_eval_ds'], splits['forget'], shuffle=False)
            forget_acc, _ = evaluate(umodel, forget_eval_loader)
            retain_eval_loader = get_loader(info['train_eval_ds'], splits['retain'], shuffle=False)
            retain_acc, _ = evaluate(umodel, retain_eval_loader)

            nonmember_idx = np.random.RandomState(seed).choice(info['n_test'], size=FORGET_SIZE, replace=False)
            member_ref_mean = np.mean([r[forget_idx] for r in train_per_ref], axis=0)
            nonmember_ref_mean = np.mean([r[nonmember_idx] for r in test_per_ref], axis=0)
            member_ref_list = [r[forget_idx] for r in train_per_ref]
            nonmember_ref_list = [r[nonmember_idx] for r in test_per_ref]

            _, member_losses = evaluate(umodel, forget_eval_loader)
            nonmember_loader = get_loader(info['test_ds'], nonmember_idx.tolist(), shuffle=False)
            _, nonmember_losses = evaluate(umodel, nonmember_loader)

            strat = stratified_mia(member_losses, nonmember_losses,
                                   member_ref_mean, nonmember_ref_mean,
                                   member_ref_list, nonmember_ref_list,
                                   train_quints[forget_idx], test_quints[nonmember_idx])

            alpha_results.append({
                'alpha': alpha, 'seed': seed,
                'test_acc': float(test_acc), 'retain_acc': float(retain_acc), 'forget_acc': float(forget_acc),
                'aggregate_auc': strat['aggregate']['best_auc'],
                'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
            })
            log(f"  alpha={alpha}, seed={seed}: WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}, FA={forget_acc:.4f}")

    ablation_results['alpha'] = alpha_results

    # --- Ablation 2: Random-weight control ---
    log("\n--- Ablation: Random-weight control ---")
    random_weight_results = []
    for seed in SEEDS:
        splits = create_splits(info['n_train'], seed)
        forget_idx = np.array(splits['forget'])
        orig_model = original_models[(ds_name, seed)]
        forget_loader = get_loader(info['train_ds'], splits['forget'], shuffle=False)
        retain_loader = get_loader(info['train_ds'], splits['retain'], shuffle=True)

        forget_diff = train_diff[forget_idx]
        # Shuffle difficulty scores to break correspondence
        perm = np.random.RandomState(seed + 9999).permutation(len(forget_diff))
        random_weights = compute_dau_weights(forget_diff[perm], alpha=1.0)

        model_path = os.path.join(MODEL_DIR, f'ablation_random_weight_{ds_name}_s{seed}.pt')
        if os.path.exists(model_path):
            umodel = get_model(ds_name)
            umodel.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        else:
            set_seed(seed)
            umodel = unlearn_ga(orig_model, forget_loader, retain_loader, ds_name, sample_weights=random_weights)
            torch.save(umodel.state_dict(), model_path)

        test_acc, _ = evaluate(umodel, get_loader(info['test_ds'], shuffle=False))
        forget_acc, _ = evaluate(umodel, get_loader(info['train_eval_ds'], splits['forget'], shuffle=False))
        retain_acc, _ = evaluate(umodel, get_loader(info['train_eval_ds'], splits['retain'], shuffle=False))

        forget_eval_loader = get_loader(info['train_eval_ds'], splits['forget'], shuffle=False)
        nonmember_idx = np.random.RandomState(seed).choice(info['n_test'], size=FORGET_SIZE, replace=False)
        member_ref_mean = np.mean([r[forget_idx] for r in train_per_ref], axis=0)
        nonmember_ref_mean = np.mean([r[nonmember_idx] for r in test_per_ref], axis=0)
        member_ref_list = [r[forget_idx] for r in train_per_ref]
        nonmember_ref_list = [r[nonmember_idx] for r in test_per_ref]

        _, member_losses = evaluate(umodel, forget_eval_loader)
        nonmember_loader = get_loader(info['test_ds'], nonmember_idx.tolist(), shuffle=False)
        _, nonmember_losses = evaluate(umodel, nonmember_loader)

        strat = stratified_mia(member_losses, nonmember_losses,
                               member_ref_mean, nonmember_ref_mean,
                               member_ref_list, nonmember_ref_list,
                               train_quints[forget_idx], test_quints[nonmember_idx])
        random_weight_results.append({
            'seed': seed, 'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
            'aggregate_auc': strat['aggregate']['best_auc'],
            'test_acc': float(test_acc), 'forget_acc': float(forget_acc),
        })
        log(f"  Random-weight seed={seed}: WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")

    ablation_results['random_weight'] = random_weight_results

    # --- Ablation 3: Stratification granularity ---
    log("\n--- Ablation: Stratification granularity ---")
    strata_results = []
    seed = 42
    splits = create_splits(info['n_train'], seed)
    forget_idx = np.array(splits['forget'])
    nonmember_idx = np.random.RandomState(seed).choice(info['n_test'], size=FORGET_SIZE, replace=False)
    member_ref_mean = np.mean([r[forget_idx] for r in train_per_ref], axis=0)
    nonmember_ref_mean = np.mean([r[nonmember_idx] for r in test_per_ref], axis=0)
    member_ref_list = [r[forget_idx] for r in train_per_ref]
    nonmember_ref_list = [r[nonmember_idx] for r in test_per_ref]

    for n_strata in [3, 5, 10]:
        pcts = np.percentile(train_diff, np.linspace(0, 100, n_strata + 1)[1:-1])
        m_quints = np.digitize(train_diff[forget_idx], pcts)
        nm_quints = np.digitize(test_diff[nonmember_idx], pcts)

        for method in ['ga', 'scrub']:
            model = unlearned_models[(ds_name, seed, method)]
            forget_eval_loader = get_loader(info['train_eval_ds'], splits['forget'], shuffle=False)
            nonmember_loader = get_loader(info['test_ds'], nonmember_idx.tolist(), shuffle=False)
            _, ml = evaluate(model, forget_eval_loader)
            _, nml = evaluate(model, nonmember_loader)

            strat = stratified_mia(ml, nml, member_ref_mean, nonmember_ref_mean,
                                   member_ref_list, nonmember_ref_list,
                                   m_quints, nm_quints, n_strata=n_strata)
            strata_results.append({
                'n_strata': n_strata, 'method': method,
                'wq_auc': strat['wq_auc'], 'dg': strat['dg'], 'max_spread': strat['max_spread'],
            })
            log(f"  {n_strata} strata, {method}: WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")

    ablation_results['strata'] = strata_results

    # --- Ablation 4: Forget set size ---
    log("\n--- Ablation: Forget set size ---")
    forget_size_results = []
    for fsize in [500, 1000, 2500]:
        seed = 42
        n_train = info['n_train']
        non_ref = list(range(REF_POOL_SIZE, n_train))
        rng = np.random.RandomState(seed)
        f_idx = sorted(rng.choice(non_ref, size=fsize, replace=False).tolist())
        r_idx = sorted(set(non_ref) - set(f_idx))
        f_idx_arr = np.array(f_idx)

        for method in ['ga', 'scrub']:
            model_path = os.path.join(MODEL_DIR, f'ablation_fsize_{fsize}_{method}_{ds_name}_s{seed}.pt')
            if os.path.exists(model_path):
                umodel = get_model(ds_name)
                umodel.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            else:
                set_seed(seed)
                f_loader = get_loader(info['train_ds'], f_idx, shuffle=False)
                r_loader = get_loader(info['train_ds'], r_idx, shuffle=True)
                umodel = UNLEARN_FNS[method](original_models[(ds_name, seed)], f_loader, r_loader, ds_name)
                torch.save(umodel.state_dict(), model_path)

            f_eval = get_loader(info['train_eval_ds'], f_idx, shuffle=False)
            nm_size = min(fsize, info['n_test'])
            nm_idx = np.random.RandomState(seed).choice(info['n_test'], size=nm_size, replace=False)
            nm_loader = get_loader(info['test_ds'], nm_idx.tolist(), shuffle=False)

            _, ml = evaluate(umodel, f_eval)
            _, nml = evaluate(umodel, nm_loader)

            mrm = np.mean([r[f_idx_arr] for r in train_per_ref], axis=0)
            nmrm = np.mean([r[nm_idx] for r in test_per_ref], axis=0)
            mrl = [r[f_idx_arr] for r in train_per_ref]
            nmrl = [r[nm_idx] for r in test_per_ref]
            mq = train_quints[f_idx_arr]
            nmq = test_quints[nm_idx]

            strat = stratified_mia(ml, nml, mrm, nmrm, mrl, nmrl, mq, nmq)
            forget_size_results.append({
                'forget_size': fsize, 'method': method,
                'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                'aggregate_auc': strat['aggregate']['best_auc'],
            })
            log(f"  fsize={fsize}, {method}: WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")

    ablation_results['forget_size'] = forget_size_results

    # --- Ablation 5: K reference models ---
    log("\n--- Ablation: Number of reference models K ---")
    k_results = []
    for K in [2, 4]:
        ref_subset = ref_models[ds_name][:K]
        train_eval_loader = get_loader(info['train_eval_ds'], shuffle=False)
        test_loader = get_loader(info['test_ds'], shuffle=False)

        k_train_losses = []
        k_test_losses = []
        for rm in ref_subset:
            _, tl = evaluate(rm, train_eval_loader)
            k_train_losses.append(tl)
            _, tsl = evaluate(rm, test_loader)
            k_test_losses.append(tsl)

        k_diff = np.mean(k_train_losses, axis=0)
        k_test_diff = np.mean(k_test_losses, axis=0)

        # Correlation with K=4
        if K < 4:
            corr = stats.spearmanr(k_diff, train_diff).correlation
        else:
            corr = 1.0

        # Quintile stability
        k_pcts = np.percentile(k_diff, [20, 40, 60, 80])
        k_quints = np.digitize(k_diff, k_pcts)
        stability = (k_quints == train_quints).mean()

        seed = 42
        splits_42 = create_splits(info['n_train'], seed)
        f_idx = np.array(splits_42['forget'])
        nm_idx = np.random.RandomState(seed).choice(info['n_test'], size=FORGET_SIZE, replace=False)

        for method in ['ga', 'scrub']:
            model = unlearned_models[(ds_name, seed, method)]
            f_eval = get_loader(info['train_eval_ds'], splits_42['forget'], shuffle=False)
            nm_loader = get_loader(info['test_ds'], nm_idx.tolist(), shuffle=False)
            _, ml = evaluate(model, f_eval)
            _, nml = evaluate(model, nm_loader)

            mrm = np.mean([k_train_losses[i][f_idx] for i in range(K)], axis=0)
            nmrm = np.mean([k_test_losses[i][nm_idx] for i in range(K)], axis=0)
            mrl = [k_train_losses[i][f_idx] for i in range(K)]
            nmrl = [k_test_losses[i][nm_idx] for i in range(K)]
            mq = k_quints[f_idx]
            k_test_quints = np.digitize(k_test_diff, k_pcts)
            nmq = k_test_quints[nm_idx]

            strat = stratified_mia(ml, nml, mrm, nmrm, mrl, nmrl, mq, nmq)
            k_results.append({
                'K': K, 'method': method,
                'spearman_corr': float(corr), 'quintile_stability': float(stability),
                'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
            })
            log(f"  K={K}, {method}: corr={corr:.4f}, stability={stability:.4f}, WQ={strat['wq_auc']:.4f}")

    ablation_results['K'] = k_results

    save_json(ablation_results, os.path.join(RESULT_DIR, 'ablation_results.json'))
    log(f"\nPhase 6 complete. Time: {(time.time()-start_time)/60:.1f} min")

    # ========================================================================
    # PHASE 7: Statistical Analysis + Final Results
    # ========================================================================
    log("\n" + "=" * 70)
    log("PHASE 7: Statistical Analysis")
    log("=" * 70)

    stat_tests = {}

    # Test 1: WQ-AUC > Aggregate AUC
    log("\n--- Test 1: WQ-AUC > Aggregate AUC ---")
    wq_gt_agg = []
    for method in ['ft', 'ga', 'rl', 'scrub', 'neggrad']:
        for ds_name in ['cifar10', 'cifar100', 'purchase100']:
            entries = [r for r in dcua_results if r['method'] == method and r['dataset'] == ds_name]
            diffs = [r['wq_auc'] - r['aggregate_auc'] for r in entries]
            if len(diffs) >= 2:
                t_stat, p_val = stats.ttest_1samp(diffs, 0, alternative='greater')
                mean_diff = np.mean(diffs)
                wq_gt_agg.append({
                    'method': method, 'dataset': ds_name,
                    'mean_diff': float(mean_diff), 'p_value': float(p_val),
                    'significant': bool(p_val < 0.05 / 15),  # Bonferroni
                    'effect_gt_005': bool(mean_diff > 0.05),
                })
                log(f"  {method}/{ds_name}: diff={mean_diff:.4f}, p={p_val:.4f}")

    stat_tests['wq_gt_aggregate'] = wq_gt_agg

    # Test 2: Difficulty Gap significance
    log("\n--- Test 2: Difficulty Gap > 0 ---")
    dg_tests = []
    for method in ['ft', 'ga', 'rl', 'scrub', 'neggrad']:
        for ds_name in ['cifar10', 'cifar100', 'purchase100']:
            entries = [r for r in dcua_results if r['method'] == method and r['dataset'] == ds_name]
            dgs = [r['dg'] for r in entries]
            if len(dgs) >= 2:
                t_stat, p_val = stats.ttest_1samp(dgs, 0, alternative='greater')
                dg_tests.append({
                    'method': method, 'dataset': ds_name,
                    'mean_dg': float(np.mean(dgs)), 'std_dg': float(np.std(dgs)),
                    'p_value': float(p_val),
                    'significant': bool(p_val < 0.05 / 15),
                })
                log(f"  {method}/{ds_name}: DG={np.mean(dgs):.4f}±{np.std(dgs):.4f}, p={p_val:.4f}")

    stat_tests['dg_significance'] = dg_tests

    # Test 3: DAU effectiveness (honest)
    log("\n--- Test 3: DAU effectiveness ---")
    dau_tests = []
    for method in ['ga', 'scrub']:
        for ds_name in ['cifar10', 'cifar100', 'purchase100']:
            std_entries = [r for r in dcua_results if r['method'] == method and r['dataset'] == ds_name]
            dau_entries = [r for r in dau_results if r['method'] == f'{method}_dau' and r['dataset'] == ds_name]

            if len(std_entries) >= 2 and len(dau_entries) >= 2:
                std_wqs = sorted([(r['seed'], r['wq_auc']) for r in std_entries])
                dau_wqs = sorted([(r['seed'], r['wq_auc']) for r in dau_entries])
                # Paired comparison
                diffs = [s[1] - d[1] for s, d in zip(std_wqs, dau_wqs)]  # positive = DAU improves
                mean_improvement = np.mean(diffs)
                if len(diffs) >= 2:
                    t_stat, p_val = stats.ttest_1samp(diffs, 0, alternative='greater')
                else:
                    p_val = 1.0
                dau_tests.append({
                    'method': method, 'dataset': ds_name,
                    'mean_wq_improvement': float(mean_improvement),
                    'p_value': float(p_val),
                    'significant': bool(p_val < 0.05 / 6),
                    'dau_helps': bool(mean_improvement > 0),
                })
                log(f"  {method}_dau/{ds_name}: Δ_WQ={mean_improvement:.4f}, p={p_val:.4f}, helps={mean_improvement > 0}")

    stat_tests['dau_effectiveness'] = dau_tests

    # Test 4: Retrain as gold standard (AUC ≈ 0.5)
    log("\n--- Test 4: Retrain gold standard ---")
    retrain_aucs = []
    for ds_name in ['cifar10', 'cifar100', 'purchase100']:
        entries = [r for r in dcua_results if r['method'] == 'retrain' and r['dataset'] == ds_name]
        agg_aucs = [r['aggregate_auc'] for r in entries]
        wq_aucs = [r['wq_auc'] for r in entries]
        retrain_aucs.append({
            'dataset': ds_name,
            'mean_agg_auc': float(np.mean(agg_aucs)),
            'mean_wq_auc': float(np.mean(wq_aucs)),
        })
        log(f"  retrain/{ds_name}: Agg={np.mean(agg_aucs):.4f}, WQ={np.mean(wq_aucs):.4f}")

    stat_tests['retrain_baseline'] = retrain_aucs

    save_json(stat_tests, os.path.join(RESULT_DIR, 'statistical_tests.json'))

    # ========================================================================
    # Compile final results.json
    # ========================================================================
    log("\n" + "=" * 70)
    log("Compiling final results.json")
    log("=" * 70)

    final_results = {
        'experiment_config': {
            'datasets': ['cifar10', 'cifar100', 'purchase100'],
            'seeds': SEEDS,
            'unlearning_methods': ['ft', 'ga', 'rl', 'scrub', 'neggrad'],
            'defense_methods': ['ga_dau', 'scrub_dau'],
            'forget_size': FORGET_SIZE,
            'n_reference_models': 4,
            'total_runtime_minutes': (time.time() - start_time) / 60,
        },
        'main_results': {},
        'statistical_tests': stat_tests,
        'ablations': ablation_results,
    }

    # Compile main results table
    for ds_name in ['cifar10', 'cifar100', 'purchase100']:
        ds_results = {}
        for method in ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad']:
            entries = [r for r in dcua_results if r['method'] == method and r['dataset'] == ds_name]
            if entries:
                ds_results[method] = {
                    'aggregate_auc': {'mean': float(np.mean([e['aggregate_auc'] for e in entries])),
                                      'std': float(np.std([e['aggregate_auc'] for e in entries]))},
                    'wq_auc': {'mean': float(np.mean([e['wq_auc'] for e in entries])),
                               'std': float(np.std([e['wq_auc'] for e in entries]))},
                    'dg': {'mean': float(np.mean([e['dg'] for e in entries])),
                           'std': float(np.std([e['dg'] for e in entries]))},
                }

        # Add utility metrics from unlearning_results
        for method in ['ft', 'ga', 'rl', 'scrub', 'neggrad']:
            u_entries = [r for r in unlearning_results if r['method'] == method and r['dataset'] == ds_name]
            if u_entries and method in ds_results:
                ds_results[method]['retain_acc'] = {
                    'mean': float(np.mean([e['retain_acc'] for e in u_entries])),
                    'std': float(np.std([e['retain_acc'] for e in u_entries])),
                }
                ds_results[method]['test_acc'] = {
                    'mean': float(np.mean([e['test_acc'] for e in u_entries])),
                    'std': float(np.std([e['test_acc'] for e in u_entries])),
                }
                ds_results[method]['forget_acc'] = {
                    'mean': float(np.mean([e['forget_acc'] for e in u_entries])),
                    'std': float(np.std([e['forget_acc'] for e in u_entries])),
                }

        # Add DAU results
        for method in ['ga', 'scrub']:
            key = f'{method}_dau'
            entries = [r for r in dau_results if r['method'] == key and r['dataset'] == ds_name]
            if entries:
                ds_results[key] = {
                    'aggregate_auc': {'mean': float(np.mean([e['aggregate_auc'] for e in entries])),
                                      'std': float(np.std([e['aggregate_auc'] for e in entries]))},
                    'wq_auc': {'mean': float(np.mean([e['wq_auc'] for e in entries])),
                               'std': float(np.std([e['wq_auc'] for e in entries]))},
                    'dg': {'mean': float(np.mean([e['dg'] for e in entries])),
                           'std': float(np.std([e['dg'] for e in entries]))},
                    'retain_acc': {'mean': float(np.mean([e['retain_acc'] for e in entries])),
                                   'std': float(np.std([e['retain_acc'] for e in entries]))},
                    'test_acc': {'mean': float(np.mean([e['test_acc'] for e in entries])),
                                 'std': float(np.std([e['test_acc'] for e in entries]))},
                    'forget_acc': {'mean': float(np.mean([e['forget_acc'] for e in entries])),
                                   'std': float(np.std([e['forget_acc'] for e in entries]))},
                }

        final_results['main_results'][ds_name] = ds_results

    save_json(final_results, os.path.join(RESULT_DIR, 'final_results.json'))

    total_time = (time.time() - start_time) / 60
    log(f"\n{'='*70}")
    log(f"ALL EXPERIMENTS COMPLETE. Total time: {total_time:.1f} min")
    log(f"{'='*70}")

    return final_results


if __name__ == '__main__':
    results = main()
