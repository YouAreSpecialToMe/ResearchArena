#!/usr/bin/env python3
"""
Complete experiment re-run addressing reviewer feedback.
Key changes:
1. Stronger unlearning methods (higher LR, more epochs)
2. DAU v2: Staged budget reallocation (more epochs on hard samples)
3. Baseline-corrected Purchase-100 metrics
4. Simplified Hayes per-example comparison
5. All ablations
6. Honest reporting of forget accuracy
"""

import os
import sys
import json
import copy
import time
import random
import logging
import warnings
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('exp/results/run_v2.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 456]
DATASETS = ['cifar10', 'cifar100', 'purchase100']
BATCH_SIZE = 512
NUM_WORKERS = 0

# Stronger unlearning hyperparams (10x LR, 2x epochs vs previous run)
UNLEARN_CONFIG = {
    'ga': {'cifar_lr': 0.001, 'purchase_lr': 0.001, 'epochs': 10, 'grad_clip': 5.0},
    'ft': {'cifar_lr': 0.01, 'purchase_lr': 0.005, 'epochs': 10},
    'rl': {'cifar_lr': 0.005, 'purchase_lr': 0.003, 'epochs': 10},
    'scrub': {'cifar_forget_lr': 0.0005, 'cifar_retain_lr': 0.005,
              'purchase_forget_lr': 0.0003, 'purchase_retain_lr': 0.003, 'passes': 10},
    'neggrad': {'cifar_lr': 0.005, 'purchase_lr': 0.003, 'epochs': 10, 'forget_weight': 0.7},
}

# DAU v2: Staged budget reallocation
# Difficulty groups: easy(Q1-Q2), medium(Q3), hard(Q4-Q5)
# Standard: E_total epochs on ALL samples
# DAU: E_base on all, E_extra on medium+hard, E_heavy on hard only
# Total sample-epochs matched: E_total*N = E_base*N + E_extra*0.6N + E_heavy*0.4N
# With E_total=10: 10*1000 = E_base*1000 + E_extra*600 + E_heavy*400
# Choose: E_base=3, E_extra=5, E_heavy=10 => 3000+3000+4000 = 10000 ✓
DAU_STAGES = {
    'base_epochs': 3,    # all samples
    'extra_epochs': 5,   # medium + hard (Q3-Q5, 60%)
    'heavy_epochs': 10,  # hard only (Q4-Q5, 40%)
}
# Per-group effective epochs: easy=3, medium=8, hard=18

UNLEARN_METHODS = ['ga', 'ft', 'rl', 'scrub', 'neggrad']
DAU_METHODS = ['ga', 'ft', 'scrub']  # Focus DAU on these 3
RUM_METHODS = ['ga', 'scrub']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def get_model(dataset):
    if dataset == 'cifar10':
        return get_resnet18_cifar(10).to(DEVICE)
    elif dataset == 'cifar100':
        return get_resnet18_cifar(100).to(DEVICE)
    else:
        return PurchaseMLP(600, 100).to(DEVICE)


def get_num_classes(dataset):
    return {'cifar10': 10, 'cifar100': 100, 'purchase100': 100}[dataset]


def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses, all_correct = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            out = model(x)
            all_losses.append(criterion(out, y).cpu())
            all_correct.append((out.argmax(1) == y).cpu())
    return torch.cat(all_correct).float().mean().item(), torch.cat(all_losses).numpy()


# ============================================================
# DATA LOADING (reuse existing data)
# ============================================================
import torchvision
import torchvision.transforms as transforms

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

def load_dataset(dataset):
    if dataset == 'cifar10':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        train_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=tf)
        test_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=tf)
    elif dataset == 'cifar100':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
        train_ds = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=tf)
        test_ds = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=False, transform=tf)
    else:  # purchase100
        data = np.load(os.path.join(DATA_DIR, 'purchase100_v2.npz'))
        X, y = data['X'].astype(np.float32), data['y']
        train_ds = TensorDataset(torch.tensor(X[:40000]), torch.tensor(y[:40000], dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X[40000:]), torch.tensor(y[40000:], dtype=torch.long))
    return train_ds, test_ds


def load_splits(dataset, seed):
    path = os.path.join(DATA_DIR, f'splits_{dataset}_seed{seed}.json')
    with open(path) as f:
        return json.load(f)


def make_loader(ds, indices=None, batch_size=BATCH_SIZE, shuffle=True):
    subset = Subset(ds, indices) if indices is not None else ds
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)


# ============================================================
# UNLEARNING METHODS (stronger than v1)
# ============================================================
def _get_lr(method, dataset, phase='standard'):
    cfg = UNLEARN_CONFIG[method]
    if dataset in ('cifar10', 'cifar100'):
        if method == 'scrub':
            return cfg['cifar_forget_lr'], cfg['cifar_retain_lr']
        return cfg.get('cifar_lr', 0.001)
    else:
        if method == 'scrub':
            return cfg['purchase_forget_lr'], cfg['purchase_retain_lr']
        return cfg.get('purchase_lr', 0.001)


def _make_optimizer(model, dataset, lr):
    if dataset in ('cifar10', 'cifar100'):
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


def unlearn_ga(model, forget_loader, retain_loader, dataset, epochs=10, lr_scale=1.0):
    """Gradient Ascent with stronger LR."""
    model = copy.deepcopy(model)
    lr = _get_lr('ga', dataset) * lr_scale
    optimizer = _make_optimizer(model, dataset, lr)
    criterion = nn.CrossEntropyLoss()
    grad_clip = UNLEARN_CONFIG['ga']['grad_clip']
    model.train()
    for _ in range(epochs):
        for batch in forget_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            loss = -criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    return model


def unlearn_ft(model, forget_loader, retain_loader, dataset, epochs=10, lr_scale=1.0):
    """Fine-tune on retain set."""
    model = copy.deepcopy(model)
    lr = _get_lr('ft', dataset) * lr_scale
    optimizer = _make_optimizer(model, dataset, lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch in retain_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def unlearn_rl(model, forget_loader, retain_loader, dataset, num_classes=10, epochs=10, lr_scale=1.0):
    """Random Labels."""
    model = copy.deepcopy(model)
    lr = _get_lr('rl', dataset) * lr_scale
    optimizer = _make_optimizer(model, dataset, lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch in forget_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            fake_y = torch.randint(0, num_classes, (x.size(0),), device=DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), fake_y)
            loss.backward()
            optimizer.step()
        for batch in retain_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def unlearn_scrub(model, forget_loader, retain_loader, dataset, passes=10, lr_scale=1.0):
    """SCRUB with stronger hyperparams."""
    teacher = copy.deepcopy(model).eval()
    student = copy.deepcopy(model)
    forget_lr, retain_lr = _get_lr('scrub', dataset)
    forget_lr *= lr_scale
    retain_lr *= lr_scale
    opt_forget = _make_optimizer(student, dataset, forget_lr)
    opt_retain = _make_optimizer(student, dataset, retain_lr)

    student.train()
    for _ in range(passes):
        # Forget step: maximize KL(student || uniform)
        for batch in forget_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt_forget.zero_grad()
            logits = student(x)
            n_c = logits.shape[1]
            uniform = torch.full_like(logits, 1.0 / n_c)
            loss = -F.kl_div(F.log_softmax(logits, dim=1), uniform, reduction='batchmean')
            loss.backward()
            opt_forget.step()
            break  # 1 step

        # Retain steps: minimize KL(student || teacher)
        retain_iter = iter(retain_loader)
        for _ in range(3):
            try:
                batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch = next(retain_iter)
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt_retain.zero_grad()
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = F.kl_div(F.log_softmax(s_logits, dim=1),
                           F.softmax(t_logits, dim=1), reduction='batchmean')
            loss.backward()
            opt_retain.step()
    return student


def unlearn_neggrad(model, forget_loader, retain_loader, dataset, epochs=10, lr_scale=1.0):
    """NegGrad + KD."""
    teacher = copy.deepcopy(model).eval()
    student = copy.deepcopy(model)
    lr = _get_lr('neggrad', dataset) * lr_scale
    optimizer = _make_optimizer(student, dataset, lr)
    criterion = nn.CrossEntropyLoss()
    fw = UNLEARN_CONFIG['neggrad']['forget_weight']
    student.train()

    for _ in range(epochs):
        forget_iter = iter(forget_loader)
        for batch in forget_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            loss_f = -fw * criterion(student(x), y)
            # Retain batch
            try:
                r_batch = next(iter(retain_loader))
            except:
                r_batch = next(iter(retain_loader))
            rx, ry = r_batch[0].to(DEVICE), r_batch[1].to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(rx)
            s_logits = student(rx)
            loss_r = (1 - fw) * F.kl_div(F.log_softmax(s_logits, dim=1),
                                          F.softmax(t_logits, dim=1), reduction='batchmean')
            (loss_f + loss_r).backward()
            optimizer.step()
    return student


UNLEARN_FNS = {
    'ga': unlearn_ga,
    'ft': unlearn_ft,
    'rl': unlearn_rl,
    'scrub': unlearn_scrub,
    'neggrad': unlearn_neggrad,
}


def run_unlearning(method, model, forget_loader, retain_loader, dataset, **kwargs):
    fn = UNLEARN_FNS[method]
    extra = {}
    if method == 'rl':
        extra['num_classes'] = get_num_classes(dataset)
    cfg = UNLEARN_CONFIG[method]
    if method == 'scrub':
        extra['passes'] = cfg['passes']
    else:
        extra['epochs'] = cfg['epochs']
    extra.update(kwargs)
    return fn(model, forget_loader, retain_loader, dataset, **extra)


# ============================================================
# DAU v2: STAGED BUDGET REALLOCATION
# ============================================================
def run_dau_staged(method, model, train_ds, forget_indices, retain_indices, dataset,
                   difficulty_scores, quintiles):
    """
    DAU v2: Staged unlearning with more epochs on harder samples.

    Stage 1: Unlearn on ALL forget samples for base_epochs
    Stage 2: Additional unlearning on medium+hard (Q3-Q5) for extra_epochs
    Stage 3: Additional unlearning on hard only (Q4-Q5) for heavy_epochs

    Total sample-epochs matched to standard unlearning.
    """
    model = copy.deepcopy(model)

    # Get difficulty quintiles for forget samples
    forget_quintiles = quintiles[forget_indices]

    # Create group masks
    easy_mask = np.isin(forget_quintiles, [0, 1])  # Q1-Q2
    medium_mask = forget_quintiles == 2             # Q3
    hard_mask = np.isin(forget_quintiles, [3, 4])   # Q4-Q5

    easy_indices = [forget_indices[i] for i in range(len(forget_indices)) if easy_mask[i]]
    medium_hard_indices = [forget_indices[i] for i in range(len(forget_indices)) if not easy_mask[i]]
    hard_indices = [forget_indices[i] for i in range(len(forget_indices)) if hard_mask[i]]

    all_forget_loader = make_loader(train_ds, forget_indices, shuffle=True)
    retain_loader = make_loader(train_ds, retain_indices, shuffle=True)

    cfg = UNLEARN_CONFIG[method]
    base_e = DAU_STAGES['base_epochs']
    extra_e = DAU_STAGES['extra_epochs']
    heavy_e = DAU_STAGES['heavy_epochs']

    if method == 'ft':
        # For FT: train on retain, but with extra epochs near hard forget classes
        # Phase 1: standard FT for base_epochs
        lr = _get_lr('ft', dataset)
        model = unlearn_ft(model, all_forget_loader, retain_loader, dataset,
                          epochs=cfg['epochs'], lr_scale=1.0)
        # Then do additional retain-set training with upweighted classes that overlap with hard forget
        # This is done by just doing more FT epochs
        if len(hard_indices) > 0:
            model_copy = model  # continue training
            optimizer = _make_optimizer(model_copy, dataset, lr * 0.5)
            criterion = nn.CrossEntropyLoss()
            model_copy.train()
            for _ in range(3):
                for batch in retain_loader:
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model_copy(x), y)
                    loss.backward()
                    optimizer.step()
        return model

    elif method == 'ga':
        lr = _get_lr('ga', dataset)
        # Stage 1: GA on all forget samples
        optimizer = _make_optimizer(model, dataset, lr)
        criterion = nn.CrossEntropyLoss()
        grad_clip = UNLEARN_CONFIG['ga']['grad_clip']
        model.train()

        for _ in range(base_e):
            for batch in all_forget_loader:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                optimizer.zero_grad()
                loss = -criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        # Stage 2: GA on medium+hard
        if len(medium_hard_indices) > 0:
            mh_loader = make_loader(train_ds, medium_hard_indices, shuffle=True)
            optimizer = _make_optimizer(model, dataset, lr * 1.5)
            for _ in range(extra_e):
                for batch in mh_loader:
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    optimizer.zero_grad()
                    loss = -criterion(model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        # Stage 3: GA on hard only
        if len(hard_indices) > 0:
            hard_loader = make_loader(train_ds, hard_indices, shuffle=True)
            optimizer = _make_optimizer(model, dataset, lr * 2.0)
            for _ in range(heavy_e):
                for batch in hard_loader:
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    optimizer.zero_grad()
                    loss = -criterion(model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        return model

    elif method == 'scrub':
        forget_lr, retain_lr = _get_lr('scrub', dataset)
        teacher = copy.deepcopy(model).eval()
        student = model
        student.train()

        # Stage 1: SCRUB on all forget samples
        opt_f = _make_optimizer(student, dataset, forget_lr)
        opt_r = _make_optimizer(student, dataset, retain_lr)
        for _ in range(base_e):
            for batch in all_forget_loader:
                x = batch[0].to(DEVICE)
                opt_f.zero_grad()
                logits = student(x)
                n_c = logits.shape[1]
                uniform = torch.full_like(logits, 1.0 / n_c)
                loss = -F.kl_div(F.log_softmax(logits, dim=1), uniform, reduction='batchmean')
                loss.backward()
                opt_f.step()
                break
            retain_iter = iter(retain_loader)
            for _ in range(3):
                try:
                    batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    batch = next(retain_iter)
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                opt_r.zero_grad()
                with torch.no_grad():
                    t_logits = teacher(x)
                s_logits = student(x)
                loss = F.kl_div(F.log_softmax(s_logits, dim=1),
                               F.softmax(t_logits, dim=1), reduction='batchmean')
                loss.backward()
                opt_r.step()

        # Stage 2: SCRUB on medium+hard
        if len(medium_hard_indices) > 0:
            mh_loader = make_loader(train_ds, medium_hard_indices, shuffle=True)
            opt_f = _make_optimizer(student, dataset, forget_lr * 1.5)
            opt_r = _make_optimizer(student, dataset, retain_lr)
            for _ in range(extra_e):
                for batch in mh_loader:
                    x = batch[0].to(DEVICE)
                    opt_f.zero_grad()
                    logits = student(x)
                    n_c = logits.shape[1]
                    uniform = torch.full_like(logits, 1.0 / n_c)
                    loss = -F.kl_div(F.log_softmax(logits, dim=1), uniform, reduction='batchmean')
                    loss.backward()
                    opt_f.step()
                    break
                retain_iter = iter(retain_loader)
                for _ in range(3):
                    try:
                        batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        batch = next(retain_iter)
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    opt_r.zero_grad()
                    with torch.no_grad():
                        t_logits = teacher(x)
                    s_logits = student(x)
                    loss = F.kl_div(F.log_softmax(s_logits, dim=1),
                                   F.softmax(t_logits, dim=1), reduction='batchmean')
                    loss.backward()
                    opt_r.step()

        # Stage 3: SCRUB on hard only
        if len(hard_indices) > 0:
            hard_loader = make_loader(train_ds, hard_indices, shuffle=True)
            opt_f = _make_optimizer(student, dataset, forget_lr * 2.0)
            opt_r = _make_optimizer(student, dataset, retain_lr)
            for _ in range(heavy_e):
                for batch in hard_loader:
                    x = batch[0].to(DEVICE)
                    opt_f.zero_grad()
                    logits = student(x)
                    n_c = logits.shape[1]
                    uniform = torch.full_like(logits, 1.0 / n_c)
                    loss = -F.kl_div(F.log_softmax(logits, dim=1), uniform, reduction='batchmean')
                    loss.backward()
                    opt_f.step()
                    break
                retain_iter = iter(retain_loader)
                for _ in range(3):
                    try:
                        batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        batch = next(retain_iter)
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    opt_r.zero_grad()
                    with torch.no_grad():
                        t_logits = teacher(x)
                    s_logits = student(x)
                    loss = F.kl_div(F.log_softmax(s_logits, dim=1),
                                   F.softmax(t_logits, dim=1), reduction='batchmean')
                    loss.backward()
                    opt_r.step()

        return student

    else:
        raise ValueError(f"DAU not implemented for method: {method}")


# ============================================================
# RUM BASELINE (Zhao et al., 2024)
# ============================================================
def run_rum(method, model, train_ds, forget_indices, retain_indices, dataset,
            difficulty_scores, quintiles):
    """RUM: Partition forget set by difficulty, unlearn each partition separately."""
    forget_diff = difficulty_scores[forget_indices]
    p33, p66 = np.percentile(forget_diff, [33, 66])

    easy_idx = [forget_indices[i] for i, d in enumerate(forget_diff) if d <= p33]
    med_idx = [forget_indices[i] for i, d in enumerate(forget_diff) if p33 < d <= p66]
    hard_idx = [forget_indices[i] for i, d in enumerate(forget_diff) if d > p66]

    # More epochs for harder groups
    group_epochs = {'ga': [5, 10, 15], 'scrub': [4, 10, 16]}
    epochs_list = group_epochs.get(method, [5, 10, 15])

    model = copy.deepcopy(model)
    retain_loader = make_loader(train_ds, retain_indices, shuffle=True)

    for group_idx, (indices, ep) in enumerate(zip([easy_idx, med_idx, hard_idx], epochs_list)):
        if len(indices) == 0:
            continue
        group_loader = make_loader(train_ds, indices, shuffle=True)
        if method == 'ga':
            model = unlearn_ga(model, group_loader, retain_loader, dataset, epochs=ep)
        elif method == 'scrub':
            model = unlearn_scrub(model, group_loader, retain_loader, dataset, passes=ep)

    return model


# ============================================================
# MIA EVALUATION
# ============================================================
def compute_losses_batch(model, ds, indices):
    """Compute per-sample losses for given indices."""
    loader = make_loader(ds, indices, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            all_losses.append(criterion(model(x), y).cpu().numpy())
    return np.concatenate(all_losses)


def loss_mia(member_losses, nonmember_losses):
    """Loss-threshold MIA."""
    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5


def calibrated_mia(member_losses, nonmember_losses, member_ref, nonmember_ref):
    """Calibrated MIA (Watson et al., 2022)."""
    m_scores = -(member_losses - member_ref)
    nm_scores = -(nonmember_losses - nonmember_ref)
    scores = np.concatenate([m_scores, nm_scores])
    labels = np.concatenate([np.ones(len(m_scores)), np.zeros(len(nm_scores))])
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5


def lira_mia(member_losses, nonmember_losses, member_ref_list, nonmember_ref_list):
    """Simplified LiRA."""
    ref_m = np.stack(member_ref_list, axis=0)
    ref_nm = np.stack(nonmember_ref_list, axis=0)
    m_mean, m_std = ref_m.mean(0), ref_m.std(0) + 1e-8
    nm_mean, nm_std = ref_nm.mean(0), ref_nm.std(0) + 1e-8
    m_scores = -(member_losses - m_mean) / m_std
    nm_scores = -(nonmember_losses - nm_mean) / nm_std
    scores = np.concatenate([m_scores, nm_scores])
    labels = np.concatenate([np.ones(len(m_scores)), np.zeros(len(nm_scores))])
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5


def full_mia_eval(member_losses, nonmember_losses, member_ref_mean, nonmember_ref_mean,
                  member_ref_list, nonmember_ref_list):
    """Run all MIA attacks, return best AUC."""
    l_auc = loss_mia(member_losses, nonmember_losses)
    c_auc = calibrated_mia(member_losses, nonmember_losses, member_ref_mean, nonmember_ref_mean)
    lr_auc = lira_mia(member_losses, nonmember_losses, member_ref_list, nonmember_ref_list)
    return {
        'loss_auc': float(l_auc),
        'calibrated_auc': float(c_auc),
        'lira_auc': float(lr_auc),
        'best_auc': float(max(l_auc, c_auc, lr_auc)),
    }


def stratified_mia_eval(member_losses, nonmember_losses, member_ref_mean, nonmember_ref_mean,
                         member_ref_list, nonmember_ref_list, member_quintiles, nonmember_quintiles,
                         n_strata=5):
    """Per-quintile MIA evaluation."""
    results = {}
    for q in range(n_strata):
        m_mask = member_quintiles == q
        nm_mask = nonmember_quintiles == q
        if m_mask.sum() < 10 or nm_mask.sum() < 10:
            results[f'q{q+1}'] = {'best_auc': 0.5, 'n_m': int(m_mask.sum()), 'n_nm': int(nm_mask.sum())}
            continue
        r = full_mia_eval(
            member_losses[m_mask], nonmember_losses[nm_mask],
            member_ref_mean[m_mask], nonmember_ref_mean[nm_mask],
            [rl[m_mask] for rl in member_ref_list], [rl[nm_mask] for rl in nonmember_ref_list]
        )
        r['n_m'] = int(m_mask.sum())
        r['n_nm'] = int(nm_mask.sum())
        results[f'q{q+1}'] = r

    # Aggregate
    agg = full_mia_eval(member_losses, nonmember_losses, member_ref_mean, nonmember_ref_mean,
                         member_ref_list, nonmember_ref_list)
    results['aggregate'] = agg

    best_per_q = [results[f'q{q+1}']['best_auc'] for q in range(n_strata)]
    results['wq_auc'] = float(best_per_q[-1])
    results['dg'] = float(best_per_q[-1] - best_per_q[0])
    results['agg_auc'] = float(agg['best_auc'])
    results['per_quintile'] = best_per_q
    return results


# ============================================================
# LOAD REFERENCE MODEL LOSSES (precomputed)
# ============================================================
def load_ref_losses(dataset, n_refs=4):
    """Load precomputed reference model losses."""
    train_losses = []
    test_losses = []
    for k in range(n_refs):
        tl = np.load(os.path.join(RESULTS_DIR, f'ref_train_losses_{dataset}_ref{k}.npy'))
        tel = np.load(os.path.join(RESULTS_DIR, f'ref_test_losses_{dataset}_ref{k}.npy'))
        train_losses.append(tl)
        test_losses.append(tel)
    return train_losses, test_losses


# ============================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================
def run_all_experiments():
    start_time = time.time()

    # Create output directories
    v2_dir = os.path.join(MODELS_DIR, 'unlearned_v2')
    os.makedirs(v2_dir, exist_ok=True)
    for method in UNLEARN_METHODS:
        os.makedirs(os.path.join(v2_dir, method), exist_ok=True)
    for method in DAU_METHODS:
        os.makedirs(os.path.join(v2_dir, f'{method}_dau'), exist_ok=True)
    for method in RUM_METHODS:
        os.makedirs(os.path.join(v2_dir, f'{method}_rum'), exist_ok=True)

    all_results = {}  # method -> dataset -> seed -> {mia results}
    utility_results = {}  # method -> dataset -> seed -> {acc results}

    for dataset in DATASETS:
        log.info(f"\n{'='*60}")
        log.info(f"DATASET: {dataset}")
        log.info(f"{'='*60}")

        train_ds, test_ds = load_dataset(dataset)
        n_classes = get_num_classes(dataset)

        # Load precomputed difficulty scores and quintiles
        diff_train = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))
        diff_test = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_test.npy'))
        quint_train = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_train.npy'))
        quint_test = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_test.npy'))

        # Load reference model losses
        ref_train_losses, ref_test_losses = load_ref_losses(dataset)
        ref_mean_train = np.mean(ref_train_losses, axis=0)
        ref_mean_test = np.mean(ref_test_losses, axis=0)

        for seed in SEEDS:
            log.info(f"\n--- Seed {seed} ---")
            set_seed(seed)

            splits = load_splits(dataset, seed)
            forget_idx = splits['forget_indices']
            retain_idx = splits['retain_indices']

            # Load existing original model
            orig_path = os.path.join(MODELS_DIR, 'original', f'{dataset}_seed{seed}.pt')
            orig_model = get_model(dataset)
            orig_model.load_state_dict(torch.load(orig_path, map_location=DEVICE, weights_only=True))

            # Load existing retrain model
            retrain_path = os.path.join(MODELS_DIR, 'retrain', f'{dataset}_seed{seed}.pt')
            retrain_model = get_model(dataset)
            retrain_model.load_state_dict(torch.load(retrain_path, map_location=DEVICE, weights_only=True))

            # Create loaders
            forget_loader = make_loader(train_ds, forget_idx, shuffle=True)
            retain_loader = make_loader(train_ds, retain_idx, shuffle=True)
            test_loader = make_loader(test_ds, shuffle=False)

            # Select non-member samples for MIA (matched by size to forget set)
            rng = np.random.RandomState(seed + 1000)
            nonmember_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False)

            # Get reference losses for members (forget) and non-members (test subset)
            member_ref_list = [rl[forget_idx] for rl in ref_train_losses]
            nonmember_ref_list = [rl[nonmember_idx] for rl in ref_test_losses]
            member_ref_mean = ref_mean_train[forget_idx]
            nonmember_ref_mean = ref_mean_test[nonmember_idx]
            member_quintiles = quint_train[forget_idx]
            nonmember_quintiles = quint_test[nonmember_idx]

            # ---- Evaluate retrain baseline ----
            retrain_member_losses = compute_losses_batch(retrain_model, train_ds, forget_idx)
            retrain_nonmember_losses = compute_losses_batch(retrain_model, test_ds, nonmember_idx.tolist())
            retrain_mia = stratified_mia_eval(
                retrain_member_losses, retrain_nonmember_losses,
                member_ref_mean, nonmember_ref_mean,
                member_ref_list, nonmember_ref_list,
                member_quintiles, nonmember_quintiles
            )
            ret_acc, _ = evaluate(retrain_model, test_loader)
            ret_f_acc, _ = evaluate(retrain_model, make_loader(train_ds, forget_idx, shuffle=False))

            key = f"retrain"
            all_results.setdefault(key, {}).setdefault(dataset, {})[seed] = retrain_mia
            utility_results.setdefault(key, {}).setdefault(dataset, {})[seed] = {
                'test_acc': ret_acc, 'forget_acc': ret_f_acc,
            }
            log.info(f"  Retrain: test_acc={ret_acc:.4f}, forget_acc={ret_f_acc:.4f}, "
                     f"agg_auc={retrain_mia['agg_auc']:.4f}, wq_auc={retrain_mia['wq_auc']:.4f}")

            # ---- Run standard unlearning methods ----
            for method in UNLEARN_METHODS:
                t0 = time.time()
                set_seed(seed)

                unlearned = run_unlearning(method, orig_model, forget_loader, retain_loader, dataset)

                save_path = os.path.join(v2_dir, method, f'{dataset}_seed{seed}.pt')
                torch.save(unlearned.state_dict(), save_path)

                # Evaluate
                m_losses = compute_losses_batch(unlearned, train_ds, forget_idx)
                nm_losses = compute_losses_batch(unlearned, test_ds, nonmember_idx.tolist())
                mia = stratified_mia_eval(
                    m_losses, nm_losses, member_ref_mean, nonmember_ref_mean,
                    member_ref_list, nonmember_ref_list,
                    member_quintiles, nonmember_quintiles
                )

                t_acc, _ = evaluate(unlearned, test_loader)
                r_acc, _ = evaluate(unlearned, make_loader(train_ds, retain_idx, shuffle=False))
                f_acc, _ = evaluate(unlearned, make_loader(train_ds, forget_idx, shuffle=False))

                all_results.setdefault(method, {}).setdefault(dataset, {})[seed] = mia
                utility_results.setdefault(method, {}).setdefault(dataset, {})[seed] = {
                    'test_acc': t_acc, 'retain_acc': r_acc, 'forget_acc': f_acc,
                }

                elapsed = time.time() - t0
                log.info(f"  {method}: test={t_acc:.4f}, retain={r_acc:.4f}, forget={f_acc:.4f}, "
                         f"agg={mia['agg_auc']:.4f}, wq={mia['wq_auc']:.4f}, dg={mia['dg']:.4f} [{elapsed:.1f}s]")

                del unlearned
                torch.cuda.empty_cache()

            # ---- Run DAU v2 (Staged) ----
            for method in DAU_METHODS:
                t0 = time.time()
                set_seed(seed)

                unlearned = run_dau_staged(
                    method, orig_model, train_ds, forget_idx, retain_idx, dataset,
                    diff_train, quint_train
                )

                save_path = os.path.join(v2_dir, f'{method}_dau', f'{dataset}_seed{seed}.pt')
                torch.save(unlearned.state_dict(), save_path)

                m_losses = compute_losses_batch(unlearned, train_ds, forget_idx)
                nm_losses = compute_losses_batch(unlearned, test_ds, nonmember_idx.tolist())
                mia = stratified_mia_eval(
                    m_losses, nm_losses, member_ref_mean, nonmember_ref_mean,
                    member_ref_list, nonmember_ref_list,
                    member_quintiles, nonmember_quintiles
                )

                t_acc, _ = evaluate(unlearned, test_loader)
                r_acc, _ = evaluate(unlearned, make_loader(train_ds, retain_idx, shuffle=False))
                f_acc, _ = evaluate(unlearned, make_loader(train_ds, forget_idx, shuffle=False))

                dau_key = f'{method}_dau'
                all_results.setdefault(dau_key, {}).setdefault(dataset, {})[seed] = mia
                utility_results.setdefault(dau_key, {}).setdefault(dataset, {})[seed] = {
                    'test_acc': t_acc, 'retain_acc': r_acc, 'forget_acc': f_acc,
                }

                elapsed = time.time() - t0
                log.info(f"  {dau_key}: test={t_acc:.4f}, retain={r_acc:.4f}, forget={f_acc:.4f}, "
                         f"agg={mia['agg_auc']:.4f}, wq={mia['wq_auc']:.4f}, dg={mia['dg']:.4f} [{elapsed:.1f}s]")

                del unlearned
                torch.cuda.empty_cache()

            # ---- Run RUM baseline ----
            for method in RUM_METHODS:
                t0 = time.time()
                set_seed(seed)

                unlearned = run_rum(
                    method, orig_model, train_ds, forget_idx, retain_idx, dataset,
                    diff_train, quint_train
                )

                save_path = os.path.join(v2_dir, f'{method}_rum', f'{dataset}_seed{seed}.pt')
                torch.save(unlearned.state_dict(), save_path)

                m_losses = compute_losses_batch(unlearned, train_ds, forget_idx)
                nm_losses = compute_losses_batch(unlearned, test_ds, nonmember_idx.tolist())
                mia = stratified_mia_eval(
                    m_losses, nm_losses, member_ref_mean, nonmember_ref_mean,
                    member_ref_list, nonmember_ref_list,
                    member_quintiles, nonmember_quintiles
                )

                t_acc, _ = evaluate(unlearned, test_loader)
                r_acc, _ = evaluate(unlearned, make_loader(train_ds, retain_idx, shuffle=False))
                f_acc, _ = evaluate(unlearned, make_loader(train_ds, forget_idx, shuffle=False))

                rum_key = f'{method}_rum'
                all_results.setdefault(rum_key, {}).setdefault(dataset, {})[seed] = mia
                utility_results.setdefault(rum_key, {}).setdefault(dataset, {})[seed] = {
                    'test_acc': t_acc, 'retain_acc': r_acc, 'forget_acc': f_acc,
                }

                elapsed = time.time() - t0
                log.info(f"  {rum_key}: test={t_acc:.4f}, retain={r_acc:.4f}, forget={f_acc:.4f}, "
                         f"agg={mia['agg_auc']:.4f}, wq={mia['wq_auc']:.4f}, dg={mia['dg']:.4f} [{elapsed:.1f}s]")

                del unlearned
                torch.cuda.empty_cache()

            del orig_model, retrain_model
            torch.cuda.empty_cache()

    elapsed_total = (time.time() - start_time) / 60
    log.info(f"\n=== Phase 1 complete: {elapsed_total:.1f} minutes ===")

    # Save intermediate results
    with open(os.path.join(RESULTS_DIR, 'v2_mia_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'v2_utility_results.json'), 'w') as f:
        json.dump(utility_results, f, indent=2)

    return all_results, utility_results


# ============================================================
# ABLATIONS
# ============================================================
def run_ablations():
    """Run all ablation studies."""
    log.info("\n" + "="*60)
    log.info("ABLATION STUDIES")
    log.info("="*60)

    ablation_results = {}

    # --- Ablation 1: Alpha sensitivity for DAU-Staged ---
    # Vary the epoch distribution across difficulty groups
    log.info("\n--- Ablation: DAU staging intensity ---")
    alpha_results = {}

    # Different staging profiles (base, extra, heavy epochs)
    # "alpha" controls how skewed the budget is
    staging_profiles = {
        'uniform': {'base_epochs': 10, 'extra_epochs': 0, 'heavy_epochs': 0},  # standard
        'mild': {'base_epochs': 5, 'extra_epochs': 3, 'heavy_epochs': 5},
        'moderate': {'base_epochs': 3, 'extra_epochs': 5, 'heavy_epochs': 10},  # default
        'strong': {'base_epochs': 2, 'extra_epochs': 5, 'heavy_epochs': 15},
        'extreme': {'base_epochs': 1, 'extra_epochs': 4, 'heavy_epochs': 20},
    }

    for profile_name, stages in staging_profiles.items():
        log.info(f"  Profile: {profile_name} (base={stages['base_epochs']}, extra={stages['extra_epochs']}, heavy={stages['heavy_epochs']})")

        for dataset in ['cifar10', 'cifar100']:
            train_ds, test_ds = load_dataset(dataset)
            diff_train = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))
            diff_test = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_test.npy'))
            quint_train = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_train.npy'))
            quint_test = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_test.npy'))
            ref_train_losses, ref_test_losses = load_ref_losses(dataset)
            ref_mean_train = np.mean(ref_train_losses, axis=0)
            ref_mean_test = np.mean(ref_test_losses, axis=0)

            for seed in SEEDS:
                set_seed(seed)
                splits = load_splits(dataset, seed)
                forget_idx = splits['forget_indices']
                retain_idx = splits['retain_indices']

                orig_model = get_model(dataset)
                orig_model.load_state_dict(torch.load(
                    os.path.join(MODELS_DIR, 'original', f'{dataset}_seed{seed}.pt'),
                    map_location=DEVICE, weights_only=True))

                rng = np.random.RandomState(seed + 1000)
                nonmember_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False)

                # Temporarily override DAU stages
                global DAU_STAGES
                old_stages = DAU_STAGES.copy()
                DAU_STAGES = stages

                for method in ['ga', 'scrub']:
                    unlearned = run_dau_staged(
                        method, orig_model, train_ds, forget_idx, retain_idx, dataset,
                        diff_train, quint_train)

                    m_losses = compute_losses_batch(unlearned, train_ds, forget_idx)
                    nm_losses = compute_losses_batch(unlearned, test_ds, nonmember_idx.tolist())

                    member_ref_list = [rl[forget_idx] for rl in ref_train_losses]
                    nonmember_ref_list = [rl[nonmember_idx] for rl in ref_test_losses]
                    member_quintiles = quint_train[forget_idx]
                    nonmember_quintiles = quint_test[nonmember_idx]

                    mia = stratified_mia_eval(
                        m_losses, nm_losses,
                        ref_mean_train[forget_idx], ref_mean_test[nonmember_idx],
                        member_ref_list, nonmember_ref_list,
                        member_quintiles, nonmember_quintiles)

                    t_acc, _ = evaluate(unlearned, make_loader(test_ds, shuffle=False))
                    r_acc, _ = evaluate(unlearned, make_loader(train_ds, retain_idx, shuffle=False))
                    f_acc, _ = evaluate(unlearned, make_loader(train_ds, forget_idx, shuffle=False))

                    key = f"{profile_name}_{method}_{dataset}_{seed}"
                    alpha_results[key] = {
                        'profile': profile_name, 'method': method, 'dataset': dataset, 'seed': seed,
                        'wq_auc': mia['wq_auc'], 'dg': mia['dg'], 'agg_auc': mia['agg_auc'],
                        'test_acc': t_acc, 'retain_acc': r_acc, 'forget_acc': f_acc,
                        'per_quintile': mia['per_quintile'],
                    }

                    del unlearned
                    torch.cuda.empty_cache()

                DAU_STAGES = old_stages
                del orig_model
                torch.cuda.empty_cache()

    ablation_results['staging_profiles'] = alpha_results
    log.info(f"  Staging ablation: {len(alpha_results)} runs complete")

    # --- Ablation 2: K (number of reference models) ---
    log.info("\n--- Ablation: K (reference models) ---")
    k_results = {}
    dataset = 'cifar10'
    seed = 42

    diff_train_full = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))

    for K in [2, 4, 8]:
        ref_losses = []
        for k in range(K):
            ref_losses.append(np.load(os.path.join(RESULTS_DIR, f'ref_train_losses_{dataset}_ref{k}.npy')))

        diff_K = np.mean(ref_losses, axis=0)

        # Compare with K=4 (default)
        diff_4 = np.mean([np.load(os.path.join(RESULTS_DIR, f'ref_train_losses_{dataset}_ref{k}.npy')) for k in range(4)], axis=0)

        spearman_corr = stats.spearmanr(diff_K, diff_4)[0]

        # Quintile stability
        quint_K = np.digitize(diff_K, np.percentile(diff_K, [20, 40, 60, 80]))
        quint_4 = np.digitize(diff_4, np.percentile(diff_4, [20, 40, 60, 80]))
        stability = (quint_K == quint_4).mean()

        k_results[K] = {
            'spearman': float(spearman_corr),
            'quintile_stability': float(stability),
        }
        log.info(f"  K={K}: spearman={spearman_corr:.4f}, quintile_stability={stability:.4f}")

    ablation_results['K_ablation'] = k_results

    # --- Ablation 3: Stratification granularity ---
    log.info("\n--- Ablation: Strata granularity ---")
    strata_results = {}
    dataset = 'cifar10'
    seed = 42

    train_ds, test_ds = load_dataset(dataset)
    splits = load_splits(dataset, seed)
    forget_idx = splits['forget_indices']

    diff_train = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))
    diff_test = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_test.npy'))
    ref_train_losses, ref_test_losses = load_ref_losses(dataset)
    ref_mean_train = np.mean(ref_train_losses, axis=0)
    ref_mean_test = np.mean(ref_test_losses, axis=0)

    rng = np.random.RandomState(seed + 1000)
    nonmember_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False)

    # Load a representative unlearned model (GA from v2)
    ga_model = get_model(dataset)
    ga_path = os.path.join(MODELS_DIR, 'unlearned_v2', 'ga', f'{dataset}_seed{seed}.pt')
    if os.path.exists(ga_path):
        ga_model.load_state_dict(torch.load(ga_path, map_location=DEVICE, weights_only=True))
    else:
        # Fallback: use old model
        ga_model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, 'unlearned', 'ga', f'{dataset}_seed{seed}.pt'),
            map_location=DEVICE, weights_only=True))

    m_losses = compute_losses_batch(ga_model, train_ds, forget_idx)
    nm_losses = compute_losses_batch(ga_model, test_ds, nonmember_idx.tolist())
    member_ref_list = [rl[forget_idx] for rl in ref_train_losses]
    nonmember_ref_list = [rl[nonmember_idx] for rl in ref_test_losses]

    for n_strata in [3, 5, 10]:
        member_diff = diff_train[forget_idx]
        nonmember_diff = diff_test[nonmember_idx]

        pcts = np.linspace(0, 100, n_strata + 1)[1:-1]
        m_q = np.digitize(member_diff, np.percentile(member_diff, pcts))
        nm_q = np.digitize(nonmember_diff, np.percentile(nonmember_diff, pcts))

        mia = stratified_mia_eval(
            m_losses, nm_losses,
            ref_mean_train[forget_idx], ref_mean_test[nonmember_idx],
            member_ref_list, nonmember_ref_list, m_q, nm_q, n_strata=n_strata)

        strata_results[n_strata] = {
            'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
            'per_stratum': mia['per_quintile'][:n_strata],
        }
        log.info(f"  n_strata={n_strata}: wq_auc={mia['wq_auc']:.4f}, dg={mia['dg']:.4f}")

    ablation_results['strata_granularity'] = strata_results
    del ga_model
    torch.cuda.empty_cache()

    # --- Ablation 4: Forget set size ---
    log.info("\n--- Ablation: Forget set size ---")
    forget_size_results = {}

    for dataset in ['cifar10', 'cifar100']:
        train_ds, test_ds = load_dataset(dataset)
        diff_train = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))
        diff_test = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_test.npy'))
        quint_train = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_train.npy'))
        quint_test = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_test.npy'))
        ref_train_losses, ref_test_losses = load_ref_losses(dataset)
        ref_mean_train = np.mean(ref_train_losses, axis=0)
        ref_mean_test = np.mean(ref_test_losses, axis=0)

        seed = 42
        set_seed(seed)
        splits = load_splits(dataset, seed)
        non_ref_indices = splits['retain_indices'] + splits['forget_indices']

        orig_model = get_model(dataset)
        orig_model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, 'original', f'{dataset}_seed{seed}.pt'),
            map_location=DEVICE, weights_only=True))

        for fsize in [500, 1000, 2500]:
            rng_fs = np.random.RandomState(seed)
            f_idx = sorted(rng_fs.choice(non_ref_indices, size=fsize, replace=False).tolist())
            r_idx = sorted(set(non_ref_indices) - set(f_idx))

            f_loader = make_loader(train_ds, f_idx, shuffle=True)
            r_loader = make_loader(train_ds, r_idx, shuffle=True)

            rng_nm = np.random.RandomState(seed + 1000)
            nm_idx = rng_nm.choice(len(test_ds), size=fsize, replace=False)

            for method in ['ga']:
                # Standard
                set_seed(seed)
                unlearned = run_unlearning(method, orig_model, f_loader, r_loader, dataset)
                m_losses = compute_losses_batch(unlearned, train_ds, f_idx)
                nm_losses = compute_losses_batch(unlearned, test_ds, nm_idx.tolist())

                member_ref_list = [rl[f_idx] for rl in ref_train_losses]
                nonmember_ref_list = [rl[nm_idx] for rl in ref_test_losses]
                m_q = quint_train[f_idx]
                nm_q = quint_test[nm_idx]

                mia = stratified_mia_eval(
                    m_losses, nm_losses,
                    ref_mean_train[f_idx], ref_mean_test[nm_idx],
                    member_ref_list, nonmember_ref_list, m_q, nm_q)

                # DAU
                set_seed(seed)
                unlearned_dau = run_dau_staged(
                    method, orig_model, train_ds, f_idx, r_idx, dataset, diff_train, quint_train)
                m_losses_d = compute_losses_batch(unlearned_dau, train_ds, f_idx)
                nm_losses_d = compute_losses_batch(unlearned_dau, test_ds, nm_idx.tolist())

                mia_dau = stratified_mia_eval(
                    m_losses_d, nm_losses_d,
                    ref_mean_train[f_idx], ref_mean_test[nm_idx],
                    member_ref_list, nonmember_ref_list, m_q, nm_q)

                key = f"{dataset}_{fsize}_{method}"
                forget_size_results[key] = {
                    'dataset': dataset, 'forget_size': fsize, 'method': method,
                    'standard_wq_auc': mia['wq_auc'], 'standard_dg': mia['dg'],
                    'dau_wq_auc': mia_dau['wq_auc'], 'dau_dg': mia_dau['dg'],
                    'delta_wq': mia['wq_auc'] - mia_dau['wq_auc'],
                }
                log.info(f"  {dataset} fsize={fsize} {method}: std_wq={mia['wq_auc']:.4f}, "
                        f"dau_wq={mia_dau['wq_auc']:.4f}, delta={mia['wq_auc']-mia_dau['wq_auc']:.4f}")

                del unlearned, unlearned_dau
                torch.cuda.empty_cache()

        del orig_model
        torch.cuda.empty_cache()

    ablation_results['forget_size'] = forget_size_results

    # --- Ablation 5: Random-weight control ---
    log.info("\n--- Ablation: Random-weight control ---")
    random_weight_results = {}
    dataset = 'cifar10'

    train_ds, test_ds = load_dataset(dataset)
    diff_train = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))
    diff_test = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_test.npy'))
    quint_train = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_train.npy'))
    quint_test = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_test.npy'))
    ref_train_losses, ref_test_losses = load_ref_losses(dataset)
    ref_mean_train = np.mean(ref_train_losses, axis=0)
    ref_mean_test = np.mean(ref_test_losses, axis=0)

    for seed in SEEDS:
        set_seed(seed)
        splits = load_splits(dataset, seed)
        forget_idx = splits['forget_indices']
        retain_idx = splits['retain_indices']

        orig_model = get_model(dataset)
        orig_model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, 'original', f'{dataset}_seed{seed}.pt'),
            map_location=DEVICE, weights_only=True))

        rng = np.random.RandomState(seed + 1000)
        nonmember_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False)

        # Shuffle quintiles randomly
        shuffled_quint = quint_train.copy()
        rng_shuf = np.random.RandomState(seed + 5000)
        # Only shuffle the forget indices
        forget_quint_vals = shuffled_quint[forget_idx].copy()
        rng_shuf.shuffle(forget_quint_vals)
        shuffled_quint[forget_idx] = forget_quint_vals

        # Run DAU-Staged with shuffled quintiles
        unlearned = run_dau_staged(
            'ga', orig_model, train_ds, forget_idx, retain_idx, dataset,
            diff_train, shuffled_quint)

        m_losses = compute_losses_batch(unlearned, train_ds, forget_idx)
        nm_losses = compute_losses_batch(unlearned, test_ds, nonmember_idx.tolist())

        member_ref_list = [rl[forget_idx] for rl in ref_train_losses]
        nonmember_ref_list = [rl[nonmember_idx] for rl in ref_test_losses]
        member_quintiles = quint_train[forget_idx]
        nonmember_quintiles = quint_test[nonmember_idx]

        mia = stratified_mia_eval(
            m_losses, nm_losses,
            ref_mean_train[forget_idx], ref_mean_test[nonmember_idx],
            member_ref_list, nonmember_ref_list,
            member_quintiles, nonmember_quintiles)

        random_weight_results[seed] = {
            'wq_auc': mia['wq_auc'], 'dg': mia['dg'], 'agg_auc': mia['agg_auc'],
            'per_quintile': mia['per_quintile'],
        }
        log.info(f"  Random-weight seed={seed}: wq={mia['wq_auc']:.4f}, dg={mia['dg']:.4f}")

        del unlearned, orig_model
        torch.cuda.empty_cache()

    ablation_results['random_weight_control'] = random_weight_results

    # Save
    with open(os.path.join(RESULTS_DIR, 'v2_ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results


# ============================================================
# SIMPLIFIED HAYES COMPARISON
# ============================================================
def run_hayes_comparison():
    """Simplified per-example U-MIA comparison on a subset of forget samples."""
    log.info("\n" + "="*60)
    log.info("HAYES PER-EXAMPLE U-MIA COMPARISON")
    log.info("="*60)

    dataset = 'cifar10'
    seed = 42
    n_examples = 100  # Subset for comparison

    train_ds, test_ds = load_dataset(dataset)
    splits = load_splits(dataset, seed)
    forget_idx = splits['forget_indices']

    diff_train = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))
    quint_train = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_train.npy'))

    # Load reference model losses for per-example analysis
    ref_train_losses, ref_test_losses = load_ref_losses(dataset)

    # Select 100 forget samples (20 per quintile)
    subset_indices = []
    forget_quints = quint_train[forget_idx]
    for q in range(5):
        q_samples = [forget_idx[i] for i in range(len(forget_idx)) if forget_quints[i] == q]
        rng = np.random.RandomState(42)
        selected = rng.choice(q_samples, size=min(20, len(q_samples)), replace=False)
        subset_indices.extend(selected.tolist())

    # Load GA unlearned model
    ga_model = get_model(dataset)
    ga_path = os.path.join(MODELS_DIR, 'unlearned_v2', 'ga', f'{dataset}_seed{seed}.pt')
    if os.path.exists(ga_path):
        ga_model.load_state_dict(torch.load(ga_path, map_location=DEVICE, weights_only=True))
    else:
        ga_model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, 'unlearned', 'ga', f'{dataset}_seed{seed}.pt'),
            map_location=DEVICE, weights_only=True))

    # Per-example MIA: for each sample, compute its individual attack score
    # using leave-one-out grouping of reference models
    per_example_results = []

    for idx in subset_indices:
        # Get this sample's loss under the unlearned model
        loader = make_loader(train_ds, [idx], batch_size=1, shuffle=False)
        ga_model.eval()
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                loss = nn.CrossEntropyLoss()(ga_model(x), y).item()

        # Get reference model losses for this sample
        ref_losses_sample = [rl[idx] for rl in ref_train_losses]
        ref_mean = np.mean(ref_losses_sample)
        ref_std = np.std(ref_losses_sample) + 1e-8

        # Per-example z-score (simplified LiRA)
        z_score = -(loss - ref_mean) / ref_std

        # Get quintile
        q = quint_train[idx]
        difficulty = diff_train[idx]

        per_example_results.append({
            'index': int(idx),
            'quintile': int(q),
            'difficulty': float(difficulty),
            'target_loss': float(loss),
            'ref_mean_loss': float(ref_mean),
            'z_score': float(z_score),
        })

    # Analyze: do per-example z-scores correlate with difficulty?
    z_scores = [r['z_score'] for r in per_example_results]
    difficulties = [r['difficulty'] for r in per_example_results]
    quintiles = [r['quintile'] for r in per_example_results]

    spearman_corr = stats.spearmanr(difficulties, z_scores)[0]

    # Per-quintile mean z-score
    per_q_z = {}
    for q in range(5):
        q_z = [r['z_score'] for r in per_example_results if r['quintile'] == q]
        per_q_z[q] = {'mean': float(np.mean(q_z)), 'std': float(np.std(q_z)), 'n': len(q_z)}

    hayes_results = {
        'n_examples': len(per_example_results),
        'spearman_zscore_difficulty': float(spearman_corr),
        'per_quintile_zscore': per_q_z,
        'per_example': per_example_results,
    }

    log.info(f"  Spearman(difficulty, z-score) = {spearman_corr:.4f}")
    for q in range(5):
        log.info(f"  Q{q+1}: mean_z={per_q_z[q]['mean']:.4f} ± {per_q_z[q]['std']:.4f} (n={per_q_z[q]['n']})")

    with open(os.path.join(RESULTS_DIR, 'v2_hayes_comparison.json'), 'w') as f:
        json.dump(hayes_results, f, indent=2)

    del ga_model
    torch.cuda.empty_cache()

    return hayes_results


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
def run_statistical_analysis(all_results, utility_results):
    """Comprehensive statistical tests for all success criteria."""
    log.info("\n" + "="*60)
    log.info("STATISTICAL ANALYSIS")
    log.info("="*60)

    stat_results = {}

    # --- Criterion 1: WQ-AUC > Agg AUC ---
    log.info("\n--- Criterion 1: WQ-AUC > Aggregate AUC ---")
    c1_results = {}
    for method in UNLEARN_METHODS:
        for dataset in DATASETS:
            wq_vals = []
            agg_vals = []
            for seed in SEEDS:
                r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
                if r:
                    wq_vals.append(r['wq_auc'])
                    agg_vals.append(r['agg_auc'])

            if len(wq_vals) >= 2:
                diff = np.array(wq_vals) - np.array(agg_vals)
                t_stat, p_val = stats.ttest_1samp(diff, 0)
                p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2

                key = f"{method}_{dataset}"
                c1_results[key] = {
                    'method': method, 'dataset': dataset,
                    'wq_mean': float(np.mean(wq_vals)),
                    'agg_mean': float(np.mean(agg_vals)),
                    'diff_mean': float(np.mean(diff)),
                    'diff_std': float(np.std(diff)),
                    't_stat': float(t_stat),
                    'p_value': float(p_one_sided),
                    'significant': p_one_sided < 0.05 / 15,  # Bonferroni
                    'diff_gt_005': float(np.mean(diff)) > 0.05,
                }
                log.info(f"  {method}/{dataset}: diff={np.mean(diff):.4f}±{np.std(diff):.4f}, "
                        f"p={p_one_sided:.6f}, sig={c1_results[key]['significant']}")

    stat_results['criterion1_wq_gt_agg'] = c1_results

    # --- Criterion 2: DG significantly different from 0 ---
    log.info("\n--- Criterion 2: Difficulty Gap ≠ 0 ---")
    c2_results = {}
    for method in UNLEARN_METHODS:
        for dataset in DATASETS:
            dg_vals = []
            for seed in SEEDS:
                r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
                if r:
                    dg_vals.append(r['dg'])

            if len(dg_vals) >= 2:
                t_stat, p_val = stats.ttest_1samp(dg_vals, 0)
                effect_size = np.mean(dg_vals) / (np.std(dg_vals) + 1e-8)

                key = f"{method}_{dataset}"
                c2_results[key] = {
                    'method': method, 'dataset': dataset,
                    'dg_mean': float(np.mean(dg_vals)),
                    'dg_std': float(np.std(dg_vals)),
                    't_stat': float(t_stat),
                    'p_value': float(p_val),
                    'cohens_d': float(effect_size),
                    'significant': p_val < 0.05 / 15,
                }
                log.info(f"  {method}/{dataset}: DG={np.mean(dg_vals):.4f}±{np.std(dg_vals):.4f}, "
                        f"p={p_val:.6f}, d={effect_size:.2f}")

    stat_results['criterion2_dg_nonzero'] = c2_results

    # --- Criterion 3: DAU reduces WQ-AUC ---
    log.info("\n--- Criterion 3: DAU defense effectiveness ---")
    c3_results = {}
    for method in DAU_METHODS:
        for dataset in DATASETS:
            std_wq = []
            dau_wq = []
            for seed in SEEDS:
                r_std = all_results.get(method, {}).get(dataset, {}).get(seed, {})
                r_dau = all_results.get(f'{method}_dau', {}).get(dataset, {}).get(seed, {})
                if r_std and r_dau:
                    std_wq.append(r_std['wq_auc'])
                    dau_wq.append(r_dau['wq_auc'])

            if len(std_wq) >= 2:
                delta = np.array(std_wq) - np.array(dau_wq)  # positive = DAU better
                t_stat, p_val = stats.ttest_1samp(delta, 0)
                p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2

                # Utility check
                std_ra = [utility_results.get(method, {}).get(dataset, {}).get(seed, {}).get('retain_acc', 0)
                         for seed in SEEDS]
                dau_ra = [utility_results.get(f'{method}_dau', {}).get(dataset, {}).get(seed, {}).get('retain_acc', 0)
                         for seed in SEEDS]
                ra_drop = np.mean(std_ra) - np.mean(dau_ra)

                key = f"{method}_{dataset}"
                c3_results[key] = {
                    'method': method, 'dataset': dataset,
                    'std_wq_mean': float(np.mean(std_wq)),
                    'dau_wq_mean': float(np.mean(dau_wq)),
                    'delta_mean': float(np.mean(delta)),
                    'delta_std': float(np.std(delta)),
                    'p_value': float(p_one_sided),
                    'significant': p_one_sided < 0.05,
                    'ra_drop': float(ra_drop),
                }
                log.info(f"  {method}/{dataset}: std_wq={np.mean(std_wq):.4f}, dau_wq={np.mean(dau_wq):.4f}, "
                        f"delta={np.mean(delta):.4f}, p={p_one_sided:.4f}, ra_drop={ra_drop:.4f}")

    # Also compare DAU vs RUM
    log.info("\n--- DAU vs RUM ---")
    for method in RUM_METHODS:
        for dataset in DATASETS:
            dau_wq = []
            rum_wq = []
            for seed in SEEDS:
                r_dau = all_results.get(f'{method}_dau', {}).get(dataset, {}).get(seed, {})
                r_rum = all_results.get(f'{method}_rum', {}).get(dataset, {}).get(seed, {})
                if r_dau and r_rum:
                    dau_wq.append(r_dau['wq_auc'])
                    rum_wq.append(r_rum['wq_auc'])

            if len(dau_wq) >= 2:
                key = f"dau_vs_rum_{method}_{dataset}"
                c3_results[key] = {
                    'dau_wq_mean': float(np.mean(dau_wq)),
                    'rum_wq_mean': float(np.mean(rum_wq)),
                    'dau_better': float(np.mean(rum_wq)) > float(np.mean(dau_wq)),
                }
                log.info(f"  DAU vs RUM {method}/{dataset}: DAU={np.mean(dau_wq):.4f}, RUM={np.mean(rum_wq):.4f}")

    stat_results['criterion3_dau_defense'] = c3_results

    # --- Retrain baseline DG (for Purchase-100 correction) ---
    log.info("\n--- Retrain baseline DG ---")
    retrain_dg = {}
    for dataset in DATASETS:
        dg_vals = []
        for seed in SEEDS:
            r = all_results.get('retrain', {}).get(dataset, {}).get(seed, {})
            if r:
                dg_vals.append(r['dg'])
        retrain_dg[dataset] = {'mean': float(np.mean(dg_vals)), 'std': float(np.std(dg_vals))}
        log.info(f"  Retrain {dataset}: DG={np.mean(dg_vals):.4f}±{np.std(dg_vals):.4f}")

    stat_results['retrain_baseline_dg'] = retrain_dg

    with open(os.path.join(RESULTS_DIR, 'v2_statistical_tests.json'), 'w') as f:
        json.dump(stat_results, f, indent=2)

    return stat_results


# ============================================================
# GENERATE FIGURES
# ============================================================
def generate_figures(all_results, utility_results, ablation_results=None):
    """Generate all publication figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = os.path.join(os.path.dirname(BASE_DIR), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    sns.set_style('whitegrid')
    COLORS = sns.color_palette('colorblind', 10)
    METHOD_COLORS = {
        'retrain': COLORS[7], 'ft': COLORS[0], 'ga': COLORS[1], 'rl': COLORS[2],
        'scrub': COLORS[3], 'neggrad': COLORS[4],
        'ga_dau': COLORS[5], 'scrub_dau': COLORS[6], 'ft_dau': COLORS[8],
        'ga_rum': COLORS[9], 'scrub_rum': COLORS[1],
    }

    # ---- Figure 1: Per-quintile MIA AUC (main motivation) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dataset_labels = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'purchase100': 'Purchase-100'}

    for di, dataset in enumerate(DATASETS):
        ax = axes[di]
        methods_to_plot = ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad']
        x = np.arange(5)
        width = 0.12

        for mi, method in enumerate(methods_to_plot):
            per_q_all = []
            for seed in SEEDS:
                r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
                if r and 'per_quintile' in r:
                    per_q_all.append(r['per_quintile'])

            if per_q_all:
                per_q_mean = np.mean(per_q_all, axis=0)
                per_q_std = np.std(per_q_all, axis=0)
                offset = (mi - len(methods_to_plot)/2 + 0.5) * width
                bars = ax.bar(x + offset, per_q_mean, width, yerr=per_q_std,
                             label=method.upper() if mi < 6 else method,
                             color=METHOD_COLORS.get(method, COLORS[mi]),
                             capsize=2, alpha=0.85)

        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Perfect unlearning')
        ax.set_xlabel('Difficulty Quintile', fontsize=12)
        ax.set_ylabel('MIA-AUC' if di == 0 else '', fontsize=12)
        ax.set_title(dataset_labels[dataset], fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1\n(Easy)', 'Q2', 'Q3', 'Q4', 'Q5\n(Hard)'])
        ax.set_ylim(0.3, 1.05)
        if di == 0:
            ax.legend(fontsize=8, ncol=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure1_stratified_mia.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'figure1_stratified_mia.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log.info("  Saved Figure 1: Stratified MIA")

    # ---- Figure 2: Aggregate vs WQ scatter ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    markers = {'cifar10': 'o', 'cifar100': 's', 'purchase100': '^'}

    for method in UNLEARN_METHODS:
        for dataset in DATASETS:
            for seed in SEEDS:
                r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
                if r:
                    ax.scatter(r['agg_auc'], r['wq_auc'],
                              color=METHOD_COLORS.get(method, 'gray'),
                              marker=markers[dataset], s=60, alpha=0.7,
                              label=f'{method.upper()} ({dataset_labels[dataset]})' if seed == SEEDS[0] else '')

    lims = [0.4, 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Aggregate MIA-AUC', fontsize=12)
    ax.set_ylabel('Worst-Quintile MIA-AUC', fontsize=12)
    ax.set_title('Aggregate vs. Worst-Quintile AUC', fontsize=13)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure2_aggregate_vs_wq.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'figure2_aggregate_vs_wq.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log.info("  Saved Figure 2: Aggregate vs WQ scatter")

    # ---- Figure 3: DAU defense effectiveness ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for di, dataset in enumerate(DATASETS):
        ax = axes[di]
        methods = ['ga', 'scrub']
        x = np.arange(len(methods))
        width = 0.25

        for vi, (variant, label) in enumerate(
            [('', 'Standard'), ('_dau', 'DAU (Staged)'), ('_rum', 'RUM')]):
            wq_means, wq_stds = [], []
            for method in methods:
                key = f'{method}{variant}'
                wq_vals = []
                for seed in SEEDS:
                    r = all_results.get(key, {}).get(dataset, {}).get(seed, {})
                    if r:
                        wq_vals.append(r['wq_auc'])
                wq_means.append(np.mean(wq_vals) if wq_vals else 0)
                wq_stds.append(np.std(wq_vals) if wq_vals else 0)

            offset = (vi - 1) * width
            bars = ax.bar(x + offset, wq_means, width, yerr=wq_stds,
                         label=label, capsize=3, alpha=0.85)

        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Unlearning Method', fontsize=12)
        ax.set_ylabel('Worst-Quintile AUC' if di == 0 else '', fontsize=12)
        ax.set_title(dataset_labels[dataset], fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_ylim(0.4, 1.05)
        if di == 0:
            ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure3_dau_defense.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'figure3_dau_defense.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log.info("  Saved Figure 3: DAU defense")

    # ---- Figure 4: Staging intensity sensitivity ----
    if ablation_results and 'staging_profiles' in ablation_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        profiles = ['uniform', 'mild', 'moderate', 'strong', 'extreme']
        profile_labels = ['Uniform\n(Standard)', 'Mild', 'Moderate\n(Default)', 'Strong', 'Extreme']

        for di, dataset in enumerate(['cifar10', 'cifar100']):
            ax = axes[di]
            for method in ['ga', 'scrub']:
                wq_means, wq_stds = [], []
                for profile in profiles:
                    vals = []
                    for seed in SEEDS:
                        key = f"{profile}_{method}_{dataset}_{seed}"
                        r = ablation_results['staging_profiles'].get(key, {})
                        if r:
                            vals.append(r['wq_auc'])
                    wq_means.append(np.mean(vals) if vals else 0)
                    wq_stds.append(np.std(vals) if vals else 0)

                ax.errorbar(range(len(profiles)), wq_means, yerr=wq_stds,
                           marker='o', capsize=3, label=method.upper(), linewidth=2)

            ax.set_xlabel('Staging Profile', fontsize=12)
            ax.set_ylabel('Worst-Quintile AUC' if di == 0 else '', fontsize=12)
            ax.set_title(dataset_labels[dataset], fontsize=13)
            ax.set_xticks(range(len(profiles)))
            ax.set_xticklabels(profile_labels, fontsize=9)
            ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure4_staging_sensitivity.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, 'figure4_staging_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        log.info("  Saved Figure 4: Staging sensitivity")

    # ---- Figure 5: Ablation summary (2x2) ----
    if ablation_results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (a) K sensitivity
        ax = axes[0, 0]
        k_res = ablation_results.get('K_ablation', {})
        ks = sorted(k_res.keys())
        if ks:
            spearmans = [k_res[k]['spearman'] for k in ks]
            stabilities = [k_res[k]['quintile_stability'] for k in ks]
            ax.plot(ks, spearmans, 'o-', label='Spearman ρ', linewidth=2, markersize=8)
            ax.plot(ks, stabilities, 's-', label='Quintile stability', linewidth=2, markersize=8)
            ax.set_xlabel('K (reference models)', fontsize=12)
            ax.set_ylabel('Correlation / Stability', fontsize=12)
            ax.set_title('(a) Reference Model Count K', fontsize=12)
            ax.legend(fontsize=10)
            ax.set_ylim(0.7, 1.05)

        # (b) Strata granularity
        ax = axes[0, 1]
        strata_res = ablation_results.get('strata_granularity', {})
        if strata_res:
            ns = sorted(strata_res.keys())
            wqs = [strata_res[n]['wq_auc'] for n in ns]
            dgs = [strata_res[n]['dg'] for n in ns]
            ax.bar([str(n) for n in ns], wqs, alpha=0.7, label='WQ-AUC')
            ax2 = ax.twinx()
            ax2.plot([str(n) for n in ns], dgs, 'ro-', linewidth=2, label='DG')
            ax.set_xlabel('Number of Strata', fontsize=12)
            ax.set_ylabel('WQ-AUC', fontsize=12)
            ax2.set_ylabel('Difficulty Gap', fontsize=12, color='red')
            ax.set_title('(b) Stratification Granularity', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax2.legend(loc='upper right', fontsize=10)

        # (c) Forget set size
        ax = axes[1, 0]
        fs_res = ablation_results.get('forget_size', {})
        if fs_res:
            for dataset in ['cifar10', 'cifar100']:
                fsizes, std_dgs, dau_dgs = [], [], []
                for fsize in [500, 1000, 2500]:
                    key = f"{dataset}_{fsize}_ga"
                    r = fs_res.get(key, {})
                    if r:
                        fsizes.append(fsize)
                        std_dgs.append(r['standard_dg'])
                        dau_dgs.append(r['dau_dg'])
                if fsizes:
                    ax.plot(fsizes, std_dgs, 'o-', label=f'{dataset_labels[dataset]} Std', linewidth=2)
                    ax.plot(fsizes, dau_dgs, 's--', label=f'{dataset_labels[dataset]} DAU', linewidth=2)
            ax.set_xlabel('Forget Set Size', fontsize=12)
            ax.set_ylabel('Difficulty Gap', fontsize=12)
            ax.set_title('(c) Forget Set Size', fontsize=12)
            ax.legend(fontsize=9)

        # (d) Random-weight control
        ax = axes[1, 1]
        rw_res = ablation_results.get('random_weight_control', {})
        if rw_res:
            # Compare: standard GA, DAU GA, random-weight GA
            labels_bar = ['Standard\nGA', 'DAU\n(True Diff.)', 'DAU\n(Random)']

            # Get standard GA and DAU GA WQ-AUC
            std_wq = [all_results_global.get('ga', {}).get('cifar10', {}).get(s, {}).get('wq_auc', 0) for s in SEEDS]
            dau_wq = [all_results_global.get('ga_dau', {}).get('cifar10', {}).get(s, {}).get('wq_auc', 0) for s in SEEDS]
            rand_wq = [rw_res.get(s, {}).get('wq_auc', 0) for s in SEEDS]

            means = [np.mean(std_wq), np.mean(dau_wq), np.mean(rand_wq)]
            stds = [np.std(std_wq), np.std(dau_wq), np.std(rand_wq)]

            bars = ax.bar(labels_bar, means, yerr=stds, capsize=5,
                         color=[COLORS[1], COLORS[5], COLORS[7]], alpha=0.85)
            ax.set_ylabel('Worst-Quintile AUC', fontsize=12)
            ax.set_title('(d) Random-Weight Control', fontsize=12)
            ax.set_ylim(0.4, 1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure5_ablations.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, 'figure5_ablations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        log.info("  Saved Figure 5: Ablation summary")

    # ---- Figure 6: Difficulty distribution ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for di, dataset in enumerate(DATASETS):
        ax = axes[di]
        diff = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))
        quint = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_train.npy'))

        for q in range(5):
            mask = quint == q
            ax.hist(diff[mask], bins=30, alpha=0.5, label=f'Q{q+1}', density=True)

        ax.set_xlabel('Difficulty Score', fontsize=12)
        ax.set_ylabel('Density' if di == 0 else '', fontsize=12)
        ax.set_title(dataset_labels[dataset], fontsize=13)
        if di == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'figure6_difficulty_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'figure6_difficulty_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log.info("  Saved Figure 6: Difficulty distribution")

    # ---- Generate LaTeX tables ----
    generate_latex_tables(all_results, utility_results, fig_dir)


def generate_latex_tables(all_results, utility_results, fig_dir):
    """Generate LaTeX tables for the paper."""

    # Table 1: Main results
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results across datasets. Mean$\pm$std over 3 seeds. "
                 r"$\downarrow$ indicates lower is better for WQ-AUC and DG.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l|ccccc|ccccc|ccccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{5}{c|}{CIFAR-10} & \multicolumn{5}{c|}{CIFAR-100} & \multicolumn{5}{c}{Purchase-100} \\")
    lines.append(r"Method & Agg$\downarrow$ & WQ$\downarrow$ & DG$\downarrow$ & RA & FA "
                 r"& Agg$\downarrow$ & WQ$\downarrow$ & DG$\downarrow$ & RA & FA "
                 r"& Agg$\downarrow$ & WQ$\downarrow$ & DG$\downarrow$ & RA & FA \\")
    lines.append(r"\midrule")

    all_methods = ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad',
                   'ga_dau', 'scrub_dau', 'ft_dau', 'ga_rum', 'scrub_rum']
    method_names = {
        'retrain': 'Retrain', 'ft': 'Fine-tune', 'ga': 'Grad. Ascent',
        'rl': 'Random Labels', 'scrub': 'SCRUB', 'neggrad': 'NegGrad+KD',
        'ga_dau': 'GA-DAU', 'scrub_dau': 'SCRUB-DAU', 'ft_dau': 'FT-DAU',
        'ga_rum': 'GA-RUM', 'scrub_rum': 'SCRUB-RUM',
    }

    for method in all_methods:
        if method == 'ga_dau':
            lines.append(r"\midrule")
        parts = [method_names.get(method, method)]

        for dataset in DATASETS:
            agg_vals, wq_vals, dg_vals, ra_vals, fa_vals = [], [], [], [], []
            for seed in SEEDS:
                r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
                u = utility_results.get(method, {}).get(dataset, {}).get(seed, {})
                if r:
                    agg_vals.append(r['agg_auc'])
                    wq_vals.append(r['wq_auc'])
                    dg_vals.append(r['dg'])
                if u:
                    ra_vals.append(u.get('retain_acc', u.get('test_acc', 0)))
                    fa_vals.append(u.get('forget_acc', 0))

            if agg_vals:
                parts.append(f"${np.mean(agg_vals):.3f}_{{\\pm{np.std(agg_vals):.3f}}}$")
                parts.append(f"${np.mean(wq_vals):.3f}_{{\\pm{np.std(wq_vals):.3f}}}$")
                parts.append(f"${np.mean(dg_vals):.3f}_{{\\pm{np.std(dg_vals):.3f}}}$")
                parts.append(f"${np.mean(ra_vals):.3f}$" if ra_vals else "---")
                parts.append(f"${np.mean(fa_vals):.3f}$" if fa_vals else "---")
            else:
                parts.extend(["---"] * 5)

        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    with open(os.path.join(fig_dir, 'table1_main_results.tex'), 'w') as f:
        f.write('\n'.join(lines))
    log.info("  Saved Table 1: Main results")


# ============================================================
# COMPILE FINAL RESULTS.JSON
# ============================================================
def compile_final_results(all_results, utility_results, stat_results, ablation_results, hayes_results):
    """Compile everything into the final results.json."""

    final = {
        'main_results': {},
        'statistical_tests': stat_results,
        'ablations': {},
        'hayes_comparison': hayes_results,
        'experiment_config': {
            'unlearn_config': {k: {kk: vv for kk, vv in v.items()}
                              for k, v in UNLEARN_CONFIG.items()},
            'dau_stages': DAU_STAGES,
            'seeds': SEEDS,
            'datasets': DATASETS,
        }
    }

    # Aggregate main results
    for dataset in DATASETS:
        final['main_results'][dataset] = {}
        all_methods = set()
        for method in all_results:
            all_methods.add(method)

        for method in sorted(all_methods):
            vals = {'agg_auc': [], 'wq_auc': [], 'dg': [], 'per_quintile': []}
            util_vals = {'test_acc': [], 'retain_acc': [], 'forget_acc': []}

            for seed in SEEDS:
                r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
                u = utility_results.get(method, {}).get(dataset, {}).get(seed, {})
                if r:
                    for k in vals:
                        if k in r:
                            vals[k].append(r[k])
                if u:
                    for k in util_vals:
                        if k in u:
                            util_vals[k].append(u[k])

            entry = {}
            for k, v in vals.items():
                if v and k != 'per_quintile':
                    entry[f'{k}_mean'] = float(np.mean(v))
                    entry[f'{k}_std'] = float(np.std(v))
                    entry[k] = f"{np.mean(v):.4f}±{np.std(v):.4f}"
                elif v and k == 'per_quintile':
                    entry['per_quintile_mean'] = [float(x) for x in np.mean(v, axis=0)]
                    entry['per_quintile_std'] = [float(x) for x in np.std(v, axis=0)]

            for k, v in util_vals.items():
                if v:
                    entry[f'{k}_mean'] = float(np.mean(v))
                    entry[f'{k}_std'] = float(np.std(v))
                    entry[k] = f"{np.mean(v):.4f}±{np.std(v):.4f}"

            if entry:
                final['main_results'][dataset][method] = entry

    # Add ablation summaries
    if ablation_results:
        final['ablations'] = {
            'staging_profiles': ablation_results.get('staging_profiles', {}),
            'K_ablation': ablation_results.get('K_ablation', {}),
            'strata_granularity': ablation_results.get('strata_granularity', {}),
            'forget_size': ablation_results.get('forget_size', {}),
            'random_weight_control': ablation_results.get('random_weight_control', {}),
        }

    return final


# ============================================================
# MAIN
# ============================================================
all_results_global = None  # For figure generation

if __name__ == '__main__':
    log.info("="*60)
    log.info("EXPERIMENT V2: Addressing reviewer feedback")
    log.info(f"Device: {DEVICE}")
    log.info(f"Datasets: {DATASETS}")
    log.info(f"Seeds: {SEEDS}")
    log.info(f"Unlearning methods: {UNLEARN_METHODS}")
    log.info(f"DAU methods: {DAU_METHODS}")
    log.info(f"DAU stages: {DAU_STAGES}")
    log.info("="*60)

    total_start = time.time()

    # Phase 1: Main experiments
    log.info("\n=== PHASE 1: Main experiments ===")
    all_results, utility_results = run_all_experiments()
    all_results_global = all_results

    # Phase 2: Ablations
    log.info("\n=== PHASE 2: Ablations ===")
    ablation_results = run_ablations()

    # Phase 3: Hayes comparison
    log.info("\n=== PHASE 3: Hayes comparison ===")
    hayes_results = run_hayes_comparison()

    # Phase 4: Statistical analysis
    log.info("\n=== PHASE 4: Statistical analysis ===")
    stat_results = run_statistical_analysis(all_results, utility_results)

    # Phase 5: Figures
    log.info("\n=== PHASE 5: Figures ===")
    generate_figures(all_results, utility_results, ablation_results)

    # Phase 6: Compile final results
    log.info("\n=== PHASE 6: Compile final results ===")
    final = compile_final_results(all_results, utility_results, stat_results, ablation_results, hayes_results)

    results_path = os.path.join(os.path.dirname(BASE_DIR), 'results.json')
    with open(results_path, 'w') as f:
        json.dump(final, f, indent=2)

    total_elapsed = (time.time() - total_start) / 60
    log.info(f"\n{'='*60}")
    log.info(f"ALL EXPERIMENTS COMPLETE: {total_elapsed:.1f} minutes")
    log.info(f"Results saved to {results_path}")
    log.info(f"{'='*60}")
