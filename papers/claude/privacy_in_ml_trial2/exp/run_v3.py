#!/usr/bin/env python3
"""
Difficulty-Aware Unlearning (DAU) — Complete Experiment Pipeline v3.

Fixes from v1/v2:
  1. Stronger unlearning hyperparameters (forget_acc was 100% → target near random chance)
  2. Fixed DAU weight mapping (use indexed datasets, not batch_idx arithmetic)
  3. Single authoritative pipeline producing one results.json

Reuses existing: reference models, original models, retrain models, difficulty scores.
"""

import os, sys, json, time, copy, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings('ignore')

# Ensure we're in workspace root
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

from exp.shared.config import *
from exp.shared.models import get_model
from exp.shared.utils import set_seed, load_dataset, create_splits, get_loader, evaluate_model, train_model
from exp.shared.mia import compute_losses, run_all_attacks, stratified_mia

RESULTS_DIR = 'exp/results_v3'
MODELS_DIR = 'exp/models_v3'
FIGURES_DIR = 'figures'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
for m in ['ft', 'ga', 'rl', 'scrub', 'neggrad',
          'ft_dau', 'ga_dau', 'rl_dau', 'scrub_dau', 'neggrad_dau',
          'ga_rum', 'scrub_rum']:
    os.makedirs(f'{MODELS_DIR}/{m}', exist_ok=True)

LOG_FILE = f'{RESULTS_DIR}/run_v3.log'

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

# ============================================================
# Indexed Dataset Wrapper (for proper DAU weight mapping)
# ============================================================
class IndexedSubset(Dataset):
    """Wraps a dataset subset and returns (x, y, local_idx) where local_idx
    is the position within the provided indices list."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, y, idx  # idx is the local index into self.indices


# ============================================================
# UNLEARNING METHODS (v3 — stronger hyperparameters)
# ============================================================

def get_unlearn_params(method, dataset):
    """Return calibrated hyperparameters for each method × dataset.

    Calibrated via grid search on CIFAR-10 seed 42 to achieve meaningful
    forgetting (FA < 0.97) while preserving utility (RA > 0.90, TA > 0.85).
    Phase transition analysis shows GA exhibits a sharp collapse boundary;
    we use the GA+FT interleaved approach to push toward that boundary safely.
    """
    params = {
        ('ft', 'cifar10'):      {'lr': 0.01, 'epochs': 15},
        ('ft', 'cifar100'):     {'lr': 0.01, 'epochs': 15},
        ('ft', 'purchase100'):  {'lr': 0.005, 'epochs': 15},
        ('ga', 'cifar10'):      {'ga_lr': 0.02, 'ft_lr': 0.01, 'epochs': 30},
        ('ga', 'cifar100'):     {'ga_lr': 0.02, 'ft_lr': 0.01, 'epochs': 30},
        ('ga', 'purchase100'):  {'ga_lr': 0.01, 'ft_lr': 0.005, 'epochs': 25},
        ('rl', 'cifar10'):      {'lr': 0.01, 'epochs': 15},
        ('rl', 'cifar100'):     {'lr': 0.01, 'epochs': 15},
        ('rl', 'purchase100'):  {'lr': 0.005, 'epochs': 15},
        ('scrub', 'cifar10'):   {'forget_lr': 0.01, 'retain_lr': 0.01, 'passes': 20},
        ('scrub', 'cifar100'):  {'forget_lr': 0.01, 'retain_lr': 0.01, 'passes': 20},
        ('scrub', 'purchase100'): {'forget_lr': 0.005, 'retain_lr': 0.005, 'passes': 15},
        ('neggrad', 'cifar10'):   {'lr': 0.01, 'epochs': 20, 'forget_w': 2.0},
        ('neggrad', 'cifar100'):  {'lr': 0.01, 'epochs': 20, 'forget_w': 2.0},
        ('neggrad', 'purchase100'): {'lr': 0.005, 'epochs': 15, 'forget_w': 2.0},
    }
    return params.get((method, dataset), {})


def unlearn_finetune(model, forget_loader, retain_loader, dataset, epochs=None,
                     sample_weights=None, device=DEVICE):
    """Fine-tune on retain set only."""
    p = get_unlearn_params('ft', dataset)
    epochs = epochs or p.get('epochs', 15)
    lr = p.get('lr', 0.01)
    model = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch in retain_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def unlearn_ga(model, forget_loader, retain_loader, dataset, epochs=None,
               sample_weights=None, device=DEVICE):
    """Gradient ascent interleaved with full retain-set fine-tuning.

    Calibrated approach: alternate GA on forget set with FT on retain set
    each epoch. This pushes toward the phase transition boundary while
    the retain FT prevents catastrophic collapse.
    """
    p = get_unlearn_params('ga', dataset)
    epochs = epochs or p.get('epochs', 30)
    ga_lr = p.get('ga_lr', 0.02)
    ft_lr = p.get('ft_lr', 0.01)

    model = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        ga_opt = optim.SGD(model.parameters(), lr=ga_lr, momentum=0.9, weight_decay=5e-4)
        ft_opt = optim.SGD(model.parameters(), lr=ft_lr, momentum=0.9, weight_decay=5e-4)
    else:
        ga_opt = optim.Adam(model.parameters(), lr=ga_lr, weight_decay=1e-4)
        ft_opt = optim.Adam(model.parameters(), lr=ft_lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(reduction='none')
    retain_criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        # GA step on forget set
        for batch in forget_loader:
            if len(batch) == 3:
                x, y, idxs = batch[0].to(device), batch[1].to(device), batch[2]
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                idxs = None
            ga_opt.zero_grad()
            losses = criterion(model(x), y)
            if sample_weights is not None and idxs is not None:
                w = sample_weights[idxs].to(device)
                losses = losses * w
            loss = -losses.mean()
            loss.backward()
            ga_opt.step()

        # Full FT epoch on retain set
        for batch in retain_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            ft_opt.zero_grad()
            loss = retain_criterion(model(x), y)
            loss.backward()
            ft_opt.step()

    return model


def unlearn_random_labels(model, forget_loader, retain_loader, dataset, epochs=None,
                          num_classes=10, sample_weights=None, device=DEVICE):
    """Train on forget set with random labels + retain with true labels."""
    p = get_unlearn_params('rl', dataset)
    epochs = epochs or p.get('epochs', 15)
    lr = p.get('lr', 0.01)

    model = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_retain = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch in forget_loader:
            if len(batch) == 3:
                x, y, idxs = batch[0].to(device), batch[1].to(device), batch[2]
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                idxs = None
            fake_y = torch.randint(0, num_classes, (x.size(0),), device=device)
            optimizer.zero_grad()
            losses = criterion(model(x), fake_y)
            if sample_weights is not None and idxs is not None:
                w = sample_weights[idxs].to(device)
                losses = losses * w
            loss = losses.mean()
            loss.backward()
            optimizer.step()
        for batch in retain_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion_retain(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def unlearn_scrub(model, forget_loader, retain_loader, dataset, passes=None,
                  sample_weights=None, device=DEVICE):
    """SCRUB: teacher-student with gradient ascent on forget set.

    Uses CE-based gradient ascent (not just KL to uniform) for stronger forgetting,
    interleaved with KL(student||teacher) on retain set.
    """
    p = get_unlearn_params('scrub', dataset)
    passes = passes or p.get('passes', 20)
    forget_lr = p.get('forget_lr', 0.01)
    retain_lr = p.get('retain_lr', 0.01)

    teacher = copy.deepcopy(model)
    teacher.eval()
    student = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        opt_forget = optim.SGD(student.parameters(), lr=forget_lr, momentum=0.9, weight_decay=5e-4)
        opt_retain = optim.SGD(student.parameters(), lr=retain_lr, momentum=0.9, weight_decay=5e-4)
    else:
        opt_forget = optim.Adam(student.parameters(), lr=forget_lr, weight_decay=1e-4)
        opt_retain = optim.Adam(student.parameters(), lr=retain_lr, weight_decay=1e-4)
    student.train()
    criterion = nn.CrossEntropyLoss(reduction='none')

    for pass_idx in range(passes):
        # Forget step: gradient ascent on CE loss for all forget batches
        for batch in forget_loader:
            if len(batch) == 3:
                x, y, idxs = batch[0].to(device), batch[1].to(device), batch[2]
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                idxs = None
            opt_forget.zero_grad()
            losses = criterion(student(x), y)
            if sample_weights is not None and idxs is not None:
                w = sample_weights[idxs].to(device)
                losses = losses * w
            loss = -losses.mean()
            loss.backward()
            opt_forget.step()

        # Retain step: 5 batches of KL(student || teacher)
        retain_iter = iter(retain_loader)
        for i in range(5):
            try:
                batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch = next(retain_iter)
            x, y = batch[0].to(device), batch[1].to(device)
            opt_retain.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss = F.kl_div(F.log_softmax(student_logits, dim=1),
                           F.softmax(teacher_logits, dim=1), reduction='batchmean')
            loss.backward()
            opt_retain.step()

    return student


def unlearn_neggrad_kd(model, forget_loader, retain_loader, dataset, epochs=None,
                       sample_weights=None, device=DEVICE):
    """NegGrad + KD with calibrated forget weight."""
    p = get_unlearn_params('neggrad', dataset)
    epochs = epochs or p.get('epochs', 20)
    lr = p.get('lr', 0.01)
    forget_w = p.get('forget_w', 2.0)

    teacher = copy.deepcopy(model)
    teacher.eval()
    student = copy.deepcopy(model)
    if dataset in ('cifar10', 'cifar100'):
        optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    student.train()

    for epoch in range(epochs):
        retain_iter = iter(retain_loader)
        for batch in forget_loader:
            if len(batch) == 3:
                x, y, idxs = batch[0].to(device), batch[1].to(device), batch[2]
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                idxs = None
            optimizer.zero_grad()
            losses = criterion(student(x), y)
            if sample_weights is not None and idxs is not None:
                w = sample_weights[idxs].to(device)
                losses = losses * w
            loss_forget = -forget_w * losses.mean()

            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)
            rx, ry = r_batch[0].to(device), r_batch[1].to(device)
            with torch.no_grad():
                teacher_logits = teacher(rx)
            student_logits = student(rx)
            loss_retain = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits, dim=1), reduction='batchmean')
            loss = loss_forget + loss_retain
            loss.backward()
            optimizer.step()

    return student


UNLEARN_FNS = {
    'ft': unlearn_finetune,
    'ga': unlearn_ga,
    'rl': unlearn_random_labels,
    'scrub': unlearn_scrub,
    'neggrad': unlearn_neggrad_kd,
}


def compute_dau_weights(difficulty_scores, alpha=1.0):
    """DAU weights: w(x) = 1 + alpha * (d(x) - mean) / std, clamped [0.1, 10]."""
    d_mean = difficulty_scores.mean()
    d_std = difficulty_scores.std() + 1e-8
    weights = 1.0 + alpha * (difficulty_scores - d_mean) / d_std
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_rum_epochs(difficulty_scores, base_epochs, n_groups=3):
    """RUM: partition into groups, return per-group epoch counts."""
    percs = np.percentile(difficulty_scores, [100/n_groups * i for i in range(1, n_groups)])
    groups = np.digitize(difficulty_scores, percs)
    # easy=fewer epochs, hard=more
    epoch_map = {0: max(3, base_epochs // 2), 1: base_epochs, 2: base_epochs * 2}
    return groups, epoch_map


# ============================================================
# MIA EVALUATION
# ============================================================

def get_ref_losses(dataset_name, n_ref=4, split='train'):
    """Load pre-computed reference model losses."""
    losses = []
    for k in range(n_ref):
        path = f'exp/results/ref_{split}_losses_{dataset_name}_ref{k}.npy'
        losses.append(np.load(path))
    return losses


def run_mia_evaluation(model, forget_indices, test_indices, train_eval_ds, test_ds,
                       dataset_name, difficulty_train, difficulty_test,
                       quintiles_train, quintiles_test, n_ref=4, device=DEVICE):
    """Full MIA evaluation: aggregate + stratified."""
    # Get forget and test loaders (no shuffle, for consistent indexing)
    forget_loader = get_loader(train_eval_ds, forget_indices, batch_size=512, shuffle=False)
    test_loader = get_loader(test_ds, test_indices, batch_size=512, shuffle=False)

    # Compute losses on unlearned model
    member_losses = compute_losses(model, forget_loader, device)
    nonmember_losses = compute_losses(model, test_loader, device)

    # Reference model losses
    ref_train_losses = get_ref_losses(dataset_name, n_ref, 'train')
    ref_test_losses = get_ref_losses(dataset_name, n_ref, 'test')

    # Extract losses for forget/test indices
    member_ref_list = [r[forget_indices] for r in ref_train_losses]
    nonmember_ref_list = [r[test_indices] for r in ref_test_losses]

    member_ref_mean = np.mean([r[forget_indices] for r in ref_train_losses], axis=0)
    nonmember_ref_mean = np.mean([r[test_indices] for r in ref_test_losses], axis=0)

    # Quintiles for forget and test sets
    member_quintiles = quintiles_train[forget_indices]
    nonmember_quintiles = quintiles_test[test_indices]

    # Run stratified MIA
    results = stratified_mia(
        member_losses, nonmember_losses,
        member_ref_mean, nonmember_ref_mean,
        member_ref_list, nonmember_ref_list,
        member_quintiles, nonmember_quintiles,
        n_strata=5
    )

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_unlearning_for_config(dataset_name, seed, original_model, train_ds, train_eval_ds,
                              test_ds, splits, difficulty_train, difficulty_test,
                              quintiles_train, quintiles_test):
    """Run all unlearning methods + DAU + RUM for one (dataset, seed) config."""
    forget_indices = splits['forget_indices']
    retain_indices = splits['retain_indices']

    num_classes = 10 if dataset_name == 'cifar10' else 100

    # Create loaders
    retain_loader = get_loader(train_ds, retain_indices, batch_size=BATCH_SIZE, shuffle=True)
    # Standard forget loader (no indices)
    forget_loader_std = get_loader(train_ds, forget_indices, batch_size=BATCH_SIZE, shuffle=True)
    # Indexed forget loader (for DAU weight mapping)
    indexed_forget_ds = IndexedSubset(train_ds, forget_indices)
    forget_loader_idx = DataLoader(indexed_forget_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True)

    # Eval loaders
    forget_eval_loader = get_loader(train_eval_ds, forget_indices, batch_size=512, shuffle=False)

    # Difficulty scores for forget set
    forget_difficulty = difficulty_train[forget_indices]

    # Select matched non-member test indices (same size as forget, difficulty-matched per quintile)
    rng = np.random.RandomState(seed + 1000)
    n_test = len(test_ds)
    test_quintiles = quintiles_test[:n_test]
    member_quintiles = quintiles_train[forget_indices]
    test_indices_list = []
    for q in range(5):
        member_q_count = (member_quintiles == q).sum()
        nonmember_q_candidates = np.where(test_quintiles == q)[0]
        if len(nonmember_q_candidates) >= member_q_count:
            chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=False)
        else:
            chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=True)
        test_indices_list.append(chosen)
    test_indices = np.concatenate(test_indices_list)

    all_results = {}

    # --- Standard unlearning methods ---
    for method in UNLEARN_METHODS:
        model_path = f'{MODELS_DIR}/{method}/{dataset_name}_seed{seed}.pt'
        t0 = time.time()

        extra_kwargs = {}
        if method == 'rl':
            extra_kwargs['num_classes'] = num_classes
        if method == 'scrub':
            fn = UNLEARN_FNS[method]
            unlearned = fn(original_model, forget_loader_std, retain_loader, dataset_name,
                          **extra_kwargs)
        else:
            fn = UNLEARN_FNS[method]
            unlearned = fn(original_model, forget_loader_std, retain_loader, dataset_name,
                          **extra_kwargs)

        elapsed = time.time() - t0
        torch.save(unlearned.state_dict(), model_path)

        # Evaluate utility
        retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval_ds, retain_indices, 512, False))
        test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, None, 512, False))
        forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)

        # MIA evaluation
        mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                                 train_eval_ds, test_ds, dataset_name,
                                 difficulty_train, difficulty_test,
                                 quintiles_train, quintiles_test)

        all_results[method] = {
            'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
            'forget_acc': float(forget_acc),
            'aggregate_auc': mia['aggregate']['best_auc'],
            'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
            'per_quintile': {k: v['best_auc'] for k, v in mia.items() if k.startswith('q')},
            'time_sec': elapsed,
        }
        log(f"  {method}: forget_acc={forget_acc:.4f} retain_acc={retain_acc:.4f} "
            f"test_acc={test_acc:.4f} agg_auc={mia['aggregate']['best_auc']:.4f} "
            f"wq_auc={mia['wq_auc']:.4f} dg={mia['dg']:.4f} [{elapsed:.1f}s]")

        del unlearned
        torch.cuda.empty_cache()

    # --- DAU defense ---
    for alpha in [DEFAULT_ALPHA]:
        dau_weights = compute_dau_weights(forget_difficulty, alpha=alpha)

        for method in UNLEARN_METHODS:
            dau_key = f'{method}_dau'
            model_path = f'{MODELS_DIR}/{dau_key}/{dataset_name}_seed{seed}.pt'
            t0 = time.time()

            extra_kwargs = {}
            if method == 'rl':
                extra_kwargs['num_classes'] = num_classes

            fn = UNLEARN_FNS[method]
            if method in ('ft',):
                # FT doesn't use forget loader weights directly
                unlearned = fn(original_model, forget_loader_idx, retain_loader, dataset_name,
                              sample_weights=None, **extra_kwargs)
            else:
                unlearned = fn(original_model, forget_loader_idx, retain_loader, dataset_name,
                              sample_weights=dau_weights, **extra_kwargs)

            elapsed = time.time() - t0
            torch.save(unlearned.state_dict(), model_path)

            retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval_ds, retain_indices, 512, False))
            test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, None, 512, False))
            forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)

            mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                                     train_eval_ds, test_ds, dataset_name,
                                     difficulty_train, difficulty_test,
                                     quintiles_train, quintiles_test)

            std_wq = all_results[method]['wq_auc']
            all_results[dau_key] = {
                'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
                'forget_acc': float(forget_acc),
                'aggregate_auc': mia['aggregate']['best_auc'],
                'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
                'per_quintile': {k: v['best_auc'] for k, v in mia.items() if k.startswith('q')},
                'delta_wq': std_wq - mia['wq_auc'],
                'delta_dg': all_results[method]['dg'] - mia['dg'],
                'alpha': alpha,
                'time_sec': elapsed,
            }
            log(f"  {dau_key}: forget_acc={forget_acc:.4f} wq_auc={mia['wq_auc']:.4f} "
                f"Δwq={std_wq - mia['wq_auc']:+.4f} dg={mia['dg']:.4f} [{elapsed:.1f}s]")

            del unlearned
            torch.cuda.empty_cache()

    # --- RUM baseline (GA and SCRUB only) ---
    for method in ['ga', 'scrub']:
        rum_key = f'{method}_rum'
        model_path = f'{MODELS_DIR}/{rum_key}/{dataset_name}_seed{seed}.pt'
        t0 = time.time()

        base_epochs = 15 if method == 'ga' else 10
        groups, epoch_map = compute_rum_epochs(forget_difficulty, base_epochs)

        # Run unlearning per group with different epoch counts
        unlearned = copy.deepcopy(original_model)
        for g in range(3):
            g_mask = groups == g
            if g_mask.sum() == 0:
                continue
            g_indices = [forget_indices[i] for i in range(len(forget_indices)) if g_mask[i]]
            g_loader = get_loader(train_ds, g_indices, batch_size=BATCH_SIZE, shuffle=True)
            g_epochs = epoch_map[g]

            if method == 'ga':
                unlearned = unlearn_ga(unlearned, g_loader, retain_loader, dataset_name,
                                      epochs=g_epochs)
            else:
                unlearned = unlearn_scrub(unlearned, g_loader, retain_loader, dataset_name,
                                         passes=g_epochs)

        elapsed = time.time() - t0
        torch.save(unlearned.state_dict(), model_path)

        retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval_ds, retain_indices, 512, False))
        test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, None, 512, False))
        forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)

        mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                                 train_eval_ds, test_ds, dataset_name,
                                 difficulty_train, difficulty_test,
                                 quintiles_train, quintiles_test)

        std_wq = all_results[method]['wq_auc']
        all_results[rum_key] = {
            'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
            'forget_acc': float(forget_acc),
            'aggregate_auc': mia['aggregate']['best_auc'],
            'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
            'per_quintile': {k: v['best_auc'] for k, v in mia.items() if k.startswith('q')},
            'delta_wq': std_wq - mia['wq_auc'],
            'time_sec': elapsed,
        }
        log(f"  {rum_key}: forget_acc={forget_acc:.4f} wq_auc={mia['wq_auc']:.4f} "
            f"Δwq={std_wq - mia['wq_auc']:+.4f} [{elapsed:.1f}s]")

        del unlearned
        torch.cuda.empty_cache()

    return all_results


def run_retrain_eval(dataset_name, seed, train_eval_ds, test_ds, splits,
                     difficulty_train, difficulty_test, quintiles_train, quintiles_test):
    """Evaluate retrain model (gold standard)."""
    model = get_model(dataset_name, DEVICE)
    model.load_state_dict(torch.load(f'exp/models/retrain/{dataset_name}_seed{seed}.pt',
                                     map_location=DEVICE, weights_only=True))

    forget_indices = splits['forget_indices']
    retain_indices = splits['retain_indices']

    # Matched test indices
    rng = np.random.RandomState(seed + 1000)
    n_test = len(test_ds)
    test_quintiles = quintiles_test[:n_test]
    member_quintiles = quintiles_train[forget_indices]
    test_indices_list = []
    for q in range(5):
        member_q_count = (member_quintiles == q).sum()
        nonmember_q_candidates = np.where(test_quintiles == q)[0]
        if len(nonmember_q_candidates) >= member_q_count:
            chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=False)
        else:
            chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=True)
        test_indices_list.append(chosen)
    test_indices = np.concatenate(test_indices_list)

    retain_acc, _ = evaluate_model(model, get_loader(train_eval_ds, retain_indices, 512, False))
    test_acc, _ = evaluate_model(model, get_loader(test_ds, None, 512, False))
    forget_acc, _ = evaluate_model(model, get_loader(train_eval_ds, forget_indices, 512, False))

    mia = run_mia_evaluation(model, forget_indices, test_indices,
                             train_eval_ds, test_ds, dataset_name,
                             difficulty_train, difficulty_test,
                             quintiles_train, quintiles_test)

    return {
        'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
        'forget_acc': float(forget_acc),
        'aggregate_auc': mia['aggregate']['best_auc'],
        'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
        'per_quintile': {k: v['best_auc'] for k, v in mia.items() if k.startswith('q')},
    }


# ============================================================
# ABLATION STUDIES
# ============================================================

def run_alpha_ablation(dataset_name, seed, original_model, train_ds, train_eval_ds,
                       test_ds, splits, difficulty_train, difficulty_test,
                       quintiles_train, quintiles_test):
    """Alpha sensitivity: run GA-DAU and SCRUB-DAU with alpha in {0.0, 0.5, 1.0, 2.0, 5.0}."""
    forget_indices = splits['forget_indices']
    retain_indices = splits['retain_indices']
    forget_difficulty = difficulty_train[forget_indices]

    indexed_forget_ds = IndexedSubset(train_ds, forget_indices)
    forget_loader_idx = DataLoader(indexed_forget_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True)
    retain_loader = get_loader(train_ds, retain_indices, batch_size=BATCH_SIZE, shuffle=True)
    forget_eval_loader = get_loader(train_eval_ds, forget_indices, batch_size=512, shuffle=False)

    # Test indices
    rng = np.random.RandomState(seed + 1000)
    n_test = len(test_ds)
    test_quintiles = quintiles_test[:n_test]
    member_quintiles = quintiles_train[forget_indices]
    test_indices_list = []
    for q in range(5):
        member_q_count = (member_quintiles == q).sum()
        nonmember_q_candidates = np.where(test_quintiles == q)[0]
        if len(nonmember_q_candidates) >= member_q_count:
            chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=False)
        else:
            chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=True)
        test_indices_list.append(chosen)
    test_indices = np.concatenate(test_indices_list)

    results = {}
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        dau_weights = compute_dau_weights(forget_difficulty, alpha=alpha)
        for method in ['ga', 'scrub']:
            key = f'{method}_alpha{alpha}'
            fn = UNLEARN_FNS[method]
            if alpha == 0.0:
                # Uniform weights = standard
                unlearned = fn(original_model, forget_loader_idx, retain_loader, dataset_name)
            else:
                unlearned = fn(original_model, forget_loader_idx, retain_loader, dataset_name,
                              sample_weights=dau_weights)

            retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval_ds, retain_indices, 512, False))
            test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, None, 512, False))
            forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)

            mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                                     train_eval_ds, test_ds, dataset_name,
                                     difficulty_train, difficulty_test,
                                     quintiles_train, quintiles_test)

            results[key] = {
                'alpha': alpha, 'method': method,
                'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
                'forget_acc': float(forget_acc),
                'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
                'aggregate_auc': mia['aggregate']['best_auc'],
            }
            log(f"    {key}: wq_auc={mia['wq_auc']:.4f} dg={mia['dg']:.4f} "
                f"forget_acc={forget_acc:.4f} retain_acc={retain_acc:.4f}")

            del unlearned
            torch.cuda.empty_cache()

    return results


def run_k_ablation(dataset_name, seed, original_model, train_ds, train_eval_ds,
                   test_ds, splits, quintiles_train, quintiles_test):
    """K ablation: test K=2, 4, 8 reference models."""
    forget_indices = splits['forget_indices']
    retain_indices = splits['retain_indices']

    results = {}
    for K in [2, 4, 8]:
        # Recompute difficulty with K ref models
        ref_losses = []
        for k in range(K):
            path = f'exp/results/ref_train_losses_{dataset_name}_ref{k}.npy'
            if os.path.exists(path):
                ref_losses.append(np.load(path))
            else:
                log(f"    Warning: ref model {k} not found for {dataset_name}")
                break
        if len(ref_losses) < K:
            log(f"    Skipping K={K} for {dataset_name} (only {len(ref_losses)} ref models)")
            continue

        difficulty_K = np.mean(ref_losses, axis=0)

        # K=8 difficulty as ground truth
        if K == 8:
            difficulty_8 = difficulty_K

        # Spearman correlation with K=8 (if available)
        if K < 8 and 'difficulty_8' in dir():
            rho, _ = stats.spearmanr(difficulty_K, difficulty_8)
        else:
            rho = 1.0

        # Quintile stability
        quintiles_K = np.zeros(len(difficulty_K), dtype=int)
        percs = np.percentile(difficulty_K, [20, 40, 60, 80])
        for i in range(4):
            quintiles_K[difficulty_K > percs[i]] = i + 1

        # Only compute on forget set
        forget_diff = difficulty_K[forget_indices]
        forget_quintiles = quintiles_K[forget_indices]

        # Run GA-DAU with these scores
        dau_weights = compute_dau_weights(forget_diff, alpha=1.0)
        indexed_forget_ds = IndexedSubset(train_ds, forget_indices)
        forget_loader_idx = DataLoader(indexed_forget_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=0, pin_memory=True)
        retain_loader = get_loader(train_ds, retain_indices, batch_size=BATCH_SIZE, shuffle=True)

        unlearned = unlearn_ga(original_model, forget_loader_idx, retain_loader, dataset_name,
                               sample_weights=dau_weights)

        # Test indices
        rng = np.random.RandomState(seed + 1000)
        n_test = len(test_ds)
        test_quintiles_K = np.zeros(len(quintiles_test), dtype=int)
        test_diff = np.mean([np.load(f'exp/results/ref_test_losses_{dataset_name}_ref{k}.npy')
                            for k in range(K)], axis=0)
        percs_test = np.percentile(test_diff, [20, 40, 60, 80])
        for i in range(4):
            test_quintiles_K[test_diff > percs_test[i]] = i + 1

        member_quintiles = quintiles_K[forget_indices]
        test_indices_list = []
        for q in range(5):
            member_q_count = (member_quintiles == q).sum()
            nonmember_q_candidates = np.where(test_quintiles_K[:n_test] == q)[0]
            if len(nonmember_q_candidates) >= member_q_count:
                chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=False)
            else:
                chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=True)
            test_indices_list.append(chosen)
        test_indices = np.concatenate(test_indices_list)

        mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                                 train_eval_ds, test_ds, dataset_name,
                                 difficulty_K, test_diff,
                                 quintiles_K, test_quintiles_K)

        results[f'K={K}'] = {
            'K': K, 'spearman_rho': float(rho),
            'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
            'aggregate_auc': mia['aggregate']['best_auc'],
        }
        log(f"    K={K}: rho={rho:.4f} wq_auc={mia['wq_auc']:.4f}")

        del unlearned
        torch.cuda.empty_cache()

    return results


def run_strata_ablation(dataset_name, seed, original_model, train_ds, train_eval_ds,
                        test_ds, splits, difficulty_train, difficulty_test):
    """Strata granularity ablation: terciles, quintiles, deciles."""
    forget_indices = splits['forget_indices']
    retain_indices = splits['retain_indices']

    # Run GA unlearning once
    forget_loader = get_loader(train_ds, forget_indices, batch_size=BATCH_SIZE, shuffle=True)
    retain_loader = get_loader(train_ds, retain_indices, batch_size=BATCH_SIZE, shuffle=True)
    unlearned = unlearn_ga(original_model, forget_loader, retain_loader, dataset_name)

    results = {}
    for n_strata, name in [(3, 'terciles'), (5, 'quintiles'), (10, 'deciles')]:
        # Compute quintiles with given granularity
        percs = np.percentile(difficulty_train, np.linspace(100/n_strata, 100*(1-1/n_strata), n_strata-1))
        train_strata = np.digitize(difficulty_train, percs)
        percs_test = np.percentile(difficulty_test, np.linspace(100/n_strata, 100*(1-1/n_strata), n_strata-1))
        test_strata = np.digitize(difficulty_test, percs_test)

        # Matched test indices
        rng = np.random.RandomState(seed + 1000)
        n_test = len(test_ds)
        member_strata = train_strata[forget_indices]
        test_indices_list = []
        for q in range(n_strata):
            member_q_count = (member_strata == q).sum()
            nonmember_q_candidates = np.where(test_strata[:n_test] == q)[0]
            if len(nonmember_q_candidates) >= member_q_count:
                chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=False)
            else:
                chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=True)
            test_indices_list.append(chosen)
        test_indices = np.concatenate(test_indices_list)

        # Evaluate
        forget_loader_eval = get_loader(train_eval_ds, forget_indices, batch_size=512, shuffle=False)
        test_loader_eval = get_loader(test_ds, test_indices, batch_size=512, shuffle=False)
        member_losses = compute_losses(unlearned, forget_loader_eval)
        nonmember_losses = compute_losses(unlearned, test_loader_eval)

        ref_train_losses = get_ref_losses(dataset_name, 4, 'train')
        ref_test_losses = get_ref_losses(dataset_name, 4, 'test')
        member_ref_list = [r[forget_indices] for r in ref_train_losses]
        nonmember_ref_list = [r[test_indices] for r in ref_test_losses]
        member_ref_mean = np.mean(member_ref_list, axis=0)
        nonmember_ref_mean = np.mean(nonmember_ref_list, axis=0)

        mia_res = stratified_mia(member_losses, nonmember_losses,
                                  member_ref_mean, nonmember_ref_mean,
                                  member_ref_list, nonmember_ref_list,
                                  member_strata, test_strata[test_indices],
                                  n_strata=n_strata)

        per_stratum_aucs = [mia_res[f'q{q+1}']['best_auc'] for q in range(n_strata)]
        results[name] = {
            'n_strata': n_strata,
            'worst_stratum_auc': float(max(per_stratum_aucs)),
            'best_stratum_auc': float(min(per_stratum_aucs)),
            'dg': float(max(per_stratum_aucs) - min(per_stratum_aucs)),
            'per_stratum': [float(a) for a in per_stratum_aucs],
        }
        log(f"    {name}: worst={max(per_stratum_aucs):.4f} dg={max(per_stratum_aucs)-min(per_stratum_aucs):.4f}")

    del unlearned
    torch.cuda.empty_cache()
    return results


def run_random_weight_ablation(dataset_name, seed, original_model, train_ds, train_eval_ds,
                               test_ds, splits, difficulty_train, difficulty_test,
                               quintiles_train, quintiles_test):
    """Random weight control: verify DAU benefit comes from difficulty-aware weighting."""
    forget_indices = splits['forget_indices']
    retain_indices = splits['retain_indices']
    forget_difficulty = difficulty_train[forget_indices]

    indexed_forget_ds = IndexedSubset(train_ds, forget_indices)
    forget_loader_idx = DataLoader(indexed_forget_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True)
    retain_loader = get_loader(train_ds, retain_indices, batch_size=BATCH_SIZE, shuffle=True)

    # Test indices
    rng = np.random.RandomState(seed + 1000)
    n_test = len(test_ds)
    member_quintiles = quintiles_train[forget_indices]
    test_indices_list = []
    for q in range(5):
        member_q_count = (member_quintiles == q).sum()
        nonmember_q_candidates = np.where(quintiles_test[:n_test] == q)[0]
        if len(nonmember_q_candidates) >= member_q_count:
            chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=False)
        else:
            chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=True)
        test_indices_list.append(chosen)
    test_indices = np.concatenate(test_indices_list)

    # Shuffled weights
    rng2 = np.random.RandomState(seed + 2000)
    shuffled_difficulty = forget_difficulty.copy()
    rng2.shuffle(shuffled_difficulty)
    random_weights = compute_dau_weights(shuffled_difficulty, alpha=1.0)

    unlearned = unlearn_ga(original_model, forget_loader_idx, retain_loader, dataset_name,
                           sample_weights=random_weights)

    forget_eval_loader = get_loader(train_eval_ds, forget_indices, batch_size=512, shuffle=False)
    retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval_ds, retain_indices, 512, False))
    test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, None, 512, False))
    forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)

    mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                             train_eval_ds, test_ds, dataset_name,
                             difficulty_train, difficulty_test,
                             quintiles_train, quintiles_test)

    result = {
        'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
        'forget_acc': float(forget_acc),
        'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
        'aggregate_auc': mia['aggregate']['best_auc'],
    }
    log(f"    random_weights: wq_auc={mia['wq_auc']:.4f} dg={mia['dg']:.4f}")

    del unlearned
    torch.cuda.empty_cache()
    return result


def run_forget_size_ablation(dataset_name, seed, original_model, train_ds, train_eval_ds,
                             test_ds, difficulty_train, difficulty_test,
                             quintiles_train, quintiles_test):
    """Forget set size ablation: 500, 1000, 2500."""
    results = {}
    for fsize in [500, 1000, 2500]:
        splits = create_splits(dataset_name, train_ds, seed, forget_size=fsize)
        forget_indices = splits['forget_indices']
        retain_indices = splits['retain_indices']

        forget_loader = get_loader(train_ds, forget_indices, batch_size=BATCH_SIZE, shuffle=True)
        retain_loader = get_loader(train_ds, retain_indices, batch_size=BATCH_SIZE, shuffle=True)

        indexed_forget_ds = IndexedSubset(train_ds, forget_indices)
        forget_loader_idx = DataLoader(indexed_forget_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=0, pin_memory=True)

        forget_difficulty = difficulty_train[forget_indices]
        dau_weights = compute_dau_weights(forget_difficulty, alpha=1.0)

        # Test indices
        rng = np.random.RandomState(seed + 1000)
        n_test = len(test_ds)
        member_quintiles = quintiles_train[forget_indices]
        test_indices_list = []
        for q in range(5):
            member_q_count = (member_quintiles == q).sum()
            nonmember_q_candidates = np.where(quintiles_test[:n_test] == q)[0]
            if len(nonmember_q_candidates) >= member_q_count:
                chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=False)
            else:
                chosen = rng.choice(nonmember_q_candidates, size=member_q_count, replace=True)
            test_indices_list.append(chosen)
        test_indices = np.concatenate(test_indices_list)

        fsize_results = {}
        for variant, weights in [('standard', None), ('dau', dau_weights)]:
            loader = forget_loader_idx if weights is not None else forget_loader
            unlearned = unlearn_ga(original_model, loader, retain_loader, dataset_name,
                                   sample_weights=weights)

            forget_eval = get_loader(train_eval_ds, forget_indices, 512, False)
            forget_acc, _ = evaluate_model(unlearned, forget_eval)
            retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval_ds, retain_indices, 512, False))

            mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                                     train_eval_ds, test_ds, dataset_name,
                                     difficulty_train, difficulty_test,
                                     quintiles_train, quintiles_test)

            fsize_results[variant] = {
                'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
                'forget_acc': float(forget_acc), 'retain_acc': float(retain_acc),
            }

            del unlearned
            torch.cuda.empty_cache()

        results[f'size_{fsize}'] = fsize_results
        log(f"    size={fsize}: std_wq={fsize_results['standard']['wq_auc']:.4f} "
            f"dau_wq={fsize_results['dau']['wq_auc']:.4f} "
            f"Δwq={fsize_results['standard']['wq_auc'] - fsize_results['dau']['wq_auc']:+.4f}")

    return results


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def run_statistical_tests(all_data):
    """Comprehensive statistical tests for all success criteria."""
    tests = {}

    for dataset_name in DATASETS:
        dataset_tests = {}

        # SC1: WQ-AUC > Aggregate AUC
        for method in UNLEARN_METHODS:
            wq_values = [all_data[dataset_name][seed][method]['wq_auc']
                        for seed in SEEDS]
            agg_values = [all_data[dataset_name][seed][method]['aggregate_auc']
                         for seed in SEEDS]
            diff = [w - a for w, a in zip(wq_values, agg_values)]
            if len(set(diff)) > 1:
                t_stat, p_val = stats.ttest_1samp(diff, 0)
                p_val_one = p_val / 2 if t_stat > 0 else 1.0
            else:
                t_stat, p_val_one = float('inf') if diff[0] > 0 else 0.0, 0.0 if diff[0] > 0 else 1.0
            dataset_tests[f'{method}_wq_vs_agg'] = {
                'mean_diff': float(np.mean(diff)),
                'std_diff': float(np.std(diff)),
                't_stat': float(t_stat),
                'p_value': float(p_val_one),
                'significant': bool(p_val_one < 0.05 / 15),  # Bonferroni
            }

        # SC2: DG significantly > 0
        for method in UNLEARN_METHODS:
            dg_values = [all_data[dataset_name][seed][method]['dg']
                        for seed in SEEDS]
            if len(set(dg_values)) > 1:
                t_stat, p_val = stats.ttest_1samp(dg_values, 0)
                p_val_one = p_val / 2 if t_stat > 0 else 1.0
            else:
                t_stat, p_val_one = float('inf') if dg_values[0] > 0 else 0.0, 0.0 if dg_values[0] > 0 else 1.0
            dataset_tests[f'{method}_dg_positive'] = {
                'mean_dg': float(np.mean(dg_values)),
                'std_dg': float(np.std(dg_values)),
                't_stat': float(t_stat),
                'p_value': float(p_val_one),
                'significant': bool(p_val_one < 0.05 / 15),
            }

        # SC3: DAU improves WQ-AUC
        for method in UNLEARN_METHODS:
            std_wq = [all_data[dataset_name][seed][method]['wq_auc'] for seed in SEEDS]
            dau_key = f'{method}_dau'
            if dau_key in all_data[dataset_name][SEEDS[0]]:
                dau_wq = [all_data[dataset_name][seed][dau_key]['wq_auc'] for seed in SEEDS]
                diff = [s - d for s, d in zip(std_wq, dau_wq)]  # positive = DAU better
                if len(set(diff)) > 1:
                    t_stat, p_val = stats.ttest_1samp(diff, 0)
                    p_val_one = p_val / 2 if t_stat > 0 else 1.0
                else:
                    t_stat, p_val_one = float('inf') if diff[0] > 0 else 0.0, 0.0 if diff[0] > 0 else 1.0
                dataset_tests[f'{method}_dau_improvement'] = {
                    'mean_delta_wq': float(np.mean(diff)),
                    'std_delta_wq': float(np.std(diff)),
                    't_stat': float(t_stat),
                    'p_value': float(p_val_one),
                    'significant': bool(p_val_one < 0.05 / 15),
                }

        tests[dataset_name] = dataset_tests

    return tests


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    log("=" * 60)
    log("DAU Experiment Pipeline v3 — Starting")
    log("=" * 60)

    # ---- Phase 1: Load data, models, difficulty scores ----
    log("\n=== Phase 1: Loading existing assets ===")
    datasets_cache = {}
    splits_cache = {}
    difficulty_cache = {}
    quintiles_cache = {}

    for ds in DATASETS:
        log(f"Loading {ds}...")
        train_ds, test_ds, train_eval_ds = load_dataset(ds)
        datasets_cache[ds] = (train_ds, test_ds, train_eval_ds)

        difficulty_cache[f'{ds}_train'] = np.load(f'exp/results/difficulty_{ds}_train.npy')
        difficulty_cache[f'{ds}_test'] = np.load(f'exp/results/difficulty_{ds}_test.npy')
        quintiles_cache[f'{ds}_train'] = np.load(f'exp/results/quintiles_{ds}_train.npy')
        quintiles_cache[f'{ds}_test'] = np.load(f'exp/results/quintiles_{ds}_test.npy')

        for seed in SEEDS:
            splits_cache[(ds, seed)] = create_splits(ds, train_ds, seed)

        log(f"  {ds}: train={len(train_ds)}, test={len(test_ds)}, "
            f"difficulty range=[{difficulty_cache[f'{ds}_train'].min():.3f}, "
            f"{difficulty_cache[f'{ds}_train'].max():.3f}]")

    # ---- Phase 2: Main experiments ----
    log("\n=== Phase 2: Unlearning + DAU + RUM ===")
    all_data = {ds: {} for ds in DATASETS}

    for ds in DATASETS:
        train_ds, test_ds, train_eval_ds = datasets_cache[ds]
        diff_train = difficulty_cache[f'{ds}_train']
        diff_test = difficulty_cache[f'{ds}_test']
        q_train = quintiles_cache[f'{ds}_train']
        q_test = quintiles_cache[f'{ds}_test']

        for seed in SEEDS:
            log(f"\n--- {ds} seed={seed} ---")
            set_seed(seed)
            splits = splits_cache[(ds, seed)]

            # Load original model
            original_model = get_model(ds, DEVICE)
            original_model.load_state_dict(
                torch.load(f'exp/models/original/{ds}_seed{seed}.pt',
                          map_location=DEVICE, weights_only=True))

            # Retrain eval
            log("  Evaluating retrain model...")
            retrain_result = run_retrain_eval(ds, seed, train_eval_ds, test_ds, splits,
                                              diff_train, diff_test, q_train, q_test)
            log(f"  retrain: agg_auc={retrain_result['aggregate_auc']:.4f} "
                f"wq_auc={retrain_result['wq_auc']:.4f}")

            # Unlearning + DAU + RUM
            log("  Running unlearning methods...")
            results = run_unlearning_for_config(
                ds, seed, original_model, train_ds, train_eval_ds, test_ds, splits,
                diff_train, diff_test, q_train, q_test)

            results['retrain'] = retrain_result
            all_data[ds][seed] = results

            # Save intermediate results
            with open(f'{RESULTS_DIR}/intermediate_{ds}_seed{seed}.json', 'w') as f:
                json.dump(results, f, indent=2)

    # ---- Phase 3: Ablation studies ----
    log("\n=== Phase 3: Ablation studies ===")
    ablation_results = {}

    # Alpha ablation: CIFAR-10 and CIFAR-100, all seeds
    log("\n--- Alpha ablation ---")
    alpha_results = {}
    for ds in ['cifar10', 'cifar100']:
        train_ds, test_ds, train_eval_ds = datasets_cache[ds]
        diff_train = difficulty_cache[f'{ds}_train']
        diff_test = difficulty_cache[f'{ds}_test']
        q_train = quintiles_cache[f'{ds}_train']
        q_test = quintiles_cache[f'{ds}_test']
        alpha_results[ds] = {}
        for seed in SEEDS:
            set_seed(seed)
            splits = splits_cache[(ds, seed)]
            original_model = get_model(ds, DEVICE)
            original_model.load_state_dict(
                torch.load(f'exp/models/original/{ds}_seed{seed}.pt',
                          map_location=DEVICE, weights_only=True))
            log(f"  Alpha ablation: {ds} seed={seed}")
            alpha_results[ds][seed] = run_alpha_ablation(
                ds, seed, original_model, train_ds, train_eval_ds, test_ds, splits,
                diff_train, diff_test, q_train, q_test)
    ablation_results['alpha'] = alpha_results

    # K ablation: CIFAR-10 seed 42
    log("\n--- K ablation ---")
    ds = 'cifar10'
    seed = 42
    set_seed(seed)
    train_ds, test_ds, train_eval_ds = datasets_cache[ds]
    splits = splits_cache[(ds, seed)]
    original_model = get_model(ds, DEVICE)
    original_model.load_state_dict(
        torch.load(f'exp/models/original/{ds}_seed{seed}.pt',
                  map_location=DEVICE, weights_only=True))
    ablation_results['K'] = run_k_ablation(
        ds, seed, original_model, train_ds, train_eval_ds, test_ds, splits,
        quintiles_cache[f'{ds}_train'], quintiles_cache[f'{ds}_test'])

    # Strata ablation: CIFAR-10 seed 42
    log("\n--- Strata ablation ---")
    ablation_results['strata'] = run_strata_ablation(
        ds, seed, original_model, train_ds, train_eval_ds, test_ds, splits,
        difficulty_cache[f'{ds}_train'], difficulty_cache[f'{ds}_test'])

    # Random weight control: CIFAR-10 all seeds
    log("\n--- Random weight control ---")
    rw_results = {}
    for seed in SEEDS:
        set_seed(seed)
        splits = splits_cache[('cifar10', seed)]
        original_model = get_model('cifar10', DEVICE)
        original_model.load_state_dict(
            torch.load(f'exp/models/original/cifar10_seed{seed}.pt',
                      map_location=DEVICE, weights_only=True))
        rw_results[seed] = run_random_weight_ablation(
            'cifar10', seed, original_model, train_ds, train_eval_ds, test_ds, splits,
            difficulty_cache['cifar10_train'], difficulty_cache['cifar10_test'],
            quintiles_cache['cifar10_train'], quintiles_cache['cifar10_test'])
    ablation_results['random_weights'] = rw_results

    # Forget size ablation: CIFAR-10 seed 42
    log("\n--- Forget size ablation ---")
    set_seed(42)
    original_model = get_model('cifar10', DEVICE)
    original_model.load_state_dict(
        torch.load('exp/models/original/cifar10_seed42.pt',
                  map_location=DEVICE, weights_only=True))
    ablation_results['forget_size'] = run_forget_size_ablation(
        'cifar10', 42, original_model, train_ds, train_eval_ds, test_ds,
        difficulty_cache['cifar10_train'], difficulty_cache['cifar10_test'],
        quintiles_cache['cifar10_train'], quintiles_cache['cifar10_test'])

    with open(f'{RESULTS_DIR}/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # ---- Phase 4: Statistical tests ----
    log("\n=== Phase 4: Statistical analysis ===")
    stat_tests = run_statistical_tests(all_data)
    with open(f'{RESULTS_DIR}/statistical_tests.json', 'w') as f:
        json.dump(stat_tests, f, indent=2)

    # ---- Phase 5: Aggregate results ----
    log("\n=== Phase 5: Aggregating final results ===")
    final_results = {
        'experiment_config': {
            'datasets': DATASETS,
            'seeds': SEEDS,
            'forget_size': FORGET_SIZE,
            'n_reference_models': 4,
            'unlearning_methods': ['Fine-Tune', 'Gradient Ascent', 'Random Labels', 'SCRUB', 'NegGrad+KD'],
            'defense_methods': ['DAU (alpha=1.0)', 'RUM (3 groups)'],
            'pipeline_version': 'v3',
            'total_runtime_minutes': round((time.time() - start_time) / 60, 1),
        },
        'main_results': {},
        'ablation_results': ablation_results,
        'statistical_tests': stat_tests,
    }

    # Aggregate across seeds
    for ds in DATASETS:
        ds_results = {}
        all_methods = set()
        for seed in SEEDS:
            all_methods.update(all_data[ds][seed].keys())

        for method in sorted(all_methods):
            values = [all_data[ds][seed].get(method) for seed in SEEDS if method in all_data[ds][seed]]
            if not values:
                continue
            agg = {}
            for key in ['aggregate_auc', 'wq_auc', 'dg', 'retain_acc', 'test_acc', 'forget_acc']:
                vals = [v[key] for v in values if key in v]
                if vals:
                    agg[key] = {'mean': round(float(np.mean(vals)), 4),
                                'std': round(float(np.std(vals)), 4)}
            if 'delta_wq' in values[0]:
                vals = [v['delta_wq'] for v in values]
                agg['delta_wq'] = {'mean': round(float(np.mean(vals)), 4),
                                   'std': round(float(np.std(vals)), 4)}
            # Overestimation
            if 'aggregate_auc' in agg and 'wq_auc' in agg:
                overest = [v['wq_auc'] - v['aggregate_auc'] for v in values
                          if 'wq_auc' in v and 'aggregate_auc' in v]
                if overest:
                    agg['overestimation'] = {'mean': round(float(np.mean(overest)), 4),
                                             'std': round(float(np.std(overest)), 4)}
            # Per-quintile
            if 'per_quintile' in values[0]:
                pq = {}
                for qk in values[0]['per_quintile']:
                    qvals = [v['per_quintile'][qk] for v in values if qk in v.get('per_quintile', {})]
                    if qvals:
                        pq[qk] = {'mean': round(float(np.mean(qvals)), 4),
                                  'std': round(float(np.std(qvals)), 4)}
                agg['per_quintile'] = pq

            ds_results[method] = agg

        final_results['main_results'][ds] = ds_results

    # Save raw per-seed data
    raw_data = {}
    for ds in DATASETS:
        raw_data[ds] = {}
        for seed in SEEDS:
            raw_data[ds][str(seed)] = all_data[ds][seed]
    final_results['raw_per_seed'] = raw_data

    # Save to workspace root
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    with open(f'{RESULTS_DIR}/results_v3.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    elapsed = (time.time() - start_time) / 60
    log(f"\n{'='*60}")
    log(f"Pipeline complete! Total time: {elapsed:.1f} minutes")
    log(f"Results saved to results.json")
    log(f"{'='*60}")

    # Print summary
    log("\n=== SUMMARY ===")
    for ds in DATASETS:
        log(f"\n{ds}:")
        for method in ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad',
                       'ga_dau', 'scrub_dau', 'ga_rum', 'scrub_rum']:
            if method in final_results['main_results'][ds]:
                r = final_results['main_results'][ds][method]
                parts = [f"  {method:12s}:"]
                if 'forget_acc' in r:
                    parts.append(f"FA={r['forget_acc']['mean']:.3f}")
                parts.append(f"AggAUC={r['aggregate_auc']['mean']:.3f}")
                parts.append(f"WQ-AUC={r['wq_auc']['mean']:.3f}")
                parts.append(f"DG={r['dg']['mean']:.3f}")
                if 'delta_wq' in r:
                    parts.append(f"Δwq={r['delta_wq']['mean']:+.3f}")
                log(' '.join(parts))


if __name__ == '__main__':
    main()
