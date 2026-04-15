#!/usr/bin/env python3
"""
Fix NegGrad collapse: re-run neggrad and neggrad_dau with forget_w=1.0 instead of 2.0.
Then patch results.json with corrected values.

Run AFTER the main pipeline (run_v3.py) finishes.
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

from exp.shared.config import *
from exp.shared.models import get_model
from exp.shared.utils import set_seed, load_dataset, create_splits, get_loader, evaluate_model
from exp.shared.mia import compute_losses, stratified_mia

RESULTS_DIR = 'exp/results_v3'
MODELS_DIR = 'exp/models_v3'


def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, y, idx


def unlearn_neggrad_fixed(model, forget_loader, retain_loader, dataset,
                          sample_weights=None, device=DEVICE):
    """NegGrad+KD with forget_w=1.0 (not 2.0) and interleaved retain."""
    # Corrected hyperparams
    PARAMS = {
        'cifar10':    {'lr': 0.005, 'epochs': 15, 'forget_w': 1.0},
        'cifar100':   {'lr': 0.005, 'epochs': 15, 'forget_w': 1.0},
        'purchase100': {'lr': 0.005, 'epochs': 15, 'forget_w': 1.0},
    }
    p = PARAMS[dataset]
    epochs = p['epochs']
    lr = p['lr']
    forget_w = p['forget_w']

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


def compute_dau_weights(difficulty_scores, alpha=1.0):
    d_mean = difficulty_scores.mean()
    d_std = difficulty_scores.std() + 1e-8
    weights = 1.0 + alpha * (difficulty_scores - d_mean) / d_std
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def get_ref_losses(dataset_name, n_ref=4, split='train'):
    losses = []
    for k in range(n_ref):
        path = f'exp/results/ref_{split}_losses_{dataset_name}_ref{k}.npy'
        losses.append(np.load(path))
    return losses


def run_mia_evaluation(model, forget_indices, test_indices, train_eval_ds, test_ds,
                       dataset_name, difficulty_train, difficulty_test,
                       quintiles_train, quintiles_test, n_ref=4, device=DEVICE):
    forget_loader = get_loader(train_eval_ds, forget_indices, batch_size=512, shuffle=False)
    test_loader = get_loader(test_ds, test_indices, batch_size=512, shuffle=False)

    member_losses = compute_losses(model, forget_loader, device)
    nonmember_losses = compute_losses(model, test_loader, device)

    ref_train_losses = get_ref_losses(dataset_name, n_ref, 'train')
    ref_test_losses = get_ref_losses(dataset_name, n_ref, 'test')

    member_ref_list = [r[forget_indices] for r in ref_train_losses]
    nonmember_ref_list = [r[test_indices] for r in ref_test_losses]
    member_ref_mean = np.mean([r[forget_indices] for r in ref_train_losses], axis=0)
    nonmember_ref_mean = np.mean([r[test_indices] for r in ref_test_losses], axis=0)

    member_quintiles = quintiles_train[forget_indices]
    nonmember_quintiles = quintiles_test[test_indices]

    results = stratified_mia(
        member_losses, nonmember_losses,
        member_ref_mean, nonmember_ref_mean,
        member_ref_list, nonmember_ref_list,
        member_quintiles, nonmember_quintiles,
        n_strata=5
    )
    return results


def main():
    start_time = time.time()
    log("NegGrad fix: re-running with forget_w=1.0")

    # Load all datasets and caches
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

    # Re-run neggrad and neggrad_dau for all configs
    neggrad_results = {}
    for ds in DATASETS:
        train_ds, test_ds, train_eval_ds = datasets_cache[ds]
        diff_train = difficulty_cache[f'{ds}_train']
        diff_test = difficulty_cache[f'{ds}_test']
        q_train = quintiles_cache[f'{ds}_train']
        q_test = quintiles_cache[f'{ds}_test']
        neggrad_results[ds] = {}

        for seed in SEEDS:
            log(f"\n--- {ds} seed={seed} ---")
            set_seed(seed)
            splits = splits_cache[(ds, seed)]
            forget_indices = splits['forget_indices']
            retain_indices = splits['retain_indices']
            num_classes = 10 if ds == 'cifar10' else 100

            original_model = get_model(ds, DEVICE)
            original_model.load_state_dict(
                torch.load(f'exp/models/original/{ds}_seed{seed}.pt',
                          map_location=DEVICE, weights_only=True))

            retain_loader = get_loader(train_ds, retain_indices, batch_size=BATCH_SIZE, shuffle=True)
            forget_loader_std = get_loader(train_ds, forget_indices, batch_size=BATCH_SIZE, shuffle=True)
            indexed_forget_ds = IndexedSubset(train_ds, forget_indices)
            forget_loader_idx = DataLoader(indexed_forget_ds, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=0, pin_memory=True)
            forget_eval_loader = get_loader(train_eval_ds, forget_indices, batch_size=512, shuffle=False)

            # Test indices (same RNG as main pipeline)
            rng = np.random.RandomState(seed + 1000)
            n_test = len(test_ds)
            test_quintiles = q_test[:n_test]
            member_quintiles = q_train[forget_indices]
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

            forget_difficulty = diff_train[forget_indices]
            seed_results = {}

            # --- Standard neggrad ---
            t0 = time.time()
            unlearned = unlearn_neggrad_fixed(original_model, forget_loader_std, retain_loader, ds)
            elapsed = time.time() - t0

            retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval_ds, retain_indices, 512, False))
            test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, None, 512, False))
            forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)

            mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                                     train_eval_ds, test_ds, ds,
                                     diff_train, diff_test, q_train, q_test)

            seed_results['neggrad'] = {
                'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
                'forget_acc': float(forget_acc),
                'aggregate_auc': mia['aggregate']['best_auc'],
                'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
                'per_quintile': {k: v['best_auc'] for k, v in mia.items() if k.startswith('q')},
                'time_sec': elapsed,
            }
            log(f"  neggrad: FA={forget_acc:.4f} RA={retain_acc:.4f} TA={test_acc:.4f} "
                f"agg={mia['aggregate']['best_auc']:.4f} wq={mia['wq_auc']:.4f} dg={mia['dg']:.4f}")

            torch.save(unlearned.state_dict(), f'{MODELS_DIR}/neggrad/{ds}_seed{seed}.pt')
            base_wq = mia['wq_auc']
            del unlearned
            torch.cuda.empty_cache()

            # --- NegGrad + DAU ---
            dau_weights = compute_dau_weights(forget_difficulty, alpha=1.0)
            t0 = time.time()
            unlearned = unlearn_neggrad_fixed(original_model, forget_loader_idx, retain_loader, ds,
                                              sample_weights=dau_weights)
            elapsed = time.time() - t0

            retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval_ds, retain_indices, 512, False))
            test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, None, 512, False))
            forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)

            mia = run_mia_evaluation(unlearned, forget_indices, test_indices,
                                     train_eval_ds, test_ds, ds,
                                     diff_train, diff_test, q_train, q_test)

            delta_wq = base_wq - mia['wq_auc']
            seed_results['neggrad_dau'] = {
                'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
                'forget_acc': float(forget_acc),
                'aggregate_auc': mia['aggregate']['best_auc'],
                'wq_auc': mia['wq_auc'], 'dg': mia['dg'],
                'per_quintile': {k: v['best_auc'] for k, v in mia.items() if k.startswith('q')},
                'delta_wq': delta_wq,
                'time_sec': elapsed,
            }
            log(f"  neggrad_dau: FA={forget_acc:.4f} wq={mia['wq_auc']:.4f} "
                f"Δwq={delta_wq:+.4f} dg={mia['dg']:.4f}")

            torch.save(unlearned.state_dict(), f'{MODELS_DIR}/neggrad_dau/{ds}_seed{seed}.pt')
            del unlearned
            torch.cuda.empty_cache()

            neggrad_results[ds][seed] = seed_results

    # Save standalone results
    with open(f'{RESULTS_DIR}/neggrad_fix_results.json', 'w') as f:
        json.dump(neggrad_results, f, indent=2, default=str)

    # Patch main results if they exist
    results_path = 'results.json'
    if os.path.exists(results_path):
        log("\nPatching results.json...")
        with open(results_path) as f:
            results = json.load(f)

        for ds in DATASETS:
            for seed in SEEDS:
                seed_key = str(seed)
                if ds in results.get('raw_per_seed', {}) and seed_key in results['raw_per_seed'][ds]:
                    for method in ['neggrad', 'neggrad_dau']:
                        if method in neggrad_results[ds].get(seed, {}):
                            results['raw_per_seed'][ds][seed_key][method] = neggrad_results[ds][seed][method]
                            log(f"  Patched {ds}/{seed}/{method}")

        # Re-aggregate
        for ds in DATASETS:
            for method in ['neggrad', 'neggrad_dau']:
                values = []
                for seed in SEEDS:
                    seed_key = str(seed)
                    if seed_key in results.get('raw_per_seed', {}).get(ds, {}):
                        v = results['raw_per_seed'][ds][seed_key].get(method)
                        if v:
                            values.append(v)
                if values:
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
                    if 'aggregate_auc' in agg and 'wq_auc' in agg:
                        overest = [v['wq_auc'] - v['aggregate_auc'] for v in values]
                        agg['overestimation'] = {'mean': round(float(np.mean(overest)), 4),
                                                 'std': round(float(np.std(overest)), 4)}
                    if 'per_quintile' in values[0]:
                        pq = {}
                        for qk in values[0]['per_quintile']:
                            qvals = [v['per_quintile'][qk] for v in values if qk in v.get('per_quintile', {})]
                            if qvals:
                                pq[qk] = {'mean': round(float(np.mean(qvals)), 4),
                                           'std': round(float(np.std(qvals)), 4)}
                        agg['per_quintile'] = pq
                    results['main_results'][ds][method] = agg
                    log(f"  Re-aggregated {ds}/{method}: wq_auc={agg['wq_auc']['mean']:.4f}")

        # Re-run statistical tests with patched data
        log("  Re-running statistical tests...")
        from scipy import stats as scipy_stats
        # Rebuild all_data from raw_per_seed
        all_data = {}
        for ds in DATASETS:
            all_data[ds] = {}
            for seed in SEEDS:
                seed_key = str(seed)
                if seed_key in results['raw_per_seed'].get(ds, {}):
                    all_data[ds][seed] = results['raw_per_seed'][ds][seed_key]

        # Simple paired t-tests for DAU improvement
        stat_tests = results.get('statistical_tests', {})
        for ds in DATASETS:
            for base in ['ga', 'scrub', 'neggrad']:
                dau_key = f'{base}_dau'
                base_vals = [all_data[ds][s][base]['wq_auc'] for s in SEEDS
                            if base in all_data[ds].get(s, {})]
                dau_vals = [all_data[ds][s][dau_key]['wq_auc'] for s in SEEDS
                           if dau_key in all_data[ds].get(s, {})]
                if len(base_vals) == len(dau_vals) == 3:
                    # delta = base - dau (positive = DAU improved = lower wq_auc)
                    deltas = [b - d for b, d in zip(base_vals, dau_vals)]
                    t_stat, p_val = scipy_stats.ttest_1samp(deltas, 0)
                    key = f'{ds}_{base}_dau_wq_improvement'
                    stat_tests[key] = {
                        'base_wq': [round(v, 4) for v in base_vals],
                        'dau_wq': [round(v, 4) for v in dau_vals],
                        'deltas': [round(d, 4) for d in deltas],
                        't_stat': round(float(t_stat), 4),
                        'p_value': round(float(p_val), 4),
                        'significant_005': p_val < 0.05,
                    }
        results['statistical_tests'] = stat_tests

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        with open(f'{RESULTS_DIR}/results_v3.json', 'w') as f:
            json.dump(results, f, indent=2)
        log("  Results patched successfully!")
    else:
        log(f"WARNING: {results_path} not found. Save standalone results only.")

    elapsed = (time.time() - start_time) / 60
    log(f"\nNegGrad fix complete! Time: {elapsed:.1f} minutes")


if __name__ == '__main__':
    main()
