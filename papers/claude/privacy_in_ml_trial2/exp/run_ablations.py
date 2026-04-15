#!/usr/bin/env python3
"""Ablation experiments for DAU.

Ablation 1: Number of reference models K (2, 4, 8)
Ablation 2: DAU strength alpha (0.5, 2.0, 5.0) — 0.0 and 1.0 already in baselines
Ablation 3: Stratification granularity (terciles, quintiles, deciles)
Ablation 4: Forget set size (500, 1000, 2500)
Ablation 5: Random-weight control
"""
import os
import sys
import json
import time
import copy
import gc
import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.shared.config import *
from exp.shared.models import get_model
from exp.shared.utils import set_seed, load_dataset, create_splits, train_model, evaluate_model, get_loader
from exp.shared.unlearning import run_unlearning
from exp.shared.mia import compute_losses, run_all_attacks, stratified_mia
from exp.shared.dau import compute_dau_weights

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def log(msg):
    t = time.strftime('%H:%M:%S')
    print(f"[{t}] {msg}", flush=True)

def get_num_classes(dataset):
    return {'cifar10': 10, 'cifar100': 100, 'purchase100': 100}[dataset]


def load_ref_losses(ds, k_indices):
    """Load reference model losses for given indices."""
    train_losses = [np.load(f'exp/results/ref_train_losses_{ds}_ref{k}.npy') for k in k_indices]
    test_losses = [np.load(f'exp/results/ref_test_losses_{ds}_ref{k}.npy') for k in k_indices]
    return train_losses, test_losses


# ============================================================
# Ablation 1: K (number of reference models)
# ============================================================
def ablation_K():
    log("=== Ablation 1: Number of Reference Models K ===")
    ds = 'cifar10'
    seed = 42
    results = {}

    # Train 4 extra ref models for K=8
    train_ds, test_ds, train_eval = load_dataset(ds)
    non_ref = list(range(REF_POOL_SIZE, len(train_ds)))

    for k, rseed in enumerate(REF_SEEDS_EXTRA):
        model_path = f'exp/models/reference/{ds}_ref{k+4}.pt'
        if os.path.exists(model_path):
            log(f"  Extra ref model {k+4} exists.")
            continue
        log(f"  Training extra ref model {k+4} for {ds}...")
        set_seed(rseed)
        model = get_model(ds)
        rng = np.random.RandomState(rseed)
        subset_idx = rng.choice(non_ref, size=int(0.8 * len(non_ref)), replace=False).tolist()
        loader = get_loader(train_ds, indices=subset_idx)
        model = train_model(model, loader, ds, verbose=False)
        torch.save(model.state_dict(), model_path)

        # Compute and save losses
        train_eval_loader = get_loader(train_eval, shuffle=False)
        test_loader = get_loader(test_ds, shuffle=False)
        _, train_losses = evaluate_model(model, train_eval_loader)
        _, test_losses_arr = evaluate_model(model, test_loader)
        np.save(f'exp/results/ref_train_losses_{ds}_ref{k+4}.npy', train_losses)
        np.save(f'exp/results/ref_test_losses_{ds}_ref{k+4}.npy', test_losses_arr)
        del model
        torch.cuda.empty_cache()

    # Compare K=2, 4, 8
    for K in [2, 4, 8]:
        k_indices = list(range(K))
        train_losses_list, test_losses_list = load_ref_losses(ds, k_indices)

        diff_K = np.mean(train_losses_list, axis=0)

        # Compare to K=8 baseline
        if K < 8:
            train_losses_8, _ = load_ref_losses(ds, list(range(8)))
            diff_8 = np.mean(train_losses_8, axis=0)
            from scipy.stats import spearmanr
            rho, _ = spearmanr(diff_K, diff_8)
        else:
            rho = 1.0

        # Quintile stability vs K=8
        percentiles_K = np.percentile(diff_K, [20, 40, 60, 80])
        quint_K = np.digitize(diff_K, percentiles_K)

        if K == 8:
            quint_8 = quint_K
        else:
            diff_8_full = np.mean(load_ref_losses(ds, list(range(8)))[0], axis=0)
            p8 = np.percentile(diff_8_full, [20, 40, 60, 80])
            quint_8 = np.digitize(diff_8_full, p8)

        stability = (quint_K == quint_8).mean()

        # Run stratified MIA on GA and SCRUB unlearned models
        splits = create_splits(ds, train_ds, seed)
        forget_idx = splits['forget_indices']
        rng = np.random.RandomState(seed)
        nm_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False).tolist()

        forget_quintiles = quint_K[forget_idx]
        test_diff_K = np.mean(test_losses_list, axis=0)
        nm_quint = np.digitize(test_diff_K[nm_idx], percentiles_K)

        forget_ref_losses = [tl[forget_idx] for tl in train_losses_list]
        nm_ref_losses = [tl[nm_idx] for tl in test_losses_list]
        forget_ref_mean = np.mean(forget_ref_losses, axis=0)
        nm_ref_mean = np.mean(nm_ref_losses, axis=0)

        forget_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)
        nm_loader = get_loader(test_ds, indices=nm_idx, shuffle=False)

        method_results = {}
        for method in ['ga', 'scrub']:
            model_path = f'exp/models/unlearned/{method}/{ds}_seed{seed}.pt'
            if not os.path.exists(model_path):
                continue
            model = get_model(ds)
            model.load_state_dict(torch.load(model_path, weights_only=True))

            m_losses = compute_losses(model, forget_loader)
            nm_losses = compute_losses(model, nm_loader)

            strat = stratified_mia(m_losses, nm_losses, forget_ref_mean, nm_ref_mean,
                                   forget_ref_losses, nm_ref_losses,
                                   forget_quintiles, nm_quint)
            method_results[method] = {
                'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                'agg_auc': strat['aggregate']['best_auc'],
            }
            del model
            torch.cuda.empty_cache()

        results[f'K={K}'] = {
            'spearman_rho': float(rho),
            'quintile_stability': float(stability),
            'methods': method_results,
        }
        log(f"  K={K}: rho={rho:.4f}, stability={stability:.4f}")

    save_json(results, 'exp/results/ablation_K.json')
    log("Ablation 1 complete.")


# ============================================================
# Ablation 2: Alpha sensitivity
# ============================================================
def ablation_alpha():
    log("=== Ablation 2: Alpha Sensitivity ===")
    results = []
    alphas = [0.5, 2.0, 5.0]  # 0.0=baseline, 1.0=default already exist

    for ds in ['cifar10', 'cifar100']:
        train_ds, test_ds, train_eval = load_dataset(ds)
        diff_train = np.load(f'exp/results/difficulty_{ds}_train.npy')
        diff_test = np.load(f'exp/results/difficulty_{ds}_test.npy')
        train_quint = np.load(f'exp/results/quintiles_{ds}_train.npy')
        test_quint = np.load(f'exp/results/quintiles_{ds}_test.npy')
        nc = get_num_classes(ds)

        ref_train_losses = [np.load(f'exp/results/ref_train_losses_{ds}_ref{k}.npy') for k in range(4)]
        ref_test_losses = [np.load(f'exp/results/ref_test_losses_{ds}_ref{k}.npy') for k in range(4)]

        for seed in SEEDS:
            splits = create_splits(ds, train_ds, seed)
            forget_idx = splits['forget_indices']
            retain_idx = splits['retain_indices']

            orig_model = get_model(ds)
            orig_model.load_state_dict(torch.load(
                f'exp/models/original/{ds}_seed{seed}.pt', weights_only=True))

            forget_diff = diff_train[forget_idx]
            forget_loader = get_loader(train_ds, indices=forget_idx, shuffle=False)
            retain_loader = get_loader(train_ds, indices=retain_idx, shuffle=True)

            rng = np.random.RandomState(seed)
            nm_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False).tolist()
            nm_loader = get_loader(test_ds, indices=nm_idx, shuffle=False)
            forget_eval_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)

            forget_quintiles = train_quint[forget_idx]
            nm_quintiles = test_quint[nm_idx]
            forget_ref = [rl[forget_idx] for rl in ref_train_losses]
            nm_ref = [rl[nm_idx] for rl in ref_test_losses]
            forget_ref_mean = np.mean(forget_ref, axis=0)
            nm_ref_mean = np.mean(nm_ref, axis=0)

            for method in ['ga', 'scrub']:
                for alpha in alphas:
                    log(f"  Alpha ablation: {method}/{ds}_s{seed}, alpha={alpha}")
                    dau_weights = compute_dau_weights(forget_diff, alpha=alpha)
                    set_seed(seed)
                    unlearned = run_unlearning(orig_model, forget_loader, retain_loader,
                                              method, ds, sample_weights=dau_weights, num_classes=nc)

                    # Evaluate
                    test_loader = get_loader(test_ds, shuffle=False)
                    test_acc, _ = evaluate_model(unlearned, test_loader)
                    retain_eval = get_loader(train_eval, indices=retain_idx, shuffle=False)
                    retain_acc, _ = evaluate_model(unlearned, retain_eval)

                    m_losses = compute_losses(unlearned, forget_eval_loader)
                    nm_losses = compute_losses(unlearned, nm_loader)

                    strat = stratified_mia(m_losses, nm_losses, forget_ref_mean, nm_ref_mean,
                                           forget_ref, nm_ref, forget_quintiles, nm_quintiles)

                    results.append({
                        'method': method, 'dataset': ds, 'seed': seed, 'alpha': alpha,
                        'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                        'agg_auc': strat['aggregate']['best_auc'],
                        'test_acc': float(test_acc), 'retain_acc': float(retain_acc),
                    })

                    del unlearned
                    torch.cuda.empty_cache()

            del orig_model
            torch.cuda.empty_cache()
            gc.collect()

    save_json(results, 'exp/results/ablation_alpha.json')
    log("Ablation 2 complete.")


# ============================================================
# Ablation 3: Stratification granularity
# ============================================================
def ablation_strata():
    log("=== Ablation 3: Stratification Granularity ===")
    ds = 'cifar10'
    seed = 42
    results = {}

    train_ds, test_ds, train_eval = load_dataset(ds)
    diff_train = np.load(f'exp/results/difficulty_{ds}_train.npy')
    diff_test = np.load(f'exp/results/difficulty_{ds}_test.npy')
    ref_train_losses = [np.load(f'exp/results/ref_train_losses_{ds}_ref{k}.npy') for k in range(4)]
    ref_test_losses = [np.load(f'exp/results/ref_test_losses_{ds}_ref{k}.npy') for k in range(4)]

    splits = create_splits(ds, train_ds, seed)
    forget_idx = splits['forget_indices']
    rng = np.random.RandomState(seed)
    nm_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False).tolist()

    forget_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)
    nm_loader = get_loader(test_ds, indices=nm_idx, shuffle=False)

    for n_strata, name in [(3, 'terciles'), (5, 'quintiles'), (10, 'deciles')]:
        pcts = [100/n_strata * i for i in range(1, n_strata)]
        train_pcts = np.percentile(diff_train, pcts)
        f_quint = np.digitize(diff_train[forget_idx], train_pcts)
        nm_quint = np.digitize(diff_test[nm_idx], train_pcts)

        forget_ref = [rl[forget_idx] for rl in ref_train_losses]
        nm_ref = [rl[nm_idx] for rl in ref_test_losses]
        forget_ref_mean = np.mean(forget_ref, axis=0)
        nm_ref_mean = np.mean(nm_ref, axis=0)

        method_results = {}
        for method in UNLEARN_METHODS + ['retrain']:
            if method == 'retrain':
                model_path = f'exp/models/retrain/{ds}_seed{seed}.pt'
            else:
                model_path = f'exp/models/unlearned/{method}/{ds}_seed{seed}.pt'
            if not os.path.exists(model_path):
                continue

            model = get_model(ds)
            model.load_state_dict(torch.load(model_path, weights_only=True))

            m_losses = compute_losses(model, forget_loader)
            nm_losses = compute_losses(model, nm_loader)

            strat = stratified_mia(m_losses, nm_losses, forget_ref_mean, nm_ref_mean,
                                   forget_ref, nm_ref, f_quint, nm_quint, n_strata=n_strata)
            method_results[method] = {
                'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                'max_spread': strat['max_spread'],
                'per_stratum': {k: v for k, v in strat.items()
                               if k.startswith('q') and isinstance(v, dict)},
            }
            del model
            torch.cuda.empty_cache()

        results[name] = {'n_strata': n_strata, 'methods': method_results}
        log(f"  {name}: completed")

    save_json(results, 'exp/results/ablation_strata.json')
    log("Ablation 3 complete.")


# ============================================================
# Ablation 4: Forget set size
# ============================================================
def ablation_forget_size():
    log("=== Ablation 4: Forget Set Size ===")
    results = []
    seed = 42

    for ds in ['cifar10', 'cifar100']:
        train_ds, test_ds, train_eval = load_dataset(ds)
        diff_train = np.load(f'exp/results/difficulty_{ds}_train.npy')
        diff_test = np.load(f'exp/results/difficulty_{ds}_test.npy')
        train_quint = np.load(f'exp/results/quintiles_{ds}_train.npy')
        test_quint = np.load(f'exp/results/quintiles_{ds}_test.npy')
        nc = get_num_classes(ds)

        ref_train_losses = [np.load(f'exp/results/ref_train_losses_{ds}_ref{k}.npy') for k in range(4)]
        ref_test_losses = [np.load(f'exp/results/ref_test_losses_{ds}_ref{k}.npy') for k in range(4)]

        for fsize in [500, 2500]:  # 1000 already done
            log(f"  Forget size ablation: {ds}, size={fsize}")
            splits = create_splits(ds, train_ds, seed, forget_size=fsize)
            forget_idx = splits['forget_indices']
            retain_idx = splits['retain_indices']

            orig_model = get_model(ds)
            orig_model.load_state_dict(torch.load(
                f'exp/models/original/{ds}_seed{seed}.pt', weights_only=True))

            forget_diff = diff_train[forget_idx]
            forget_loader = get_loader(train_ds, indices=forget_idx, shuffle=False)
            retain_loader = get_loader(train_ds, indices=retain_idx, shuffle=True)

            rng = np.random.RandomState(seed)
            nm_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False).tolist()
            nm_loader = get_loader(test_ds, indices=nm_idx, shuffle=False)
            forget_eval_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)

            forget_quintiles = train_quint[forget_idx]
            nm_quintiles = test_quint[nm_idx]
            forget_ref = [rl[forget_idx] for rl in ref_train_losses]
            nm_ref = [rl[nm_idx] for rl in ref_test_losses]
            forget_ref_mean = np.mean(forget_ref, axis=0)
            nm_ref_mean = np.mean(nm_ref, axis=0)

            # Also train retrain model for this forget size
            set_seed(seed + 2000 + fsize)
            retrain_model = get_model(ds)
            retrain_loader = get_loader(train_ds, indices=retain_idx)
            retrain_model = train_model(retrain_model, retrain_loader, ds, verbose=False)

            for method in ['ga', 'scrub']:
                for variant in ['standard', 'dau']:
                    set_seed(seed)
                    if variant == 'dau':
                        dau_w = compute_dau_weights(forget_diff, alpha=1.0)
                        unlearned = run_unlearning(orig_model, forget_loader, retain_loader,
                                                   method, ds, sample_weights=dau_w, num_classes=nc)
                    else:
                        unlearned = run_unlearning(orig_model, forget_loader, retain_loader,
                                                   method, ds, num_classes=nc)

                    test_loader = get_loader(test_ds, shuffle=False)
                    test_acc, _ = evaluate_model(unlearned, test_loader)
                    retain_eval = get_loader(train_eval, indices=retain_idx, shuffle=False)
                    retain_acc, _ = evaluate_model(unlearned, retain_eval)

                    m_losses = compute_losses(unlearned, forget_eval_loader)
                    nm_losses = compute_losses(unlearned, nm_loader)

                    strat = stratified_mia(m_losses, nm_losses, forget_ref_mean, nm_ref_mean,
                                           forget_ref, nm_ref, forget_quintiles, nm_quintiles)

                    results.append({
                        'method': method, 'variant': variant, 'dataset': ds, 'seed': seed,
                        'forget_size': fsize, 'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                        'agg_auc': strat['aggregate']['best_auc'],
                        'test_acc': float(test_acc), 'retain_acc': float(retain_acc),
                    })
                    del unlearned
                    torch.cuda.empty_cache()

            # Retrain baseline
            m_losses = compute_losses(retrain_model, forget_eval_loader)
            nm_losses = compute_losses(retrain_model, nm_loader)
            strat = stratified_mia(m_losses, nm_losses, forget_ref_mean, nm_ref_mean,
                                   forget_ref, nm_ref, forget_quintiles, nm_quintiles)
            test_acc, _ = evaluate_model(retrain_model, get_loader(test_ds, shuffle=False))
            results.append({
                'method': 'retrain', 'variant': 'standard', 'dataset': ds, 'seed': seed,
                'forget_size': fsize, 'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                'agg_auc': strat['aggregate']['best_auc'],
                'test_acc': float(test_acc), 'retain_acc': float(test_acc),
            })

            del orig_model, retrain_model
            torch.cuda.empty_cache()
            gc.collect()

    save_json(results, 'exp/results/ablation_forget_size.json')
    log("Ablation 4 complete.")


# ============================================================
# Ablation 5: Random-weight control
# ============================================================
def ablation_random_weights():
    log("=== Ablation 5: Random-Weight Control ===")
    ds = 'cifar10'
    results = []

    train_ds, test_ds, train_eval = load_dataset(ds)
    diff_train = np.load(f'exp/results/difficulty_{ds}_train.npy')
    diff_test = np.load(f'exp/results/difficulty_{ds}_test.npy')
    train_quint = np.load(f'exp/results/quintiles_{ds}_train.npy')
    test_quint = np.load(f'exp/results/quintiles_{ds}_test.npy')

    ref_train_losses = [np.load(f'exp/results/ref_train_losses_{ds}_ref{k}.npy') for k in range(4)]
    ref_test_losses = [np.load(f'exp/results/ref_test_losses_{ds}_ref{k}.npy') for k in range(4)]

    for seed in SEEDS:
        splits = create_splits(ds, train_ds, seed)
        forget_idx = splits['forget_indices']
        retain_idx = splits['retain_indices']

        orig_model = get_model(ds)
        orig_model.load_state_dict(torch.load(
            f'exp/models/original/{ds}_seed{seed}.pt', weights_only=True))

        forget_diff = diff_train[forget_idx]
        forget_loader = get_loader(train_ds, indices=forget_idx, shuffle=False)
        retain_loader = get_loader(train_ds, indices=retain_idx, shuffle=True)

        rng = np.random.RandomState(seed)
        nm_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False).tolist()
        nm_loader = get_loader(test_ds, indices=nm_idx, shuffle=False)
        forget_eval_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)

        forget_quintiles = train_quint[forget_idx]
        nm_quintiles = test_quint[nm_idx]
        forget_ref = [rl[forget_idx] for rl in ref_train_losses]
        nm_ref = [rl[nm_idx] for rl in ref_test_losses]
        forget_ref_mean = np.mean(forget_ref, axis=0)
        nm_ref_mean = np.mean(nm_ref, axis=0)

        # Random permutation of difficulty scores
        perm_diff = forget_diff.copy()
        np.random.RandomState(seed + 999).shuffle(perm_diff)
        random_weights = compute_dau_weights(perm_diff, alpha=1.0)

        set_seed(seed)
        unlearned = run_unlearning(orig_model, forget_loader, retain_loader,
                                   'ga', ds, sample_weights=random_weights, num_classes=10)

        test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, shuffle=False))
        retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval, indices=retain_idx, shuffle=False))

        m_losses = compute_losses(unlearned, forget_eval_loader)
        nm_losses = compute_losses(unlearned, nm_loader)
        strat = stratified_mia(m_losses, nm_losses, forget_ref_mean, nm_ref_mean,
                               forget_ref, nm_ref, forget_quintiles, nm_quintiles)

        results.append({
            'method': 'ga', 'variant': 'random_weights', 'dataset': ds, 'seed': seed,
            'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
            'agg_auc': strat['aggregate']['best_auc'],
            'test_acc': float(test_acc), 'retain_acc': float(retain_acc),
        })
        log(f"  Random weights {ds}_s{seed}: WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")

        del unlearned, orig_model
        torch.cuda.empty_cache()
        gc.collect()

    save_json(results, 'exp/results/ablation_random_weights.json')
    log("Ablation 5 complete.")


if __name__ == '__main__':
    start_time = time.time()
    log("Starting ablation experiments")

    ablation_K()
    ablation_alpha()
    ablation_strata()
    ablation_forget_size()
    ablation_random_weights()

    total = (time.time() - start_time) / 60
    log(f"All ablations complete in {total:.1f} minutes")
