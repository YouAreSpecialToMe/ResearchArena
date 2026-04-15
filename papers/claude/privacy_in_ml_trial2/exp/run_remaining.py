#!/usr/bin/env python3
"""Resume pipeline: train remaining models and run full evaluation.
Skips any already-saved checkpoints.
"""
import os
import sys
import json
import time
import copy
import gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.shared.config import *
from exp.shared.models import get_model
from exp.shared.utils import set_seed, load_dataset, create_splits, train_model, evaluate_model, get_loader
from exp.shared.unlearning import run_unlearning
from exp.shared.mia import compute_losses, run_all_attacks, stratified_mia
from exp.shared.dau import compute_dau_weights, compute_rum_groups

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

# Focus on CIFAR-10 and CIFAR-100 only (Purchase-100 has generalization issues)
FOCUS_DATASETS = ['cifar10', 'cifar100']

def main():
    start_time = time.time()

    # ---- Load datasets ----
    log("Loading datasets...")
    all_data = {}
    for ds in FOCUS_DATASETS:
        train_ds, test_ds, train_eval = load_dataset(ds)
        all_data[ds] = {
            'train': train_ds, 'test': test_ds, 'train_eval': train_eval,
            'splits': {},
        }
        for seed in SEEDS:
            all_data[ds]['splits'][seed] = create_splits(ds, train_ds, seed)

    # Also load purchase100 for completeness
    train_ds, test_ds, train_eval = load_dataset('purchase100')
    all_data['purchase100'] = {
        'train': train_ds, 'test': test_ds, 'train_eval': train_eval,
        'splits': {},
    }
    for seed in SEEDS:
        all_data['purchase100']['splits'][seed] = create_splits('purchase100', train_ds, seed)

    # ---- Load difficulty scores ----
    log("Loading difficulty scores...")
    difficulty_scores = {}
    for ds in DATASETS:
        diff_train = np.load(f'exp/results/difficulty_{ds}_train.npy')
        diff_test = np.load(f'exp/results/difficulty_{ds}_test.npy')
        train_quint = np.load(f'exp/results/quintiles_{ds}_train.npy')
        test_quint = np.load(f'exp/results/quintiles_{ds}_test.npy')
        ref_train = [np.load(f'exp/results/ref_train_losses_{ds}_ref{k}.npy') for k in range(4)]
        ref_test = [np.load(f'exp/results/ref_test_losses_{ds}_ref{k}.npy') for k in range(4)]
        difficulty_scores[ds] = {
            'train': diff_train, 'test': diff_test,
            'train_quintiles': train_quint, 'test_quintiles': test_quint,
            'ref_train_losses': ref_train, 'ref_test_losses': ref_test,
        }

    # ---- Phase 3: Train remaining original + retrain models ----
    log("=== Phase 3: Train Remaining Models ===")
    training_log = []

    for ds in FOCUS_DATASETS:
        train_ds = all_data[ds]['train']
        test_ds = all_data[ds]['test']

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            non_ref = splits['non_ref_indices']
            retain = splits['retain_indices']

            for model_type, indices, seed_offset in [('original', non_ref, 0), ('retrain', retain, 1000)]:
                model_path = f'exp/models/{model_type}/{ds}_seed{seed}.pt'
                if os.path.exists(model_path):
                    log(f"  {model_type} exists: {ds} seed={seed}")
                    continue

                log(f"Training {model_type}: {ds} seed={seed}...")
                set_seed(seed + seed_offset)
                model = get_model(ds)
                loader = get_loader(train_ds, indices=indices)
                t0 = time.time()
                model = train_model(model, loader, ds, verbose=True)
                elapsed = time.time() - t0
                torch.save(model.state_dict(), model_path)

                test_loader = get_loader(test_ds, shuffle=False)
                test_acc, _ = evaluate_model(model, test_loader)
                log(f"  {model_type} {ds} s{seed}: test_acc={test_acc:.4f}, time={elapsed:.0f}s")
                training_log.append({
                    'model_id': f'{model_type}_{ds}_s{seed}', 'dataset': ds, 'seed': seed,
                    'type': model_type, 'test_acc': float(test_acc), 'time_sec': elapsed
                })
                del model
                torch.cuda.empty_cache()

    # Also train Purchase models (faster)
    ds = 'purchase100'
    train_ds = all_data[ds]['train']
    test_ds = all_data[ds]['test']
    for seed in SEEDS:
        splits = all_data[ds]['splits'][seed]
        for model_type, indices, seed_offset in [('original', splits['non_ref_indices'], 0), ('retrain', splits['retain_indices'], 1000)]:
            model_path = f'exp/models/{model_type}/{ds}_seed{seed}.pt'
            if os.path.exists(model_path):
                log(f"  {model_type} exists: {ds} seed={seed}")
                continue
            log(f"Training {model_type}: {ds} seed={seed}...")
            set_seed(seed + seed_offset)
            model = get_model(ds)
            loader = get_loader(train_ds, indices=indices)
            t0 = time.time()
            model = train_model(model, loader, ds, verbose=True)
            elapsed = time.time() - t0
            torch.save(model.state_dict(), model_path)
            test_acc, _ = evaluate_model(model, get_loader(test_ds, shuffle=False))
            log(f"  {model_type} {ds} s{seed}: test_acc={test_acc:.4f}")
            training_log.append({'model_id': f'{model_type}_{ds}_s{seed}', 'dataset': ds, 'seed': seed,
                                 'type': model_type, 'test_acc': float(test_acc), 'time_sec': elapsed})
            del model; torch.cuda.empty_cache()

    save_json(training_log, 'exp/results/model_training_log.json')
    log(f"Phase 3 done: {(time.time()-start_time)/60:.1f} min elapsed")

    # ---- Phase 4: Unlearning baselines ----
    log("=== Phase 4: Unlearning Baselines ===")
    ul_results = []

    for ds in DATASETS:
        train_ds = all_data[ds]['train']
        train_eval = all_data[ds]['train_eval']
        test_ds = all_data[ds]['test']
        nc = get_num_classes(ds)

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            forget_idx = splits['forget_indices']
            retain_idx = splits['retain_indices']

            orig_path = f'exp/models/original/{ds}_seed{seed}.pt'
            if not os.path.exists(orig_path):
                log(f"  Original model not found: {ds} s{seed}, skipping")
                continue

            orig_model = get_model(ds)
            orig_model.load_state_dict(torch.load(orig_path, weights_only=True))

            forget_loader = get_loader(train_ds, indices=forget_idx, shuffle=True)
            retain_loader = get_loader(train_ds, indices=retain_idx, shuffle=True)

            for method in UNLEARN_METHODS:
                save_path = f'exp/models/unlearned/{method}/{ds}_seed{seed}.pt'
                if os.path.exists(save_path):
                    log(f"  {method}/{ds}_s{seed} exists")
                    continue

                log(f"Unlearning {method}/{ds} s{seed}...")
                set_seed(seed)
                t0 = time.time()
                unlearned = run_unlearning(orig_model, forget_loader, retain_loader,
                                          method, ds, num_classes=nc)
                elapsed = time.time() - t0
                torch.save(unlearned.state_dict(), save_path)

                test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, shuffle=False))
                forget_acc, _ = evaluate_model(unlearned, get_loader(train_eval, indices=forget_idx, shuffle=False))
                retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval, indices=retain_idx, shuffle=False))
                log(f"  {method}/{ds}_s{seed}: TA={test_acc:.4f}, FA={forget_acc:.4f}, RA={retain_acc:.4f}")
                ul_results.append({
                    'method': method, 'dataset': ds, 'seed': seed,
                    'test_acc': float(test_acc), 'forget_acc': float(forget_acc),
                    'retain_acc': float(retain_acc), 'time_sec': elapsed,
                })
                del unlearned; torch.cuda.empty_cache()

            del orig_model; torch.cuda.empty_cache(); gc.collect()

    save_json(ul_results, 'exp/results/unlearning_baselines.json')
    log(f"Phase 4 done: {(time.time()-start_time)/60:.1f} min elapsed")

    # ---- Phase 5: MIA evaluation ----
    log("=== Phase 5: MIA Evaluation ===")
    agg_results = []
    strat_results = []

    for ds in DATASETS:
        train_eval = all_data[ds]['train_eval']
        test_ds = all_data[ds]['test']
        dsc = difficulty_scores[ds]

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            forget_idx = splits['forget_indices']
            rng = np.random.RandomState(seed)
            nm_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False).tolist()

            forget_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)
            nm_loader = get_loader(test_ds, indices=nm_idx, shuffle=False)

            f_quint = dsc['train_quintiles'][forget_idx]
            nm_quint = dsc['test_quintiles'][nm_idx]
            f_ref = [rl[forget_idx] for rl in dsc['ref_train_losses']]
            nm_ref = [rl[nm_idx] for rl in dsc['ref_test_losses']]
            f_ref_mean = np.mean(f_ref, axis=0)
            nm_ref_mean = np.mean(nm_ref, axis=0)

            configs = [('retrain', f'exp/models/retrain/{ds}_seed{seed}.pt')]
            for m in UNLEARN_METHODS:
                configs.append((m, f'exp/models/unlearned/{m}/{ds}_seed{seed}.pt'))

            for method, path in configs:
                if not os.path.exists(path):
                    continue
                model = get_model(ds)
                model.load_state_dict(torch.load(path, weights_only=True))
                m_losses = compute_losses(model, forget_loader)
                nm_losses = compute_losses(model, nm_loader)

                agg = run_all_attacks(m_losses, nm_losses, f_ref_mean, nm_ref_mean, f_ref, nm_ref)
                agg.update({'method': method, 'dataset': ds, 'seed': seed, 'variant': 'standard'})
                agg_results.append(agg)

                strat = stratified_mia(m_losses, nm_losses, f_ref_mean, nm_ref_mean,
                                       f_ref, nm_ref, f_quint, nm_quint)
                strat.update({'method': method, 'dataset': ds, 'seed': seed, 'variant': 'standard'})
                strat_results.append(strat)
                log(f"  MIA {method}/{ds}_s{seed}: agg={agg['best_auc']:.4f}, WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")
                del model; torch.cuda.empty_cache()

    save_json(agg_results, 'exp/results/mia_aggregate.json')
    save_json(strat_results, 'exp/results/mia_stratified.json')
    log(f"Phase 5 done: {(time.time()-start_time)/60:.1f} min elapsed")

    # ---- Phase 6: DAU + RUM ----
    log("=== Phase 6: DAU Defense + RUM ===")
    dau_results = []

    for ds in DATASETS:
        train_ds = all_data[ds]['train']
        train_eval = all_data[ds]['train_eval']
        test_ds = all_data[ds]['test']
        diff_train = difficulty_scores[ds]['train']
        nc = get_num_classes(ds)

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            forget_idx = splits['forget_indices']
            retain_idx = splits['retain_indices']

            orig_path = f'exp/models/original/{ds}_seed{seed}.pt'
            if not os.path.exists(orig_path):
                continue
            orig_model = get_model(ds)
            orig_model.load_state_dict(torch.load(orig_path, weights_only=True))

            forget_diff = diff_train[forget_idx]
            dau_weights = compute_dau_weights(forget_diff, alpha=DEFAULT_ALPHA)
            forget_loader = get_loader(train_ds, indices=forget_idx, shuffle=False)
            retain_loader = get_loader(train_ds, indices=retain_idx, shuffle=True)

            # DAU for all methods
            for method in UNLEARN_METHODS:
                save_path = f'exp/models/unlearned/{method}_dau/{ds}_seed{seed}.pt'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.exists(save_path):
                    log(f"  DAU {method}/{ds}_s{seed} exists")
                    continue
                log(f"DAU {method}/{ds} s{seed}...")
                set_seed(seed)
                unlearned = run_unlearning(orig_model, forget_loader, retain_loader,
                                          method, ds, sample_weights=dau_weights, num_classes=nc)
                torch.save(unlearned.state_dict(), save_path)
                test_acc, _ = evaluate_model(unlearned, get_loader(test_ds, shuffle=False))
                retain_acc, _ = evaluate_model(unlearned, get_loader(train_eval, indices=retain_idx, shuffle=False))
                log(f"  DAU-{method}/{ds}_s{seed}: TA={test_acc:.4f}, RA={retain_acc:.4f}")
                dau_results.append({
                    'method': f'{method}_dau', 'dataset': ds, 'seed': seed,
                    'test_acc': float(test_acc), 'retain_acc': float(retain_acc), 'alpha': DEFAULT_ALPHA,
                })
                del unlearned; torch.cuda.empty_cache()

            # RUM for GA and SCRUB
            for method in ['ga', 'scrub']:
                save_path = f'exp/models/unlearned/{method}_rum/{ds}_seed{seed}.pt'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.exists(save_path):
                    log(f"  RUM {method}/{ds}_s{seed} exists")
                    continue
                log(f"RUM {method}/{ds} s{seed}...")
                set_seed(seed)
                groups = compute_rum_groups(forget_diff)
                epoch_map = {0: 3, 1: 5, 2: 8}
                current = copy.deepcopy(orig_model)
                for g in range(3):
                    g_mask = groups == g
                    g_idx = np.array(forget_idx)[g_mask].tolist()
                    if not g_idx:
                        continue
                    g_loader = get_loader(train_ds, indices=g_idx, shuffle=False)
                    current = run_unlearning(current, g_loader, retain_loader,
                                            method, ds, epochs=epoch_map[g], num_classes=nc)
                torch.save(current.state_dict(), save_path)
                test_acc, _ = evaluate_model(current, get_loader(test_ds, shuffle=False))
                retain_acc, _ = evaluate_model(current, get_loader(train_eval, indices=retain_idx, shuffle=False))
                log(f"  RUM-{method}/{ds}_s{seed}: TA={test_acc:.4f}, RA={retain_acc:.4f}")
                dau_results.append({
                    'method': f'{method}_rum', 'dataset': ds, 'seed': seed,
                    'test_acc': float(test_acc), 'retain_acc': float(retain_acc),
                })
                del current; torch.cuda.empty_cache()

            del orig_model; torch.cuda.empty_cache(); gc.collect()

    save_json(dau_results, 'exp/results/dau_rum_baselines.json')
    log(f"Phase 6 done: {(time.time()-start_time)/60:.1f} min elapsed")

    # ---- Phase 7: MIA on DAU/RUM ----
    log("=== Phase 7: MIA on DAU/RUM ===")
    agg_def = []
    strat_def = []

    dau_methods = [f'{m}_dau' for m in UNLEARN_METHODS] + ['ga_rum', 'scrub_rum']

    for ds in DATASETS:
        train_eval = all_data[ds]['train_eval']
        test_ds = all_data[ds]['test']
        dsc = difficulty_scores[ds]

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            forget_idx = splits['forget_indices']
            rng = np.random.RandomState(seed)
            nm_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False).tolist()

            forget_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)
            nm_loader = get_loader(test_ds, indices=nm_idx, shuffle=False)

            f_quint = dsc['train_quintiles'][forget_idx]
            nm_quint = dsc['test_quintiles'][nm_idx]
            f_ref = [rl[forget_idx] for rl in dsc['ref_train_losses']]
            nm_ref = [rl[nm_idx] for rl in dsc['ref_test_losses']]
            f_ref_mean = np.mean(f_ref, axis=0)
            nm_ref_mean = np.mean(nm_ref, axis=0)

            for method in dau_methods:
                path = f'exp/models/unlearned/{method}/{ds}_seed{seed}.pt'
                if not os.path.exists(path):
                    continue
                model = get_model(ds)
                model.load_state_dict(torch.load(path, weights_only=True))
                m_losses = compute_losses(model, forget_loader)
                nm_losses = compute_losses(model, nm_loader)

                agg = run_all_attacks(m_losses, nm_losses, f_ref_mean, nm_ref_mean, f_ref, nm_ref)
                agg.update({'method': method, 'dataset': ds, 'seed': seed, 'variant': 'defense'})
                agg_def.append(agg)

                strat = stratified_mia(m_losses, nm_losses, f_ref_mean, nm_ref_mean,
                                       f_ref, nm_ref, f_quint, nm_quint)
                strat.update({'method': method, 'dataset': ds, 'seed': seed, 'variant': 'defense'})
                strat_def.append(strat)
                log(f"  MIA {method}/{ds}_s{seed}: agg={agg['best_auc']:.4f}, WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")
                del model; torch.cuda.empty_cache()

    save_json(agg_def, 'exp/results/mia_aggregate_defense.json')
    save_json(strat_def, 'exp/results/mia_stratified_defense.json')

    total = (time.time() - start_time) / 60
    log(f"All phases complete in {total:.1f} minutes")


if __name__ == '__main__':
    main()
