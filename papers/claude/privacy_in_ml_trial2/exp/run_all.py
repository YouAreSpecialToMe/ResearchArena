#!/usr/bin/env python3
"""Main experiment pipeline for Difficulty-Aware Unlearning (DAU).

Phases:
1. Data preparation
2. Train reference models + compute difficulty scores
3. Train original + retrain models
4. Run unlearning baselines
5. MIA evaluation (aggregate + stratified)
6. DAU defense + RUM baseline
7. MIA evaluation on DAU/RUM models
"""
import os
import sys
import json
import time
import copy
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

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

# ============================================================
# Phase 1: Data preparation
# ============================================================
def phase1_data():
    log("=== Phase 1: Data Preparation ===")
    all_data = {}
    data_stats = {}

    for ds in DATASETS:
        log(f"Loading {ds}...")
        train_ds, test_ds, train_eval_ds = load_dataset(ds)
        all_data[ds] = {
            'train': train_ds,
            'test': test_ds,
            'train_eval': train_eval_ds,
            'splits': {},
        }

        # Create splits for each seed
        for seed in SEEDS:
            splits = create_splits(ds, train_ds, seed)
            all_data[ds]['splits'][seed] = splits
            save_json(splits, f'exp/data/splits_{ds}_seed{seed}.json')

        # Stats
        n_train = len(train_ds)
        n_test = len(test_ds)
        data_stats[ds] = {
            'n_train': n_train, 'n_test': n_test,
            'n_forget': FORGET_SIZE, 'n_retain': n_train - REF_POOL_SIZE - FORGET_SIZE,
            'n_ref_pool': REF_POOL_SIZE, 'n_classes': get_num_classes(ds),
        }

    save_json(data_stats, 'exp/results/data_stats.json')
    log("Phase 1 complete.")
    return all_data

# ============================================================
# Phase 2: Reference models + difficulty scores
# ============================================================
def phase2_reference(all_data):
    log("=== Phase 2: Reference Models & Difficulty Scores ===")
    ref_quality = {}
    difficulty_scores = {}

    for ds in DATASETS:
        log(f"Training reference models for {ds}...")
        train_ds = all_data[ds]['train']
        test_ds = all_data[ds]['test']
        train_eval = all_data[ds]['train_eval']
        non_ref_indices = list(range(REF_POOL_SIZE, len(train_ds)))

        ref_models = []
        for k, rseed in enumerate(REF_SEEDS):
            model_path = f'exp/models/reference/{ds}_ref{k}.pt'
            if os.path.exists(model_path):
                log(f"  Loading existing ref model {k}...")
                model = get_model(ds)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                ref_models.append(model)
                # Evaluate
                test_loader = get_loader(test_ds, shuffle=False)
                acc, _ = evaluate_model(model, test_loader)
                ref_quality[f'{ds}_ref{k}'] = {'test_acc': float(acc)}
                continue

            set_seed(rseed)
            model = get_model(ds)

            # Random 80% subset of non-ref-pool data
            rng = np.random.RandomState(rseed)
            subset_size = int(0.8 * len(non_ref_indices))
            subset_idx = rng.choice(non_ref_indices, size=subset_size, replace=False).tolist()

            loader = get_loader(train_ds, indices=subset_idx)
            model = train_model(model, loader, ds, verbose=True)

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

            # Evaluate
            test_loader = get_loader(test_ds, shuffle=False)
            acc, _ = evaluate_model(model, test_loader)
            ref_quality[f'{ds}_ref{k}'] = {'test_acc': float(acc)}
            log(f"  Ref model {k}: test_acc={acc:.4f}")
            ref_models.append(model)

        # Compute difficulty scores on full training set
        log(f"Computing difficulty scores for {ds}...")
        train_eval_loader = get_loader(train_eval, shuffle=False)
        test_loader = get_loader(test_ds, shuffle=False)

        all_train_losses = []
        all_test_losses = []
        for model in ref_models:
            _, train_losses = evaluate_model(model, train_eval_loader)
            _, test_losses = evaluate_model(model, test_loader)
            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)

        # Mean loss across ref models = difficulty score
        train_diff = np.mean(all_train_losses, axis=0)
        test_diff = np.mean(all_test_losses, axis=0)

        np.save(f'exp/results/difficulty_{ds}_train.npy', train_diff)
        np.save(f'exp/results/difficulty_{ds}_test.npy', test_diff)

        # Store per-ref-model losses for LiRA
        for k, losses in enumerate(all_train_losses):
            np.save(f'exp/results/ref_train_losses_{ds}_ref{k}.npy', losses)
        for k, losses in enumerate(all_test_losses):
            np.save(f'exp/results/ref_test_losses_{ds}_ref{k}.npy', losses)

        # Quintile assignments
        percentiles = np.percentile(train_diff, [20, 40, 60, 80])
        quintiles = np.digitize(train_diff, percentiles)  # 0-4
        np.save(f'exp/results/quintiles_{ds}_train.npy', quintiles)

        test_quintiles = np.digitize(test_diff, percentiles)
        np.save(f'exp/results/quintiles_{ds}_test.npy', test_quintiles)

        # Validate difficulty scores
        for q in range(5):
            mask = quintiles == q
            q_diff_mean = train_diff[mask].mean()
            log(f"  {ds} Q{q+1}: n={mask.sum()}, mean_difficulty={q_diff_mean:.4f}")

        difficulty_scores[ds] = {
            'train': train_diff, 'test': test_diff,
            'train_quintiles': quintiles, 'test_quintiles': test_quintiles,
            'ref_train_losses': all_train_losses, 'ref_test_losses': all_test_losses,
        }

        # Free ref models from GPU
        del ref_models
        torch.cuda.empty_cache()
        gc.collect()

    save_json(ref_quality, 'exp/results/reference_model_quality.json')
    log("Phase 2 complete.")
    return difficulty_scores

# ============================================================
# Phase 3: Train original + retrain models
# ============================================================
def phase3_train(all_data):
    log("=== Phase 3: Train Original & Retrain Models ===")
    training_log = []

    for ds in DATASETS:
        train_ds = all_data[ds]['train']
        test_ds = all_data[ds]['test']

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            non_ref = splits['non_ref_indices']
            retain = splits['retain_indices']

            # Original model (trained on all non-ref data)
            orig_path = f'exp/models/original/{ds}_seed{seed}.pt'
            if not os.path.exists(orig_path):
                log(f"Training original model: {ds} seed={seed}...")
                set_seed(seed)
                model = get_model(ds)
                loader = get_loader(train_ds, indices=non_ref)
                t0 = time.time()
                model = train_model(model, loader, ds)
                elapsed = time.time() - t0

                torch.save(model.state_dict(), orig_path)
                test_loader = get_loader(test_ds, shuffle=False)
                test_acc, _ = evaluate_model(model, test_loader)
                log(f"  Original {ds} s{seed}: test_acc={test_acc:.4f}, time={elapsed:.0f}s")
                training_log.append({
                    'model_id': f'original_{ds}_s{seed}', 'dataset': ds, 'seed': seed,
                    'type': 'original', 'test_acc': float(test_acc), 'time_sec': elapsed
                })
            else:
                log(f"  Original model exists: {ds} seed={seed}")

            # Retrain model (trained on retain only)
            retrain_path = f'exp/models/retrain/{ds}_seed{seed}.pt'
            if not os.path.exists(retrain_path):
                log(f"Training retrain model: {ds} seed={seed}...")
                set_seed(seed + 1000)  # Different init
                model = get_model(ds)
                loader = get_loader(train_ds, indices=retain)
                t0 = time.time()
                model = train_model(model, loader, ds)
                elapsed = time.time() - t0

                torch.save(model.state_dict(), retrain_path)
                test_loader = get_loader(test_ds, shuffle=False)
                test_acc, _ = evaluate_model(model, test_loader)
                log(f"  Retrain {ds} s{seed}: test_acc={test_acc:.4f}, time={elapsed:.0f}s")
                training_log.append({
                    'model_id': f'retrain_{ds}_s{seed}', 'dataset': ds, 'seed': seed,
                    'type': 'retrain', 'test_acc': float(test_acc), 'time_sec': elapsed
                })
            else:
                log(f"  Retrain model exists: {ds} seed={seed}")

            torch.cuda.empty_cache()
            gc.collect()

    save_json(training_log, 'exp/results/model_training_log.json')
    log("Phase 3 complete.")

# ============================================================
# Phase 4: Unlearning baselines
# ============================================================
def phase4_unlearning(all_data):
    log("=== Phase 4: Unlearning Baselines ===")
    results = []

    for ds in DATASETS:
        train_ds = all_data[ds]['train']
        train_eval = all_data[ds]['train_eval']
        test_ds = all_data[ds]['test']
        nc = get_num_classes(ds)

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            forget_idx = splits['forget_indices']
            retain_idx = splits['retain_indices']

            # Load original model
            orig_path = f'exp/models/original/{ds}_seed{seed}.pt'
            orig_model = get_model(ds)
            orig_model.load_state_dict(torch.load(orig_path, weights_only=True))

            forget_loader = get_loader(train_ds, indices=forget_idx, shuffle=True)
            retain_loader = get_loader(train_ds, indices=retain_idx, shuffle=True)

            for method in UNLEARN_METHODS:
                save_path = f'exp/models/unlearned/{method}/{ds}_seed{seed}.pt'
                if os.path.exists(save_path):
                    log(f"  Unlearned model exists: {method}/{ds}_s{seed}")
                    continue

                log(f"Unlearning {method} on {ds} seed={seed}...")
                t0 = time.time()
                set_seed(seed)
                unlearned = run_unlearning(orig_model, forget_loader, retain_loader,
                                          method, ds, num_classes=nc)
                elapsed = time.time() - t0

                torch.save(unlearned.state_dict(), save_path)

                # Evaluate
                test_loader = get_loader(test_ds, shuffle=False)
                test_acc, _ = evaluate_model(unlearned, test_loader)
                forget_eval_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)
                forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)
                retain_eval_loader = get_loader(train_eval, indices=retain_idx, shuffle=False)
                retain_acc, _ = evaluate_model(unlearned, retain_eval_loader)

                log(f"  {method}/{ds}_s{seed}: TA={test_acc:.4f}, FA={forget_acc:.4f}, RA={retain_acc:.4f}, time={elapsed:.0f}s")
                results.append({
                    'method': method, 'dataset': ds, 'seed': seed,
                    'test_acc': float(test_acc), 'forget_acc': float(forget_acc),
                    'retain_acc': float(retain_acc), 'time_sec': elapsed,
                })

                del unlearned
                torch.cuda.empty_cache()

            del orig_model
            torch.cuda.empty_cache()
            gc.collect()

    save_json(results, 'exp/results/unlearning_baselines.json')
    log("Phase 4 complete.")

# ============================================================
# Phase 5: MIA evaluation (aggregate + stratified)
# ============================================================
def phase5_mia(all_data, difficulty_scores):
    log("=== Phase 5: MIA Evaluation ===")
    aggregate_results = []
    stratified_results = []

    for ds in DATASETS:
        train_eval = all_data[ds]['train_eval']
        test_ds = all_data[ds]['test']
        diff_train = difficulty_scores[ds]['train']
        diff_test = difficulty_scores[ds]['test']
        train_quintiles = difficulty_scores[ds]['train_quintiles']
        test_quintiles = difficulty_scores[ds]['test_quintiles']
        ref_train_losses = difficulty_scores[ds]['ref_train_losses']
        ref_test_losses = difficulty_scores[ds]['ref_test_losses']

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            forget_idx = splits['forget_indices']

            # Select matched non-members from test set
            n_forget = len(forget_idx)
            rng = np.random.RandomState(seed)
            nonmember_idx = rng.choice(len(test_ds), size=n_forget, replace=False).tolist()

            # Create loaders
            forget_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)
            nonmember_loader = get_loader(test_ds, indices=nonmember_idx, shuffle=False)

            # Forget set difficulty quintiles
            forget_quintiles = train_quintiles[forget_idx]

            # Difficulty-matched non-member quintiles
            nonmember_quintiles = test_quintiles[nonmember_idx]

            # Reference model losses for forget set and nonmembers
            forget_ref_losses = [ref_train_losses[k][forget_idx] for k in range(len(REF_SEEDS))]
            nonmember_ref_losses = [ref_test_losses[k][nonmember_idx] for k in range(len(REF_SEEDS))]
            forget_ref_mean = np.mean(forget_ref_losses, axis=0)
            nonmember_ref_mean = np.mean(nonmember_ref_losses, axis=0)

            # Evaluate all model types: retrain + 5 unlearning methods
            model_configs = [('retrain', f'exp/models/retrain/{ds}_seed{seed}.pt')]
            for method in UNLEARN_METHODS:
                model_configs.append((method, f'exp/models/unlearned/{method}/{ds}_seed{seed}.pt'))

            for method, model_path in model_configs:
                if not os.path.exists(model_path):
                    log(f"  Skipping MIA for {method}/{ds}_s{seed} (model not found)")
                    continue

                model = get_model(ds)
                model.load_state_dict(torch.load(model_path, weights_only=True))

                # Compute losses
                member_losses = compute_losses(model, forget_loader)
                nm_losses = compute_losses(model, nonmember_loader)

                # Aggregate MIA
                agg = run_all_attacks(member_losses, nm_losses,
                                      forget_ref_mean, nonmember_ref_mean,
                                      forget_ref_losses, nonmember_ref_losses)
                agg.update({'method': method, 'dataset': ds, 'seed': seed, 'variant': 'standard'})
                aggregate_results.append(agg)

                # Stratified MIA
                strat = stratified_mia(member_losses, nm_losses,
                                       forget_ref_mean, nonmember_ref_mean,
                                       forget_ref_losses, nonmember_ref_losses,
                                       forget_quintiles, nonmember_quintiles)
                strat.update({'method': method, 'dataset': ds, 'seed': seed, 'variant': 'standard'})
                stratified_results.append(strat)

                log(f"  MIA {method}/{ds}_s{seed}: agg_best={agg['best_auc']:.4f}, "
                    f"WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")

                del model
                torch.cuda.empty_cache()

    save_json(aggregate_results, 'exp/results/mia_aggregate.json')
    save_json(stratified_results, 'exp/results/mia_stratified.json')
    log("Phase 5 complete.")
    return aggregate_results, stratified_results

# ============================================================
# Phase 6: DAU defense + RUM baseline
# ============================================================
def phase6_dau_rum(all_data, difficulty_scores):
    log("=== Phase 6: DAU Defense + RUM Baseline ===")
    results = []

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

            # Load original model
            orig_model = get_model(ds)
            orig_model.load_state_dict(torch.load(
                f'exp/models/original/{ds}_seed{seed}.pt', weights_only=True))

            # DAU weights for forget set
            forget_diff = diff_train[forget_idx]
            dau_weights = compute_dau_weights(forget_diff, alpha=DEFAULT_ALPHA)

            forget_loader = get_loader(train_ds, indices=forget_idx, shuffle=False)  # No shuffle for weight alignment
            retain_loader = get_loader(train_ds, indices=retain_idx, shuffle=True)

            # DAU for all methods
            for method in UNLEARN_METHODS:
                save_path = f'exp/models/unlearned/{method}_dau/{ds}_seed{seed}.pt'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.exists(save_path):
                    log(f"  DAU model exists: {method}_dau/{ds}_s{seed}")
                    continue

                log(f"Running DAU {method} on {ds} seed={seed}...")
                set_seed(seed)
                t0 = time.time()
                unlearned = run_unlearning(orig_model, forget_loader, retain_loader,
                                          method, ds, sample_weights=dau_weights, num_classes=nc)
                elapsed = time.time() - t0
                torch.save(unlearned.state_dict(), save_path)

                # Quick eval
                test_loader = get_loader(test_ds, shuffle=False)
                test_acc, _ = evaluate_model(unlearned, test_loader)
                retain_eval_loader = get_loader(train_eval, indices=retain_idx, shuffle=False)
                retain_acc, _ = evaluate_model(unlearned, retain_eval_loader)
                log(f"  DAU-{method}/{ds}_s{seed}: TA={test_acc:.4f}, RA={retain_acc:.4f}")
                results.append({
                    'method': f'{method}_dau', 'dataset': ds, 'seed': seed,
                    'test_acc': float(test_acc), 'retain_acc': float(retain_acc),
                    'time_sec': elapsed, 'alpha': DEFAULT_ALPHA,
                })
                del unlearned
                torch.cuda.empty_cache()

            # RUM for GA and SCRUB
            for method in ['ga', 'scrub']:
                save_path = f'exp/models/unlearned/{method}_rum/{ds}_seed{seed}.pt'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if os.path.exists(save_path):
                    log(f"  RUM model exists: {method}_rum/{ds}_s{seed}")
                    continue

                log(f"Running RUM {method} on {ds} seed={seed}...")
                set_seed(seed)
                t0 = time.time()

                # Partition forget set into 3 groups by difficulty
                groups = compute_rum_groups(forget_diff)
                epoch_map = {0: 3, 1: 5, 2: 8}  # easy/medium/hard

                # Start from original, apply unlearning per group sequentially
                current_model = copy.deepcopy(orig_model)
                for g in range(3):
                    g_mask = groups == g
                    g_indices = np.array(forget_idx)[g_mask].tolist()
                    if len(g_indices) == 0:
                        continue
                    g_loader = get_loader(train_ds, indices=g_indices, shuffle=False)
                    current_model = run_unlearning(current_model, g_loader, retain_loader,
                                                   method, ds, epochs=epoch_map[g], num_classes=nc)

                elapsed = time.time() - t0
                torch.save(current_model.state_dict(), save_path)

                test_loader = get_loader(test_ds, shuffle=False)
                test_acc, _ = evaluate_model(current_model, test_loader)
                retain_eval_loader = get_loader(train_eval, indices=retain_idx, shuffle=False)
                retain_acc, _ = evaluate_model(current_model, retain_eval_loader)
                log(f"  RUM-{method}/{ds}_s{seed}: TA={test_acc:.4f}, RA={retain_acc:.4f}")
                results.append({
                    'method': f'{method}_rum', 'dataset': ds, 'seed': seed,
                    'test_acc': float(test_acc), 'retain_acc': float(retain_acc),
                    'time_sec': elapsed,
                })
                del current_model
                torch.cuda.empty_cache()

            del orig_model
            torch.cuda.empty_cache()
            gc.collect()

    save_json(results, 'exp/results/dau_rum_baselines.json')
    log("Phase 6 complete.")

# ============================================================
# Phase 7: MIA on DAU/RUM models
# ============================================================
def phase7_mia_dau_rum(all_data, difficulty_scores):
    log("=== Phase 7: MIA on DAU/RUM Models ===")
    aggregate_results = []
    stratified_results = []

    dau_methods = [f'{m}_dau' for m in UNLEARN_METHODS]
    rum_methods = ['ga_rum', 'scrub_rum']
    all_methods = dau_methods + rum_methods

    for ds in DATASETS:
        train_eval = all_data[ds]['train_eval']
        test_ds = all_data[ds]['test']
        train_quintiles = difficulty_scores[ds]['train_quintiles']
        test_quintiles = difficulty_scores[ds]['test_quintiles']
        ref_train_losses = difficulty_scores[ds]['ref_train_losses']
        ref_test_losses = difficulty_scores[ds]['ref_test_losses']

        for seed in SEEDS:
            splits = all_data[ds]['splits'][seed]
            forget_idx = splits['forget_indices']

            rng = np.random.RandomState(seed)
            nonmember_idx = rng.choice(len(test_ds), size=len(forget_idx), replace=False).tolist()

            forget_loader = get_loader(train_eval, indices=forget_idx, shuffle=False)
            nonmember_loader = get_loader(test_ds, indices=nonmember_idx, shuffle=False)

            forget_quintiles = train_quintiles[forget_idx]
            nonmember_quintiles = test_quintiles[nonmember_idx]
            forget_ref_losses = [ref_train_losses[k][forget_idx] for k in range(len(REF_SEEDS))]
            nonmember_ref_losses = [ref_test_losses[k][nonmember_idx] for k in range(len(REF_SEEDS))]
            forget_ref_mean = np.mean(forget_ref_losses, axis=0)
            nonmember_ref_mean = np.mean(nonmember_ref_losses, axis=0)

            for method in all_methods:
                model_path = f'exp/models/unlearned/{method}/{ds}_seed{seed}.pt'
                if not os.path.exists(model_path):
                    continue

                model = get_model(ds)
                model.load_state_dict(torch.load(model_path, weights_only=True))

                member_losses = compute_losses(model, forget_loader)
                nm_losses = compute_losses(model, nonmember_loader)

                agg = run_all_attacks(member_losses, nm_losses,
                                      forget_ref_mean, nonmember_ref_mean,
                                      forget_ref_losses, nonmember_ref_losses)
                agg.update({'method': method, 'dataset': ds, 'seed': seed, 'variant': 'defense'})
                aggregate_results.append(agg)

                strat = stratified_mia(member_losses, nm_losses,
                                       forget_ref_mean, nonmember_ref_mean,
                                       forget_ref_losses, nonmember_ref_losses,
                                       forget_quintiles, nonmember_quintiles)
                strat.update({'method': method, 'dataset': ds, 'seed': seed, 'variant': 'defense'})
                stratified_results.append(strat)

                log(f"  MIA {method}/{ds}_s{seed}: agg_best={agg['best_auc']:.4f}, "
                    f"WQ={strat['wq_auc']:.4f}, DG={strat['dg']:.4f}")

                del model
                torch.cuda.empty_cache()

    save_json(aggregate_results, 'exp/results/mia_aggregate_defense.json')
    save_json(stratified_results, 'exp/results/mia_stratified_defense.json')
    log("Phase 7 complete.")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    start_time = time.time()
    log("Starting DAU experiment pipeline")

    all_data = phase1_data()
    difficulty_scores = phase2_reference(all_data)
    phase3_train(all_data)
    phase4_unlearning(all_data)
    phase5_mia(all_data, difficulty_scores)
    phase6_dau_rum(all_data, difficulty_scores)
    phase7_mia_dau_rum(all_data, difficulty_scores)

    total = (time.time() - start_time) / 60
    log(f"All phases complete in {total:.1f} minutes")
