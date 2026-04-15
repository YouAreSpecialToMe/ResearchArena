#!/usr/bin/env python3
"""Complete experiment pipeline with all bug fixes applied.

Fixes from self-review:
1. Purchase-100 regenerated with better make_classification params (v2)
2. finetune() now uses sample_weights via per-sample loss weighting
3. Complete ablation experiments
4. Proper results aggregation with mean±std
5. All figures generated

Strategy:
- Keep valid CIFAR-10/100 reference, original, retrain models
- Regenerate Purchase-100 from scratch
- Re-run ALL DAU experiments (finetune was broken)
- Complete all ablations and figures
"""
import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.shared.config import *
from exp.shared.models import get_model
from exp.shared.utils import (set_seed, load_dataset, create_splits,
                               train_model, evaluate_model, get_loader)
from exp.shared.mia import compute_losses, run_all_attacks, stratified_mia
from exp.shared.unlearning import run_unlearning, UNLEARN_FN
from exp.shared.dau import compute_dau_weights, compute_rum_groups

RESULTS_DIR = 'exp/results'
MODELS_DIR = 'exp/models'
FIGURES_DIR = 'figures'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# PHASE 0: Verify GPU
# ============================================================
def phase0_verify():
    log("PHASE 0: Verifying GPU")
    assert torch.cuda.is_available(), "CUDA not available!"
    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# PHASE 1: Data Preparation (Purchase-100 only - CIFAR cached)
# ============================================================
def phase1_data():
    log("PHASE 1: Preparing datasets")
    datasets_info = {}

    for ds_name in DATASETS:
        log(f"  Loading {ds_name}...")
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)

        # Verify Purchase-100 is now learnable
        if ds_name == 'purchase100':
            # Quick sanity: check class distribution
            if hasattr(train_ds, 'tensors'):
                labels = train_ds.tensors[1].numpy()
            else:
                labels = np.array([train_ds[i][1] for i in range(len(train_ds))])
            unique, counts = np.unique(labels, return_counts=True)
            log(f"    {ds_name}: {len(train_ds)} train, {len(test_ds)} test, {len(unique)} classes")
            log(f"    Class count range: [{counts.min()}, {counts.max()}]")

        n_classes = 10 if ds_name == 'cifar10' else 100
        datasets_info[ds_name] = {
            'n_train': len(train_ds), 'n_test': len(test_ds),
            'n_classes': n_classes
        }

    # Save data stats
    with open(f'{RESULTS_DIR}/data_stats.json', 'w') as f:
        json.dump(datasets_info, f, indent=2)

    return datasets_info

# ============================================================
# PHASE 2: Train Purchase-100 reference models (CIFAR cached)
# ============================================================
def phase2_reference_models():
    log("PHASE 2: Training reference models")
    ref_quality = {}

    # Check which reference models exist and are valid
    for ds_name in DATASETS:
        for k, rs in enumerate(REF_SEEDS):
            model_path = f'{MODELS_DIR}/reference/{ds_name}_ref{k}.pt'
            if ds_name != 'purchase100' and os.path.exists(model_path):
                log(f"  [CACHED] {ds_name} ref{k}")
                continue

            log(f"  Training {ds_name} ref{k} (seed={rs})...")
            set_seed(rs)
            train_ds, test_ds, train_eval_ds = load_dataset(ds_name)
            splits = create_splits(ds_name, train_ds, seed=rs)
            non_ref = splits['non_ref_indices']

            # Random 80% of non-ref for training
            rng = np.random.RandomState(rs)
            train_indices = sorted(rng.choice(non_ref, size=int(0.8 * len(non_ref)), replace=False).tolist())

            model = get_model(ds_name)
            train_loader = get_loader(train_ds, train_indices, shuffle=True)
            model = train_model(model, train_loader, ds_name, verbose=False)

            os.makedirs(f'{MODELS_DIR}/reference', exist_ok=True)
            torch.save(model.state_dict(), model_path)

            # Evaluate
            test_loader = get_loader(test_ds, shuffle=False)
            acc, _ = evaluate_model(model, test_loader)
            ref_quality[f'{ds_name}_ref{k}'] = {'test_acc': float(acc)}
            log(f"    Test acc: {acc:.4f}")

    # Load existing ref quality for cached models
    existing_quality = {}
    if os.path.exists(f'{RESULTS_DIR}/reference_model_quality.json'):
        existing_quality = json.load(open(f'{RESULTS_DIR}/reference_model_quality.json'))

    # Merge
    for k, v in existing_quality.items():
        if k not in ref_quality:
            ref_quality[k] = v

    with open(f'{RESULTS_DIR}/reference_model_quality.json', 'w') as f:
        json.dump(ref_quality, f, indent=2)

    return ref_quality

# ============================================================
# PHASE 3: Compute difficulty scores
# ============================================================
def phase3_difficulty_scores():
    log("PHASE 3: Computing difficulty scores")

    for ds_name in DATASETS:
        # Check if we need to recompute (Purchase-100 always, CIFAR if missing)
        train_path = f'{RESULTS_DIR}/difficulty_{ds_name}_train.npy'
        if ds_name != 'purchase100' and os.path.exists(train_path):
            log(f"  [CACHED] {ds_name} difficulty scores")
            continue

        log(f"  Computing difficulty for {ds_name}...")
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)

        train_loader = get_loader(train_eval_ds, shuffle=False)
        test_loader = get_loader(test_ds, shuffle=False)

        all_train_losses = []
        all_test_losses = []

        for k in range(len(REF_SEEDS)):
            model_path = f'{MODELS_DIR}/reference/{ds_name}_ref{k}.pt'
            model = get_model(ds_name)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

            train_losses = compute_losses(model, train_loader)
            test_losses = compute_losses(model, test_loader)

            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)

            # Save per-ref losses for LiRA
            np.save(f'{RESULTS_DIR}/ref_train_losses_{ds_name}_ref{k}.npy', train_losses)
            np.save(f'{RESULTS_DIR}/ref_test_losses_{ds_name}_ref{k}.npy', test_losses)

        # Average across reference models
        difficulty_train = np.mean(all_train_losses, axis=0)
        difficulty_test = np.mean(all_test_losses, axis=0)

        np.save(f'{RESULTS_DIR}/difficulty_{ds_name}_train.npy', difficulty_train)
        np.save(f'{RESULTS_DIR}/difficulty_{ds_name}_test.npy', difficulty_test)

        # Compute quintiles
        percentiles = np.percentile(difficulty_train, [20, 40, 60, 80])
        quintiles_train = np.digitize(difficulty_train, percentiles)
        percentiles_test = np.percentile(difficulty_test, [20, 40, 60, 80])
        quintiles_test = np.digitize(difficulty_test, percentiles_test)

        np.save(f'{RESULTS_DIR}/quintiles_{ds_name}_train.npy', quintiles_train)
        np.save(f'{RESULTS_DIR}/quintiles_{ds_name}_test.npy', quintiles_test)

        log(f"    Difficulty train: mean={difficulty_train.mean():.3f}, std={difficulty_train.std():.3f}")
        log(f"    Quintile sizes: {[int((quintiles_train==q).sum()) for q in range(5)]}")

# ============================================================
# PHASE 4: Train original + retrain models
# ============================================================
def phase4_original_retrain():
    log("PHASE 4: Training original and retrain models")
    training_log = []

    for ds_name in DATASETS:
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)
        n_classes = 10 if ds_name == 'cifar10' else 100

        for seed in SEEDS:
            splits = create_splits(ds_name, train_ds, seed)

            for model_type in ['original', 'retrain']:
                model_path = f'{MODELS_DIR}/{model_type}/{ds_name}_seed{seed}.pt'

                # Cache CIFAR models, always retrain Purchase-100
                if ds_name != 'purchase100' and os.path.exists(model_path):
                    log(f"  [CACHED] {ds_name} {model_type} seed={seed}")
                    # Still log metrics
                    model = get_model(ds_name)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                    test_loader = get_loader(test_ds, shuffle=False)
                    acc, _ = evaluate_model(model, test_loader)
                    training_log.append({
                        'model_id': f'{ds_name}_{model_type}_seed{seed}',
                        'dataset': ds_name, 'seed': seed, 'type': model_type,
                        'test_acc': float(acc), 'cached': True
                    })
                    continue

                set_seed(seed)
                log(f"  Training {ds_name} {model_type} seed={seed}...")
                t0 = time.time()

                if model_type == 'original':
                    train_indices = splits['non_ref_indices']
                else:
                    train_indices = splits['retain_indices']

                model = get_model(ds_name)
                train_loader = get_loader(train_ds, train_indices, shuffle=True)
                model = train_model(model, train_loader, ds_name, verbose=False)

                os.makedirs(f'{MODELS_DIR}/{model_type}', exist_ok=True)
                torch.save(model.state_dict(), model_path)

                test_loader = get_loader(test_ds, shuffle=False)
                acc, _ = evaluate_model(model, test_loader)
                elapsed = time.time() - t0
                log(f"    Test acc: {acc:.4f} ({elapsed:.0f}s)")

                training_log.append({
                    'model_id': f'{ds_name}_{model_type}_seed{seed}',
                    'dataset': ds_name, 'seed': seed, 'type': model_type,
                    'test_acc': float(acc), 'training_time_sec': elapsed
                })

    with open(f'{RESULTS_DIR}/model_training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)

# ============================================================
# PHASE 5: Run unlearning baselines
# ============================================================
def phase5_unlearning():
    log("PHASE 5: Running unlearning baselines")
    baselines_log = []

    for ds_name in DATASETS:
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)
        n_classes = 10 if ds_name == 'cifar10' else 100

        for seed in SEEDS:
            splits = create_splits(ds_name, train_ds, seed)

            for method in UNLEARN_METHODS:
                model_path = f'{MODELS_DIR}/unlearned/{method}/{ds_name}_seed{seed}.pt'

                # Cache CIFAR models
                if ds_name != 'purchase100' and os.path.exists(model_path):
                    log(f"  [CACHED] {ds_name} {method} seed={seed}")
                    model = get_model(ds_name)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                    test_loader = get_loader(test_ds, shuffle=False)
                    test_acc, _ = evaluate_model(model, test_loader)
                    forget_loader = get_loader(train_eval_ds, splits['forget_indices'], shuffle=False)
                    forget_acc, _ = evaluate_model(model, forget_loader)
                    retain_loader = get_loader(train_eval_ds, splits['retain_indices'][:5000], shuffle=False)
                    retain_acc, _ = evaluate_model(model, retain_loader)
                    baselines_log.append({
                        'method': method, 'dataset': ds_name, 'seed': seed,
                        'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
                        'forget_acc': float(forget_acc)
                    })
                    continue

                log(f"  Running {method} on {ds_name} seed={seed}...")
                set_seed(seed)

                # Load original model
                orig_model = get_model(ds_name)
                orig_model.load_state_dict(torch.load(
                    f'{MODELS_DIR}/original/{ds_name}_seed{seed}.pt',
                    map_location=DEVICE, weights_only=True))

                forget_loader = get_loader(train_ds, splits['forget_indices'], shuffle=True)
                retain_loader = get_loader(train_ds, splits['retain_indices'], shuffle=True)

                unlearned = run_unlearning(
                    orig_model, forget_loader, retain_loader,
                    method, ds_name, num_classes=n_classes
                )

                os.makedirs(f'{MODELS_DIR}/unlearned/{method}', exist_ok=True)
                torch.save(unlearned.state_dict(), model_path)

                # Evaluate
                test_loader = get_loader(test_ds, shuffle=False)
                test_acc, _ = evaluate_model(unlearned, test_loader)
                forget_eval_loader = get_loader(train_eval_ds, splits['forget_indices'], shuffle=False)
                forget_acc, _ = evaluate_model(unlearned, forget_eval_loader)
                retain_eval_loader = get_loader(train_eval_ds, splits['retain_indices'][:5000], shuffle=False)
                retain_acc, _ = evaluate_model(unlearned, retain_eval_loader)

                log(f"    TA={test_acc:.4f} FA={forget_acc:.4f} RA={retain_acc:.4f}")
                baselines_log.append({
                    'method': method, 'dataset': ds_name, 'seed': seed,
                    'retain_acc': float(retain_acc), 'test_acc': float(test_acc),
                    'forget_acc': float(forget_acc)
                })

    with open(f'{RESULTS_DIR}/unlearning_baselines.json', 'w') as f:
        json.dump(baselines_log, f, indent=2)

# ============================================================
# PHASE 6: MIA Evaluation (aggregate + stratified)
# ============================================================
def _load_ref_losses(ds_name, indices, split='train'):
    """Load reference model losses for given indices."""
    ref_losses = []
    for k in range(len(REF_SEEDS)):
        losses = np.load(f'{RESULTS_DIR}/ref_{split}_losses_{ds_name}_ref{k}.npy')
        ref_losses.append(losses[indices])
    return ref_losses

def _run_mia_for_model(model, ds_name, seed, method, variant, train_eval_ds, test_ds):
    """Run full MIA evaluation for a single model."""
    splits = create_splits(ds_name, train_eval_ds, seed)
    forget_indices = splits['forget_indices']

    # Non-members: equal-sized random subset of test set
    rng = np.random.RandomState(seed)
    n_forget = len(forget_indices)
    test_indices = sorted(rng.choice(len(test_ds), size=n_forget, replace=False).tolist())

    # Compute losses
    forget_loader = get_loader(train_eval_ds, forget_indices, shuffle=False, batch_size=BATCH_SIZE)
    test_subset_loader = get_loader(test_ds, test_indices, shuffle=False, batch_size=BATCH_SIZE)

    member_losses = compute_losses(model, forget_loader)
    nonmember_losses = compute_losses(model, test_subset_loader)

    # Reference model losses
    member_ref_list = _load_ref_losses(ds_name, forget_indices, 'train')
    nonmember_ref_list = _load_ref_losses(ds_name, test_indices, 'test')

    member_ref_mean = np.mean(member_ref_list, axis=0)
    nonmember_ref_mean = np.mean(nonmember_ref_list, axis=0)

    # Aggregate MIA
    agg_results = run_all_attacks(
        member_losses, nonmember_losses,
        member_ref_mean, nonmember_ref_mean,
        member_ref_list, nonmember_ref_list
    )
    agg_results.update({
        'method': method, 'dataset': ds_name, 'seed': seed, 'variant': variant
    })

    # Stratified MIA
    quintiles_train = np.load(f'{RESULTS_DIR}/quintiles_{ds_name}_train.npy')
    quintiles_test = np.load(f'{RESULTS_DIR}/quintiles_{ds_name}_test.npy')

    member_quintiles = quintiles_train[forget_indices]
    nonmember_quintiles = quintiles_test[test_indices]

    strat_results = stratified_mia(
        member_losses, nonmember_losses,
        member_ref_mean, nonmember_ref_mean,
        member_ref_list, nonmember_ref_list,
        member_quintiles, nonmember_quintiles
    )
    strat_results.update({
        'method': method, 'dataset': ds_name, 'seed': seed, 'variant': variant
    })

    return agg_results, strat_results

def phase6_mia_evaluation():
    log("PHASE 6: Running MIA evaluation on all models")
    agg_results_all = []
    strat_results_all = []

    for ds_name in DATASETS:
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)

        for seed in SEEDS:
            # Retrain
            model_path = f'{MODELS_DIR}/retrain/{ds_name}_seed{seed}.pt'
            if os.path.exists(model_path):
                model = get_model(ds_name)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                agg, strat = _run_mia_for_model(model, ds_name, seed, 'retrain', 'standard', train_eval_ds, test_ds)
                agg_results_all.append(agg)
                strat_results_all.append(strat)
                log(f"  retrain {ds_name} seed={seed}: agg_best={agg['best_auc']:.4f} WQ={strat['wq_auc']:.4f}")

            # Standard unlearning methods
            for method in UNLEARN_METHODS:
                model_path = f'{MODELS_DIR}/unlearned/{method}/{ds_name}_seed{seed}.pt'
                if not os.path.exists(model_path):
                    continue
                model = get_model(ds_name)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                agg, strat = _run_mia_for_model(model, ds_name, seed, method, 'standard', train_eval_ds, test_ds)
                agg_results_all.append(agg)
                strat_results_all.append(strat)
                log(f"  {method} {ds_name} seed={seed}: agg_best={agg['best_auc']:.4f} WQ={strat['wq_auc']:.4f} DG={strat['dg']:.4f}")

    with open(f'{RESULTS_DIR}/mia_aggregate.json', 'w') as f:
        json.dump(agg_results_all, f, indent=2)
    with open(f'{RESULTS_DIR}/mia_stratified.json', 'w') as f:
        json.dump(strat_results_all, f, indent=2)

    return agg_results_all, strat_results_all

# ============================================================
# PHASE 7: DAU + RUM Defense
# ============================================================
def phase7_dau_rum():
    log("PHASE 7: Running DAU and RUM defense experiments")

    for ds_name in DATASETS:
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)
        n_classes = 10 if ds_name == 'cifar10' else 100
        difficulty_train = np.load(f'{RESULTS_DIR}/difficulty_{ds_name}_train.npy')

        for seed in SEEDS:
            splits = create_splits(ds_name, train_ds, seed)
            forget_indices = splits['forget_indices']
            retain_indices = splits['retain_indices']

            # Load original model
            orig_model = get_model(ds_name)
            orig_model.load_state_dict(torch.load(
                f'{MODELS_DIR}/original/{ds_name}_seed{seed}.pt',
                map_location=DEVICE, weights_only=True))

            # DAU weights for forget set
            forget_difficulty = difficulty_train[forget_indices]
            dau_weights_forget = compute_dau_weights(forget_difficulty, alpha=DEFAULT_ALPHA)

            # DAU weights for retain set (for FT-DAU: upweight retain samples near hard forget classes)
            forget_labels = []
            for idx in forget_indices:
                if hasattr(train_ds, 'tensors'):
                    forget_labels.append(int(train_ds.tensors[1][idx]))
                else:
                    forget_labels.append(int(train_ds[idx][1]))
            forget_labels = np.array(forget_labels)

            # Weight retain samples: higher weight for classes that have hard forget samples
            class_difficulty = {}
            for i, idx in enumerate(forget_indices):
                c = forget_labels[i]
                if c not in class_difficulty:
                    class_difficulty[c] = []
                class_difficulty[c].append(forget_difficulty[i])
            class_mean_diff = {c: np.mean(v) for c, v in class_difficulty.items()}

            retain_weights = np.ones(len(retain_indices), dtype=np.float32)
            for i, idx in enumerate(retain_indices):
                if hasattr(train_ds, 'tensors'):
                    c = int(train_ds.tensors[1][idx])
                else:
                    c = int(train_ds[idx][1])
                if c in class_mean_diff:
                    # Upweight retain samples in classes with hard forget samples
                    d_mean = np.mean(list(class_mean_diff.values()))
                    d_std = np.std(list(class_mean_diff.values())) + 1e-8
                    retain_weights[i] = 1.0 + DEFAULT_ALPHA * (class_mean_diff[c] - d_mean) / d_std
            retain_weights = np.clip(retain_weights, 0.1, 10.0)
            dau_weights_retain = torch.tensor(retain_weights, dtype=torch.float32)

            forget_loader = get_loader(train_ds, forget_indices, shuffle=False)
            retain_loader = get_loader(train_ds, retain_indices, shuffle=False)

            for method in UNLEARN_METHODS:
                # --- DAU ---
                dau_path = f'{MODELS_DIR}/unlearned/{method}_dau/{ds_name}_seed{seed}.pt'
                os.makedirs(os.path.dirname(dau_path), exist_ok=True)

                log(f"  DAU: {method} on {ds_name} seed={seed}...")
                set_seed(seed)

                # For FT, pass retain weights; for others, pass forget weights
                if method == 'ft':
                    weights = dau_weights_retain
                else:
                    weights = dau_weights_forget

                unlearned = run_unlearning(
                    orig_model, forget_loader, retain_loader,
                    method, ds_name, sample_weights=weights, num_classes=n_classes
                )
                torch.save(unlearned.state_dict(), dau_path)

                # --- RUM (only for GA and SCRUB) ---
                if method in ('ga', 'scrub'):
                    rum_path = f'{MODELS_DIR}/unlearned/{method}_rum/{ds_name}_seed{seed}.pt'
                    os.makedirs(os.path.dirname(rum_path), exist_ok=True)

                    log(f"  RUM: {method} on {ds_name} seed={seed}...")
                    set_seed(seed)

                    groups = compute_rum_groups(forget_difficulty)
                    rum_epochs = {0: 3, 1: 5, 2: 8}  # easy/medium/hard

                    rum_model = copy.deepcopy(orig_model)
                    for g in range(3):
                        g_mask = groups == g
                        g_indices = [forget_indices[i] for i in range(len(forget_indices)) if g_mask[i]]
                        if len(g_indices) == 0:
                            continue
                        g_forget_loader = get_loader(train_ds, g_indices, shuffle=True)
                        rum_model = run_unlearning(
                            rum_model, g_forget_loader, retain_loader,
                            method, ds_name, num_classes=n_classes,
                            epochs=rum_epochs[g]
                        )
                    torch.save(rum_model.state_dict(), rum_path)

# ============================================================
# PHASE 8: MIA evaluation on DAU/RUM models
# ============================================================
def phase8_defense_mia():
    log("PHASE 8: MIA evaluation on DAU/RUM defense models")
    agg_results = []
    strat_results = []

    for ds_name in DATASETS:
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)

        for seed in SEEDS:
            for method in UNLEARN_METHODS:
                # DAU
                dau_path = f'{MODELS_DIR}/unlearned/{method}_dau/{ds_name}_seed{seed}.pt'
                if os.path.exists(dau_path):
                    model = get_model(ds_name)
                    model.load_state_dict(torch.load(dau_path, map_location=DEVICE, weights_only=True))
                    agg, strat = _run_mia_for_model(model, ds_name, seed, f'{method}_dau', 'dau', train_eval_ds, test_ds)
                    agg_results.append(agg)
                    strat_results.append(strat)
                    log(f"  {method}_dau {ds_name} seed={seed}: agg={agg['best_auc']:.4f} WQ={strat['wq_auc']:.4f}")

                # RUM
                if method in ('ga', 'scrub'):
                    rum_path = f'{MODELS_DIR}/unlearned/{method}_rum/{ds_name}_seed{seed}.pt'
                    if os.path.exists(rum_path):
                        model = get_model(ds_name)
                        model.load_state_dict(torch.load(rum_path, map_location=DEVICE, weights_only=True))
                        agg, strat = _run_mia_for_model(model, ds_name, seed, f'{method}_rum', 'rum', train_eval_ds, test_ds)
                        agg_results.append(agg)
                        strat_results.append(strat)
                        log(f"  {method}_rum {ds_name} seed={seed}: agg={agg['best_auc']:.4f} WQ={strat['wq_auc']:.4f}")

    with open(f'{RESULTS_DIR}/mia_aggregate_defense.json', 'w') as f:
        json.dump(agg_results, f, indent=2)
    with open(f'{RESULTS_DIR}/mia_stratified_defense.json', 'w') as f:
        json.dump(strat_results, f, indent=2)

# ============================================================
# PHASE 9: Ablation experiments
# ============================================================
def phase9_ablations():
    log("PHASE 9: Running ablation experiments")

    # --- Ablation 1: K sensitivity (CIFAR-10 only) ---
    log("  Ablation 1: K sensitivity")
    ablation_k = {}
    ds_name = 'cifar10'

    # Train extra reference models for K=8
    for k_extra, rs in enumerate(REF_SEEDS_EXTRA):
        k_idx = len(REF_SEEDS) + k_extra
        model_path = f'{MODELS_DIR}/reference/{ds_name}_ref{k_idx}.pt'
        if os.path.exists(model_path):
            log(f"    [CACHED] {ds_name} ref{k_idx}")
            continue

        log(f"    Training extra ref model {k_idx} (seed={rs})...")
        set_seed(rs)
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name)
        splits = create_splits(ds_name, train_ds, seed=rs)
        non_ref = splits['non_ref_indices']
        rng = np.random.RandomState(rs)
        train_indices = sorted(rng.choice(non_ref, size=int(0.8*len(non_ref)), replace=False).tolist())

        model = get_model(ds_name)
        train_loader = get_loader(train_ds, train_indices, shuffle=True)
        model = train_model(model, train_loader, ds_name, verbose=False)
        torch.save(model.state_dict(), model_path)

        # Compute losses
        train_eval_ds_loaded = load_dataset(ds_name)[2]
        train_loader_eval = get_loader(train_eval_ds_loaded, shuffle=False)
        test_loader = get_loader(test_ds, shuffle=False)

        train_losses = compute_losses(model, train_loader_eval)
        test_losses = compute_losses(model, test_loader)
        np.save(f'{RESULTS_DIR}/ref_train_losses_{ds_name}_ref{k_idx}.npy', train_losses)
        np.save(f'{RESULTS_DIR}/ref_test_losses_{ds_name}_ref{k_idx}.npy', test_losses)

    # Evaluate K=2,4,8
    train_ds, test_ds, train_eval_ds = load_dataset(ds_name)

    for K in [2, 4, 8]:
        # Compute difficulty with K models
        all_losses = []
        for k in range(K):
            losses = np.load(f'{RESULTS_DIR}/ref_train_losses_{ds_name}_ref{k}.npy')
            all_losses.append(losses)
        difficulty_K = np.mean(all_losses, axis=0)

        # Reference: K=8
        if K < 8:
            all_losses_8 = []
            for k in range(8):
                losses = np.load(f'{RESULTS_DIR}/ref_train_losses_{ds_name}_ref{k}.npy')
                all_losses_8.append(losses)
            difficulty_8 = np.mean(all_losses_8, axis=0)

            from scipy.stats import spearmanr
            rho, p = spearmanr(difficulty_K, difficulty_8)

            # Quintile stability
            pct_K = np.percentile(difficulty_K, [20,40,60,80])
            pct_8 = np.percentile(difficulty_8, [20,40,60,80])
            q_K = np.digitize(difficulty_K, pct_K)
            q_8 = np.digitize(difficulty_8, pct_8)
            stability = (q_K == q_8).mean()
        else:
            rho, stability = 1.0, 1.0

        # Run stratified MIA with this K's difficulty for GA and SCRUB (seed 42)
        seed = 42
        splits = create_splits(ds_name, train_ds, seed)
        pct_K_eval = np.percentile(difficulty_K, [20,40,60,80])
        quintiles_K = np.digitize(difficulty_K, pct_K_eval)

        difficulty_test_all = []
        for k in range(K):
            losses = np.load(f'{RESULTS_DIR}/ref_test_losses_{ds_name}_ref{k}.npy')
            difficulty_test_all.append(losses)
        difficulty_test_K = np.mean(difficulty_test_all, axis=0)
        pct_test_K = np.percentile(difficulty_test_K, [20,40,60,80])
        quintiles_test_K = np.digitize(difficulty_test_K, pct_test_K)

        k_results = {'K': K, 'spearman_rho': float(rho), 'quintile_stability': float(stability)}

        for method in ['ga', 'scrub']:
            model_path = f'{MODELS_DIR}/unlearned/{method}/{ds_name}_seed{seed}.pt'
            if not os.path.exists(model_path):
                continue
            model = get_model(ds_name)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

            forget_loader = get_loader(train_eval_ds, splits['forget_indices'], shuffle=False)
            rng = np.random.RandomState(seed)
            test_indices = sorted(rng.choice(len(test_ds), size=len(splits['forget_indices']), replace=False).tolist())
            test_sub_loader = get_loader(test_ds, test_indices, shuffle=False)

            member_losses = compute_losses(model, forget_loader)
            nonmember_losses = compute_losses(model, test_sub_loader)

            member_ref_list = [np.load(f'{RESULTS_DIR}/ref_train_losses_{ds_name}_ref{k}.npy')[splits['forget_indices']] for k in range(min(K, len(REF_SEEDS)))]
            nonmember_ref_list = [np.load(f'{RESULTS_DIR}/ref_test_losses_{ds_name}_ref{k}.npy')[test_indices] for k in range(min(K, len(REF_SEEDS)))]

            member_ref_mean = np.mean(member_ref_list, axis=0)
            nonmember_ref_mean = np.mean(nonmember_ref_list, axis=0)

            strat = stratified_mia(
                member_losses, nonmember_losses,
                member_ref_mean, nonmember_ref_mean,
                member_ref_list, nonmember_ref_list,
                quintiles_K[splits['forget_indices']], quintiles_test_K[test_indices]
            )
            k_results[f'{method}_wq_auc'] = strat['wq_auc']
            k_results[f'{method}_dg'] = strat['dg']

        ablation_k[K] = k_results
        log(f"    K={K}: rho={rho:.4f}, stability={stability:.4f}")

    with open(f'{RESULTS_DIR}/ablation_K.json', 'w') as f:
        json.dump(ablation_k, f, indent=2)

    # --- Ablation 2: Alpha sensitivity ---
    log("  Ablation 2: Alpha sensitivity")
    ablation_alpha = []

    for ds_name_ab in ['cifar10', 'cifar100']:
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name_ab)
        n_classes = 10 if ds_name_ab == 'cifar10' else 100
        difficulty_train = np.load(f'{RESULTS_DIR}/difficulty_{ds_name_ab}_train.npy')

        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for method in ['ga', 'scrub']:
                for seed in SEEDS:
                    # alpha=0.0 = uniform weights = standard baseline (skip if alpha=1.0 already done as DAU)
                    splits = create_splits(ds_name_ab, train_ds, seed)
                    forget_difficulty = difficulty_train[splits['forget_indices']]

                    if alpha == 0.0:
                        weights = torch.ones(len(splits['forget_indices']))
                    else:
                        weights = compute_dau_weights(forget_difficulty, alpha=alpha)

                    set_seed(seed)
                    orig_model = get_model(ds_name_ab)
                    orig_model.load_state_dict(torch.load(
                        f'{MODELS_DIR}/original/{ds_name_ab}_seed{seed}.pt',
                        map_location=DEVICE, weights_only=True))

                    forget_loader = get_loader(train_ds, splits['forget_indices'], shuffle=False)
                    retain_loader = get_loader(train_ds, splits['retain_indices'], shuffle=False)

                    unlearned = run_unlearning(
                        orig_model, forget_loader, retain_loader,
                        method, ds_name_ab, sample_weights=weights, num_classes=n_classes
                    )

                    # Evaluate
                    agg, strat = _run_mia_for_model(unlearned, ds_name_ab, seed, method, f'alpha_{alpha}', train_eval_ds, test_ds)

                    test_loader = get_loader(test_ds, shuffle=False)
                    retain_eval_loader = get_loader(train_eval_ds, splits['retain_indices'][:5000], shuffle=False)
                    ta, _ = evaluate_model(unlearned, test_loader)
                    ra, _ = evaluate_model(unlearned, retain_eval_loader)

                    ablation_alpha.append({
                        'dataset': ds_name_ab, 'method': method, 'seed': seed,
                        'alpha': alpha, 'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                        'agg_auc': agg['best_auc'], 'retain_acc': float(ra), 'test_acc': float(ta)
                    })

                log(f"    {ds_name_ab} {method} alpha={alpha}: done")

    with open(f'{RESULTS_DIR}/ablation_alpha.json', 'w') as f:
        json.dump(ablation_alpha, f, indent=2)

    # --- Ablation 3: Stratification granularity ---
    log("  Ablation 3: Strata granularity")
    ablation_strata = []
    ds_name = 'cifar10'
    seed = 42
    train_ds, test_ds, train_eval_ds = load_dataset(ds_name)
    splits = create_splits(ds_name, train_ds, seed)
    difficulty_train = np.load(f'{RESULTS_DIR}/difficulty_{ds_name}_train.npy')
    difficulty_test = np.load(f'{RESULTS_DIR}/difficulty_{ds_name}_test.npy')

    rng = np.random.RandomState(seed)
    test_indices = sorted(rng.choice(len(test_ds), size=len(splits['forget_indices']), replace=False).tolist())

    for n_strata in [3, 5, 10]:
        pcts = np.percentile(difficulty_train, [100*i/n_strata for i in range(1, n_strata)])
        q_train = np.digitize(difficulty_train, pcts)
        pcts_test = np.percentile(difficulty_test, [100*i/n_strata for i in range(1, n_strata)])
        q_test = np.digitize(difficulty_test, pcts_test)

        for method in UNLEARN_METHODS + ['retrain']:
            if method == 'retrain':
                model_path = f'{MODELS_DIR}/retrain/{ds_name}_seed{seed}.pt'
            else:
                model_path = f'{MODELS_DIR}/unlearned/{method}/{ds_name}_seed{seed}.pt'
            if not os.path.exists(model_path):
                continue

            model = get_model(ds_name)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

            forget_loader = get_loader(train_eval_ds, splits['forget_indices'], shuffle=False)
            test_sub_loader = get_loader(test_ds, test_indices, shuffle=False)

            member_losses = compute_losses(model, forget_loader)
            nonmember_losses = compute_losses(model, test_sub_loader)

            member_ref_list = _load_ref_losses(ds_name, splits['forget_indices'], 'train')
            nonmember_ref_list = _load_ref_losses(ds_name, test_indices, 'test')
            member_ref_mean = np.mean(member_ref_list, axis=0)
            nonmember_ref_mean = np.mean(nonmember_ref_list, axis=0)

            strat = stratified_mia(
                member_losses, nonmember_losses,
                member_ref_mean, nonmember_ref_mean,
                member_ref_list, nonmember_ref_list,
                q_train[splits['forget_indices']], q_test[test_indices],
                n_strata=n_strata
            )

            ablation_strata.append({
                'n_strata': n_strata, 'method': method,
                'wq_auc': strat['wq_auc'], 'dg': strat['dg'], 'max_spread': strat['max_spread']
            })

        log(f"    n_strata={n_strata}: done")

    with open(f'{RESULTS_DIR}/ablation_strata.json', 'w') as f:
        json.dump(ablation_strata, f, indent=2)

    # --- Ablation 4: Forget set size ---
    log("  Ablation 4: Forget set size")
    ablation_forget = []

    for ds_name_ab in ['cifar10', 'cifar100']:
        train_ds, test_ds, train_eval_ds = load_dataset(ds_name_ab)
        n_classes = 10 if ds_name_ab == 'cifar10' else 100
        difficulty_train = np.load(f'{RESULTS_DIR}/difficulty_{ds_name_ab}_train.npy')
        seed = 42

        for forget_size in [500, 1000, 2500]:
            splits = create_splits(ds_name_ab, train_ds, seed, forget_size=forget_size)

            for method in ['ga', 'scrub']:
                for variant_name, use_dau in [('standard', False), ('dau', True)]:
                    set_seed(seed)
                    orig_model = get_model(ds_name_ab)
                    orig_model.load_state_dict(torch.load(
                        f'{MODELS_DIR}/original/{ds_name_ab}_seed{seed}.pt',
                        map_location=DEVICE, weights_only=True))

                    forget_difficulty = difficulty_train[splits['forget_indices']]
                    weights = compute_dau_weights(forget_difficulty, alpha=1.0) if use_dau else None

                    forget_loader = get_loader(train_ds, splits['forget_indices'], shuffle=True if not use_dau else False)
                    retain_loader = get_loader(train_ds, splits['retain_indices'], shuffle=True if not use_dau else False)

                    unlearned = run_unlearning(
                        orig_model, forget_loader, retain_loader,
                        method, ds_name_ab, sample_weights=weights, num_classes=n_classes
                    )

                    # MIA eval
                    agg, strat = _run_mia_for_model(unlearned, ds_name_ab, seed, method, variant_name, train_eval_ds, test_ds)

                    ablation_forget.append({
                        'dataset': ds_name_ab, 'method': method, 'variant': variant_name,
                        'forget_size': forget_size, 'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                        'agg_auc': agg['best_auc']
                    })

            # Retrain for this forget size
            set_seed(seed)
            retrain_model = get_model(ds_name_ab)
            retrain_loader = get_loader(train_ds, splits['retain_indices'], shuffle=True)
            retrain_model = train_model(retrain_model, retrain_loader, ds_name_ab, verbose=False)

            agg, strat = _run_mia_for_model(retrain_model, ds_name_ab, seed, 'retrain', 'standard', train_eval_ds, test_ds)
            ablation_forget.append({
                'dataset': ds_name_ab, 'method': 'retrain', 'variant': 'standard',
                'forget_size': forget_size, 'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
                'agg_auc': agg['best_auc']
            })

            log(f"    {ds_name_ab} forget_size={forget_size}: done")

    with open(f'{RESULTS_DIR}/ablation_forget_size.json', 'w') as f:
        json.dump(ablation_forget, f, indent=2)

    # --- Ablation 5: Random-weight control ---
    log("  Ablation 5: Random-weight control")
    ablation_random = []
    ds_name = 'cifar10'
    train_ds, test_ds, train_eval_ds = load_dataset(ds_name)
    difficulty_train = np.load(f'{RESULTS_DIR}/difficulty_{ds_name}_train.npy')

    for seed in SEEDS:
        splits = create_splits(ds_name, train_ds, seed)
        forget_difficulty = difficulty_train[splits['forget_indices']]

        # Random permutation of difficulty scores
        rng = np.random.RandomState(seed + 1000)
        random_difficulty = rng.permutation(forget_difficulty)
        random_weights = compute_dau_weights(random_difficulty, alpha=1.0)

        set_seed(seed)
        orig_model = get_model(ds_name)
        orig_model.load_state_dict(torch.load(
            f'{MODELS_DIR}/original/{ds_name}_seed{seed}.pt',
            map_location=DEVICE, weights_only=True))

        forget_loader = get_loader(train_ds, splits['forget_indices'], shuffle=False)
        retain_loader = get_loader(train_ds, splits['retain_indices'], shuffle=False)

        unlearned = run_unlearning(
            orig_model, forget_loader, retain_loader,
            'ga', ds_name, sample_weights=random_weights, num_classes=10
        )

        agg, strat = _run_mia_for_model(unlearned, ds_name, seed, 'ga_random', 'random', train_eval_ds, test_ds)
        ablation_random.append({
            'seed': seed, 'wq_auc': strat['wq_auc'], 'dg': strat['dg'],
            'agg_auc': agg['best_auc']
        })
        log(f"    Random-weight seed={seed}: WQ={strat['wq_auc']:.4f}")

    with open(f'{RESULTS_DIR}/ablation_random_weights.json', 'w') as f:
        json.dump(ablation_random, f, indent=2)

# ============================================================
# PHASE 10: Statistical analysis + results.json
# ============================================================
def phase10_analysis():
    log("PHASE 10: Statistical analysis and aggregation")
    from scipy import stats

    # Load all results
    mia_std = json.load(open(f'{RESULTS_DIR}/mia_aggregate.json'))
    strat_std = json.load(open(f'{RESULTS_DIR}/mia_stratified.json'))
    mia_def = json.load(open(f'{RESULTS_DIR}/mia_aggregate_defense.json'))
    strat_def = json.load(open(f'{RESULTS_DIR}/mia_stratified_defense.json'))
    baselines = json.load(open(f'{RESULTS_DIR}/unlearning_baselines.json'))

    # Merge all results
    all_agg = mia_std + mia_def
    all_strat = strat_std + strat_def

    # Build main results table: method × dataset → mean±std across seeds
    main_table = {}
    all_methods = ['retrain'] + UNLEARN_METHODS + [f'{m}_dau' for m in UNLEARN_METHODS] + ['ga_rum', 'scrub_rum']

    for ds_name in DATASETS:
        main_table[ds_name] = {}
        for method in all_methods:
            agg_entries = [e for e in all_agg if e['dataset'] == ds_name and e['method'] == method]
            strat_entries = [e for e in all_strat if e['dataset'] == ds_name and e['method'] == method]
            bl_entries = [e for e in baselines if e['dataset'] == ds_name and e['method'] == method]

            if not agg_entries:
                continue

            agg_aucs = [e['best_auc'] for e in agg_entries]
            wq_aucs = [e['wq_auc'] for e in strat_entries] if strat_entries else []
            dgs = [e['dg'] for e in strat_entries] if strat_entries else []

            # Utility metrics from baselines
            if bl_entries:
                ras = [e['retain_acc'] for e in bl_entries]
                tas = [e['test_acc'] for e in bl_entries]
            elif method.endswith('_dau') or method.endswith('_rum'):
                # Get utility from defense models
                base_method = method.replace('_dau', '').replace('_rum', '')
                # Re-evaluate utility (stored in strat results indirectly)
                ras, tas = [], []
            else:
                ras, tas = [], []

            entry = {
                'agg_auc': f"{np.mean(agg_aucs):.4f}±{np.std(agg_aucs):.4f}",
                'agg_auc_mean': float(np.mean(agg_aucs)),
                'agg_auc_std': float(np.std(agg_aucs)),
            }
            if wq_aucs:
                entry['wq_auc'] = f"{np.mean(wq_aucs):.4f}±{np.std(wq_aucs):.4f}"
                entry['wq_auc_mean'] = float(np.mean(wq_aucs))
                entry['wq_auc_std'] = float(np.std(wq_aucs))
            if dgs:
                entry['dg'] = f"{np.mean(dgs):.4f}±{np.std(dgs):.4f}"
                entry['dg_mean'] = float(np.mean(dgs))
                entry['dg_std'] = float(np.std(dgs))
            if ras:
                entry['retain_acc'] = f"{np.mean(ras):.4f}±{np.std(ras):.4f}"
            if tas:
                entry['test_acc'] = f"{np.mean(tas):.4f}±{np.std(tas):.4f}"

            main_table[ds_name][method] = entry

    # Statistical tests
    stat_tests = []

    for ds_name in DATASETS:
        for method in UNLEARN_METHODS:
            strat_entries = [e for e in all_strat if e['dataset'] == ds_name and e['method'] == method]
            if len(strat_entries) < 2:
                continue

            wq_aucs = [e['wq_auc'] for e in strat_entries]
            agg_aucs = [e['aggregate']['best_auc'] for e in strat_entries]
            dgs = [e['dg'] for e in strat_entries]

            # Test: WQ-AUC > Aggregate AUC
            if len(wq_aucs) >= 3:
                diff = np.array(wq_aucs) - np.array(agg_aucs)
                t_stat, p_val = stats.ttest_1samp(diff, 0, alternative='greater')
                stat_tests.append({
                    'test': 'WQ > Agg', 'dataset': ds_name, 'method': method,
                    'mean_diff': float(np.mean(diff)),
                    't_stat': float(t_stat), 'p_value': float(p_val),
                    'significant_005': bool(p_val < 0.05)
                })

            # Test: DG > 0
            if len(dgs) >= 3:
                t_stat, p_val = stats.ttest_1samp(dgs, 0, alternative='greater')
                stat_tests.append({
                    'test': 'DG > 0', 'dataset': ds_name, 'method': method,
                    'mean_dg': float(np.mean(dgs)),
                    't_stat': float(t_stat), 'p_value': float(p_val),
                    'significant_005': bool(p_val < 0.05)
                })

    # DAU effectiveness tests
    for ds_name in DATASETS:
        for method in ['ga', 'scrub']:
            std_entries = [e for e in all_strat if e['dataset'] == ds_name and e['method'] == method]
            dau_entries = [e for e in all_strat if e['dataset'] == ds_name and e['method'] == f'{method}_dau']

            if len(std_entries) >= 3 and len(dau_entries) >= 3:
                std_wq = sorted([e['wq_auc'] for e in std_entries])
                dau_wq = sorted([e['wq_auc'] for e in dau_entries])

                if len(std_wq) == len(dau_wq):
                    diff = np.array(std_wq) - np.array(dau_wq)
                    t_stat, p_val = stats.ttest_1samp(diff, 0, alternative='greater')
                    stat_tests.append({
                        'test': 'DAU reduces WQ', 'dataset': ds_name, 'method': method,
                        'mean_improvement': float(np.mean(diff)),
                        't_stat': float(t_stat), 'p_value': float(p_val),
                        'significant_005': bool(p_val < 0.05)
                    })

    with open(f'{RESULTS_DIR}/statistical_tests.json', 'w') as f:
        json.dump(stat_tests, f, indent=2)
    with open(f'{RESULTS_DIR}/main_table.json', 'w') as f:
        json.dump(main_table, f, indent=2)

    # Build final results.json
    results = {
        'main_results': main_table,
        'statistical_tests': stat_tests,
        'success_criteria': _check_success_criteria(all_strat, stat_tests),
        'ablations': {
            'K': json.load(open(f'{RESULTS_DIR}/ablation_K.json')),
            'alpha': 'see ablation_alpha.json',
            'strata': 'see ablation_strata.json',
            'forget_size': 'see ablation_forget_size.json',
            'random_weights': json.load(open(f'{RESULTS_DIR}/ablation_random_weights.json'))
        }
    }

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    log("  Results saved to results.json")

def _check_success_criteria(all_strat, stat_tests):
    criteria = {}

    # SC1: WQ-AUC > Agg AUC by >0.05 for ≥3/5 methods
    sc1_tests = [t for t in stat_tests if t['test'] == 'WQ > Agg']
    sc1_pass = sum(1 for t in sc1_tests if t['mean_diff'] > 0.05 and t['significant_005'])
    criteria['SC1_wq_gt_agg'] = {
        'passing': sc1_pass,
        'total': len(sc1_tests),
        'met': sc1_pass >= 9  # 3 methods × 3 datasets
    }

    # SC2: DG significantly > 0
    sc2_tests = [t for t in stat_tests if t['test'] == 'DG > 0']
    sc2_pass = sum(1 for t in sc2_tests if t['significant_005'])
    criteria['SC2_dg_significant'] = {
        'passing': sc2_pass,
        'total': len(sc2_tests),
        'met': sc2_pass == len(sc2_tests)
    }

    # SC3: DAU reduces WQ-AUC
    sc3_tests = [t for t in stat_tests if t['test'] == 'DAU reduces WQ']
    sc3_pass = sum(1 for t in sc3_tests if t['mean_improvement'] > 0)
    criteria['SC3_dau_improves'] = {
        'passing': sc3_pass,
        'total': len(sc3_tests),
        'details': [{
            'method': t['method'], 'dataset': t['dataset'],
            'improvement': t['mean_improvement'], 'p_value': t['p_value']
        } for t in sc3_tests]
    }

    return criteria

# ============================================================
# PHASE 11: Generate figures
# ============================================================
def phase11_figures():
    log("PHASE 11: Generating figures")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 12,
        'legend.fontsize': 9, 'figure.dpi': 300
    })
    sns.set_style('whitegrid')
    palette = sns.color_palette('colorblind', 8)

    # Load data
    strat_std = json.load(open(f'{RESULTS_DIR}/mia_stratified.json'))
    strat_def = json.load(open(f'{RESULTS_DIR}/mia_stratified_defense.json'))
    all_strat = strat_std + strat_def

    # --- Figure 1: Stratified MIA (main motivation) ---
    log("  Figure 1: Stratified MIA per quintile")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax_idx, ds_name in enumerate(DATASETS):
        ax = axes[ax_idx]
        methods_to_plot = ['retrain'] + UNLEARN_METHODS

        for m_idx, method in enumerate(methods_to_plot):
            entries = [e for e in strat_std if e['dataset'] == ds_name and e['method'] == method]
            if not entries:
                continue

            # Average across seeds
            q_aucs = []
            for q in range(5):
                vals = [e[f'q{q+1}']['best_auc'] for e in entries if f'q{q+1}' in e]
                q_aucs.append((np.mean(vals), np.std(vals)))

            means = [v[0] for v in q_aucs]
            stds = [v[1] for v in q_aucs]
            x = np.arange(5) + m_idx * 0.12

            ax.bar(x, means, width=0.1, yerr=stds, label=method.upper(),
                   color=palette[m_idx], capsize=2, alpha=0.85)

        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Perfect')
        ax.set_xticks(np.arange(5) + 0.3)
        ax.set_xticklabels(['Q1\n(Easy)', 'Q2', 'Q3', 'Q4', 'Q5\n(Hard)'])
        ax.set_title(ds_name.upper())
        ax.set_ylabel('MIA-AUC' if ax_idx == 0 else '')
        if ax_idx == 0:
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Difficulty-Dependent Privacy Gap in Machine Unlearning', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure1_stratified_mia.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/figure1_stratified_mia.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 2: Aggregate vs WQ scatter ---
    log("  Figure 2: Aggregate vs WQ-AUC scatter")
    fig, ax = plt.subplots(figsize=(7, 6))

    method_colors = {m: palette[i] for i, m in enumerate(UNLEARN_METHODS)}
    ds_markers = {'cifar10': 'o', 'cifar100': 's', 'purchase100': 'D'}

    for entry in strat_std:
        if entry['method'] == 'retrain':
            continue
        method = entry['method']
        ds = entry['dataset']
        agg_auc = entry['aggregate']['best_auc']
        wq_auc = entry['wq_auc']
        ax.scatter(agg_auc, wq_auc, color=method_colors.get(method, 'gray'),
                  marker=ds_markers.get(ds, 'o'), s=60, alpha=0.8)

    lims = [0.45, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Aggregate MIA-AUC')
    ax.set_ylabel('Worst-Quintile MIA-AUC')
    ax.set_title('Aggregate Metrics Underestimate Privacy Risk')

    # Custom legends
    from matplotlib.lines import Line2D
    method_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=method_colors[m],
                             markersize=8, label=m.upper()) for m in UNLEARN_METHODS]
    ds_handles = [Line2D([0],[0], marker=ds_markers[d], color='w', markerfacecolor='gray',
                         markersize=8, label=d) for d in DATASETS]
    ax.legend(handles=method_handles + ds_handles + [Line2D([0],[0], linestyle='--', color='black', label='y=x')],
              fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure2_aggregate_vs_wq.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/figure2_aggregate_vs_wq.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 3: DAU defense effectiveness ---
    log("  Figure 3: DAU defense comparison")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax_idx, ds_name in enumerate(DATASETS):
        ax = axes[ax_idx]
        methods = ['ga', 'scrub']
        variants = ['standard', 'dau', 'rum']
        variant_labels = ['Standard', 'DAU', 'RUM']
        variant_colors = [palette[0], palette[2], palette[4]]

        x = np.arange(len(methods))
        width = 0.25

        for v_idx, (variant, vlabel, vcolor) in enumerate(zip(variants, variant_labels, variant_colors)):
            means, stds = [], []
            for method in methods:
                if variant == 'standard':
                    entries = [e for e in all_strat if e['dataset'] == ds_name and e['method'] == method and e.get('variant') == 'standard']
                elif variant == 'dau':
                    entries = [e for e in all_strat if e['dataset'] == ds_name and e['method'] == f'{method}_dau']
                else:
                    entries = [e for e in all_strat if e['dataset'] == ds_name and e['method'] == f'{method}_rum']

                if entries:
                    wqs = [e['wq_auc'] for e in entries]
                    means.append(np.mean(wqs))
                    stds.append(np.std(wqs))
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + v_idx * width, means, width, yerr=stds, label=vlabel,
                   color=vcolor, capsize=3, alpha=0.85)

        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_title(ds_name.upper())
        ax.set_ylabel('Worst-Quintile AUC' if ax_idx == 0 else '')
        if ax_idx == 0:
            ax.legend()

    fig.suptitle('DAU Defense: Worst-Quintile AUC Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure3_dau_defense.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/figure3_dau_defense.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 4: Alpha sensitivity ---
    log("  Figure 4: Alpha sensitivity")
    ablation_alpha = json.load(open(f'{RESULTS_DIR}/ablation_alpha.json'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, ds_name in enumerate(['cifar10', 'cifar100']):
        ax = axes[ax_idx]
        ax2 = ax.twinx()

        for m_idx, method in enumerate(['ga', 'scrub']):
            entries = [e for e in ablation_alpha if e['dataset'] == ds_name and e['method'] == method]
            if not entries:
                continue

            alphas_all = sorted(set(e['alpha'] for e in entries))
            wq_means, wq_stds, ra_means = [], [], []

            for alpha in alphas_all:
                a_entries = [e for e in entries if e['alpha'] == alpha]
                wqs = [e['wq_auc'] for e in a_entries]
                ras = [e['retain_acc'] for e in a_entries]
                wq_means.append(np.mean(wqs))
                wq_stds.append(np.std(wqs))
                ra_means.append(np.mean(ras))

            color = palette[m_idx * 2]
            ax.errorbar(alphas_all, wq_means, yerr=wq_stds, marker='o',
                       label=f'{method.upper()} WQ-AUC', color=color, linewidth=2)
            ax2.plot(alphas_all, ra_means, marker='s', linestyle='--',
                    label=f'{method.upper()} RA', color=color, alpha=0.6)

        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('α (DAU strength)')
        ax.set_ylabel('WQ-AUC (↓ better)')
        ax2.set_ylabel('Retain Accuracy (↑ better)')
        ax.set_title(ds_name.upper())
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure4_alpha_sensitivity.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/figure4_alpha_sensitivity.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 5: Ablation summary ---
    log("  Figure 5: Ablation summary")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) K vs metrics
    ax = axes[0, 0]
    ablation_k = json.load(open(f'{RESULTS_DIR}/ablation_K.json'))
    Ks = sorted([int(k) for k in ablation_k.keys()])
    rhos = [ablation_k[str(k)]['spearman_rho'] for k in Ks]
    stabs = [ablation_k[str(k)]['quintile_stability'] for k in Ks]
    ax.plot(Ks, rhos, 'o-', label='Spearman ρ', color=palette[0], linewidth=2)
    ax.plot(Ks, stabs, 's--', label='Quintile stability', color=palette[2], linewidth=2)
    ax.set_xlabel('K (# reference models)')
    ax.set_ylabel('Correlation / Stability')
    ax.set_title('(a) Reference Model Count')
    ax.legend()
    ax.set_ylim([0.7, 1.05])

    # (b) Strata granularity
    ax = axes[0, 1]
    ablation_strata = json.load(open(f'{RESULTS_DIR}/ablation_strata.json'))
    for m_idx, method in enumerate(['ga', 'scrub']):
        for n_strata in [3, 5, 10]:
            entries = [e for e in ablation_strata if e['method'] == method and e['n_strata'] == n_strata]
            if entries:
                ax.bar(n_strata + m_idx*0.8 - 0.4, entries[0]['wq_auc'], 0.7,
                       label=f'{method.upper()}' if n_strata == 3 else '',
                       color=palette[m_idx*2], alpha=0.85)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Number of Strata')
    ax.set_ylabel('Worst-Stratum AUC')
    ax.set_title('(b) Stratification Granularity')
    ax.legend()

    # (c) Forget set size vs DG
    ax = axes[1, 0]
    ablation_forget = json.load(open(f'{RESULTS_DIR}/ablation_forget_size.json'))
    for m_idx, method in enumerate(['ga', 'scrub']):
        for variant in ['standard', 'dau']:
            entries = [e for e in ablation_forget
                      if e['method'] == method and e['variant'] == variant and e['dataset'] == 'cifar10']
            if entries:
                sizes = sorted(set(e['forget_size'] for e in entries))
                dgs = [next(e['dg'] for e in entries if e['forget_size'] == s) for s in sizes]
                ls = '-' if variant == 'standard' else '--'
                ax.plot(sizes, dgs, f'o{ls}', label=f'{method.upper()}-{variant.upper()}',
                       color=palette[m_idx*2], linewidth=2)
    ax.set_xlabel('Forget Set Size')
    ax.set_ylabel('Difficulty Gap (DG)')
    ax.set_title('(c) Forget Set Size Effect')
    ax.legend(fontsize=8)

    # (d) Random-weight control
    ax = axes[1, 1]
    ablation_random = json.load(open(f'{RESULTS_DIR}/ablation_random_weights.json'))

    # Get standard GA and DAU GA WQ-AUCs for CIFAR-10
    std_wqs = [e['wq_auc'] for e in all_strat if e['dataset'] == 'cifar10' and e['method'] == 'ga']
    dau_wqs = [e['wq_auc'] for e in all_strat if e['dataset'] == 'cifar10' and e['method'] == 'ga_dau']
    rand_wqs = [e['wq_auc'] for e in ablation_random]

    x = [0, 1, 2]
    means = [np.mean(std_wqs) if std_wqs else 0,
             np.mean(rand_wqs) if rand_wqs else 0,
             np.mean(dau_wqs) if dau_wqs else 0]
    stds_plot = [np.std(std_wqs) if std_wqs else 0,
            np.std(rand_wqs) if rand_wqs else 0,
            np.std(dau_wqs) if dau_wqs else 0]
    colors = [palette[0], palette[4], palette[2]]

    ax.bar(x, means, yerr=stds_plot, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard\nGA', 'Random\nWeights', 'DAU\n(Difficulty)'])
    ax.set_ylabel('WQ-AUC')
    ax.set_title('(d) DAU Specificity Control')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure5_ablations.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/figure5_ablations.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 6: Difficulty distribution ---
    log("  Figure 6: Difficulty distribution")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax_idx, ds_name in enumerate(DATASETS):
        ax = axes[ax_idx]
        diff = np.load(f'{RESULTS_DIR}/difficulty_{ds_name}_train.npy')
        quintiles = np.load(f'{RESULTS_DIR}/quintiles_{ds_name}_train.npy')

        for q in range(5):
            mask = quintiles == q
            ax.hist(diff[mask], bins=30, alpha=0.5, label=f'Q{q+1}', color=palette[q],
                   density=True)

        ax.set_xlabel('Difficulty Score (Avg CE Loss)')
        ax.set_ylabel('Density' if ax_idx == 0 else '')
        ax.set_title(ds_name.upper())
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Per-Sample Difficulty Score Distribution', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure6_difficulty_distribution.pdf', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/figure6_difficulty_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Table 1: Main results (LaTeX) ---
    log("  Generating LaTeX tables")
    main_table = json.load(open(f'{RESULTS_DIR}/main_table.json'))

    latex = "\\begin{table}[t]\n\\centering\n\\caption{Main Results: Aggregate vs.\\ Stratified MIA Evaluation and Defense Effectiveness}\n"
    latex += "\\label{tab:main_results}\n\\resizebox{\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{l" + "ccccc" * len(DATASETS) + "}\n\\toprule\n"

    # Header
    header = "Method"
    for ds in DATASETS:
        header += f" & \\multicolumn{{5}}{{c}}{{{ds.upper()}}}"
    header += " \\\\\n"

    subheader = ""
    for _ in DATASETS:
        subheader += " & Agg AUC & WQ-AUC & DG & RA & TA"
    subheader += " \\\\\n\\midrule\n"

    latex += header + "\\cmidrule(lr){2-6}" + "".join(f"\\cmidrule(lr){{{2+5*i}-{6+5*i}}}" for i in range(1, len(DATASETS))) + "\n"
    latex += subheader

    display_methods = ['retrain'] + UNLEARN_METHODS + ['ga_dau', 'scrub_dau', 'ga_rum', 'scrub_rum']
    method_names = {
        'retrain': 'Retrain', 'ft': 'Fine-Tune', 'ga': 'Grad. Ascent',
        'rl': 'Random Labels', 'scrub': 'SCRUB', 'neggrad': 'NegGrad+KD',
        'ga_dau': 'GA-DAU', 'scrub_dau': 'SCRUB-DAU',
        'ga_rum': 'GA-RUM', 'scrub_rum': 'SCRUB-RUM'
    }

    for method in display_methods:
        row = method_names.get(method, method)
        for ds in DATASETS:
            entry = main_table.get(ds, {}).get(method, {})
            row += f" & {entry.get('agg_auc', '-')}"
            row += f" & {entry.get('wq_auc', '-')}"
            row += f" & {entry.get('dg', '-')}"
            row += f" & {entry.get('retain_acc', '-')}"
            row += f" & {entry.get('test_acc', '-')}"
        row += " \\\\\n"

        if method == 'retrain' or method == 'neggrad':
            row += "\\midrule\n"

        latex += row

    latex += "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"

    with open(f'{FIGURES_DIR}/table1_main_results.tex', 'w') as f:
        f.write(latex)

    log("  All figures generated!")

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    t_start = time.time()

    phase0_verify()
    phase1_data()
    phase2_reference_models()
    phase3_difficulty_scores()
    phase4_original_retrain()
    phase5_unlearning()
    phase6_mia_evaluation()
    phase7_dau_rum()
    phase8_defense_mia()
    phase9_ablations()
    phase10_analysis()
    phase11_figures()

    elapsed = (time.time() - t_start) / 60
    log(f"COMPLETE! Total time: {elapsed:.1f} minutes")
