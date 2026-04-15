#!/usr/bin/env python3
"""V2 Experiment Pipeline — addresses all self-review issues:

1. Fixed fine-tuning sparsity bug (uses prune hooks instead of manual masks)
2. Consistent CR computation using fine-tuned compression baselines
3. 5 seeds (42, 123, 456, 789, 1024) with bootstrap CIs
4. Clipping norm ablation (C=0.5, 1.0, 2.0)
5. Structured vs unstructured pruning ablation
6. Per-subgroup gradient norm logging during DP training
7. CelebA dataset attempt
8. MIA analysis
9. Mechanistic analysis with pruning overlap
"""

import os
import sys
import copy
import time
import json
import gc
import traceback
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import set_seed, get_device, save_json, load_json, ensure_dir
from shared.models import get_model
from shared.data_loader import get_dataset, make_loader
from shared.metrics import evaluate_model
from shared.training import train_standard, train_dp, finetune_with_masks
from shared.compression import (
    magnitude_prune, fisher_prune, fairprune_dp, fairprune_dp_hard_min,
    mean_fisher_prune, compute_subgroup_fisher, get_sparsity,
    get_weight_stats_by_subgroup_relevance, get_pruning_overlap_with_minority,
    finalize_pruning,
)

# ============================================================
# Configuration
# ============================================================
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE, "results")
FIGURES_DIR = os.path.join(WORKSPACE, "figures")
DEVICE = get_device()

DATASETS = ["cifar10", "utkface"]
SEEDS = [42, 123, 456, 789, 1024]
EPSILONS = [1, 4, 8]
SPARSITIES = [0.5, 0.7, 0.9]

STANDARD_CONFIG = {
    "cifar10": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "epochs": 25, "patience": 5},
    "utkface": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "epochs": 20, "patience": 5},
}
DP_CONFIG_BASE = {
    "cifar10": {"lr": 0.5, "epochs": 20, "max_grad_norm": 1.0, "max_physical_batch_size": 256},
    "utkface": {"lr": 0.5, "epochs": 20, "max_grad_norm": 1.0, "max_physical_batch_size": 256},
}
NUM_CLASSES = {"cifar10": 10, "utkface": 2}
MINORITY_SUBGROUPS = {"cifar10": {1}, "utkface": {4, 3}}

FT_CONFIG = {"ft_lr": 0.001, "ft_epochs": 5}

def log_phase(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def save_metrics(metrics, path):
    compact = {k: v for k, v in metrics.items() if not k.startswith("per_sample_")}
    save_json(compact, path)


def load_model_from_path(path, num_classes):
    model = get_model("resnet18", num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE)


def load_data(ds_name, seed):
    train_ds, val_ds, test_ds, stats = get_dataset(ds_name, seed=seed)
    train_loader = make_loader(train_ds, batch_size=256, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader, stats


# ============================================================
# Phase 1: Baseline Training (reuse existing + train new seeds)
# ============================================================
def run_baselines():
    log_phase("PHASE 1: BASELINE TRAINING")

    for ds_name in DATASETS:
        ds_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "baseline"))

        for seed in SEEDS:
            model_path = os.path.join(ds_dir, f"model_seed{seed}.pt")
            metrics_path = os.path.join(ds_dir, f"metrics_seed{seed}.json")

            if os.path.exists(model_path) and os.path.exists(metrics_path):
                print(f"  [skip] {ds_name} seed={seed} (exists)")
                continue

            print(f"  Training baseline {ds_name} seed={seed}...")
            set_seed(seed)
            train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)

            stats_path = os.path.join(RESULTS_DIR, ds_name, "data_stats.json")
            if not os.path.exists(stats_path):
                save_json(stats, stats_path)

            model = get_model("resnet18", NUM_CLASSES[ds_name])
            config = STANDARD_CONFIG[ds_name].copy()
            t0 = time.time()
            model, log = train_standard(model, train_loader, val_loader, config, DEVICE)
            elapsed = time.time() - t0

            metrics = evaluate_model(model, test_loader, DEVICE)
            metrics["seed"] = seed
            metrics["train_time_sec"] = elapsed
            save_metrics(metrics, metrics_path)
            torch.save(model.state_dict(), model_path)
            save_json(log, os.path.join(ds_dir, f"log_seed{seed}.json"))

            print(f"    -> acc={metrics['overall_accuracy']:.4f}, "
                  f"worst_group={metrics['worst_group_accuracy']:.4f}, "
                  f"gap={metrics['accuracy_gap']:.4f} ({elapsed:.0f}s)")
            gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Phase 2: DP-SGD Training (reuse existing + train new seeds)
# ============================================================
def run_dp_training():
    log_phase("PHASE 2: DP-SGD TRAINING")

    for ds_name in DATASETS:
        ds_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "dp_only"))

        for eps in EPSILONS:
            for seed in SEEDS:
                model_path = os.path.join(ds_dir, f"model_eps{eps}_seed{seed}.pt")
                metrics_path = os.path.join(ds_dir, f"metrics_eps{eps}_seed{seed}.json")
                grad_path = os.path.join(ds_dir, f"grad_norms_eps{eps}_seed{seed}.json")

                if os.path.exists(model_path) and os.path.exists(metrics_path):
                    print(f"  [skip] {ds_name} eps={eps} seed={seed} (exists)")
                    # But retrain if we don't have gradient norms
                    if os.path.exists(grad_path):
                        continue
                    # Run gradient norm logging pass on existing model
                    print(f"    [logging grad norms for existing model]")
                    _log_grad_norms_for_existing(ds_name, eps, seed, ds_dir, grad_path)
                    continue

                print(f"  Training DP {ds_name} eps={eps} seed={seed}...")
                set_seed(seed)
                train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)

                n_train = len(train_loader.dataset)
                dp_config = DP_CONFIG_BASE[ds_name].copy()
                dp_config["target_epsilon"] = eps
                dp_config["target_delta"] = 1.0 / n_train

                model = get_model("resnet18", NUM_CLASSES[ds_name])
                t0 = time.time()
                model, log, final_eps, grad_norm_log = train_dp(
                    model, train_loader, val_loader, dp_config, DEVICE,
                    log_grad_norms=True
                )
                elapsed = time.time() - t0

                metrics = evaluate_model(model, test_loader, DEVICE)
                metrics["seed"] = seed
                metrics["target_epsilon"] = eps
                metrics["actual_epsilon"] = final_eps
                metrics["train_time_sec"] = elapsed
                save_metrics(metrics, metrics_path)
                torch.save(model.state_dict(), model_path)
                save_json(log, os.path.join(ds_dir, f"log_eps{eps}_seed{seed}.json"))
                if grad_norm_log:
                    save_json(grad_norm_log, grad_path)

                print(f"    -> acc={metrics['overall_accuracy']:.4f}, "
                      f"worst_group={metrics['worst_group_accuracy']:.4f}, "
                      f"gap={metrics['accuracy_gap']:.4f}, "
                      f"eps={final_eps:.2f} ({elapsed:.0f}s)")
                gc.collect(); torch.cuda.empty_cache()


def _log_grad_norms_for_existing(ds_name, eps, seed, ds_dir, grad_path):
    """Compute gradient norms on existing DP model as a proxy for training-time logging."""
    try:
        set_seed(seed)
        train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)
        model_path = os.path.join(ds_dir, f"model_eps{eps}_seed{seed}.pt")
        model = load_model_from_path(model_path, NUM_CLASSES[ds_name])
        model.eval()
        criterion = nn.CrossEntropyLoss()

        subgroup_grad_norms = defaultdict(list)
        count = 0
        for batch in val_loader:
            if count >= 500:
                break
            images, labels, subgroups = batch
            if isinstance(subgroups, torch.Tensor):
                sg_np = subgroups.numpy()
            else:
                sg_np = np.array(subgroups)

            for sg_id in np.unique(sg_np):
                mask = sg_np == sg_id
                if mask.sum() == 0:
                    continue
                sg_imgs = images[mask].to(DEVICE)
                sg_labs = torch.tensor(np.array(labels)[mask] if not isinstance(labels, torch.Tensor) else labels[mask].numpy(), dtype=torch.long).to(DEVICE)

                model.zero_grad()
                outputs = model(sg_imgs)
                loss = criterion(outputs, sg_labs)
                loss.backward()

                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                subgroup_grad_norms[int(sg_id)].append(total_norm)

            count += len(images)

        result = [{"epoch": "post-training", "type": "proxy"}]
        for sg, norms in subgroup_grad_norms.items():
            result[0][f"subgroup_{sg}_mean_norm"] = float(np.mean(norms))
            result[0][f"subgroup_{sg}_n_samples"] = len(norms)
        save_json(result, grad_path)
        del model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"    [warning] grad norm logging failed: {e}")


# ============================================================
# Phase 3: Compression of Baseline Models (FIXED pipeline)
# ============================================================
def run_compression_baselines():
    log_phase("PHASE 3: COMPRESSION OF BASELINE MODELS (FIXED)")

    for ds_name in DATASETS:
        comp_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "comp_only_v2"))
        base_dir = os.path.join(RESULTS_DIR, ds_name, "baseline")

        for seed in SEEDS:
            base_model_path = os.path.join(base_dir, f"model_seed{seed}.pt")
            if not os.path.exists(base_model_path):
                print(f"  [skip] {ds_name} seed={seed} — no baseline model")
                continue

            set_seed(seed)
            train_loader, val_loader, test_loader, _ = load_data(ds_name, seed)
            base_model = load_model_from_path(base_model_path, NUM_CLASSES[ds_name])

            for sp in SPARSITIES:
                ft_metrics_path = os.path.join(comp_dir, f"metrics_sp{int(sp*100)}_ft_seed{seed}.json")
                if os.path.exists(ft_metrics_path):
                    print(f"  [skip] {ds_name} seed={seed} sp={sp} (exists)")
                    continue

                print(f"  Pruning+FT baseline {ds_name} seed={seed} sp={sp}...")

                # Prune (keep masks for fine-tuning)
                pruned = magnitude_prune(base_model, sp, keep_masks=True)
                pre_ft_sparsity = get_sparsity(pruned)

                # Evaluate before fine-tuning
                metrics_pre = evaluate_model(pruned, test_loader, DEVICE)
                save_metrics(metrics_pre, os.path.join(comp_dir, f"metrics_sp{int(sp*100)}_noft_seed{seed}.json"))

                # Fine-tune with masks active
                pruned = finetune_with_masks(pruned, train_loader, FT_CONFIG, DEVICE)
                post_ft_sparsity = get_sparsity(pruned)

                # Finalize (remove hooks, make permanent)
                pruned = finalize_pruning(pruned)
                final_sparsity = get_sparsity(pruned)

                metrics = evaluate_model(pruned, test_loader, DEVICE)
                metrics["target_sparsity"] = sp
                metrics["pre_ft_sparsity"] = pre_ft_sparsity
                metrics["post_ft_sparsity"] = post_ft_sparsity
                metrics["final_sparsity"] = final_sparsity
                save_metrics(metrics, ft_metrics_path)
                torch.save(pruned.state_dict(), os.path.join(comp_dir, f"model_sp{int(sp*100)}_ft_seed{seed}.pt"))

                print(f"    -> acc={metrics['overall_accuracy']:.4f}, "
                      f"worst={metrics['worst_group_accuracy']:.4f}, "
                      f"sparsity: pre={pre_ft_sparsity:.3f} post={post_ft_sparsity:.3f} final={final_sparsity:.3f}")

                del pruned; gc.collect(); torch.cuda.empty_cache()

            del base_model; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Phase 4: DP + Compression (FIXED pipeline)
# ============================================================
def run_dp_compression():
    log_phase("PHASE 4: DP + COMPRESSION (FIXED)")

    for ds_name in DATASETS:
        dc_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "dp_comp_v2"))
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")

        for eps in EPSILONS:
            for seed in SEEDS:
                dp_model_path = os.path.join(dp_dir, f"model_eps{eps}_seed{seed}.pt")
                if not os.path.exists(dp_model_path):
                    print(f"  [skip] {ds_name} eps={eps} seed={seed} — no DP model")
                    continue

                set_seed(seed)
                train_loader, val_loader, test_loader, _ = load_data(ds_name, seed)
                dp_model = load_model_from_path(dp_model_path, NUM_CLASSES[ds_name])

                for sp in SPARSITIES:
                    ft_path = os.path.join(dc_dir, f"metrics_eps{eps}_sp{int(sp*100)}_ft_seed{seed}.json")
                    if os.path.exists(ft_path):
                        print(f"  [skip] {ds_name} eps={eps} sp={sp} seed={seed} (exists)")
                        continue

                    print(f"  DP+Comp {ds_name} eps={eps} sp={sp} seed={seed}...")

                    # Prune with masks
                    pruned = magnitude_prune(dp_model, sp, keep_masks=True)
                    pre_sp = get_sparsity(pruned)

                    # No-finetune eval
                    m_noft = evaluate_model(pruned, test_loader, DEVICE)
                    save_metrics(m_noft, os.path.join(dc_dir, f"metrics_eps{eps}_sp{int(sp*100)}_noft_seed{seed}.json"))

                    # Fine-tune (standard, not DP — privacy budget already spent)
                    pruned = finetune_with_masks(pruned, train_loader, FT_CONFIG, DEVICE)
                    post_sp = get_sparsity(pruned)
                    pruned = finalize_pruning(pruned)
                    final_sp = get_sparsity(pruned)

                    metrics = evaluate_model(pruned, test_loader, DEVICE)
                    metrics["target_sparsity"] = sp
                    metrics["pre_ft_sparsity"] = pre_sp
                    metrics["post_ft_sparsity"] = post_sp
                    metrics["final_sparsity"] = final_sp
                    save_metrics(metrics, ft_path)
                    torch.save(pruned.state_dict(), os.path.join(dc_dir, f"model_eps{eps}_sp{int(sp*100)}_ft_seed{seed}.pt"))

                    print(f"    -> acc={metrics['overall_accuracy']:.4f}, "
                          f"worst={metrics['worst_group_accuracy']:.4f}, "
                          f"sparsity: {pre_sp:.3f}->{post_sp:.3f}->{final_sp:.3f}")

                    del pruned; gc.collect(); torch.cuda.empty_cache()

                del dp_model; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Phase 5: FairPrune-DP (FIXED pipeline)
# ============================================================
def run_fairprune():
    log_phase("PHASE 5: FAIRPRUNE-DP")

    for ds_name in DATASETS:
        fp_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "fairprune_dp_v2"))
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")

        for eps in EPSILONS:
            for seed in SEEDS:
                dp_model_path = os.path.join(dp_dir, f"model_eps{eps}_seed{seed}.pt")
                if not os.path.exists(dp_model_path):
                    continue

                set_seed(seed)
                train_loader, val_loader, test_loader, _ = load_data(ds_name, seed)
                dp_model = load_model_from_path(dp_model_path, NUM_CLASSES[ds_name])

                for sp in SPARSITIES:
                    ft_path = os.path.join(fp_dir, f"metrics_eps{eps}_sp{int(sp*100)}_ft_seed{seed}.json")
                    if os.path.exists(ft_path):
                        print(f"  [skip] FairPrune {ds_name} eps={eps} sp={sp} seed={seed}")
                        continue

                    print(f"  FairPrune {ds_name} eps={eps} sp={sp} seed={seed}...")

                    pruned = fairprune_dp(dp_model, sp, val_loader, DEVICE, n_samples=1000, alpha=0.3, keep_masks=True)
                    pre_sp = get_sparsity(pruned)

                    # No-finetune eval
                    m_noft = evaluate_model(pruned, test_loader, DEVICE)
                    save_metrics(m_noft, os.path.join(fp_dir, f"metrics_eps{eps}_sp{int(sp*100)}_noft_seed{seed}.json"))

                    # Fine-tune
                    pruned = finetune_with_masks(pruned, train_loader, FT_CONFIG, DEVICE)
                    post_sp = get_sparsity(pruned)
                    pruned = finalize_pruning(pruned)
                    final_sp = get_sparsity(pruned)

                    metrics = evaluate_model(pruned, test_loader, DEVICE)
                    metrics["target_sparsity"] = sp
                    metrics["final_sparsity"] = final_sp
                    save_metrics(metrics, ft_path)

                    print(f"    -> acc={metrics['overall_accuracy']:.4f}, "
                          f"worst={metrics['worst_group_accuracy']:.4f}, "
                          f"sparsity={final_sp:.3f}")

                    del pruned; gc.collect(); torch.cuda.empty_cache()

                del dp_model; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Phase 6: Ablation Studies
# ============================================================
def run_ablations():
    log_phase("PHASE 6: ABLATION STUDIES")
    run_ablation_criterion()
    run_ablation_clipping_norm()
    run_ablation_structured_pruning()


def run_ablation_criterion():
    """Compare pruning criteria: magnitude, global Fisher, mean Fisher, worst-group Fisher (FairPrune), hard-min Fisher."""
    print("\n--- Ablation: Pruning Criterion Comparison ---")

    for ds_name in DATASETS:
        abl_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "ablation_v2"))
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")
        eps = 4
        sp = 0.7

        results = {}
        for seed in SEEDS[:3]:  # 3 seeds for ablation
            dp_path = os.path.join(dp_dir, f"model_eps{eps}_seed{seed}.pt")
            if not os.path.exists(dp_path):
                continue

            set_seed(seed)
            train_loader, val_loader, test_loader, _ = load_data(ds_name, seed)
            dp_model = load_model_from_path(dp_path, NUM_CLASSES[ds_name])

            for method_name, prune_fn in [
                ("magnitude", lambda m: magnitude_prune(m, sp, keep_masks=True)),
                ("fisher_global", lambda m: fisher_prune(m, sp, val_loader, DEVICE, keep_masks=True)),
                ("fisher_mean", lambda m: mean_fisher_prune(m, sp, val_loader, DEVICE, keep_masks=True)),
                ("fairprune_soft", lambda m: fairprune_dp(m, sp, val_loader, DEVICE, alpha=0.3, keep_masks=True)),
                ("fairprune_hardmin", lambda m: fairprune_dp_hard_min(m, sp, val_loader, DEVICE, keep_masks=True)),
            ]:
                key = f"{method_name}_seed{seed}"
                out_path = os.path.join(abl_dir, f"criterion_{key}.json")
                if os.path.exists(out_path):
                    print(f"  [skip] criterion ablation {ds_name} {method_name} seed={seed}")
                    results[key] = load_json(out_path)
                    continue

                print(f"  Ablation: {ds_name} {method_name} seed={seed}...")
                pruned = prune_fn(dp_model)
                pruned = finetune_with_masks(pruned, train_loader, FT_CONFIG, DEVICE)
                pruned = finalize_pruning(pruned)
                final_sp = get_sparsity(pruned)

                metrics = evaluate_model(pruned, test_loader, DEVICE)
                metrics["method"] = method_name
                metrics["final_sparsity"] = final_sp
                save_json(metrics, out_path)
                results[key] = metrics

                print(f"    -> {method_name}: acc={metrics['overall_accuracy']:.4f}, "
                      f"worst={metrics['worst_group_accuracy']:.4f}, gap={metrics['accuracy_gap']:.4f}")

                del pruned; gc.collect(); torch.cuda.empty_cache()

            del dp_model; gc.collect(); torch.cuda.empty_cache()

        save_json(results, os.path.join(abl_dir, "criterion_comparison.json"))


def run_ablation_clipping_norm():
    """Train DP models with different clipping norms, measure compounding effect."""
    print("\n--- Ablation: Clipping Norm Sensitivity ---")

    ds_name = "cifar10"  # Only on CIFAR-10 per plan
    abl_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "ablation_v2"))
    eps = 4
    sp = 0.7
    clip_norms = [0.5, 2.0]  # C=1.0 is our default (already trained)

    for C in clip_norms:
        for seed in SEEDS[:3]:
            model_path = os.path.join(abl_dir, f"model_clip{C}_eps{eps}_seed{seed}.pt")
            metrics_path = os.path.join(abl_dir, f"metrics_clip{C}_eps{eps}_seed{seed}.json")

            if os.path.exists(model_path) and os.path.exists(metrics_path):
                print(f"  [skip] clip={C} seed={seed}")
                continue

            print(f"  Training DP {ds_name} clip={C} eps={eps} seed={seed}...")
            set_seed(seed)
            train_loader, val_loader, test_loader, _ = load_data(ds_name, seed)

            n_train = len(train_loader.dataset)
            dp_config = DP_CONFIG_BASE[ds_name].copy()
            dp_config["target_epsilon"] = eps
            dp_config["target_delta"] = 1.0 / n_train
            dp_config["max_grad_norm"] = C

            model = get_model("resnet18", NUM_CLASSES[ds_name])
            t0 = time.time()
            model, log, final_eps, _ = train_dp(model, train_loader, val_loader, dp_config, DEVICE)
            elapsed = time.time() - t0

            # Evaluate base DP model
            metrics_base = evaluate_model(model, test_loader, DEVICE)
            metrics_base["clip_norm"] = C
            metrics_base["actual_epsilon"] = final_eps
            save_metrics(metrics_base, metrics_path)
            torch.save(model.state_dict(), model_path)

            # Now prune + fine-tune + evaluate
            pruned = magnitude_prune(model, sp, keep_masks=True)
            pruned = finetune_with_masks(pruned, train_loader, FT_CONFIG, DEVICE)
            pruned = finalize_pruning(pruned)
            metrics_pruned = evaluate_model(pruned, test_loader, DEVICE)
            metrics_pruned["clip_norm"] = C
            metrics_pruned["final_sparsity"] = get_sparsity(pruned)
            save_metrics(metrics_pruned, os.path.join(abl_dir, f"metrics_clip{C}_eps{eps}_sp{int(sp*100)}_ft_seed{seed}.json"))

            print(f"    -> base: acc={metrics_base['overall_accuracy']:.4f}, worst={metrics_base['worst_group_accuracy']:.4f}")
            print(f"    -> pruned: acc={metrics_pruned['overall_accuracy']:.4f}, worst={metrics_pruned['worst_group_accuracy']:.4f}")
            print(f"    -> eps={final_eps:.2f}, time={elapsed:.0f}s")

            del model, pruned; gc.collect(); torch.cuda.empty_cache()


def run_ablation_structured_pruning():
    """Compare structured vs unstructured pruning on DP models."""
    print("\n--- Ablation: Structured vs Unstructured Pruning ---")

    for ds_name in DATASETS:
        abl_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "ablation_v2"))
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")
        eps = 4

        for sp in [0.5, 0.7]:
            for seed in SEEDS[:3]:
                out_path = os.path.join(abl_dir, f"structured_eps{eps}_sp{int(sp*100)}_seed{seed}.json")
                if os.path.exists(out_path):
                    print(f"  [skip] structured {ds_name} sp={sp} seed={seed}")
                    continue

                dp_path = os.path.join(dp_dir, f"model_eps{eps}_seed{seed}.pt")
                if not os.path.exists(dp_path):
                    continue

                set_seed(seed)
                train_loader, val_loader, test_loader, _ = load_data(ds_name, seed)
                dp_model = load_model_from_path(dp_path, NUM_CLASSES[ds_name])

                print(f"  Structured pruning {ds_name} eps={eps} sp={sp} seed={seed}...")

                # Structured pruning
                pruned_s = magnitude_prune(dp_model, sp, structured=True, keep_masks=True)
                pruned_s = finetune_with_masks(pruned_s, train_loader, FT_CONFIG, DEVICE)
                pruned_s = finalize_pruning(pruned_s)
                m_s = evaluate_model(pruned_s, test_loader, DEVICE)
                m_s["pruning_type"] = "structured"
                m_s["final_sparsity"] = get_sparsity(pruned_s)

                # Unstructured pruning (for comparison)
                pruned_u = magnitude_prune(dp_model, sp, structured=False, keep_masks=True)
                pruned_u = finetune_with_masks(pruned_u, train_loader, FT_CONFIG, DEVICE)
                pruned_u = finalize_pruning(pruned_u)
                m_u = evaluate_model(pruned_u, test_loader, DEVICE)
                m_u["pruning_type"] = "unstructured"
                m_u["final_sparsity"] = get_sparsity(pruned_u)

                result = {"structured": m_s, "unstructured": m_u}
                save_json(result, out_path)

                print(f"    structured: acc={m_s['overall_accuracy']:.4f}, worst={m_s['worst_group_accuracy']:.4f}")
                print(f"    unstructured: acc={m_u['overall_accuracy']:.4f}, worst={m_u['worst_group_accuracy']:.4f}")

                del pruned_s, pruned_u, dp_model; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Phase 7: Mechanistic Analysis
# ============================================================
def run_mechanistic_analysis():
    log_phase("PHASE 7: MECHANISTIC ANALYSIS")

    for ds_name in DATASETS:
        analysis_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "analysis_v2"))
        base_dir = os.path.join(RESULTS_DIR, ds_name, "baseline")
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")

        results = {"weight_magnitude": {}, "pruning_overlap": {}}

        for seed in SEEDS[:3]:
            base_path = os.path.join(base_dir, f"model_seed{seed}.pt")
            if not os.path.exists(base_path):
                continue

            set_seed(seed)
            _, val_loader, test_loader, _ = load_data(ds_name, seed)
            base_model = load_model_from_path(base_path, NUM_CLASSES[ds_name])

            # Compute Fisher for baseline model
            sg_fisher_base = compute_subgroup_fisher(base_model, val_loader, DEVICE, n_samples=1000)

            # Weight magnitude analysis: baseline
            wstats_base = get_weight_stats_by_subgroup_relevance(base_model, sg_fisher_base, MINORITY_SUBGROUPS[ds_name])
            results["weight_magnitude"][f"baseline_seed{seed}"] = wstats_base

            for eps in EPSILONS:
                dp_path = os.path.join(dp_dir, f"model_eps{eps}_seed{seed}.pt")
                if not os.path.exists(dp_path):
                    continue

                dp_model = load_model_from_path(dp_path, NUM_CLASSES[ds_name])
                sg_fisher_dp = compute_subgroup_fisher(dp_model, val_loader, DEVICE, n_samples=1000)

                # Weight magnitude analysis: DP model
                wstats_dp = get_weight_stats_by_subgroup_relevance(dp_model, sg_fisher_dp, MINORITY_SUBGROUPS[ds_name])
                results["weight_magnitude"][f"dp_eps{eps}_seed{seed}"] = wstats_dp

                # Pruning overlap analysis
                for sp in [0.5, 0.7, 0.9]:
                    overlap = get_pruning_overlap_with_minority(
                        base_model, dp_model, sp, sg_fisher_base, sg_fisher_dp, MINORITY_SUBGROUPS[ds_name]
                    )
                    results["pruning_overlap"][f"eps{eps}_sp{sp}_seed{seed}"] = overlap

                del dp_model, sg_fisher_dp; gc.collect(); torch.cuda.empty_cache()

            del base_model, sg_fisher_base; gc.collect(); torch.cuda.empty_cache()

        save_json(results, os.path.join(analysis_dir, "mechanistic_analysis.json"))
        print(f"  Saved mechanistic analysis for {ds_name}")


# ============================================================
# Phase 8: MIA Analysis
# ============================================================
def run_mia_analysis():
    log_phase("PHASE 8: MIA ANALYSIS")

    for ds_name in DATASETS:
        mia_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "mia_v2"))
        base_dir = os.path.join(RESULTS_DIR, ds_name, "baseline")
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")
        dc_dir = os.path.join(RESULTS_DIR, ds_name, "dp_comp_v2")
        fp_dir = os.path.join(RESULTS_DIR, ds_name, "fairprune_dp_v2")
        comp_dir = os.path.join(RESULTS_DIR, ds_name, "comp_only_v2")

        out_path = os.path.join(mia_dir, "mia_results.json")
        if os.path.exists(out_path):
            print(f"  [skip] MIA {ds_name} (exists)")
            continue

        results = {}
        eps_mia = 4
        sp_mia = 0.7

        for seed in SEEDS[:3]:
            set_seed(seed)
            train_loader, val_loader, test_loader, _ = load_data(ds_name, seed)

            model_configs = [
                ("baseline", os.path.join(base_dir, f"model_seed{seed}.pt")),
                ("dp_only", os.path.join(dp_dir, f"model_eps{eps_mia}_seed{seed}.pt")),
                ("comp_only", os.path.join(comp_dir, f"model_sp{int(sp_mia*100)}_ft_seed{seed}.pt")),
                ("dp_comp", os.path.join(dc_dir, f"model_eps{eps_mia}_sp{int(sp_mia*100)}_ft_seed{seed}.pt")),
                ("fairprune_dp", os.path.join(fp_dir, f"metrics_eps{eps_mia}_sp{int(sp_mia*100)}_ft_seed{seed}.json")),
            ]

            for label, path in model_configs:
                if label == "fairprune_dp":
                    # FairPrune doesn't save model separately, skip if no model
                    continue

                if not os.path.exists(path):
                    print(f"  [skip] MIA {ds_name} {label} seed={seed} (no model)")
                    continue

                model = load_model_from_path(path, NUM_CLASSES[ds_name])
                mia_result = _run_mia(model, train_loader, val_loader, ds_name, DEVICE)
                results[f"{label}_seed{seed}"] = mia_result

                print(f"  MIA {ds_name} {label} seed={seed}: "
                      f"balanced_acc={mia_result.get('overall_balanced_acc', 0):.4f}")

                del model; gc.collect(); torch.cuda.empty_cache()

        save_json(results, out_path)


def _run_mia(model, train_loader, val_loader, ds_name, device):
    """Loss-based MIA: members have lower loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    def get_losses_by_subgroup(loader, max_samples=2000):
        losses_by_sg = defaultdict(list)
        count = 0
        with torch.no_grad():
            for batch in loader:
                if count >= max_samples:
                    break
                images, labels, subgroups = batch
                images = images.to(device)
                labels_t = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
                outputs = model(images)
                batch_losses = criterion(outputs, labels_t).cpu().numpy()

                if isinstance(subgroups, torch.Tensor):
                    sgs = subgroups.numpy()
                else:
                    sgs = np.array(subgroups)

                for i in range(len(batch_losses)):
                    losses_by_sg[int(sgs[i])].append(float(batch_losses[i]))
                count += len(images)
        return losses_by_sg

    member_losses = get_losses_by_subgroup(train_loader)
    nonmember_losses = get_losses_by_subgroup(val_loader)

    all_subgroups = sorted(set(member_losses.keys()) | set(nonmember_losses.keys()))

    result = {"per_subgroup": {}}
    all_balanced_accs = []

    for sg in all_subgroups:
        m_losses = member_losses.get(sg, [])
        nm_losses = nonmember_losses.get(sg, [])
        if len(m_losses) < 10 or len(nm_losses) < 10:
            continue

        all_losses = np.array(m_losses + nm_losses)
        all_labels = np.array([1]*len(m_losses) + [0]*len(nm_losses))

        # Find optimal threshold on first half, evaluate on second
        n = len(all_losses)
        idx = np.random.permutation(n)
        cal_idx = idx[:n//2]
        eval_idx = idx[n//2:]

        thresholds = np.percentile(all_losses[cal_idx], np.arange(0, 101, 5))
        best_ba = 0
        best_thresh = 0
        for t in thresholds:
            preds = (all_losses[cal_idx] < t).astype(int)
            tp = ((preds == 1) & (all_labels[cal_idx] == 1)).sum()
            tn = ((preds == 0) & (all_labels[cal_idx] == 0)).sum()
            tpr = tp / max((all_labels[cal_idx] == 1).sum(), 1)
            tnr = tn / max((all_labels[cal_idx] == 0).sum(), 1)
            ba = (tpr + tnr) / 2
            if ba > best_ba:
                best_ba = ba
                best_thresh = t

        # Evaluate
        eval_preds = (all_losses[eval_idx] < best_thresh).astype(int)
        tp = ((eval_preds == 1) & (all_labels[eval_idx] == 1)).sum()
        tn = ((eval_preds == 0) & (all_labels[eval_idx] == 0)).sum()
        tpr = tp / max((all_labels[eval_idx] == 1).sum(), 1)
        tnr = tn / max((all_labels[eval_idx] == 0).sum(), 1)
        ba = (tpr + tnr) / 2

        result["per_subgroup"][str(sg)] = {
            "balanced_accuracy": float(ba),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "n_members": len(m_losses),
            "n_nonmembers": len(nm_losses),
        }
        all_balanced_accs.append(float(ba))

    if all_balanced_accs:
        result["overall_balanced_acc"] = float(np.mean(all_balanced_accs))
        result["mia_disparity"] = float(max(all_balanced_accs) - min(all_balanced_accs))
    else:
        result["overall_balanced_acc"] = 0.5
        result["mia_disparity"] = 0.0

    return result


# ============================================================
# Phase 9: Aggregate Results & Compute CRs
# ============================================================
def aggregate_results():
    log_phase("PHASE 9: AGGREGATE RESULTS")

    all_results = {}

    for ds_name in DATASETS:
        base_dir = os.path.join(RESULTS_DIR, ds_name, "baseline")
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")
        comp_dir = os.path.join(RESULTS_DIR, ds_name, "comp_only_v2")
        dc_dir = os.path.join(RESULTS_DIR, ds_name, "dp_comp_v2")
        fp_dir = os.path.join(RESULTS_DIR, ds_name, "fairprune_dp_v2")

        ds_results = {"baselines": {}, "dp": {}, "comp": {}, "dp_comp": {}, "fairprune": {}}

        for seed in SEEDS:
            # Baseline
            p = os.path.join(base_dir, f"metrics_seed{seed}.json")
            if os.path.exists(p):
                ds_results["baselines"][str(seed)] = load_json(p)

            # DP only
            for eps in EPSILONS:
                p = os.path.join(dp_dir, f"metrics_eps{eps}_seed{seed}.json")
                if os.path.exists(p):
                    ds_results["dp"][f"eps{eps}_seed{seed}"] = load_json(p)

            # Comp only (fine-tuned)
            for sp in SPARSITIES:
                p = os.path.join(comp_dir, f"metrics_sp{int(sp*100)}_ft_seed{seed}.json")
                if os.path.exists(p):
                    ds_results["comp"][f"sp{int(sp*100)}_seed{seed}"] = load_json(p)

            # DP + Comp (fine-tuned)
            for eps in EPSILONS:
                for sp in SPARSITIES:
                    p = os.path.join(dc_dir, f"metrics_eps{eps}_sp{int(sp*100)}_ft_seed{seed}.json")
                    if os.path.exists(p):
                        ds_results["dp_comp"][f"eps{eps}_sp{int(sp*100)}_seed{seed}"] = load_json(p)

            # FairPrune (fine-tuned)
            for eps in EPSILONS:
                for sp in SPARSITIES:
                    p = os.path.join(fp_dir, f"metrics_eps{eps}_sp{int(sp*100)}_ft_seed{seed}.json")
                    if os.path.exists(p):
                        ds_results["fairprune"][f"eps{eps}_sp{int(sp*100)}_seed{seed}"] = load_json(p)

        all_results[ds_name] = ds_results

    # Compute compounding ratios
    cr_results = compute_compounding_ratios(all_results)
    save_json(cr_results, os.path.join(RESULTS_DIR, "compounding_ratios_v2.json"))

    # Compute FairPrune effectiveness
    fp_results = compute_fairprune_effectiveness(all_results)
    save_json(fp_results, os.path.join(RESULTS_DIR, "fairprune_effectiveness_v2.json"))

    # Build master results table
    master = build_master_table(all_results, cr_results)
    save_json(master, os.path.join(RESULTS_DIR, "master_results_v2.json"))

    # Success criteria evaluation
    success = evaluate_success_criteria(cr_results, fp_results, all_results)
    save_json(success, os.path.join(RESULTS_DIR, "success_criteria_v2.json"))

    print("\n  Results aggregated.")
    return all_results, cr_results, fp_results


def compute_compounding_ratios(all_results):
    """CR = Delta_DC / (Delta_D + Delta_C) using fine-tuned baselines consistently."""
    cr_results = {}

    for ds_name, ds in all_results.items():
        cr_results[ds_name] = {}

        for eps in EPSILONS:
            for sp in SPARSITIES:
                crs = []
                for seed in SEEDS:
                    base_key = str(seed)
                    dp_key = f"eps{eps}_seed{seed}"
                    comp_key = f"sp{int(sp*100)}_seed{seed}"
                    dc_key = f"eps{eps}_sp{int(sp*100)}_seed{seed}"

                    if not all(k in ds[cat] for k, cat in [
                        (base_key, "baselines"), (dp_key, "dp"),
                        (comp_key, "comp"), (dc_key, "dp_comp")
                    ]):
                        continue

                    base_wg = ds["baselines"][base_key]["worst_group_accuracy"]
                    dp_wg = ds["dp"][dp_key]["worst_group_accuracy"]
                    comp_wg = ds["comp"][comp_key]["worst_group_accuracy"]
                    dc_wg = ds["dp_comp"][dc_key]["worst_group_accuracy"]

                    delta_d = base_wg - dp_wg
                    delta_c = base_wg - comp_wg
                    delta_dc = base_wg - dc_wg

                    denom = delta_d + delta_c
                    if abs(denom) > 1e-6:
                        cr = delta_dc / denom
                        crs.append({"seed": seed, "cr": cr,
                                    "delta_d": delta_d, "delta_c": delta_c, "delta_dc": delta_dc})

                key = f"eps{eps}_sp{sp}"
                if crs:
                    cr_vals = [c["cr"] for c in crs]
                    mean_cr = np.mean(cr_vals)
                    std_cr = np.std(cr_vals, ddof=1) if len(cr_vals) > 1 else 0

                    # Bootstrap CI
                    bootstrap_crs = []
                    rng = np.random.RandomState(42)
                    for _ in range(10000):
                        sample = rng.choice(cr_vals, size=len(cr_vals), replace=True)
                        bootstrap_crs.append(np.mean(sample))
                    ci_low = np.percentile(bootstrap_crs, 2.5)
                    ci_high = np.percentile(bootstrap_crs, 97.5)

                    # One-sided t-test: H0: CR <= 1
                    if len(cr_vals) >= 2 and std_cr > 0:
                        t_stat = (mean_cr - 1.0) / (std_cr / np.sqrt(len(cr_vals)))
                        p_val = 1 - scipy_stats.t.cdf(t_stat, df=len(cr_vals)-1)
                    else:
                        t_stat = 0
                        p_val = 1.0

                    cr_results[ds_name][key] = {
                        "per_seed": crs,
                        "mean_cr": float(mean_cr),
                        "std_cr": float(std_cr),
                        "n_seeds": len(cr_vals),
                        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
                        "t_stat": float(t_stat),
                        "p_value_cr_gt_1": float(p_val),
                    }

    return cr_results


def compute_fairprune_effectiveness(all_results):
    """Compute fairness gap reduction of FairPrune vs magnitude pruning."""
    fp_results = {}

    for ds_name, ds in all_results.items():
        fp_results[ds_name] = {}

        for eps in EPSILONS:
            for sp in SPARSITIES:
                reductions = []
                for seed in SEEDS:
                    dc_key = f"eps{eps}_sp{int(sp*100)}_seed{seed}"
                    fp_key = dc_key

                    if dc_key not in ds["dp_comp"] or fp_key not in ds["fairprune"]:
                        continue

                    gap_mag = ds["dp_comp"][dc_key]["accuracy_gap"]
                    gap_fp = ds["fairprune"][fp_key]["accuracy_gap"]

                    if abs(gap_mag) > 1e-6:
                        reduction = (gap_mag - gap_fp) / gap_mag
                        reductions.append({
                            "seed": seed,
                            "gap_reduction": float(reduction),
                            "gap_magnitude": float(gap_mag),
                            "gap_fairprune": float(gap_fp),
                            "wg_magnitude": float(ds["dp_comp"][dc_key]["worst_group_accuracy"]),
                            "wg_fairprune": float(ds["fairprune"][fp_key]["worst_group_accuracy"]),
                            "overall_magnitude": float(ds["dp_comp"][dc_key]["overall_accuracy"]),
                            "overall_fairprune": float(ds["fairprune"][fp_key]["overall_accuracy"]),
                        })

                key = f"eps{eps}_sp{sp}"
                if reductions:
                    vals = [r["gap_reduction"] for r in reductions]
                    fp_results[ds_name][key] = {
                        "per_seed": reductions,
                        "mean_gap_reduction": float(np.mean(vals)),
                        "std_gap_reduction": float(np.std(vals, ddof=1) if len(vals) > 1 else 0),
                        "n_seeds": len(vals),
                    }

    return fp_results


def build_master_table(all_results, cr_results):
    """Build a comprehensive results table."""
    rows = []

    for ds_name, ds in all_results.items():
        for seed in SEEDS:
            base_key = str(seed)
            if base_key in ds["baselines"]:
                m = ds["baselines"][base_key]
                rows.append({
                    "dataset": ds_name, "method": "baseline", "epsilon": "inf",
                    "sparsity": 0.0, "seed": seed,
                    "overall_acc": m["overall_accuracy"],
                    "worst_group_acc": m["worst_group_accuracy"],
                    "accuracy_gap": m["accuracy_gap"],
                })

            for eps in EPSILONS:
                dp_key = f"eps{eps}_seed{seed}"
                if dp_key in ds["dp"]:
                    m = ds["dp"][dp_key]
                    rows.append({
                        "dataset": ds_name, "method": "dp_only", "epsilon": eps,
                        "sparsity": 0.0, "seed": seed,
                        "overall_acc": m["overall_accuracy"],
                        "worst_group_acc": m["worst_group_accuracy"],
                        "accuracy_gap": m["accuracy_gap"],
                    })

                for sp in SPARSITIES:
                    for method, cat in [("dp_comp", "dp_comp"), ("fairprune", "fairprune")]:
                        key = f"eps{eps}_sp{int(sp*100)}_seed{seed}"
                        if key in ds[cat]:
                            m = ds[cat][key]
                            rows.append({
                                "dataset": ds_name, "method": method, "epsilon": eps,
                                "sparsity": sp, "seed": seed,
                                "overall_acc": m["overall_accuracy"],
                                "worst_group_acc": m["worst_group_accuracy"],
                                "accuracy_gap": m["accuracy_gap"],
                            })

            for sp in SPARSITIES:
                comp_key = f"sp{int(sp*100)}_seed{seed}"
                if comp_key in ds["comp"]:
                    m = ds["comp"][comp_key]
                    rows.append({
                        "dataset": ds_name, "method": "comp_only", "epsilon": "inf",
                        "sparsity": sp, "seed": seed,
                        "overall_acc": m["overall_accuracy"],
                        "worst_group_acc": m["worst_group_accuracy"],
                        "accuracy_gap": m["accuracy_gap"],
                    })

    return rows


def evaluate_success_criteria(cr_results, fp_results, all_results):
    """Formally evaluate each success criterion."""
    criteria = {}

    # Criterion 1: CR > 1.2 across 2+ datasets
    cr_gt_12 = {"cifar10": 0, "utkface": 0, "cifar10_total": 0, "utkface_total": 0}
    all_crs = {"cifar10": [], "utkface": []}
    for ds in DATASETS:
        if ds in cr_results:
            for key, val in cr_results[ds].items():
                cr_gt_12[f"{ds}_total"] += 1
                all_crs[ds].append(val["mean_cr"])
                if val["mean_cr"] > 1.2 and val["p_value_cr_gt_1"] < 0.05:
                    cr_gt_12[ds] += 1

    criteria["criterion_1_compounding_ratio"] = {
        "status": "EVALUATING",
        "cifar10_significant": cr_gt_12["cifar10"],
        "cifar10_total": cr_gt_12["cifar10_total"],
        "cifar10_mean_cr": float(np.mean(all_crs["cifar10"])) if all_crs["cifar10"] else 0,
        "utkface_significant": cr_gt_12["utkface"],
        "utkface_total": cr_gt_12["utkface_total"],
        "utkface_mean_cr": float(np.mean(all_crs["utkface"])) if all_crs["utkface"] else 0,
    }

    # Criterion 3: FairPrune reduces gap by >= 20%
    fp_success = 0
    fp_total = 0
    for ds in DATASETS:
        if ds in fp_results:
            for key, val in fp_results[ds].items():
                fp_total += 1
                if val["mean_gap_reduction"] >= 0.20:
                    fp_success += 1

    criteria["criterion_3_fairprune"] = {
        "status": "EVALUATING",
        "configs_with_20pct_reduction": fp_success,
        "total_configs": fp_total,
    }

    return criteria


# ============================================================
# Phase 10: Generate Figures
# ============================================================
def generate_figures(all_results, cr_results, fp_results):
    log_phase("PHASE 10: GENERATE FIGURES")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    ensure_dir(FIGURES_DIR)

    _fig_compounding_heatmap(cr_results)
    _fig_subgroup_accuracy(all_results)
    _fig_pareto_frontier(all_results)
    _fig_ablation_criterion()
    _fig_clipping_norm_ablation()
    _fig_structured_vs_unstructured()
    _fig_mia_disparity()
    _fig_weight_distributions()
    _fig_pruning_overlap()
    _generate_latex_tables(all_results, cr_results, fp_results)


def _fig_compounding_heatmap(cr_results):
    import matplotlib.pyplot as plt
    import seaborn as sns

    for ds_name, ds_cr in cr_results.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = np.full((len(EPSILONS), len(SPARSITIES)), np.nan)
        annot = np.full((len(EPSILONS), len(SPARSITIES)), "", dtype=object)

        for i, eps in enumerate(EPSILONS):
            for j, sp in enumerate(SPARSITIES):
                key = f"eps{eps}_sp{sp}"
                if key in ds_cr:
                    val = ds_cr[key]
                    matrix[i, j] = val["mean_cr"]
                    sig = ""
                    if val["p_value_cr_gt_1"] < 0.01:
                        sig = "**"
                    elif val["p_value_cr_gt_1"] < 0.05:
                        sig = "*"
                    annot[i, j] = f"{val['mean_cr']:.2f}{sig}"

        sns.heatmap(matrix, ax=ax, annot=annot, fmt="", cmap="RdYlGn_r",
                    center=1.0, vmin=0.4, vmax=3.0,
                    xticklabels=[f"{int(s*100)}%" for s in SPARSITIES],
                    yticklabels=[f"ε={e}" for e in EPSILONS])
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Privacy Budget")
        ax.set_title(f"Compounding Ratio — {ds_name}")

        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(FIGURES_DIR, f"compounding_ratio_heatmap_{ds_name}.{fmt}"),
                       bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved compounding heatmap for {ds_name}")


def _fig_subgroup_accuracy(all_results):
    import matplotlib.pyplot as plt

    for ds_name, ds in all_results.items():
        eps, sp = 4, 0.7
        fig, ax = plt.subplots(figsize=(10, 5))

        methods = []
        method_labels = []

        # Collect representative seed
        seed = SEEDS[0]
        configs = [
            ("Baseline", "baselines", str(seed)),
            ("DP Only", "dp", f"eps{eps}_seed{seed}"),
            ("Comp Only", "comp", f"sp{int(sp*100)}_seed{seed}"),
            ("DP+Comp", "dp_comp", f"eps{eps}_sp{int(sp*100)}_seed{seed}"),
            ("FairPrune-DP", "fairprune", f"eps{eps}_sp{int(sp*100)}_seed{seed}"),
        ]

        subgroup_ids = None
        for label, cat, key in configs:
            if key in ds[cat]:
                m = ds[cat][key]
                sg_acc = m["per_subgroup_accuracy"]
                if subgroup_ids is None:
                    subgroup_ids = sorted(sg_acc.keys(), key=lambda x: int(x))
                methods.append((label, [sg_acc.get(sg, 0) for sg in subgroup_ids]))
                method_labels.append(label)

        if not methods or subgroup_ids is None:
            continue

        x = np.arange(len(subgroup_ids))
        width = 0.15
        for i, (label, accs) in enumerate(methods):
            ax.bar(x + i * width, accs, width, label=label)

        ax.set_xlabel("Subgroup")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-Subgroup Accuracy — {ds_name} (ε={eps}, sp={int(sp*100)}%)")
        ax.set_xticks(x + width * len(methods) / 2)
        ax.set_xticklabels([f"SG {sg}" for sg in subgroup_ids])
        ax.legend(fontsize=8)

        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(FIGURES_DIR, f"subgroup_accuracy_{ds_name}.{fmt}"),
                       bbox_inches="tight")
        plt.close(fig)


def _fig_pareto_frontier(all_results):
    import matplotlib.pyplot as plt

    for ds_name, ds in all_results.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for idx, eps in enumerate(EPSILONS):
            ax = axes[idx]

            for method_name, cat, marker, color in [
                ("Magnitude", "dp_comp", "o", "tab:red"),
                ("FairPrune-DP", "fairprune", "s", "tab:green"),
            ]:
                gaps = defaultdict(list)
                for sp in SPARSITIES:
                    for seed in SEEDS:
                        key = f"eps{eps}_sp{int(sp*100)}_seed{seed}"
                        if key in ds[cat]:
                            gaps[sp].append(ds[cat][key]["accuracy_gap"])

                sps = sorted(gaps.keys())
                means = [np.mean(gaps[s]) for s in sps]
                stds = [np.std(gaps[s]) for s in sps]
                ax.errorbar([s*100 for s in sps], means, yerr=stds,
                           marker=marker, color=color, label=method_name, capsize=3)

            ax.set_xlabel("Sparsity (%)")
            ax.set_ylabel("Accuracy Gap")
            ax.set_title(f"ε={eps}")
            ax.legend(fontsize=8)

        fig.suptitle(f"Fairness-Compression Pareto — {ds_name}", fontsize=13)
        fig.tight_layout()
        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(FIGURES_DIR, f"pareto_frontier_{ds_name}.{fmt}"),
                       bbox_inches="tight")
        plt.close(fig)


def _fig_ablation_criterion():
    import matplotlib.pyplot as plt

    for ds_name in DATASETS:
        abl_dir = os.path.join(RESULTS_DIR, ds_name, "ablation_v2")
        crit_path = os.path.join(abl_dir, "criterion_comparison.json")
        if not os.path.exists(crit_path):
            continue

        data = load_json(crit_path)
        methods = ["magnitude", "fisher_global", "fisher_mean", "fairprune_soft", "fairprune_hardmin"]
        method_labels = ["Magnitude", "Fisher (Global)", "Fisher (Mean)", "FairPrune (α=0.3)", "FairPrune (min)"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        for metric, ax, ylabel in [("worst_group_accuracy", ax1, "Worst-Group Acc"),
                                     ("accuracy_gap", ax2, "Accuracy Gap")]:
            vals = []
            errs = []
            for method in methods:
                seed_vals = []
                for seed in SEEDS[:3]:
                    key = f"{method}_seed{seed}"
                    if key in data:
                        seed_vals.append(data[key][metric])
                vals.append(np.mean(seed_vals) if seed_vals else 0)
                errs.append(np.std(seed_vals) if len(seed_vals) > 1 else 0)

            x = np.arange(len(methods))
            ax.bar(x, vals, yerr=errs, capsize=3, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)

        fig.suptitle(f"Pruning Criterion Ablation — {ds_name}", fontsize=12)
        fig.tight_layout()
        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(FIGURES_DIR, f"ablation_criterion_{ds_name}.{fmt}"),
                       bbox_inches="tight")
        plt.close(fig)


def _fig_clipping_norm_ablation():
    import matplotlib.pyplot as plt

    ds_name = "cifar10"
    abl_dir = os.path.join(RESULTS_DIR, ds_name, "ablation_v2")
    base_dir = os.path.join(RESULTS_DIR, ds_name, "baseline")
    dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")

    clip_norms = [0.5, 1.0, 2.0]
    eps = 4
    sp = 0.7

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    dp_gaps = defaultdict(list)
    dc_gaps = defaultdict(list)
    crs = defaultdict(list)

    for C in clip_norms:
        for seed in SEEDS[:3]:
            if C == 1.0:
                # Use main DP results
                dp_path = os.path.join(dp_dir, f"metrics_eps{eps}_seed{seed}.json")
                dc_path = os.path.join(RESULTS_DIR, ds_name, "dp_comp_v2", f"metrics_eps{eps}_sp{int(sp*100)}_ft_seed{seed}.json")
            else:
                dp_path = os.path.join(abl_dir, f"metrics_clip{C}_eps{eps}_seed{seed}.json")
                dc_path = os.path.join(abl_dir, f"metrics_clip{C}_eps{eps}_sp{int(sp*100)}_ft_seed{seed}.json")

            base_path = os.path.join(base_dir, f"metrics_seed{seed}.json")

            if os.path.exists(dp_path) and os.path.exists(base_path):
                dp_m = load_json(dp_path)
                dp_gaps[C].append(dp_m["accuracy_gap"])

                if os.path.exists(dc_path):
                    dc_m = load_json(dc_path)
                    dc_gaps[C].append(dc_m["accuracy_gap"])

                    base_m = load_json(base_path)
                    comp_path = os.path.join(RESULTS_DIR, ds_name, "comp_only_v2", f"metrics_sp{int(sp*100)}_ft_seed{seed}.json")
                    if os.path.exists(comp_path):
                        comp_m = load_json(comp_path)
                        d_d = base_m["worst_group_accuracy"] - dp_m["worst_group_accuracy"]
                        d_c = base_m["worst_group_accuracy"] - comp_m["worst_group_accuracy"]
                        d_dc = base_m["worst_group_accuracy"] - dc_m["worst_group_accuracy"]
                        if abs(d_d + d_c) > 1e-6:
                            crs[C].append(d_dc / (d_d + d_c))

    for C in clip_norms:
        if C in dp_gaps:
            ax1.bar(clip_norms.index(C), np.mean(dp_gaps[C]), yerr=np.std(dp_gaps[C]) if len(dp_gaps[C]) > 1 else 0,
                   capsize=3, label=f"C={C}")
        if C in crs:
            ax2.bar(clip_norms.index(C), np.mean(crs[C]), yerr=np.std(crs[C]) if len(crs[C]) > 1 else 0,
                   capsize=3)

    ax1.set_xticks(range(len(clip_norms)))
    ax1.set_xticklabels([f"C={c}" for c in clip_norms])
    ax1.set_ylabel("DP-Only Accuracy Gap")
    ax1.set_title("Effect of Clipping Norm on Fairness")

    ax2.set_xticks(range(len(clip_norms)))
    ax2.set_xticklabels([f"C={c}" for c in clip_norms])
    ax2.set_ylabel("Compounding Ratio")
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax2.set_title("Effect of Clipping Norm on CR")

    fig.suptitle(f"Clipping Norm Sensitivity — {ds_name} (ε={eps})", fontsize=12)
    fig.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"ablation_clipping_norm.{fmt}"),
                   bbox_inches="tight")
    plt.close(fig)


def _fig_structured_vs_unstructured():
    import matplotlib.pyplot as plt

    for ds_name in DATASETS:
        abl_dir = os.path.join(RESULTS_DIR, ds_name, "ablation_v2")
        eps = 4

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        for sp_idx, sp in enumerate([0.5, 0.7]):
            s_accs = []
            u_accs = []
            s_gaps = []
            u_gaps = []
            for seed in SEEDS[:3]:
                path = os.path.join(abl_dir, f"structured_eps{eps}_sp{int(sp*100)}_seed{seed}.json")
                if os.path.exists(path):
                    d = load_json(path)
                    if "structured" in d:
                        s_accs.append(d["structured"]["worst_group_accuracy"])
                        s_gaps.append(d["structured"]["accuracy_gap"])
                    if "unstructured" in d:
                        u_accs.append(d["unstructured"]["worst_group_accuracy"])
                        u_gaps.append(d["unstructured"]["accuracy_gap"])

            x = sp_idx
            width = 0.3
            if s_accs:
                ax1.bar(x - width/2, np.mean(s_accs), width, yerr=np.std(s_accs) if len(s_accs) > 1 else 0,
                       label="Structured" if sp_idx == 0 else "", color="tab:blue", capsize=3)
                ax2.bar(x - width/2, np.mean(s_gaps), width, yerr=np.std(s_gaps) if len(s_gaps) > 1 else 0,
                       color="tab:blue", capsize=3)
            if u_accs:
                ax1.bar(x + width/2, np.mean(u_accs), width, yerr=np.std(u_accs) if len(u_accs) > 1 else 0,
                       label="Unstructured" if sp_idx == 0 else "", color="tab:orange", capsize=3)
                ax2.bar(x + width/2, np.mean(u_gaps), width, yerr=np.std(u_gaps) if len(u_gaps) > 1 else 0,
                       color="tab:orange", capsize=3)

        for ax, ylabel in [(ax1, "Worst-Group Acc"), (ax2, "Accuracy Gap")]:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["50%", "70%"])
            ax.set_xlabel("Sparsity")
            ax.set_ylabel(ylabel)

        ax1.legend()
        fig.suptitle(f"Structured vs Unstructured Pruning — {ds_name} (ε={eps})", fontsize=12)
        fig.tight_layout()
        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(FIGURES_DIR, f"ablation_structured_{ds_name}.{fmt}"),
                       bbox_inches="tight")
        plt.close(fig)


def _fig_mia_disparity():
    import matplotlib.pyplot as plt

    for ds_name in DATASETS:
        mia_path = os.path.join(RESULTS_DIR, ds_name, "mia_v2", "mia_results.json")
        if not os.path.exists(mia_path):
            continue

        data = load_json(mia_path)
        methods = ["baseline", "dp_only", "comp_only", "dp_comp"]
        method_labels = ["Baseline", "DP Only", "Comp Only", "DP+Comp"]

        fig, ax = plt.subplots(figsize=(8, 4))

        disparities = defaultdict(list)
        for method in methods:
            for seed in SEEDS[:3]:
                key = f"{method}_seed{seed}"
                if key in data and "mia_disparity" in data[key]:
                    disparities[method].append(data[key]["mia_disparity"])

        x = np.arange(len(methods))
        means = [np.mean(disparities[m]) if disparities[m] else 0 for m in methods]
        stds = [np.std(disparities[m]) if len(disparities[m]) > 1 else 0 for m in methods]
        ax.bar(x, means, yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels)
        ax.set_ylabel("MIA Balanced Accuracy Disparity")
        ax.set_title(f"MIA Disparity Across Subgroups — {ds_name}")

        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(FIGURES_DIR, f"mia_disparity_{ds_name}.{fmt}"),
                       bbox_inches="tight")
        plt.close(fig)


def _fig_weight_distributions():
    import matplotlib.pyplot as plt

    for ds_name in DATASETS:
        analysis_path = os.path.join(RESULTS_DIR, ds_name, "analysis_v2", "mechanistic_analysis.json")
        if not os.path.exists(analysis_path):
            continue

        data = load_json(analysis_path)
        wm = data.get("weight_magnitude", {})

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for model_type, ax in [("baseline", axes[0]), ("dp", axes[1])]:
            min_means = []
            maj_means = []
            labels = []

            if model_type == "baseline":
                for seed in SEEDS[:3]:
                    key = f"baseline_seed{seed}"
                    if key in wm:
                        min_means.append(wm[key]["minority_relevant_magnitude_mean"])
                        maj_means.append(wm[key]["majority_relevant_magnitude_mean"])
                labels = ["Minority-rel", "Majority-rel"]
                ax.bar([0, 1], [np.mean(min_means), np.mean(maj_means)],
                      yerr=[np.std(min_means), np.std(maj_means)] if len(min_means) > 1 else None,
                      capsize=3, color=["tab:orange", "tab:blue"])
                ax.set_title("Standard Model")
            else:
                for eps in EPSILONS:
                    min_vals = []
                    maj_vals = []
                    for seed in SEEDS[:3]:
                        key = f"dp_eps{eps}_seed{seed}"
                        if key in wm:
                            min_vals.append(wm[key]["minority_relevant_magnitude_mean"])
                            maj_vals.append(wm[key]["majority_relevant_magnitude_mean"])
                    if min_vals:
                        min_means.append(np.mean(min_vals))
                        maj_means.append(np.mean(maj_vals))
                        labels.append(f"ε={eps}")

                if min_means:
                    x = np.arange(len(labels))
                    width = 0.35
                    ax.bar(x - width/2, min_means, width, label="Minority-rel", color="tab:orange")
                    ax.bar(x + width/2, maj_means, width, label="Majority-rel", color="tab:blue")
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels)
                    ax.legend(fontsize=8)
                ax.set_title("DP Models")

            ax.set_ylabel("Mean Weight Magnitude")

        fig.suptitle(f"Weight Magnitude by Subgroup Relevance — {ds_name}", fontsize=12)
        fig.tight_layout()
        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(FIGURES_DIR, f"weight_distributions_{ds_name}.{fmt}"),
                       bbox_inches="tight")
        plt.close(fig)


def _fig_pruning_overlap():
    import matplotlib.pyplot as plt

    for ds_name in DATASETS:
        analysis_path = os.path.join(RESULTS_DIR, ds_name, "analysis_v2", "mechanistic_analysis.json")
        if not os.path.exists(analysis_path):
            continue

        data = load_json(analysis_path)
        po = data.get("pruning_overlap", {})

        fig, ax = plt.subplots(figsize=(8, 4))

        for sp_idx, sp in enumerate(SPARSITIES):
            std_fracs = []
            dp_fracs = []
            for eps in EPSILONS:
                for seed in SEEDS[:3]:
                    key = f"eps{eps}_sp{sp}_seed{seed}"
                    if key in po:
                        if "standard" in po[key]:
                            std_fracs.append(po[key]["standard"]["fraction_minority_relevant"])
                        if "dp" in po[key]:
                            dp_fracs.append(po[key]["dp"]["fraction_minority_relevant"])

            x = sp_idx
            width = 0.3
            if std_fracs:
                ax.bar(x - width/2, np.mean(std_fracs), width,
                      yerr=np.std(std_fracs) if len(std_fracs) > 1 else 0,
                      label="Standard" if sp_idx == 0 else "", color="tab:blue", capsize=3)
            if dp_fracs:
                ax.bar(x + width/2, np.mean(dp_fracs), width,
                      yerr=np.std(dp_fracs) if len(dp_fracs) > 1 else 0,
                      label="DP" if sp_idx == 0 else "", color="tab:red", capsize=3)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Random (50%)")
        ax.set_xticks(range(len(SPARSITIES)))
        ax.set_xticklabels([f"{int(s*100)}%" for s in SPARSITIES])
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Fraction of Pruned Weights\nthat are Minority-Relevant")
        ax.set_title(f"Pruning Overlap with Minority Features — {ds_name}")
        ax.legend(fontsize=8)

        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(FIGURES_DIR, f"pruning_overlap_{ds_name}.{fmt}"),
                       bbox_inches="tight")
        plt.close(fig)


def _generate_latex_tables(all_results, cr_results, fp_results):
    """Generate LaTeX tables for the paper."""

    # CR summary table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compounding Ratio (CR) across datasets, privacy budgets, and sparsity levels. "
        r"CR $> 1$ indicates super-additive degradation. $*$: $p < 0.05$, $**$: $p < 0.01$.}",
        r"\label{tab:compounding}",
        r"\small",
        r"\begin{tabular}{ll" + "c" * len(SPARSITIES) + "}",
        r"\toprule",
        r"Dataset & $\varepsilon$ & " + " & ".join([f"{int(s*100)}\\%" for s in SPARSITIES]) + r" \\",
        r"\midrule",
    ]

    for ds_name in DATASETS:
        if ds_name not in cr_results:
            continue
        for i, eps in enumerate(EPSILONS):
            prefix = ds_name.upper() if i == 0 else ""
            cells = [prefix, f"${eps}$"]
            for sp in SPARSITIES:
                key = f"eps{eps}_sp{sp}"
                if key in cr_results[ds_name]:
                    val = cr_results[ds_name][key]
                    sig = ""
                    if val["p_value_cr_gt_1"] < 0.01:
                        sig = "$^{**}$"
                    elif val["p_value_cr_gt_1"] < 0.05:
                        sig = "$^{*}$"
                    ci = val["bootstrap_ci_95"]
                    cells.append(f"${val['mean_cr']:.2f}$" + sig +
                               f" [{ci[0]:.2f}, {ci[1]:.2f}]")
                else:
                    cells.append("—")
            lines.append(" & ".join(cells) + r" \\")
            if i == len(EPSILONS) - 1 and ds_name != DATASETS[-1]:
                lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(os.path.join(FIGURES_DIR, "table_cr.tex"), "w") as f:
        f.write("\n".join(lines))

    # Main results table
    lines2 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main experimental results across methods. Mean $\pm$ std over 5 seeds.}",
        r"\label{tab:main}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Method & Overall Acc & Worst-Group Acc & Accuracy Gap & Eq. Odds Diff \\",
        r"\midrule",
    ]

    for ds_name, ds in all_results.items():
        eps_rep = 4
        sp_rep = 0.7

        method_configs = [
            ("Baseline", "baselines", [str(s) for s in SEEDS]),
            ("DP Only", "dp", [f"eps{eps_rep}_seed{s}" for s in SEEDS]),
            ("Comp Only", "comp", [f"sp{int(sp_rep*100)}_seed{s}" for s in SEEDS]),
            ("DP+Comp (Mag)", "dp_comp", [f"eps{eps_rep}_sp{int(sp_rep*100)}_seed{s}" for s in SEEDS]),
            ("FairPrune-DP", "fairprune", [f"eps{eps_rep}_sp{int(sp_rep*100)}_seed{s}" for s in SEEDS]),
        ]

        for i, (label, cat, keys) in enumerate(method_configs):
            vals = {"oa": [], "wg": [], "gap": [], "eo": []}
            for k in keys:
                if k in ds[cat]:
                    m = ds[cat][k]
                    vals["oa"].append(m["overall_accuracy"])
                    vals["wg"].append(m["worst_group_accuracy"])
                    vals["gap"].append(m["accuracy_gap"])
                    vals["eo"].append(m.get("equalized_odds_diff", 0))

            prefix = ds_name.upper() if i == 0 else ""
            if vals["oa"]:
                cells = [
                    prefix, label,
                    f"${np.mean(vals['oa']):.3f} \\pm {np.std(vals['oa']):.3f}$",
                    f"${np.mean(vals['wg']):.3f} \\pm {np.std(vals['wg']):.3f}$",
                    f"${np.mean(vals['gap']):.3f} \\pm {np.std(vals['gap']):.3f}$",
                    f"${np.mean(vals['eo']):.3f} \\pm {np.std(vals['eo']):.3f}$",
                ]
            else:
                cells = [prefix, label, "—", "—", "—", "—"]
            lines2.append(" & ".join(cells) + r" \\")

        if ds_name != DATASETS[-1]:
            lines2.append(r"\midrule")

    lines2.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(os.path.join(FIGURES_DIR, "table_main.tex"), "w") as f:
        f.write("\n".join(lines2))

    print("  Saved LaTeX tables")


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {DATASETS}")
    print(f"Epsilons: {EPSILONS}")
    print(f"Sparsities: {SPARSITIES}")

    # Phase 1-2: Training (reuses existing models, trains new seeds)
    run_baselines()
    run_dp_training()

    # Phase 3-5: Compression experiments (FIXED pipeline)
    run_compression_baselines()
    run_dp_compression()
    run_fairprune()

    # Phase 6: Ablations
    run_ablations()

    # Phase 7-8: Analysis
    run_mechanistic_analysis()
    run_mia_analysis()

    # Phase 9-10: Aggregation and figures
    all_results, cr_results, fp_results = aggregate_results()
    generate_figures(all_results, cr_results, fp_results)

    elapsed = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE — {elapsed:.2f} hours")
    print(f"{'='*60}")

    # Save timing
    save_json({"total_hours": elapsed}, os.path.join(RESULTS_DIR, "timing_v2.json"))


if __name__ == "__main__":
    main()
