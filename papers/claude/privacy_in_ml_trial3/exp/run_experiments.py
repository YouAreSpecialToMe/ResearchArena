#!/usr/bin/env python3
"""Complete experiment pipeline for 'The Compounding Cost' paper.

Runs all experiments end-to-end:
1. Baseline training (no DP, no compression)
2. DP-SGD training at varying epsilon
3. Compression of baseline models (magnitude pruning)
4. DP + Compression (central experiment - compounding ratio)
5. FairPrune-DP (proposed method)
6. Ablation studies
7. Mechanistic analysis
8. MIA analysis
9. Figure generation
10. Results aggregation
"""

import os
import sys
import json
import copy
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from scipy import stats as scipy_stats

# Add shared modules
sys.path.insert(0, os.path.dirname(__file__))
from shared.data_loader import get_dataset, make_loader
from shared.models import get_model
from shared.training import train_standard, train_dp, finetune_standard
from shared.metrics import evaluate_model
from shared.compression import (
    magnitude_prune, fisher_prune, fairprune_dp, mean_fisher_prune,
    get_sparsity, compute_subgroup_fisher, get_weight_stats_by_subgroup_relevance
)

# ============================================================
# Configuration
# ============================================================
WORKSPACE = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(WORKSPACE, "results")
FIGURES_DIR = os.path.join(WORKSPACE, "figures")

DATASETS = ["cifar10", "utkface"]
SEEDS = [42, 123, 456]
EPSILONS = [1, 4, 8]
SPARSITIES = [0.5, 0.7, 0.9]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configs
STANDARD_CONFIG = {
    "cifar10": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "epochs": 25, "patience": 5},
    "utkface": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "epochs": 20, "patience": 5},
}
DP_CONFIG_BASE = {
    "cifar10": {"lr": 0.5, "epochs": 20, "max_grad_norm": 1.0, "max_physical_batch_size": 256},
    "utkface": {"lr": 0.5, "epochs": 20, "max_grad_norm": 1.0, "max_physical_batch_size": 256},
}
NUM_CLASSES = {"cifar10": 10, "utkface": 2}

# Minority subgroup IDs for mechanistic analysis
MINORITY_SUBGROUPS = {
    "cifar10": {1},       # minority class group
    "utkface": {4, 3},    # Others and Indian (smallest groups)
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path):
    """Save dict as JSON, handling numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)


def save_metrics(metrics, path):
    """Save evaluation metrics (without per-sample data for space)."""
    compact = {k: v for k, v in metrics.items()
               if not k.startswith("per_sample_")}
    save_json(compact, path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset_name, seed):
    """Load dataset and create data loaders."""
    train_ds, val_ds, test_ds, stats = get_dataset(dataset_name, seed=seed)
    train_loader = make_loader(train_ds, batch_size=256, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader, stats


# ============================================================
# Phase 1: Baseline Training
# ============================================================
def run_baselines():
    print("\n" + "="*60)
    print("PHASE 1: BASELINE TRAINING")
    print("="*60)

    for ds_name in DATASETS:
        ds_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "baseline"))

        for seed in SEEDS:
            out_path = os.path.join(ds_dir, f"metrics_seed{seed}.json")
            model_path = os.path.join(ds_dir, f"model_seed{seed}.pt")
            if os.path.exists(out_path) and os.path.exists(model_path):
                print(f"  [skip] {ds_name} seed={seed} (already done)")
                continue

            print(f"  Training baseline {ds_name} seed={seed}...")
            set_seed(seed)
            train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)

            # Save data stats (once per dataset)
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
            save_metrics(metrics, out_path)
            torch.save(model.state_dict(), model_path)

            print(f"    acc={metrics['overall_accuracy']:.4f}, "
                  f"worst={metrics['worst_group_accuracy']:.4f}, "
                  f"gap={metrics['accuracy_gap']:.4f}, time={elapsed:.0f}s")
            del model
            torch.cuda.empty_cache()
            gc.collect()


# ============================================================
# Phase 2: DP-SGD Training
# ============================================================
def run_dp_training():
    print("\n" + "="*60)
    print("PHASE 2: DP-SGD TRAINING")
    print("="*60)

    for ds_name in DATASETS:
        ds_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "dp_only"))

        for eps in EPSILONS:
            for seed in SEEDS:
                out_path = os.path.join(ds_dir, f"metrics_eps{eps}_seed{seed}.json")
                model_path = os.path.join(ds_dir, f"model_eps{eps}_seed{seed}.pt")
                if os.path.exists(out_path) and os.path.exists(model_path):
                    print(f"  [skip] {ds_name} eps={eps} seed={seed} (already done)")
                    continue

                print(f"  Training DP {ds_name} eps={eps} seed={seed}...")
                set_seed(seed)
                train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)
                train_size = stats["train_size"]

                model = get_model("resnet18", NUM_CLASSES[ds_name])
                dp_config = DP_CONFIG_BASE[ds_name].copy()
                dp_config["target_epsilon"] = eps
                dp_config["target_delta"] = 1.0 / train_size

                t0 = time.time()
                model, log, final_eps = train_dp(model, train_loader, val_loader, dp_config, DEVICE)
                elapsed = time.time() - t0

                metrics = evaluate_model(model, test_loader, DEVICE)
                metrics["seed"] = seed
                metrics["target_epsilon"] = eps
                metrics["final_epsilon"] = final_eps
                metrics["train_time_sec"] = elapsed
                save_metrics(metrics, out_path)
                torch.save(model.state_dict(), model_path)

                print(f"    acc={metrics['overall_accuracy']:.4f}, "
                      f"worst={metrics['worst_group_accuracy']:.4f}, "
                      f"gap={metrics['accuracy_gap']:.4f}, "
                      f"eps={final_eps:.2f}, time={elapsed:.0f}s")
                del model
                torch.cuda.empty_cache()
                gc.collect()


# ============================================================
# Phase 3: Compression of Baseline Models
# ============================================================
def run_compression_baseline():
    print("\n" + "="*60)
    print("PHASE 3: COMPRESSION OF BASELINE MODELS")
    print("="*60)

    for ds_name in DATASETS:
        base_dir = os.path.join(RESULTS_DIR, ds_name, "baseline")
        comp_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "comp_only"))

        for seed in SEEDS:
            model_path = os.path.join(base_dir, f"model_seed{seed}.pt")
            if not os.path.exists(model_path):
                print(f"  [skip] {ds_name} seed={seed} - no baseline model")
                continue

            set_seed(seed)
            train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)
            base_model = get_model("resnet18", NUM_CLASSES[ds_name])
            base_model.load_state_dict(torch.load(model_path, map_location="cpu"))

            for sp in SPARSITIES:
                sp_str = f"{int(sp*100)}"

                # Pruned (no fine-tune)
                out_path = os.path.join(comp_dir, f"metrics_sp{sp_str}_seed{seed}.json")
                if not os.path.exists(out_path):
                    pruned = magnitude_prune(base_model, sp)
                    pruned = pruned.to(DEVICE)
                    metrics = evaluate_model(pruned, test_loader, DEVICE)
                    metrics["seed"] = seed
                    metrics["sparsity"] = sp
                    metrics["actual_sparsity"] = get_sparsity(pruned)
                    metrics["finetuned"] = False
                    save_metrics(metrics, out_path)
                    print(f"  {ds_name} sp={sp_str}% seed={seed} (no-ft): "
                          f"acc={metrics['overall_accuracy']:.4f}, "
                          f"worst={metrics['worst_group_accuracy']:.4f}")
                    del pruned
                    torch.cuda.empty_cache()

                # Pruned + fine-tuned
                out_path_ft = os.path.join(comp_dir, f"metrics_sp{sp_str}_ft_seed{seed}.json")
                model_path_ft = os.path.join(comp_dir, f"model_sp{sp_str}_ft_seed{seed}.pt")
                if not os.path.exists(out_path_ft):
                    pruned_ft = magnitude_prune(base_model, sp)
                    pruned_ft = finetune_standard(pruned_ft, train_loader,
                                                   {"ft_lr": 0.001, "ft_epochs": 5}, DEVICE)
                    metrics_ft = evaluate_model(pruned_ft, test_loader, DEVICE)
                    metrics_ft["seed"] = seed
                    metrics_ft["sparsity"] = sp
                    metrics_ft["actual_sparsity"] = get_sparsity(pruned_ft)
                    metrics_ft["finetuned"] = True
                    save_metrics(metrics_ft, out_path_ft)
                    torch.save(pruned_ft.state_dict(), model_path_ft)
                    print(f"  {ds_name} sp={sp_str}% seed={seed} (ft): "
                          f"acc={metrics_ft['overall_accuracy']:.4f}, "
                          f"worst={metrics_ft['worst_group_accuracy']:.4f}")
                    del pruned_ft
                    torch.cuda.empty_cache()

            del base_model
            gc.collect()


# ============================================================
# Phase 4: DP + Compression (Central Experiment)
# ============================================================
def run_dp_compression():
    print("\n" + "="*60)
    print("PHASE 4: DP + COMPRESSION (COMPOUNDING RATIO)")
    print("="*60)

    for ds_name in DATASETS:
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")
        dc_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "dp_comp"))

        for eps in EPSILONS:
            for seed in SEEDS:
                model_path = os.path.join(dp_dir, f"model_eps{eps}_seed{seed}.pt")
                if not os.path.exists(model_path):
                    print(f"  [skip] {ds_name} eps={eps} seed={seed} - no DP model")
                    continue

                set_seed(seed)
                train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)
                dp_model = get_model("resnet18", NUM_CLASSES[ds_name])
                dp_model.load_state_dict(torch.load(model_path, map_location="cpu"))

                for sp in SPARSITIES:
                    sp_str = f"{int(sp*100)}"

                    # Pruned DP model (no fine-tune)
                    out_path = os.path.join(dc_dir, f"metrics_eps{eps}_sp{sp_str}_seed{seed}.json")
                    if not os.path.exists(out_path):
                        pruned = magnitude_prune(dp_model, sp)
                        pruned = pruned.to(DEVICE)
                        metrics = evaluate_model(pruned, test_loader, DEVICE)
                        metrics["seed"] = seed
                        metrics["epsilon"] = eps
                        metrics["sparsity"] = sp
                        metrics["actual_sparsity"] = get_sparsity(pruned)
                        metrics["finetuned"] = False
                        save_metrics(metrics, out_path)
                        print(f"  {ds_name} eps={eps} sp={sp_str}% seed={seed} (no-ft): "
                              f"acc={metrics['overall_accuracy']:.4f}, "
                              f"worst={metrics['worst_group_accuracy']:.4f}")
                        del pruned
                        torch.cuda.empty_cache()

                    # Pruned DP model + standard fine-tune (no additional DP budget)
                    out_path_ft = os.path.join(dc_dir, f"metrics_eps{eps}_sp{sp_str}_ft_seed{seed}.json")
                    if not os.path.exists(out_path_ft):
                        pruned_ft = magnitude_prune(dp_model, sp)
                        pruned_ft = finetune_standard(pruned_ft, train_loader,
                                                       {"ft_lr": 0.001, "ft_epochs": 5}, DEVICE)
                        metrics_ft = evaluate_model(pruned_ft, test_loader, DEVICE)
                        metrics_ft["seed"] = seed
                        metrics_ft["epsilon"] = eps
                        metrics_ft["sparsity"] = sp
                        metrics_ft["actual_sparsity"] = get_sparsity(pruned_ft)
                        metrics_ft["finetuned"] = True
                        save_metrics(metrics_ft, out_path_ft)
                        print(f"  {ds_name} eps={eps} sp={sp_str}% seed={seed} (ft): "
                              f"acc={metrics_ft['overall_accuracy']:.4f}, "
                              f"worst={metrics_ft['worst_group_accuracy']:.4f}")
                        del pruned_ft
                        torch.cuda.empty_cache()

                del dp_model
                gc.collect()


# ============================================================
# Phase 5: FairPrune-DP
# ============================================================
def run_fairprune():
    print("\n" + "="*60)
    print("PHASE 5: FAIRPRUNE-DP")
    print("="*60)

    for ds_name in DATASETS:
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")
        fp_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "fairprune_dp"))

        for eps in EPSILONS:
            for seed in SEEDS:
                model_path = os.path.join(dp_dir, f"model_eps{eps}_seed{seed}.pt")
                if not os.path.exists(model_path):
                    continue

                set_seed(seed)
                train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)
                dp_model = get_model("resnet18", NUM_CLASSES[ds_name])
                dp_model.load_state_dict(torch.load(model_path, map_location="cpu"))

                for sp in SPARSITIES:
                    sp_str = f"{int(sp*100)}"

                    # FairPrune-DP (worst-group Fisher)
                    out_path = os.path.join(fp_dir, f"metrics_eps{eps}_sp{sp_str}_seed{seed}.json")
                    if not os.path.exists(out_path):
                        fp_model = fairprune_dp(dp_model, sp, val_loader, DEVICE, n_samples=1000)
                        fp_model = fp_model.to(DEVICE)
                        metrics = evaluate_model(fp_model, test_loader, DEVICE)
                        metrics["seed"] = seed
                        metrics["epsilon"] = eps
                        metrics["sparsity"] = sp
                        metrics["method"] = "fairprune_dp"
                        metrics["finetuned"] = False
                        save_metrics(metrics, out_path)
                        print(f"  FairPrune {ds_name} eps={eps} sp={sp_str}% seed={seed}: "
                              f"acc={metrics['overall_accuracy']:.4f}, "
                              f"worst={metrics['worst_group_accuracy']:.4f}")
                        del fp_model
                        torch.cuda.empty_cache()

                    # FairPrune-DP + fine-tune
                    out_path_ft = os.path.join(fp_dir, f"metrics_eps{eps}_sp{sp_str}_ft_seed{seed}.json")
                    if not os.path.exists(out_path_ft):
                        fp_model_ft = fairprune_dp(dp_model, sp, val_loader, DEVICE, n_samples=1000)
                        fp_model_ft = finetune_standard(fp_model_ft, train_loader,
                                                         {"ft_lr": 0.001, "ft_epochs": 5}, DEVICE)
                        metrics_ft = evaluate_model(fp_model_ft, test_loader, DEVICE)
                        metrics_ft["seed"] = seed
                        metrics_ft["epsilon"] = eps
                        metrics_ft["sparsity"] = sp
                        metrics_ft["method"] = "fairprune_dp"
                        metrics_ft["finetuned"] = True
                        save_metrics(metrics_ft, out_path_ft)
                        print(f"  FairPrune+ft {ds_name} eps={eps} sp={sp_str}% seed={seed}: "
                              f"acc={metrics_ft['overall_accuracy']:.4f}, "
                              f"worst={metrics_ft['worst_group_accuracy']:.4f}")
                        del fp_model_ft
                        torch.cuda.empty_cache()

                del dp_model
                gc.collect()


# ============================================================
# Phase 6: Ablation Studies
# ============================================================
def run_ablations():
    print("\n" + "="*60)
    print("PHASE 6: ABLATION STUDIES")
    print("="*60)

    # Ablation 1: Pruning criterion comparison (magnitude vs Fisher vs mean-Fisher vs FairPrune)
    # at eps=4, sparsity=70% for both datasets
    eps_abl = 4
    sp_abl = 0.7
    sp_str = "70"

    for ds_name in DATASETS:
        abl_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "ablation"))
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")

        criterion_results = {}
        for seed in SEEDS:
            model_path = os.path.join(dp_dir, f"model_eps{eps_abl}_seed{seed}.pt")
            if not os.path.exists(model_path):
                continue

            set_seed(seed)
            train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)
            dp_model = get_model("resnet18", NUM_CLASSES[ds_name])
            dp_model.load_state_dict(torch.load(model_path, map_location="cpu"))

            for method_name, prune_fn in [
                ("magnitude", lambda m: magnitude_prune(m, sp_abl)),
                ("fisher_global", lambda m: fisher_prune(m, sp_abl, val_loader, DEVICE, 1000)),
                ("fisher_mean", lambda m: mean_fisher_prune(m, sp_abl, val_loader, DEVICE, 1000)),
                ("fairprune_dp", lambda m: fairprune_dp(m, sp_abl, val_loader, DEVICE, 1000)),
            ]:
                key = f"{method_name}_seed{seed}"
                out_path = os.path.join(abl_dir, f"criterion_{method_name}_seed{seed}.json")
                if os.path.exists(out_path):
                    with open(out_path) as f:
                        criterion_results[key] = json.load(f)
                    continue

                pruned = prune_fn(dp_model)
                pruned = pruned.to(DEVICE)
                metrics = evaluate_model(pruned, test_loader, DEVICE)
                metrics["method"] = method_name
                metrics["seed"] = seed
                save_metrics(metrics, out_path)
                criterion_results[key] = metrics
                print(f"  Ablation {ds_name} {method_name} seed={seed}: "
                      f"acc={metrics['overall_accuracy']:.4f}, "
                      f"worst={metrics['worst_group_accuracy']:.4f}")
                del pruned
                torch.cuda.empty_cache()

            del dp_model
            gc.collect()

        # Save summary
        save_json(criterion_results, os.path.join(abl_dir, "criterion_comparison.json"))

    # Ablation 2: Fine-tuning effect (compile from existing results)
    for ds_name in DATASETS:
        abl_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "ablation"))
        ft_comparison = {"with_finetune": {}, "without_finetune": {}}

        for variant in ["comp_only", "dp_comp", "fairprune_dp"]:
            var_dir = os.path.join(RESULTS_DIR, ds_name, variant)
            if not os.path.isdir(var_dir):
                continue
            for f in os.listdir(var_dir):
                if f.endswith(".json") and "metrics" in f:
                    with open(os.path.join(var_dir, f)) as fp:
                        data = json.load(fp)
                    key = f"{variant}_{f}"
                    if "_ft_" in f:
                        ft_comparison["with_finetune"][key] = {
                            "overall_accuracy": data.get("overall_accuracy"),
                            "worst_group_accuracy": data.get("worst_group_accuracy"),
                            "accuracy_gap": data.get("accuracy_gap"),
                        }
                    else:
                        ft_comparison["without_finetune"][key] = {
                            "overall_accuracy": data.get("overall_accuracy"),
                            "worst_group_accuracy": data.get("worst_group_accuracy"),
                            "accuracy_gap": data.get("accuracy_gap"),
                        }

        save_json(ft_comparison, os.path.join(abl_dir, "finetuning_effect.json"))
        print(f"  Saved finetuning ablation for {ds_name}")


# ============================================================
# Phase 7: Mechanistic Analysis
# ============================================================
def run_mechanistic_analysis():
    print("\n" + "="*60)
    print("PHASE 7: MECHANISTIC ANALYSIS")
    print("="*60)

    for ds_name in DATASETS:
        analysis_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "analysis"))
        base_dir = os.path.join(RESULTS_DIR, ds_name, "baseline")
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")

        seed = 42  # Use first seed for analysis
        set_seed(seed)
        train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)

        # Load baseline model
        base_path = os.path.join(base_dir, f"model_seed{seed}.pt")
        if not os.path.exists(base_path):
            print(f"  [skip] {ds_name} - no baseline model")
            continue

        base_model = get_model("resnet18", NUM_CLASSES[ds_name])
        base_model.load_state_dict(torch.load(base_path, map_location="cpu"))
        base_model = base_model.to(DEVICE)

        # Compute Fisher for baseline
        print(f"  Computing subgroup Fisher for {ds_name} baseline...")
        base_fisher = compute_subgroup_fisher(base_model, val_loader, DEVICE, n_samples=1000)

        # Weight magnitude analysis for baseline
        minority_sgs = MINORITY_SUBGROUPS[ds_name]
        base_weight_stats = get_weight_stats_by_subgroup_relevance(base_model, base_fisher, minority_sgs)

        results = {"baseline": {"weight_stats": base_weight_stats}}

        # For each epsilon, analyze DP model
        for eps in EPSILONS:
            dp_path = os.path.join(dp_dir, f"model_eps{eps}_seed{seed}.pt")
            if not os.path.exists(dp_path):
                continue

            dp_model = get_model("resnet18", NUM_CLASSES[ds_name])
            dp_model.load_state_dict(torch.load(dp_path, map_location="cpu"))
            dp_model = dp_model.to(DEVICE)

            print(f"  Computing subgroup Fisher for {ds_name} eps={eps}...")
            dp_fisher = compute_subgroup_fisher(dp_model, val_loader, DEVICE, n_samples=1000)
            dp_weight_stats = get_weight_stats_by_subgroup_relevance(dp_model, dp_fisher, minority_sgs)

            # Pruning overlap analysis
            overlap_results = {}
            for sp in SPARSITIES:
                sp_str = f"{int(sp*100)}"

                # Get pruned weight masks for both baseline and DP models
                base_pruned = magnitude_prune(base_model, sp)
                dp_pruned = magnitude_prune(dp_model, sp)

                # Count minority-relevant weights that are additionally pruned in DP model
                total_pruned_dp = 0
                total_pruned_base = 0
                minority_pruned_dp = 0
                minority_pruned_base = 0

                for name, module_dp in dp_pruned.named_modules():
                    if isinstance(module_dp, (nn.Conv2d, nn.Linear)):
                        key = name + ".weight"
                        dp_mask = (module_dp.weight.data == 0).cpu()
                        base_module = dict(base_pruned.named_modules())[name]
                        base_mask = (base_module.weight.data == 0).cpu()

                        total_pruned_dp += dp_mask.sum().item()
                        total_pruned_base += base_mask.sum().item()

                        # Check if pruned weights are minority-relevant
                        if key in base_fisher:
                            for sg in minority_sgs:
                                if sg in base_fisher:
                                    sg_fisher = base_fisher[sg].get(key, torch.zeros_like(module_dp.weight))
                                    median_f = sg_fisher.flatten().median()
                                    is_minority_relevant = (sg_fisher > median_f).cpu()
                                    minority_pruned_dp += (dp_mask & is_minority_relevant).sum().item()
                                    minority_pruned_base += (base_mask & is_minority_relevant).sum().item()

                overlap_results[sp_str] = {
                    "dp_total_pruned": total_pruned_dp,
                    "base_total_pruned": total_pruned_base,
                    "dp_minority_pruned": minority_pruned_dp,
                    "base_minority_pruned": minority_pruned_base,
                    "dp_minority_fraction": minority_pruned_dp / max(total_pruned_dp, 1),
                    "base_minority_fraction": minority_pruned_base / max(total_pruned_base, 1),
                }
                del base_pruned, dp_pruned
                torch.cuda.empty_cache()

            results[f"eps{eps}"] = {
                "weight_stats": dp_weight_stats,
                "pruning_overlap": overlap_results,
            }
            del dp_model
            gc.collect()

        save_json(results, os.path.join(analysis_dir, "mechanistic_analysis.json"))
        print(f"  Saved mechanistic analysis for {ds_name}")
        del base_model
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# Phase 8: MIA Analysis
# ============================================================
def run_mia_analysis():
    print("\n" + "="*60)
    print("PHASE 8: MEMBERSHIP INFERENCE ATTACK ANALYSIS")
    print("="*60)

    for ds_name in DATASETS:
        mia_dir = ensure_dir(os.path.join(RESULTS_DIR, ds_name, "mia"))

        seed = 42
        set_seed(seed)
        train_loader, val_loader, test_loader, stats = load_data(ds_name, seed)

        # MIA: use training data as members, test data as non-members
        # Loss-based attack: members tend to have lower loss
        def compute_mia(model, member_loader, nonmember_loader, device):
            model = model.to(device)
            model.eval()
            criterion = nn.CrossEntropyLoss(reduction='none')

            member_losses = []
            member_subgroups = []
            nonmember_losses = []
            nonmember_subgroups = []

            with torch.no_grad():
                for batch in member_loader:
                    imgs, labs, sgs = batch
                    imgs = imgs.to(device)
                    labs = torch.tensor(labs, dtype=torch.long).to(device) if not isinstance(labs, torch.Tensor) else labs.to(device)
                    losses = criterion(model(imgs), labs)
                    member_losses.extend(losses.cpu().numpy())
                    if isinstance(sgs, torch.Tensor):
                        member_subgroups.extend(sgs.numpy())
                    else:
                        member_subgroups.extend(np.array(sgs))

                for batch in nonmember_loader:
                    imgs, labs, sgs = batch
                    imgs = imgs.to(device)
                    labs = torch.tensor(labs, dtype=torch.long).to(device) if not isinstance(labs, torch.Tensor) else labs.to(device)
                    losses = criterion(model(imgs), labs)
                    nonmember_losses.extend(losses.cpu().numpy())
                    if isinstance(sgs, torch.Tensor):
                        nonmember_subgroups.extend(sgs.numpy())
                    else:
                        nonmember_subgroups.extend(np.array(sgs))

            member_losses = np.array(member_losses)
            nonmember_losses = np.array(nonmember_losses)
            member_subgroups = np.array(member_subgroups)
            nonmember_subgroups = np.array(nonmember_subgroups)

            # Find optimal threshold using all data
            all_losses = np.concatenate([member_losses, nonmember_losses])
            all_labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])
            threshold = np.median(all_losses)

            # Overall MIA accuracy
            member_pred = (member_losses < threshold).astype(int)
            nonmember_pred = (nonmember_losses >= threshold).astype(int)
            tpr = member_pred.mean()
            tnr = nonmember_pred.mean()
            balanced_acc = (tpr + tnr) / 2

            # Per-subgroup MIA
            unique_sgs = sorted(set(member_subgroups) | set(nonmember_subgroups))
            per_sg_mia = {}
            for sg in unique_sgs:
                m_mask = member_subgroups == sg
                nm_mask = nonmember_subgroups == sg
                if m_mask.sum() > 0 and nm_mask.sum() > 0:
                    sg_tpr = (member_losses[m_mask] < threshold).mean()
                    sg_tnr = (nonmember_losses[nm_mask] >= threshold).mean()
                    per_sg_mia[int(sg)] = {
                        "balanced_accuracy": float((sg_tpr + sg_tnr) / 2),
                        "tpr": float(sg_tpr),
                        "tnr": float(sg_tnr),
                        "n_members": int(m_mask.sum()),
                        "n_nonmembers": int(nm_mask.sum()),
                    }

            mia_accs = [v["balanced_accuracy"] for v in per_sg_mia.values()]
            disparity = max(mia_accs) - min(mia_accs) if len(mia_accs) >= 2 else 0.0

            return {
                "overall_balanced_accuracy": float(balanced_acc),
                "per_subgroup_mia": per_sg_mia,
                "mia_disparity": float(disparity),
                "threshold": float(threshold),
            }

        # Evaluate MIA for different model variants
        variants = []

        # Baseline
        base_path = os.path.join(RESULTS_DIR, ds_name, "baseline", f"model_seed{seed}.pt")
        if os.path.exists(base_path):
            variants.append(("baseline", base_path, {}))

        # DP-only (eps=4)
        dp_path = os.path.join(RESULTS_DIR, ds_name, "dp_only", f"model_eps4_seed{seed}.pt")
        if os.path.exists(dp_path):
            variants.append(("dp_eps4", dp_path, {}))

        # Comp-only (sp=70%, ft)
        comp_path = os.path.join(RESULTS_DIR, ds_name, "comp_only", f"model_sp70_ft_seed{seed}.pt")
        if os.path.exists(comp_path):
            variants.append(("comp_sp70", comp_path, {}))

        # DP+Comp (eps=4, sp=70%)
        # Need to recreate since we may not have saved the model
        dp4_path = os.path.join(RESULTS_DIR, ds_name, "dp_only", f"model_eps4_seed{seed}.pt")
        if os.path.exists(dp4_path):
            variants.append(("dp4_comp70", dp4_path, {"prune": 0.7}))

        # FairPrune-DP (eps=4, sp=70%)
        # Also recreate
        if os.path.exists(dp4_path):
            variants.append(("fairprune_dp4_sp70", dp4_path, {"fairprune": 0.7}))

        mia_results = {}
        for name, path, opts in variants:
            out_path = os.path.join(mia_dir, f"mia_{name}.json")
            if os.path.exists(out_path):
                with open(out_path) as f:
                    mia_results[name] = json.load(f)
                print(f"  [skip] {ds_name} MIA {name}")
                continue

            model = get_model("resnet18", NUM_CLASSES[ds_name])
            model.load_state_dict(torch.load(path, map_location="cpu"))

            if "prune" in opts:
                model = magnitude_prune(model, opts["prune"])
            elif "fairprune" in opts:
                model = fairprune_dp(model, opts["fairprune"], val_loader, DEVICE, n_samples=1000)

            result = compute_mia(model, train_loader, test_loader, DEVICE)
            result["variant"] = name
            save_json(result, out_path)
            mia_results[name] = result
            print(f"  MIA {ds_name} {name}: bal_acc={result['overall_balanced_accuracy']:.4f}, "
                  f"disparity={result['mia_disparity']:.4f}")
            del model
            torch.cuda.empty_cache()

        save_json(mia_results, os.path.join(mia_dir, "mia_summary.json"))


# ============================================================
# Phase 9: Compute Compounding Ratios
# ============================================================
def compute_compounding_ratios():
    print("\n" + "="*60)
    print("PHASE 9: COMPUTING COMPOUNDING RATIOS")
    print("="*60)

    all_ratios = {}

    for ds_name in DATASETS:
        all_ratios[ds_name] = {}
        base_dir = os.path.join(RESULTS_DIR, ds_name, "baseline")
        dp_dir = os.path.join(RESULTS_DIR, ds_name, "dp_only")
        comp_dir = os.path.join(RESULTS_DIR, ds_name, "comp_only")
        dc_dir = os.path.join(RESULTS_DIR, ds_name, "dp_comp")
        fp_dir = os.path.join(RESULTS_DIR, ds_name, "fairprune_dp")

        for eps in EPSILONS:
            all_ratios[ds_name][f"eps{eps}"] = {}
            for sp in SPARSITIES:
                sp_str = f"{int(sp*100)}"
                cr_values = []
                fp_gap_reductions = []

                for seed in SEEDS:
                    # Load metrics
                    try:
                        with open(os.path.join(base_dir, f"metrics_seed{seed}.json")) as f:
                            base = json.load(f)
                        with open(os.path.join(dp_dir, f"metrics_eps{eps}_seed{seed}.json")) as f:
                            dp = json.load(f)
                        with open(os.path.join(comp_dir, f"metrics_sp{sp_str}_ft_seed{seed}.json")) as f:
                            comp = json.load(f)
                        with open(os.path.join(dc_dir, f"metrics_eps{eps}_sp{sp_str}_ft_seed{seed}.json")) as f:
                            dc = json.load(f)
                    except FileNotFoundError:
                        continue

                    base_wga = base["worst_group_accuracy"]
                    delta_d = base_wga - dp["worst_group_accuracy"]
                    delta_c = base_wga - comp["worst_group_accuracy"]
                    delta_dc = base_wga - dc["worst_group_accuracy"]

                    denom = delta_d + delta_c
                    if denom > 0.01:  # Avoid division by near-zero
                        cr = delta_dc / denom
                        cr_values.append(cr)

                    # FairPrune gap reduction
                    try:
                        with open(os.path.join(fp_dir, f"metrics_eps{eps}_sp{sp_str}_ft_seed{seed}.json")) as f:
                            fp = json.load(f)
                        mag_gap = dc["accuracy_gap"]
                        fp_gap = fp["accuracy_gap"]
                        if mag_gap > 0.01:
                            reduction = (mag_gap - fp_gap) / mag_gap
                            fp_gap_reductions.append(reduction)
                    except FileNotFoundError:
                        pass

                if cr_values:
                    cr_mean = float(np.mean(cr_values))
                    cr_std = float(np.std(cr_values))
                    # One-sided t-test: H0: CR <= 1 vs H1: CR > 1
                    if len(cr_values) >= 2:
                        t_stat, p_value = scipy_stats.ttest_1samp(cr_values, 1.0)
                        p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
                    else:
                        t_stat, p_one_sided = 0, 1.0

                    entry = {
                        "cr_values": cr_values,
                        "cr_mean": cr_mean,
                        "cr_std": cr_std,
                        "t_stat": float(t_stat),
                        "p_value": float(p_one_sided),
                        "super_additive": cr_mean > 1.0,
                        "strong_evidence": cr_mean > 1.2 and p_one_sided < 0.05,
                    }
                else:
                    entry = {"cr_values": [], "cr_mean": None, "note": "insufficient data"}

                if fp_gap_reductions:
                    entry["fairprune_gap_reduction_mean"] = float(np.mean(fp_gap_reductions))
                    entry["fairprune_gap_reduction_std"] = float(np.std(fp_gap_reductions))

                all_ratios[ds_name][f"eps{eps}"][f"sp{sp_str}"] = entry

                if cr_values:
                    print(f"  {ds_name} eps={eps} sp={sp_str}%: CR={cr_mean:.3f}±{cr_std:.3f} "
                          f"(p={p_one_sided:.4f})")

    save_json(all_ratios, os.path.join(RESULTS_DIR, "compounding_ratios.json"))
    return all_ratios


# ============================================================
# Phase 10: Aggregate Results
# ============================================================
def aggregate_results(compounding_ratios):
    print("\n" + "="*60)
    print("PHASE 10: AGGREGATING RESULTS")
    print("="*60)

    master_results = []

    for ds_name in DATASETS:
        # Baseline
        for seed in SEEDS:
            path = os.path.join(RESULTS_DIR, ds_name, "baseline", f"metrics_seed{seed}.json")
            if os.path.exists(path):
                with open(path) as f:
                    m = json.load(f)
                master_results.append({
                    "dataset": ds_name, "method": "baseline", "epsilon": "inf",
                    "sparsity": 0.0, "finetuned": False, "seed": seed,
                    "overall_acc": m["overall_accuracy"],
                    "worst_group_acc": m["worst_group_accuracy"],
                    "accuracy_gap": m["accuracy_gap"],
                })

        # DP-only
        for eps in EPSILONS:
            for seed in SEEDS:
                path = os.path.join(RESULTS_DIR, ds_name, "dp_only", f"metrics_eps{eps}_seed{seed}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        m = json.load(f)
                    master_results.append({
                        "dataset": ds_name, "method": "dp_only", "epsilon": eps,
                        "sparsity": 0.0, "finetuned": False, "seed": seed,
                        "overall_acc": m["overall_accuracy"],
                        "worst_group_acc": m["worst_group_accuracy"],
                        "accuracy_gap": m["accuracy_gap"],
                    })

        # Compression, DP+Comp, FairPrune
        for variant, dir_name in [("comp_only", "comp_only"), ("dp_comp", "dp_comp"), ("fairprune_dp", "fairprune_dp")]:
            var_dir = os.path.join(RESULTS_DIR, ds_name, dir_name)
            if not os.path.isdir(var_dir):
                continue
            for f in sorted(os.listdir(var_dir)):
                if f.startswith("metrics_") and f.endswith(".json"):
                    with open(os.path.join(var_dir, f)) as fp:
                        m = json.load(fp)
                    # Parse filename for parameters
                    eps_val = "inf"
                    sp_val = 0.0
                    ft = "_ft_" in f
                    if "eps" in f:
                        try:
                            eps_val = int(f.split("eps")[1].split("_")[0])
                        except (ValueError, IndexError):
                            pass
                    if "sp" in f:
                        try:
                            sp_val = int(f.split("sp")[1].split("_")[0].split(".")[0]) / 100
                        except (ValueError, IndexError):
                            pass
                    master_results.append({
                        "dataset": ds_name, "method": variant, "epsilon": eps_val,
                        "sparsity": sp_val, "finetuned": ft,
                        "seed": m.get("seed", 0),
                        "overall_acc": m["overall_accuracy"],
                        "worst_group_acc": m["worst_group_accuracy"],
                        "accuracy_gap": m["accuracy_gap"],
                    })

    # Save master CSV
    import csv
    csv_path = os.path.join(RESULTS_DIR, "master_results.csv")
    if master_results:
        keys = master_results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(master_results)
        print(f"  Saved {len(master_results)} rows to master_results.csv")

    # Success criteria evaluation
    success = evaluate_success_criteria(compounding_ratios)
    save_json(success, os.path.join(RESULTS_DIR, "success_criteria_evaluation.json"))

    # Build results.json (the required output)
    results_json = build_results_json(compounding_ratios, master_results, success)
    save_json(results_json, os.path.join(WORKSPACE, "results.json"))
    print("  Saved results.json")


def evaluate_success_criteria(compounding_ratios):
    """Check each success criterion."""
    criteria = {}

    # Criterion 1: CR > 1.2 across >=2 datasets
    strong_configs = 0
    total_configs = 0
    for ds_name in DATASETS:
        for eps_key in compounding_ratios.get(ds_name, {}):
            for sp_key in compounding_ratios[ds_name][eps_key]:
                entry = compounding_ratios[ds_name][eps_key][sp_key]
                if entry.get("cr_mean") is not None:
                    total_configs += 1
                    if entry.get("strong_evidence", False):
                        strong_configs += 1

    criteria["criterion_1_super_additive"] = {
        "description": "CR > 1.2 consistently across >=2 datasets",
        "strong_configs": strong_configs,
        "total_configs": total_configs,
        "pass": strong_configs >= 2,
    }

    # Criterion 3: FairPrune reduces gap by >=20%
    gap_reductions = []
    for ds_name in DATASETS:
        for eps_key in compounding_ratios.get(ds_name, {}):
            for sp_key in compounding_ratios[ds_name][eps_key]:
                entry = compounding_ratios[ds_name][eps_key][sp_key]
                if "fairprune_gap_reduction_mean" in entry:
                    gap_reductions.append(entry["fairprune_gap_reduction_mean"])

    criteria["criterion_3_fairprune_effectiveness"] = {
        "description": "FairPrune-DP reduces gap by >=20%",
        "mean_reduction": float(np.mean(gap_reductions)) if gap_reductions else None,
        "num_configs": len(gap_reductions),
        "pass": bool(np.mean(gap_reductions) >= 0.2) if gap_reductions else False,
    }

    return criteria


def build_results_json(compounding_ratios, master_results, success_criteria):
    """Build the required results.json output."""
    import pandas as pd

    results = {
        "title": "The Compounding Cost: DP + Compression Fairness Degradation",
        "datasets": DATASETS,
        "seeds": SEEDS,
        "epsilons": EPSILONS,
        "sparsities": SPARSITIES,
    }

    # Summarize baseline performance
    for ds_name in DATASETS:
        ds_results = [r for r in master_results if r["dataset"] == ds_name and r["method"] == "baseline"]
        if ds_results:
            results[f"{ds_name}_baseline"] = {
                "overall_acc_mean": float(np.mean([r["overall_acc"] for r in ds_results])),
                "overall_acc_std": float(np.std([r["overall_acc"] for r in ds_results])),
                "worst_group_acc_mean": float(np.mean([r["worst_group_acc"] for r in ds_results])),
                "worst_group_acc_std": float(np.std([r["worst_group_acc"] for r in ds_results])),
                "accuracy_gap_mean": float(np.mean([r["accuracy_gap"] for r in ds_results])),
                "accuracy_gap_std": float(np.std([r["accuracy_gap"] for r in ds_results])),
            }

        # DP-only summary
        for eps in EPSILONS:
            dp_results = [r for r in master_results
                         if r["dataset"] == ds_name and r["method"] == "dp_only" and r["epsilon"] == eps]
            if dp_results:
                results[f"{ds_name}_dp_eps{eps}"] = {
                    "overall_acc_mean": float(np.mean([r["overall_acc"] for r in dp_results])),
                    "overall_acc_std": float(np.std([r["overall_acc"] for r in dp_results])),
                    "worst_group_acc_mean": float(np.mean([r["worst_group_acc"] for r in dp_results])),
                    "worst_group_acc_std": float(np.std([r["worst_group_acc"] for r in dp_results])),
                    "accuracy_gap_mean": float(np.mean([r["accuracy_gap"] for r in dp_results])),
                    "accuracy_gap_std": float(np.std([r["accuracy_gap"] for r in dp_results])),
                }

    results["compounding_ratios"] = compounding_ratios
    results["success_criteria"] = success_criteria

    return results


# ============================================================
# Phase 11: Generate Figures
# ============================================================
def generate_figures(compounding_ratios):
    print("\n" + "="*60)
    print("PHASE 11: GENERATING FIGURES")
    print("="*60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.dpi": 300})

    # Figure 1: Compounding ratio heatmaps
    for ds_name in DATASETS:
        if ds_name not in compounding_ratios:
            continue

        eps_labels = [str(e) for e in EPSILONS]
        sp_labels = [f"{int(s*100)}%" for s in SPARSITIES]

        cr_matrix = np.zeros((len(EPSILONS), len(SPARSITIES)))
        for i, eps in enumerate(EPSILONS):
            for j, sp in enumerate(SPARSITIES):
                sp_str = f"{int(sp*100)}"
                entry = compounding_ratios[ds_name].get(f"eps{eps}", {}).get(f"sp{sp_str}", {})
                cr_matrix[i, j] = entry.get("cr_mean", np.nan)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cr_matrix, cmap="RdYlGn_r", vmin=0.5, vmax=2.0, aspect="auto")
        ax.set_xticks(range(len(sp_labels)))
        ax.set_xticklabels(sp_labels)
        ax.set_yticks(range(len(eps_labels)))
        ax.set_yticklabels([f"ε={e}" for e in eps_labels])
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Privacy Budget")
        ax.set_title(f"Compounding Ratio ({ds_name.upper()})")

        for i in range(len(EPSILONS)):
            for j in range(len(SPARSITIES)):
                val = cr_matrix[i, j]
                if not np.isnan(val):
                    sp_str = f"{int(SPARSITIES[j]*100)}"
                    entry = compounding_ratios[ds_name].get(f"eps{EPSILONS[i]}", {}).get(f"sp{sp_str}", {})
                    p = entry.get("p_value", 1.0)
                    stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    ax.text(j, i, f"{val:.2f}{stars}", ha="center", va="center",
                            fontsize=10, fontweight="bold")

        plt.colorbar(im, ax=ax, label="CR (>1 = super-additive)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"compounding_ratio_heatmap_{ds_name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, f"compounding_ratio_heatmap_{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved compounding_ratio_heatmap_{ds_name}")

    # Figure 2: Subgroup accuracy bars at representative setting (eps=4, sp=70%)
    for ds_name in DATASETS:
        fig, ax = plt.subplots(figsize=(8, 5))
        variants = []
        variant_names = []

        for variant, label in [
            ("baseline", "Baseline"),
            ("dp_only", "DP Only (ε=4)"),
            ("comp_only", "Comp Only (70%)"),
            ("dp_comp", "DP+Comp"),
            ("fairprune_dp", "FairPrune-DP"),
        ]:
            var_dir = os.path.join(RESULTS_DIR, ds_name, variant)
            if not os.path.isdir(var_dir):
                continue

            # Find the right file
            if variant == "baseline":
                fname = "metrics_seed42.json"
            elif variant == "dp_only":
                fname = "metrics_eps4_seed42.json"
            elif variant == "comp_only":
                fname = "metrics_sp70_ft_seed42.json"
            elif variant == "dp_comp":
                fname = "metrics_eps4_sp70_ft_seed42.json"
            elif variant == "fairprune_dp":
                fname = "metrics_eps4_sp70_ft_seed42.json"

            path = os.path.join(var_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    m = json.load(f)
                variants.append(m["per_subgroup_accuracy"])
                variant_names.append(label)

        if variants:
            subgroups = sorted(variants[0].keys(), key=lambda x: int(x))
            x = np.arange(len(subgroups))
            width = 0.8 / len(variants)
            colors = sns.color_palette("Set2", len(variants))

            for i, (v, name) in enumerate(zip(variants, variant_names)):
                vals = [v.get(sg, 0) for sg in subgroups]
                ax.bar(x + i * width, vals, width, label=name, color=colors[i])

            # Get subgroup names
            stats_path = os.path.join(RESULTS_DIR, ds_name, "data_stats.json")
            sg_names = subgroups
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    ds_stats = json.load(f)
                sg_names = [ds_stats.get("subgroup_names", {}).get(sg, sg) for sg in subgroups]

            ax.set_xticks(x + width * (len(variants) - 1) / 2)
            ax.set_xticklabels(sg_names, rotation=45, ha="right")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Per-Subgroup Accuracy ({ds_name.upper()}, ε=4, 70% sparsity)")
            ax.legend(fontsize=9, loc="upper right")
            ax.set_ylim(0, 1)

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"subgroup_accuracy_{ds_name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, f"subgroup_accuracy_{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved subgroup_accuracy_{ds_name}")

    # Figure 3: Fairness-compression Pareto frontier
    for ds_name in DATASETS:
        fig, axes = plt.subplots(1, len(EPSILONS), figsize=(4 * len(EPSILONS), 4), sharey=True)
        if len(EPSILONS) == 1:
            axes = [axes]
        colors = {"magnitude": "C0", "fairprune_dp": "C1"}

        for ax_idx, eps in enumerate(EPSILONS):
            ax = axes[ax_idx]
            for method, label, color in [("dp_comp", "Magnitude", "C0"), ("fairprune_dp", "FairPrune-DP", "C1")]:
                sp_vals = []
                gap_means = []
                gap_stds = []

                for sp in SPARSITIES:
                    sp_str = f"{int(sp*100)}"
                    gaps = []
                    for seed in SEEDS:
                        suffix = f"_ft_seed{seed}.json"
                        if method == "dp_comp":
                            path = os.path.join(RESULTS_DIR, ds_name, method,
                                               f"metrics_eps{eps}_sp{sp_str}{suffix}")
                        else:
                            path = os.path.join(RESULTS_DIR, ds_name, method,
                                               f"metrics_eps{eps}_sp{sp_str}{suffix}")
                        if os.path.exists(path):
                            with open(path) as f:
                                m = json.load(f)
                            gaps.append(m["accuracy_gap"])

                    if gaps:
                        sp_vals.append(sp)
                        gap_means.append(np.mean(gaps))
                        gap_stds.append(np.std(gaps))

                if sp_vals:
                    ax.errorbar(sp_vals, gap_means, yerr=gap_stds, label=label,
                               color=color, marker="o", capsize=3)

            ax.set_xlabel("Sparsity")
            if ax_idx == 0:
                ax.set_ylabel("Accuracy Gap")
            ax.set_title(f"ε = {eps}")
            ax.legend(fontsize=9)

        fig.suptitle(f"Fairness vs Compression ({ds_name.upper()})", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"pareto_frontier_{ds_name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, f"pareto_frontier_{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved pareto_frontier_{ds_name}")

    # Figure 4: Weight distribution (mechanistic)
    for ds_name in DATASETS:
        analysis_path = os.path.join(RESULTS_DIR, ds_name, "analysis", "mechanistic_analysis.json")
        if not os.path.exists(analysis_path):
            continue

        with open(analysis_path) as f:
            analysis = json.load(f)

        fig, ax = plt.subplots(figsize=(6, 4))
        models = ["baseline"] + [f"eps{e}" for e in EPSILONS]
        model_labels = ["No DP"] + [f"ε={e}" for e in EPSILONS]
        minority_means = []
        majority_means = []

        for m in models:
            ws = analysis.get(m, {}).get("weight_stats")
            if ws:
                minority_means.append(ws["minority_relevant_magnitude_mean"])
                majority_means.append(ws["majority_relevant_magnitude_mean"])
            else:
                minority_means.append(0)
                majority_means.append(0)

        x = np.arange(len(model_labels))
        width = 0.35
        ax.bar(x - width/2, minority_means, width, label="Minority-relevant", color="C3", alpha=0.8)
        ax.bar(x + width/2, majority_means, width, label="Majority-relevant", color="C0", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("Mean |weight|")
        ax.set_title(f"Weight Magnitudes by Subgroup Relevance ({ds_name.upper()})")
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"weight_distributions_{ds_name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, f"weight_distributions_{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved weight_distributions_{ds_name}")

    # Figure 5: Pruning overlap
    for ds_name in DATASETS:
        analysis_path = os.path.join(RESULTS_DIR, ds_name, "analysis", "mechanistic_analysis.json")
        if not os.path.exists(analysis_path):
            continue

        with open(analysis_path) as f:
            analysis = json.load(f)

        fig, ax = plt.subplots(figsize=(6, 4))
        sp_labels = [f"{int(s*100)}%" for s in SPARSITIES]
        x = np.arange(len(sp_labels))
        width = 0.25

        for i, eps in enumerate(EPSILONS):
            fracs = []
            for sp in SPARSITIES:
                sp_str = f"{int(sp*100)}"
                overlap = analysis.get(f"eps{eps}", {}).get("pruning_overlap", {}).get(sp_str, {})
                fracs.append(overlap.get("dp_minority_fraction", 0))
            ax.bar(x + i * width, fracs, width, label=f"DP ε={eps}")

        # Baseline pruning overlap
        base_fracs = []
        for sp in SPARSITIES:
            sp_str = f"{int(sp*100)}"
            # Use eps1 overlap data for baseline reference
            overlap = analysis.get(f"eps{EPSILONS[0]}", {}).get("pruning_overlap", {}).get(sp_str, {})
            base_fracs.append(overlap.get("base_minority_fraction", 0))
        ax.bar(x - width, base_fracs, width, label="No DP", color="gray", alpha=0.6)

        ax.set_xticks(x + width * (len(EPSILONS) - 1) / 2)
        ax.set_xticklabels(sp_labels)
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Fraction of pruned weights\nthat are minority-relevant")
        ax.set_title(f"Pruning Overlap Analysis ({ds_name.upper()})")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"pruning_overlap_{ds_name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, f"pruning_overlap_{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved pruning_overlap_{ds_name}")

    # Figure 6: Ablation criterion comparison
    for ds_name in DATASETS:
        abl_path = os.path.join(RESULTS_DIR, ds_name, "ablation", "criterion_comparison.json")
        if not os.path.exists(abl_path):
            continue

        with open(abl_path) as f:
            abl_data = json.load(f)

        methods = ["magnitude", "fisher_global", "fisher_mean", "fairprune_dp"]
        method_labels = ["Magnitude", "Fisher (Global)", "Fisher (Mean)", "FairPrune-DP"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        for metric, ax, ylabel in [
            ("overall_accuracy", ax1, "Overall Accuracy"),
            ("worst_group_accuracy", ax2, "Worst-Group Accuracy"),
        ]:
            means = []
            stds = []
            for method in methods:
                vals = []
                for seed in SEEDS:
                    key = f"{method}_seed{seed}"
                    if key in abl_data:
                        vals.append(abl_data[key].get(metric, 0))
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)

            colors = sns.color_palette("Set2", len(methods))
            bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=3, color=colors)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 1)

        fig.suptitle(f"Pruning Criterion Comparison ({ds_name.upper()}, ε=4, 70%)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"ablation_criterion_{ds_name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, f"ablation_criterion_{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved ablation_criterion_{ds_name}")

    # Figure 7: MIA disparity
    for ds_name in DATASETS:
        mia_path = os.path.join(RESULTS_DIR, ds_name, "mia", "mia_summary.json")
        if not os.path.exists(mia_path):
            continue

        with open(mia_path) as f:
            mia_data = json.load(f)

        fig, ax = plt.subplots(figsize=(6, 4))
        variant_names = []
        disparities = []
        for name, data in mia_data.items():
            if isinstance(data, dict) and "mia_disparity" in data:
                variant_names.append(name.replace("_", "\n"))
                disparities.append(data["mia_disparity"])

        if variant_names:
            colors = sns.color_palette("Set2", len(variant_names))
            ax.bar(range(len(variant_names)), disparities, color=colors)
            ax.set_xticks(range(len(variant_names)))
            ax.set_xticklabels(variant_names, fontsize=9)
            ax.set_ylabel("MIA Disparity (max - min subgroup)")
            ax.set_title(f"MIA Vulnerability Disparity ({ds_name.upper()})")

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"mia_disparity_{ds_name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, f"mia_disparity_{ds_name}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved mia_disparity_{ds_name}")

    # Figure 8: Main results table (LaTeX)
    generate_latex_tables()


def generate_latex_tables():
    """Generate LaTeX tables for the paper."""

    # Table 1: Main results
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main results: accuracy and fairness metrics across datasets and configurations. "
        r"WGA = Worst-Group Accuracy, Gap = accuracy gap between best and worst subgroup.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llcc|cc|cc}",
        r"\toprule",
        r"& & \multicolumn{2}{c|}{Overall Acc.} & \multicolumn{2}{c|}{WGA} & \multicolumn{2}{c}{Gap} \\",
        r"Dataset & Method & Mean & Std & Mean & Std & Mean & Std \\",
        r"\midrule",
    ]

    for ds_name in DATASETS:
        first = True
        for variant, label in [
            ("baseline", "Baseline"),
            ("dp_only", "DP (ε=4)"),
            ("comp_only", "Comp (70%)"),
            ("dp_comp", "DP+Comp"),
            ("fairprune_dp", "FairPrune-DP"),
        ]:
            accs, wgas, gaps = [], [], []
            var_dir = os.path.join(RESULTS_DIR, ds_name, variant)
            if not os.path.isdir(var_dir):
                continue

            for seed in SEEDS:
                if variant == "baseline":
                    fname = f"metrics_seed{seed}.json"
                elif variant == "dp_only":
                    fname = f"metrics_eps4_seed{seed}.json"
                elif variant == "comp_only":
                    fname = f"metrics_sp70_ft_seed{seed}.json"
                elif variant == "dp_comp":
                    fname = f"metrics_eps4_sp70_ft_seed{seed}.json"
                elif variant == "fairprune_dp":
                    fname = f"metrics_eps4_sp70_ft_seed{seed}.json"

                path = os.path.join(var_dir, fname)
                if os.path.exists(path):
                    with open(path) as f:
                        m = json.load(f)
                    accs.append(m["overall_accuracy"])
                    wgas.append(m["worst_group_accuracy"])
                    gaps.append(m["accuracy_gap"])

            if accs:
                ds_label = ds_name.upper() if first else ""
                first = False
                lines.append(
                    f"{ds_label} & {label} & "
                    f"{np.mean(accs):.3f} & {np.std(accs):.3f} & "
                    f"{np.mean(wgas):.3f} & {np.std(wgas):.3f} & "
                    f"{np.mean(gaps):.3f} & {np.std(gaps):.3f} \\\\"
                )

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(os.path.join(FIGURES_DIR, "table_main_results.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  Saved table_main_results.tex")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Workspace: {WORKSPACE}")

    t_start = time.time()

    # Create directories
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)

    # Run all phases
    run_baselines()
    run_dp_training()
    run_compression_baseline()
    run_dp_compression()
    run_fairprune()
    run_ablations()
    run_mechanistic_analysis()
    run_mia_analysis()
    cr = compute_compounding_ratios()
    aggregate_results(cr)
    generate_figures(cr)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/3600:.1f} hours")
    print(f"{'='*60}")
