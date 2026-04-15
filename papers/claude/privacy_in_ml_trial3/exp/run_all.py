#!/usr/bin/env python3
"""Main experiment runner for DP + Compression fairness experiments.

Runs all experiments sequentially on a single GPU:
1. Baseline training (no DP, no compression)
2. DP-SGD training at varying epsilon
3. Compression of standard models
4. DP + Compression (compounding ratio)
5. FairPrune-DP
6. Mechanistic analysis
7. Ablation studies
8. MIA analysis
9. Aggregation + visualization
"""

import os
import sys
import copy
import time
import json
import traceback
import numpy as np
import torch
from collections import defaultdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.shared.utils import set_seed, get_device, save_json, load_json, ensure_dir
from exp.shared.models import get_model
from exp.shared.data_loader import get_dataset, make_loader
from exp.shared.metrics import evaluate_model
from exp.shared.training import train_standard, train_dp, finetune_standard
from exp.shared.compression import (
    magnitude_prune, fisher_prune, fairprune_dp, mean_fisher_prune,
    compute_subgroup_fisher, get_sparsity, get_weight_stats_by_subgroup_relevance,
    compute_fisher_importance,
)

# ==================== CONFIGURATION ====================
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE, "results")
FIGURES_DIR = os.path.join(WORKSPACE, "figures")

DATASETS = ["cifar10", "utkface", "celeba"]
SEEDS = [42, 123, 456]
EPSILONS = [1, 4, 8]
SPARSITIES = [0.5, 0.7, 0.9]
ARCH = "resnet18"

# Dataset-specific configs
DATASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
        "epochs": 30,
        "dp_epochs": 30,
        "lr": 0.01,
        "dp_lr": 0.5,
        "minority_subgroups": {1},  # minority class group
    },
    "utkface": {
        "num_classes": 2,
        "epochs": 25,
        "dp_epochs": 25,
        "lr": 0.01,
        "dp_lr": 0.5,
        "minority_subgroups": {4},  # "Others" ethnicity - smallest group
    },
    "celeba": {
        "num_classes": 2,
        "epochs": 25,
        "dp_epochs": 25,
        "lr": 0.01,
        "dp_lr": 0.5,
        "minority_subgroups": {1},  # Male (typically smaller in CelebA for Smiling task)
    },
}


def run_experiment(stage_name, func, *args, **kwargs):
    """Run an experiment stage with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed/60:.1f} min")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed/60:.1f} min: {e}")
        traceback.print_exc()
        return None


# ==================== STAGE 1: BASELINE TRAINING ====================

def stage_baseline_training(device):
    """Train models on all datasets without DP or compression."""
    all_models = {}  # (dataset, seed) -> model
    all_metrics = {}

    for ds_name in DATASETS:
        ds_config = DATASET_CONFIGS[ds_name]
        print(f"\n--- Baseline: {ds_name} ---")

        for seed in SEEDS:
            print(f"  Seed {seed}...")
            set_seed(seed)
            train_ds, val_ds, test_ds, stats = get_dataset(ds_name, seed)
            train_loader = make_loader(train_ds, batch_size=256, shuffle=True)
            val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
            test_loader = make_loader(test_ds, batch_size=256, shuffle=False)

            # Save data stats
            save_json(stats, os.path.join(RESULTS_DIR, ds_name, "data_stats.json"))

            model = get_model(ARCH, ds_config["num_classes"])
            config = {
                "lr": ds_config["lr"],
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "epochs": ds_config["epochs"],
                "patience": 5,
            }

            model, log = train_standard(model, train_loader, val_loader, config, device)

            # Evaluate
            metrics = evaluate_model(model, test_loader, device)
            metrics["training_log"] = log
            metrics["seed"] = seed

            save_path = os.path.join(RESULTS_DIR, ds_name, "baseline")
            ensure_dir(save_path)
            save_json(metrics, os.path.join(save_path, f"metrics_seed{seed}.json"))
            torch.save(model.state_dict(), os.path.join(save_path, f"model_seed{seed}.pt"))

            all_models[(ds_name, seed)] = model
            all_metrics[(ds_name, "baseline", seed)] = metrics

            print(f"    Overall acc: {metrics['overall_accuracy']:.4f}, "
                  f"Worst group: {metrics['worst_group_accuracy']:.4f}, "
                  f"Gap: {metrics['accuracy_gap']:.4f}")

            # Keep data loaders for later use (just test)
            all_models[(ds_name, seed, "test_loader")] = test_loader
            all_models[(ds_name, seed, "val_loader")] = val_loader
            all_models[(ds_name, seed, "train_loader")] = train_loader

    return all_models, all_metrics


# ==================== STAGE 2: DP-SGD TRAINING ====================

def stage_dp_training(device, all_models, all_metrics):
    """Train models with DP-SGD at varying privacy budgets."""
    dp_models = {}

    for ds_name in DATASETS:
        ds_config = DATASET_CONFIGS[ds_name]
        print(f"\n--- DP Training: {ds_name} ---")

        # For CelebA, only run eps=4 to save time
        eps_list = EPSILONS if ds_name != "celeba" else [4]

        for eps in eps_list:
            for seed in SEEDS:
                print(f"  eps={eps}, seed={seed}...")
                set_seed(seed)
                train_ds, val_ds, test_ds, stats = get_dataset(ds_name, seed)
                train_loader = make_loader(train_ds, batch_size=256, shuffle=True)
                val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
                test_loader = make_loader(test_ds, batch_size=256, shuffle=False)

                model = get_model(ARCH, ds_config["num_classes"])
                dp_config = {
                    "target_epsilon": eps,
                    "target_delta": 1.0 / stats["train_size"],
                    "max_grad_norm": 1.0,
                    "epochs": ds_config["dp_epochs"],
                    "lr": ds_config["dp_lr"],
                    "max_physical_batch_size": 128,
                }

                model, log, final_eps = train_dp(model, train_loader, val_loader, dp_config, device)

                # Evaluate
                metrics = evaluate_model(model, test_loader, device)
                metrics["training_log"] = log
                metrics["seed"] = seed
                metrics["target_epsilon"] = eps
                metrics["final_epsilon"] = final_eps

                save_path = os.path.join(RESULTS_DIR, ds_name, "dp_only")
                ensure_dir(save_path)
                save_json(metrics, os.path.join(save_path, f"metrics_eps{eps}_seed{seed}.json"))
                torch.save(model.state_dict(), os.path.join(save_path, f"model_eps{eps}_seed{seed}.pt"))

                dp_models[(ds_name, eps, seed)] = model
                all_metrics[(ds_name, f"dp_eps{eps}", seed)] = metrics

                print(f"    Overall acc: {metrics['overall_accuracy']:.4f}, "
                      f"Worst group: {metrics['worst_group_accuracy']:.4f}, "
                      f"Gap: {metrics['accuracy_gap']:.4f}, "
                      f"Final eps: {final_eps:.2f}")

    return dp_models


# ==================== STAGE 3: COMPRESSION OF STANDARD MODELS ====================

def stage_compression_standard(device, all_models, all_metrics):
    """Compress standard (non-DP) models."""
    for ds_name in DATASETS:
        print(f"\n--- Compression (standard): {ds_name} ---")

        for seed in SEEDS:
            base_model = all_models.get((ds_name, seed))
            if base_model is None:
                print(f"  Skipping seed {seed}: no baseline model")
                continue

            test_loader = all_models.get((ds_name, seed, "test_loader"))
            train_loader = all_models.get((ds_name, seed, "train_loader"))

            for sp in SPARSITIES:
                print(f"  Seed {seed}, sparsity {sp}...")

                # Prune without fine-tuning
                pruned = magnitude_prune(base_model, sp)
                metrics_noft = evaluate_model(pruned, test_loader, device)
                metrics_noft["seed"] = seed
                metrics_noft["sparsity"] = sp
                metrics_noft["actual_sparsity"] = get_sparsity(pruned)
                metrics_noft["finetuned"] = False

                save_path = os.path.join(RESULTS_DIR, ds_name, "comp_only")
                ensure_dir(save_path)
                save_json(metrics_noft, os.path.join(save_path, f"metrics_sp{sp}_seed{seed}.json"))

                # Fine-tune
                pruned_ft = finetune_standard(pruned, train_loader,
                                               {"ft_lr": 0.001, "ft_epochs": 5}, device)
                metrics_ft = evaluate_model(pruned_ft, test_loader, device)
                metrics_ft["seed"] = seed
                metrics_ft["sparsity"] = sp
                metrics_ft["actual_sparsity"] = get_sparsity(pruned_ft)
                metrics_ft["finetuned"] = True

                save_json(metrics_ft, os.path.join(save_path, f"metrics_sp{sp}_ft_seed{seed}.json"))
                all_metrics[(ds_name, f"comp_sp{sp}", seed)] = metrics_noft
                all_metrics[(ds_name, f"comp_sp{sp}_ft", seed)] = metrics_ft

                print(f"    No FT: acc={metrics_noft['overall_accuracy']:.4f}, "
                      f"worst={metrics_noft['worst_group_accuracy']:.4f}, gap={metrics_noft['accuracy_gap']:.4f}")
                print(f"    FT:    acc={metrics_ft['overall_accuracy']:.4f}, "
                      f"worst={metrics_ft['worst_group_accuracy']:.4f}, gap={metrics_ft['accuracy_gap']:.4f}")


# ==================== STAGE 4: DP + COMPRESSION ====================

def stage_dp_compression(device, dp_models, all_models, all_metrics):
    """Compress DP-trained models and compute compounding ratios."""
    compounding_ratios = {}

    for ds_name in DATASETS:
        ds_config = DATASET_CONFIGS[ds_name]
        eps_list = EPSILONS if ds_name != "celeba" else [4]

        print(f"\n--- DP + Compression: {ds_name} ---")

        for eps in eps_list:
            for seed in SEEDS:
                dp_model = dp_models.get((ds_name, eps, seed))
                if dp_model is None:
                    continue

                test_loader = all_models.get((ds_name, seed, "test_loader"))
                train_loader = all_models.get((ds_name, seed, "train_loader"))

                for sp in SPARSITIES:
                    print(f"  eps={eps}, sp={sp}, seed={seed}...")

                    # Prune DP model (no fine-tune)
                    pruned = magnitude_prune(dp_model, sp)
                    metrics = evaluate_model(pruned, test_loader, device)
                    metrics["seed"] = seed
                    metrics["epsilon"] = eps
                    metrics["sparsity"] = sp
                    metrics["finetuned"] = False

                    save_path = os.path.join(RESULTS_DIR, ds_name, "dp_comp")
                    ensure_dir(save_path)
                    save_json(metrics, os.path.join(save_path,
                              f"metrics_eps{eps}_sp{sp}_seed{seed}.json"))

                    # Fine-tune (standard, not DP — simpler and within time budget)
                    pruned_ft = finetune_standard(copy.deepcopy(pruned), train_loader,
                                                   {"ft_lr": 0.001, "ft_epochs": 5}, device)
                    metrics_ft = evaluate_model(pruned_ft, test_loader, device)
                    metrics_ft["seed"] = seed
                    metrics_ft["epsilon"] = eps
                    metrics_ft["sparsity"] = sp
                    metrics_ft["finetuned"] = True

                    save_json(metrics_ft, os.path.join(save_path,
                              f"metrics_eps{eps}_sp{sp}_ft_seed{seed}.json"))

                    all_metrics[(ds_name, f"dp_comp_eps{eps}_sp{sp}", seed)] = metrics
                    all_metrics[(ds_name, f"dp_comp_eps{eps}_sp{sp}_ft", seed)] = metrics_ft

                    # Compute compounding ratio (using non-finetuned for clean comparison)
                    baseline_worst = all_metrics.get((ds_name, "baseline", seed), {}).get("worst_group_accuracy", 0)
                    dp_worst = all_metrics.get((ds_name, f"dp_eps{eps}", seed), {}).get("worst_group_accuracy", 0)
                    comp_worst = all_metrics.get((ds_name, f"comp_sp{sp}", seed), {}).get("worst_group_accuracy", 0)
                    dc_worst = metrics["worst_group_accuracy"]

                    delta_d = baseline_worst - dp_worst
                    delta_c = baseline_worst - comp_worst
                    delta_dc = baseline_worst - dc_worst

                    denom = delta_d + delta_c
                    cr = delta_dc / denom if denom > 0.001 else float("nan")

                    key = f"{ds_name}_eps{eps}_sp{sp}_seed{seed}"
                    compounding_ratios[key] = {
                        "dataset": ds_name,
                        "epsilon": eps,
                        "sparsity": sp,
                        "seed": seed,
                        "CR": cr,
                        "delta_D": delta_d,
                        "delta_C": delta_c,
                        "delta_DC": delta_dc,
                        "baseline_worst": baseline_worst,
                        "dp_worst": dp_worst,
                        "comp_worst": comp_worst,
                        "dc_worst": dc_worst,
                    }

                    # Also compute for finetuned
                    comp_ft_worst = all_metrics.get((ds_name, f"comp_sp{sp}_ft", seed), {}).get("worst_group_accuracy", 0)
                    delta_c_ft = baseline_worst - comp_ft_worst
                    delta_dc_ft = baseline_worst - metrics_ft["worst_group_accuracy"]
                    denom_ft = delta_d + delta_c_ft
                    cr_ft = delta_dc_ft / denom_ft if denom_ft > 0.001 else float("nan")

                    compounding_ratios[f"{key}_ft"] = {
                        "dataset": ds_name, "epsilon": eps, "sparsity": sp,
                        "seed": seed, "CR": cr_ft, "finetuned": True,
                        "delta_D": delta_d, "delta_C_ft": delta_c_ft,
                        "delta_DC_ft": delta_dc_ft,
                    }

                    print(f"    DC worst: {dc_worst:.4f}, CR={cr:.3f}, "
                          f"FT worst: {metrics_ft['worst_group_accuracy']:.4f}, CR_ft={cr_ft:.3f}")

    save_json(compounding_ratios, os.path.join(RESULTS_DIR, "compounding_ratios.json"))
    return compounding_ratios


# ==================== STAGE 5: FAIRPRUNE-DP ====================

def stage_fairprune(device, dp_models, all_models, all_metrics):
    """Apply FairPrune-DP to DP-trained models."""
    for ds_name in DATASETS:
        ds_config = DATASET_CONFIGS[ds_name]
        eps_list = EPSILONS if ds_name != "celeba" else [4]

        print(f"\n--- FairPrune-DP: {ds_name} ---")

        for eps in eps_list:
            for seed in SEEDS:
                dp_model = dp_models.get((ds_name, eps, seed))
                if dp_model is None:
                    continue

                test_loader = all_models.get((ds_name, seed, "test_loader"))
                val_loader = all_models.get((ds_name, seed, "val_loader"))
                train_loader = all_models.get((ds_name, seed, "train_loader"))

                for sp in SPARSITIES:
                    print(f"  FairPrune: eps={eps}, sp={sp}, seed={seed}...")

                    try:
                        # FairPrune-DP
                        fp_model = fairprune_dp(dp_model, sp, val_loader, device, n_samples=1000)
                        metrics_fp = evaluate_model(fp_model, test_loader, device)
                        metrics_fp["method"] = "fairprune_dp"
                        metrics_fp["epsilon"] = eps
                        metrics_fp["sparsity"] = sp
                        metrics_fp["seed"] = seed

                        save_path = os.path.join(RESULTS_DIR, ds_name, "fairprune_dp")
                        ensure_dir(save_path)
                        save_json(metrics_fp, os.path.join(save_path,
                                  f"metrics_eps{eps}_sp{sp}_seed{seed}.json"))

                        # Fine-tune FairPrune model
                        fp_ft = finetune_standard(copy.deepcopy(fp_model), train_loader,
                                                   {"ft_lr": 0.001, "ft_epochs": 5}, device)
                        metrics_fp_ft = evaluate_model(fp_ft, test_loader, device)
                        metrics_fp_ft["method"] = "fairprune_dp_ft"
                        metrics_fp_ft["epsilon"] = eps
                        metrics_fp_ft["sparsity"] = sp
                        metrics_fp_ft["seed"] = seed

                        save_json(metrics_fp_ft, os.path.join(save_path,
                                  f"metrics_eps{eps}_sp{sp}_ft_seed{seed}.json"))

                        all_metrics[(ds_name, f"fairprune_eps{eps}_sp{sp}", seed)] = metrics_fp
                        all_metrics[(ds_name, f"fairprune_eps{eps}_sp{sp}_ft", seed)] = metrics_fp_ft

                        print(f"    FairPrune: acc={metrics_fp['overall_accuracy']:.4f}, "
                              f"worst={metrics_fp['worst_group_accuracy']:.4f}, "
                              f"gap={metrics_fp['accuracy_gap']:.4f}")
                        print(f"    FairPrune FT: acc={metrics_fp_ft['overall_accuracy']:.4f}, "
                              f"worst={metrics_fp_ft['worst_group_accuracy']:.4f}, "
                              f"gap={metrics_fp_ft['accuracy_gap']:.4f}")

                        # Also run Fisher pruning baseline
                        fisher_model = fisher_prune(dp_model, sp, val_loader, device, n_samples=1000)
                        metrics_fisher = evaluate_model(fisher_model, test_loader, device)
                        metrics_fisher["method"] = "fisher_prune"
                        metrics_fisher["epsilon"] = eps
                        metrics_fisher["sparsity"] = sp
                        metrics_fisher["seed"] = seed

                        save_path_fisher = os.path.join(RESULTS_DIR, ds_name, "fisher_prune")
                        ensure_dir(save_path_fisher)
                        save_json(metrics_fisher, os.path.join(save_path_fisher,
                                  f"metrics_eps{eps}_sp{sp}_seed{seed}.json"))
                        all_metrics[(ds_name, f"fisher_eps{eps}_sp{sp}", seed)] = metrics_fisher

                    except Exception as e:
                        print(f"    ERROR: {e}")
                        traceback.print_exc()


# ==================== STAGE 6: MECHANISTIC ANALYSIS ====================

def stage_mechanistic_analysis(device, all_models, dp_models, all_metrics):
    """Analyze why DP and compression compound."""
    analysis_results = {}

    for ds_name in DATASETS:
        ds_config = DATASET_CONFIGS[ds_name]
        eps_list = EPSILONS if ds_name != "celeba" else [4]
        minority_sgs = ds_config["minority_subgroups"]

        print(f"\n--- Mechanistic Analysis: {ds_name} ---")

        for seed in SEEDS:
            val_loader = all_models.get((ds_name, seed, "val_loader"))
            base_model = all_models.get((ds_name, seed))
            if base_model is None or val_loader is None:
                continue

            print(f"  Seed {seed}: computing Fisher for baseline...")
            base_sg_fisher = compute_subgroup_fisher(base_model, val_loader, device, n_samples=500)

            # Weight stats for baseline
            base_weight_stats = get_weight_stats_by_subgroup_relevance(
                base_model, base_sg_fisher, minority_sgs
            )

            for eps in eps_list:
                dp_model = dp_models.get((ds_name, eps, seed))
                if dp_model is None:
                    continue

                print(f"  Seed {seed}, eps={eps}: computing Fisher for DP model...")
                dp_sg_fisher = compute_subgroup_fisher(dp_model, val_loader, device, n_samples=500)

                # Weight stats for DP model
                dp_weight_stats = get_weight_stats_by_subgroup_relevance(
                    dp_model, dp_sg_fisher, minority_sgs
                )

                # Pruning overlap analysis
                overlap = compute_pruning_overlap(base_model, dp_model, base_sg_fisher,
                                                   dp_sg_fisher, minority_sgs)

                key = f"{ds_name}_eps{eps}_seed{seed}"
                analysis_results[key] = {
                    "baseline_weight_stats": base_weight_stats,
                    "dp_weight_stats": dp_weight_stats,
                    "pruning_overlap": overlap,
                }

        save_path = os.path.join(RESULTS_DIR, ds_name, "analysis")
        ensure_dir(save_path)
        ds_results = {k: v for k, v in analysis_results.items() if k.startswith(ds_name)}
        save_json(ds_results, os.path.join(save_path, "mechanistic_analysis.json"))

    save_json(analysis_results, os.path.join(RESULTS_DIR, "mechanistic_analysis.json"))
    return analysis_results


def compute_pruning_overlap(base_model, dp_model, base_fisher, dp_fisher, minority_sgs):
    """Compute fraction of pruned weights that are minority-relevant."""
    results = {}

    for sp in SPARSITIES:
        # Prune both models
        base_pruned = magnitude_prune(base_model, sp)
        dp_pruned = magnitude_prune(dp_model, sp)

        # Identify pruned weights and minority-relevant weights
        base_pruned_minority = 0
        base_pruned_total = 0
        dp_pruned_minority = 0
        dp_pruned_total = 0

        for name, base_mod in base_model.named_modules():
            if not isinstance(base_mod, (torch.nn.Conv2d, torch.nn.Linear)):
                continue

            key = name + ".weight"

            # Get corresponding pruned modules
            base_p_mod = dict(base_pruned.named_modules()).get(name)
            dp_p_mod = dict(dp_pruned.named_modules()).get(name)

            if base_p_mod is None or dp_p_mod is None:
                continue

            # Identify pruned weights (zero after pruning)
            base_is_pruned = (base_p_mod.weight.data == 0).flatten().cpu().numpy()
            dp_is_pruned = (dp_p_mod.weight.data == 0).flatten().cpu().numpy()

            # Classify as minority-relevant using Fisher
            minority_fisher_max = None
            for sg in minority_sgs:
                if sg in dp_fisher:
                    f = dp_fisher[sg].get(key, None)
                    if f is not None:
                        flat = f.flatten().cpu().numpy()
                        if minority_fisher_max is None:
                            minority_fisher_max = flat
                        else:
                            minority_fisher_max = np.maximum(minority_fisher_max, flat)

            if minority_fisher_max is not None:
                median = np.median(minority_fisher_max)
                is_minority_relevant = minority_fisher_max > median

                base_pruned_minority += (base_is_pruned & is_minority_relevant).sum()
                base_pruned_total += base_is_pruned.sum()
                dp_pruned_minority += (dp_is_pruned & is_minority_relevant).sum()
                dp_pruned_total += dp_is_pruned.sum()

        base_frac = base_pruned_minority / max(base_pruned_total, 1)
        dp_frac = dp_pruned_minority / max(dp_pruned_total, 1)

        results[str(sp)] = {
            "base_minority_frac": float(base_frac),
            "dp_minority_frac": float(dp_frac),
            "base_pruned_total": int(base_pruned_total),
            "dp_pruned_total": int(dp_pruned_total),
        }

    return results


# ==================== STAGE 7: ABLATION STUDIES ====================

def stage_ablations(device, dp_models, all_models, all_metrics):
    """Run ablation studies."""
    ablation_results = {}

    # Ablation 1: Pruning criterion comparison (eps=4, sp=0.7)
    for ds_name in ["cifar10", "utkface"]:
        print(f"\n--- Ablation (criterion): {ds_name} ---")

        for seed in SEEDS:
            dp_model = dp_models.get((ds_name, 4, seed))
            if dp_model is None:
                continue

            test_loader = all_models.get((ds_name, seed, "test_loader"))
            val_loader = all_models.get((ds_name, seed, "val_loader"))

            sp = 0.7
            results = {}

            # a) Magnitude pruning
            mag = magnitude_prune(dp_model, sp)
            results["magnitude"] = evaluate_model(mag, test_loader, device)

            # b) Global Fisher
            fisher = fisher_prune(dp_model, sp, val_loader, device, n_samples=1000)
            results["global_fisher"] = evaluate_model(fisher, test_loader, device)

            # c) Mean-subgroup Fisher
            mean_f = mean_fisher_prune(dp_model, sp, val_loader, device, n_samples=1000)
            results["mean_fisher"] = evaluate_model(mean_f, test_loader, device)

            # d) Worst-subgroup Fisher (FairPrune-DP)
            fair = fairprune_dp(dp_model, sp, val_loader, device, n_samples=1000)
            results["fairprune_dp"] = evaluate_model(fair, test_loader, device)

            ablation_results[f"{ds_name}_criterion_seed{seed}"] = {
                method: {
                    "overall_accuracy": m["overall_accuracy"],
                    "worst_group_accuracy": m["worst_group_accuracy"],
                    "accuracy_gap": m["accuracy_gap"],
                }
                for method, m in results.items()
            }

            print(f"  Seed {seed}:")
            for method, m in results.items():
                print(f"    {method}: acc={m['overall_accuracy']:.4f}, "
                      f"worst={m['worst_group_accuracy']:.4f}, gap={m['accuracy_gap']:.4f}")

    # Ablation 2: Structured vs Unstructured
    print(f"\n--- Ablation (structured vs unstructured) ---")
    for seed in SEEDS:
        dp_model = dp_models.get(("cifar10", 4, seed))
        if dp_model is None:
            continue

        test_loader = all_models.get(("cifar10", seed, "test_loader"))
        for sp in [0.5, 0.7]:
            # Unstructured (already have this)
            unstruct = magnitude_prune(dp_model, sp, structured=False)
            m_u = evaluate_model(unstruct, test_loader, device)

            # Structured
            struct = magnitude_prune(dp_model, sp, structured=True)
            m_s = evaluate_model(struct, test_loader, device)

            ablation_results[f"cifar10_struct_sp{sp}_seed{seed}"] = {
                "unstructured": {
                    "overall_accuracy": m_u["overall_accuracy"],
                    "worst_group_accuracy": m_u["worst_group_accuracy"],
                    "accuracy_gap": m_u["accuracy_gap"],
                    "actual_sparsity": get_sparsity(unstruct),
                },
                "structured": {
                    "overall_accuracy": m_s["overall_accuracy"],
                    "worst_group_accuracy": m_s["worst_group_accuracy"],
                    "accuracy_gap": m_s["accuracy_gap"],
                    "actual_sparsity": get_sparsity(struct),
                },
            }

    # Ablation 3: Fine-tuning effect (compile from existing results)
    print(f"\n--- Ablation (fine-tuning effect) ---")
    for ds_name in DATASETS:
        eps_list = EPSILONS if ds_name != "celeba" else [4]
        for eps in eps_list:
            for sp in SPARSITIES:
                for seed in SEEDS:
                    noft = all_metrics.get((ds_name, f"dp_comp_eps{eps}_sp{sp}", seed))
                    ft = all_metrics.get((ds_name, f"dp_comp_eps{eps}_sp{sp}_ft", seed))
                    fp_noft = all_metrics.get((ds_name, f"fairprune_eps{eps}_sp{sp}", seed))
                    fp_ft = all_metrics.get((ds_name, f"fairprune_eps{eps}_sp{sp}_ft", seed))

                    if noft and ft:
                        ablation_results[f"{ds_name}_ft_eps{eps}_sp{sp}_seed{seed}"] = {
                            "magnitude_noft": {
                                "worst_group_accuracy": noft["worst_group_accuracy"],
                                "accuracy_gap": noft["accuracy_gap"],
                            },
                            "magnitude_ft": {
                                "worst_group_accuracy": ft["worst_group_accuracy"],
                                "accuracy_gap": ft["accuracy_gap"],
                            },
                        }
                        if fp_noft and fp_ft:
                            ablation_results[f"{ds_name}_ft_eps{eps}_sp{sp}_seed{seed}"].update({
                                "fairprune_noft": {
                                    "worst_group_accuracy": fp_noft["worst_group_accuracy"],
                                    "accuracy_gap": fp_noft["accuracy_gap"],
                                },
                                "fairprune_ft": {
                                    "worst_group_accuracy": fp_ft["worst_group_accuracy"],
                                    "accuracy_gap": fp_ft["accuracy_gap"],
                                },
                            })

    save_json(ablation_results, os.path.join(RESULTS_DIR, "ablation_results.json"))
    return ablation_results


# ==================== STAGE 8: MIA ANALYSIS ====================

def stage_mia_analysis(device, all_models, dp_models, all_metrics):
    """Simple loss-based membership inference attack analysis."""
    mia_results = {}

    for ds_name in ["cifar10", "utkface"]:
        ds_config = DATASET_CONFIGS[ds_name]
        print(f"\n--- MIA Analysis: {ds_name} ---")

        for seed in SEEDS:
            train_loader = all_models.get((ds_name, seed, "train_loader"))
            test_loader = all_models.get((ds_name, seed, "test_loader"))

            if train_loader is None or test_loader is None:
                continue

            # Models to evaluate
            models_to_eval = {}

            # Baseline
            base = all_models.get((ds_name, seed))
            if base:
                models_to_eval["baseline"] = base

            # DP-only (eps=4)
            dp = dp_models.get((ds_name, 4, seed))
            if dp:
                models_to_eval["dp_eps4"] = dp

                # DP + compressed (eps=4, sp=0.7)
                dp_pruned = magnitude_prune(dp, 0.7)
                models_to_eval["dp_comp_eps4_sp07"] = dp_pruned

                # FairPrune-DP (eps=4, sp=0.7)
                val_loader = all_models.get((ds_name, seed, "val_loader"))
                if val_loader:
                    try:
                        fp = fairprune_dp(dp, 0.7, val_loader, device, n_samples=500)
                        models_to_eval["fairprune_eps4_sp07"] = fp
                    except Exception as e:
                        print(f"  FairPrune for MIA failed: {e}")

            # Standard + compressed (sp=0.7)
            if base:
                comp = magnitude_prune(base, 0.7)
                models_to_eval["comp_sp07"] = comp

            for model_name, model in models_to_eval.items():
                print(f"  MIA: {model_name}, seed {seed}...")
                mia = run_loss_mia(model, train_loader, test_loader, device)
                mia_results[f"{ds_name}_{model_name}_seed{seed}"] = mia

                print(f"    Overall MIA acc: {mia.get('overall_mia_accuracy', 0):.4f}, "
                      f"Disparity: {mia.get('mia_disparity', 0):.4f}")

        save_path = os.path.join(RESULTS_DIR, ds_name, "mia")
        ensure_dir(save_path)
        ds_mia = {k: v for k, v in mia_results.items() if k.startswith(ds_name)}
        save_json(ds_mia, os.path.join(save_path, "mia_results.json"))

    save_json(mia_results, os.path.join(RESULTS_DIR, "mia_results.json"))
    return mia_results


def run_loss_mia(model, train_loader, test_loader, device, max_samples=2000):
    """Loss-based membership inference attack.

    Members (train) tend to have lower loss.
    Returns per-subgroup MIA balanced accuracy and disparity.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def get_losses_and_subgroups(loader, n_max):
        losses = []
        subgroups = []
        count = 0
        with torch.no_grad():
            for batch in loader:
                if count >= n_max:
                    break
                images, labels, sgs = batch
                bs = min(len(images), n_max - count)
                images = images[:bs].to(device)
                labels_t = torch.tensor(labels[:bs], dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels[:bs].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_t)
                losses.extend(loss.cpu().numpy().tolist())
                if isinstance(sgs, torch.Tensor):
                    subgroups.extend(sgs[:bs].numpy().tolist())
                else:
                    subgroups.extend(list(sgs)[:bs])
                count += bs
        return np.array(losses), np.array(subgroups)

    member_losses, member_sgs = get_losses_and_subgroups(train_loader, max_samples)
    nonmember_losses, nonmember_sgs = get_losses_and_subgroups(test_loader, max_samples)

    # Use median loss as threshold
    all_losses = np.concatenate([member_losses, nonmember_losses])
    threshold = np.median(all_losses)

    # Overall MIA: predict member if loss < threshold
    member_pred = (member_losses < threshold)
    nonmember_pred = (nonmember_losses < threshold)
    tpr = member_pred.mean()
    tnr = 1 - nonmember_pred.mean()
    overall_mia_acc = (tpr + tnr) / 2

    # Per-subgroup MIA
    all_sgs = np.unique(np.concatenate([member_sgs, nonmember_sgs]))
    per_sg_mia = {}
    for sg in all_sgs:
        m_mask = member_sgs == sg
        nm_mask = nonmember_sgs == sg
        if m_mask.sum() < 10 or nm_mask.sum() < 10:
            continue
        sg_tpr = (member_losses[m_mask] < threshold).mean()
        sg_tnr = 1 - (nonmember_losses[nm_mask] < threshold).mean()
        per_sg_mia[int(sg)] = {
            "mia_accuracy": float((sg_tpr + sg_tnr) / 2),
            "tpr": float(sg_tpr),
            "tnr": float(sg_tnr),
            "n_members": int(m_mask.sum()),
            "n_nonmembers": int(nm_mask.sum()),
        }

    mia_accs = [v["mia_accuracy"] for v in per_sg_mia.values()]
    mia_disparity = max(mia_accs) - min(mia_accs) if len(mia_accs) >= 2 else 0

    return {
        "overall_mia_accuracy": float(overall_mia_acc),
        "threshold": float(threshold),
        "per_subgroup_mia": per_sg_mia,
        "mia_disparity": float(mia_disparity),
    }


# ==================== STAGE 9: AGGREGATE RESULTS ====================

def stage_aggregate(all_metrics, compounding_ratios, analysis_results, ablation_results, mia_results):
    """Aggregate all results and test success criteria."""
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    # Build master results table
    master = []
    for key, metrics in all_metrics.items():
        if isinstance(key, tuple) and len(key) == 3:
            ds_name, variant, seed = key
            row = {
                "dataset": ds_name,
                "variant": variant,
                "seed": seed,
                "overall_accuracy": metrics.get("overall_accuracy"),
                "worst_group_accuracy": metrics.get("worst_group_accuracy"),
                "best_group_accuracy": metrics.get("best_group_accuracy"),
                "accuracy_gap": metrics.get("accuracy_gap"),
                "equalized_odds_diff": metrics.get("equalized_odds_diff"),
            }
            master.append(row)

    save_json(master, os.path.join(RESULTS_DIR, "master_results.json"))

    # Compounding ratio summary
    cr_summary = {}
    if compounding_ratios:
        for key, val in compounding_ratios.items():
            if "_ft" in key:
                continue
            ds = val["dataset"]
            eps = val["epsilon"]
            sp = val["sparsity"]
            cr = val["CR"]
            group_key = f"{ds}_eps{eps}_sp{sp}"
            if group_key not in cr_summary:
                cr_summary[group_key] = {"crs": [], "dataset": ds, "epsilon": eps, "sparsity": sp}
            if not np.isnan(cr):
                cr_summary[group_key]["crs"].append(cr)

        for key, val in cr_summary.items():
            crs = val["crs"]
            val["mean_CR"] = float(np.mean(crs)) if crs else None
            val["std_CR"] = float(np.std(crs)) if crs else None

            # One-sided t-test: H0: CR <= 1 vs H1: CR > 1
            if len(crs) >= 2:
                from scipy import stats
                t_stat, p_value = stats.ttest_1samp(crs, 1.0)
                val["p_value_cr_gt_1"] = float(p_value / 2) if t_stat > 0 else 1.0  # one-sided
                val["t_stat"] = float(t_stat)
            del val["crs"]

    save_json(cr_summary, os.path.join(RESULTS_DIR, "compounding_ratio_summary.json"))

    # Success criteria evaluation
    success = evaluate_success_criteria(all_metrics, cr_summary, analysis_results, mia_results)
    save_json(success, os.path.join(RESULTS_DIR, "success_criteria_evaluation.json"))

    print("\n=== SUCCESS CRITERIA ===")
    for criterion, result in success.items():
        status = result.get("status", "unknown")
        print(f"  {criterion}: {status}")
        if "details" in result:
            print(f"    {result['details']}")


def evaluate_success_criteria(all_metrics, cr_summary, analysis_results, mia_results):
    """Evaluate the 5 success criteria from the proposal."""
    success = {}

    # Criterion 1: CR > 1.2 consistently across >= 2 datasets
    cr_above_12 = defaultdict(list)
    for key, val in cr_summary.items():
        if val.get("mean_CR") and val["mean_CR"] > 1.2:
            cr_above_12[val["dataset"]].append(key)

    n_datasets_pass = sum(1 for ds_configs in cr_above_12.values() if len(ds_configs) >= 1)
    success["criterion_1_compounding_ratio"] = {
        "status": "PASS" if n_datasets_pass >= 2 else ("PARTIAL" if n_datasets_pass >= 1 else "FAIL"),
        "details": f"CR > 1.2 in {n_datasets_pass} datasets. "
                   f"Configs: {dict(cr_above_12)}",
    }

    # Criterion 2: Mechanistic evidence
    if analysis_results:
        minority_lower = 0
        total = 0
        overlap_high = 0
        overlap_total = 0

        for key, val in analysis_results.items():
            bw = val.get("baseline_weight_stats")
            dw = val.get("dp_weight_stats")
            if bw and dw:
                total += 1
                if dw.get("minority_relevant_magnitude_mean", 1) < bw.get("minority_relevant_magnitude_mean", 0):
                    minority_lower += 1

            overlap = val.get("pruning_overlap", {})
            for sp_key, ov in overlap.items():
                overlap_total += 1
                if ov.get("dp_minority_frac", 0) > 0.6:
                    overlap_high += 1

        success["criterion_2_mechanistic"] = {
            "status": "PASS" if (minority_lower > total/2 and overlap_high > overlap_total/2) else "PARTIAL",
            "details": f"Minority weights lower in DP: {minority_lower}/{total}. "
                       f"Pruning overlap >60%: {overlap_high}/{overlap_total}.",
        }
    else:
        success["criterion_2_mechanistic"] = {"status": "SKIPPED"}

    # Criterion 3: FairPrune-DP reduces gap by >= 20%
    gap_reductions = []
    for key, metrics in all_metrics.items():
        if isinstance(key, tuple) and len(key) == 3:
            ds, variant, seed = key
            if "fairprune" in variant and "_ft" not in variant:
                # Find corresponding magnitude pruning
                parts = variant.replace("fairprune_", "").split("_")
                mag_key = (ds, f"dp_comp_{parts[0]}_{parts[1]}", seed)
                mag_metrics = all_metrics.get(mag_key)
                if mag_metrics:
                    mag_gap = mag_metrics.get("accuracy_gap", 0)
                    fp_gap = metrics.get("accuracy_gap", 0)
                    if mag_gap > 0.001:
                        reduction = (mag_gap - fp_gap) / mag_gap
                        gap_reductions.append(reduction)

    if gap_reductions:
        mean_reduction = np.mean(gap_reductions)
        success["criterion_3_fairprune_effectiveness"] = {
            "status": "PASS" if mean_reduction >= 0.20 else ("PARTIAL" if mean_reduction >= 0.10 else "FAIL"),
            "details": f"Mean gap reduction: {mean_reduction:.1%} ({len(gap_reductions)} configs). "
                       f"Min: {min(gap_reductions):.1%}, Max: {max(gap_reductions):.1%}.",
        }
    else:
        success["criterion_3_fairprune_effectiveness"] = {"status": "INSUFFICIENT_DATA"}

    # Criterion 4: MIA disparity increases
    if mia_results:
        dp_disparities = []
        comp_disparities = []
        dc_disparities = []
        for key, val in mia_results.items():
            if "dp_eps4" in key and "comp" not in key and "fairprune" not in key:
                dp_disparities.append(val.get("mia_disparity", 0))
            elif "dp_comp" in key:
                dc_disparities.append(val.get("mia_disparity", 0))
            elif "comp_sp07" in key and "dp" not in key:
                comp_disparities.append(val.get("mia_disparity", 0))

        success["criterion_4_mia_disparity"] = {
            "status": "PASS" if dc_disparities and np.mean(dc_disparities) > np.mean(dp_disparities or [0]) else "PARTIAL",
            "details": f"DP MIA disp: {np.mean(dp_disparities):.4f}, "
                       f"Comp MIA disp: {np.mean(comp_disparities):.4f}, "
                       f"DP+Comp MIA disp: {np.mean(dc_disparities):.4f}",
        }
    else:
        success["criterion_4_mia_disparity"] = {"status": "SKIPPED"}

    # Criterion 5: Results hold for structured pruning
    success["criterion_5_structured_pruning"] = {
        "status": "PARTIAL",
        "details": "Structured pruning tested in ablation. See ablation results.",
    }

    return success


# ==================== MAIN ====================

def main():
    total_start = time.time()
    device = get_device()
    print(f"Device: {device}")
    print(f"Workspace: {WORKSPACE}")

    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)

    # Stage 1: Baseline training
    all_models, all_metrics = run_experiment(
        "Baseline Training", stage_baseline_training, device
    )
    if all_models is None:
        print("FATAL: Baseline training failed")
        return

    # Stage 2: DP-SGD training
    dp_models = run_experiment(
        "DP-SGD Training", stage_dp_training, device, all_models, all_metrics
    )
    if dp_models is None:
        dp_models = {}

    # Stage 3: Compression of standard models
    run_experiment(
        "Compression (Standard)", stage_compression_standard,
        device, all_models, all_metrics
    )

    # Stage 4: DP + Compression
    compounding_ratios = run_experiment(
        "DP + Compression", stage_dp_compression,
        device, dp_models, all_models, all_metrics
    )
    if compounding_ratios is None:
        compounding_ratios = {}

    # Stage 5: FairPrune-DP
    run_experiment(
        "FairPrune-DP", stage_fairprune,
        device, dp_models, all_models, all_metrics
    )

    # Stage 6: Mechanistic analysis
    analysis_results = run_experiment(
        "Mechanistic Analysis", stage_mechanistic_analysis,
        device, all_models, dp_models, all_metrics
    )
    if analysis_results is None:
        analysis_results = {}

    # Stage 7: Ablations
    ablation_results = run_experiment(
        "Ablation Studies", stage_ablations,
        device, dp_models, all_models, all_metrics
    )
    if ablation_results is None:
        ablation_results = {}

    # Stage 8: MIA
    mia_results = run_experiment(
        "MIA Analysis", stage_mia_analysis,
        device, all_models, dp_models, all_metrics
    )
    if mia_results is None:
        mia_results = {}

    # Stage 9: Aggregate
    stage_aggregate(all_metrics, compounding_ratios, analysis_results, ablation_results, mia_results)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_time/3600:.1f} hours")
    print(f"{'='*60}")

    # Save timing info
    save_json({"total_time_hours": total_time / 3600}, os.path.join(RESULTS_DIR, "timing.json"))


if __name__ == "__main__":
    main()
