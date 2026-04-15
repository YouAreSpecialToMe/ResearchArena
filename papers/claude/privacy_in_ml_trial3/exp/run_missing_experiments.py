"""Run all missing experiments:
1. UTKFace dp_comp for eps={4,8} × sp={50,70,90} × 5 seeds
2. FairPrune-DP on CIFAR-10 and UTKFace for eps={1,4,8} × sp={50,70,90} × 5 seeds
3. Ablation: global Fisher vs magnitude vs FairPrune-DP on CIFAR-10 eps=4, sp=70
"""
import sys
import os
import json
import copy
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from shared.models import get_model
from shared.data_loader import get_dataset, make_loader
from shared.compression import (magnitude_prune, fisher_prune, fairprune_dp,
                                 mean_fisher_prune, finalize_pruning, get_sparsity)
from shared.training import finetune_with_masks
from shared.metrics import evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SEEDS = [42, 123, 456, 789, 1024]
EPSILONS = [1, 4, 8]
SPARSITIES = [50, 70, 90]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(path, num_classes):
    model = get_model("resnet18", num_classes)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def save_metrics(metrics, path):
    # Remove large per-sample data to save space
    m = {k: v for k, v in metrics.items()
         if k not in ("per_sample_losses", "per_sample_labels", "per_sample_subgroups", "per_sample_correct")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)


def run_dp_comp_prune(dataset_name, eps, sp, seed, num_classes, train_ds, test_ds):
    """Prune a DP-trained model and evaluate."""
    model_path = os.path.join(RESULTS_BASE, dataset_name, "dp_only",
                               f"model_eps{eps}_seed{seed}.pt")
    out_path = os.path.join(RESULTS_BASE, dataset_name, "dp_comp",
                             f"metrics_eps{eps}_sp{sp}_ft_seed{seed}.json")

    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} exists")
        return

    if not os.path.exists(model_path):
        print(f"  [SKIP] No DP model: {model_path}")
        return

    set_seed(seed)
    model = load_model(model_path, num_classes)

    # Prune with masks kept for fine-tuning
    sparsity = sp / 100.0
    pruned = magnitude_prune(model, sparsity, keep_masks=True)

    # Fine-tune (standard, not DP - matching existing experiments)
    train_loader = make_loader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    pruned = finetune_with_masks(pruned, train_loader, {"ft_lr": 0.001, "ft_epochs": 5}, DEVICE)
    pruned = finalize_pruning(pruned)

    # Evaluate
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    metrics = evaluate_model(pruned, test_loader, DEVICE)
    metrics["seed"] = seed
    metrics["epsilon"] = eps
    metrics["sparsity"] = sparsity
    metrics["actual_sparsity"] = get_sparsity(pruned)
    metrics["finetuned"] = True

    save_metrics(metrics, out_path)
    print(f"  [DONE] {dataset_name} dp_comp eps={eps} sp={sp} seed={seed}: "
          f"OA={metrics['overall_accuracy']:.4f}, WGA={metrics['worst_group_accuracy']:.4f}")

    del pruned, model
    torch.cuda.empty_cache()


def run_fairprune(dataset_name, eps, sp, seed, num_classes, train_ds, val_ds, test_ds):
    """Run FairPrune-DP on a DP model and evaluate."""
    model_path = os.path.join(RESULTS_BASE, dataset_name, "dp_only",
                               f"model_eps{eps}_seed{seed}.pt")
    out_dir = os.path.join(RESULTS_BASE, dataset_name, "fairprune_dp")
    out_path = os.path.join(out_dir, f"metrics_eps{eps}_sp{sp}_ft_seed{seed}.json")

    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} exists")
        return

    if not os.path.exists(model_path):
        print(f"  [SKIP] No DP model: {model_path}")
        return

    set_seed(seed)
    model = load_model(model_path, num_classes)

    # Use val set for Fisher computation
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    sparsity = sp / 100.0
    pruned = fairprune_dp(model, sparsity, val_loader, DEVICE, n_samples=2000, keep_masks=True)

    # Fine-tune
    train_loader = make_loader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    pruned = finetune_with_masks(pruned, train_loader, {"ft_lr": 0.001, "ft_epochs": 5}, DEVICE)
    pruned = finalize_pruning(pruned)

    # Evaluate
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    metrics = evaluate_model(pruned, test_loader, DEVICE)
    metrics["seed"] = seed
    metrics["epsilon"] = eps
    metrics["sparsity"] = sparsity
    metrics["actual_sparsity"] = get_sparsity(pruned)
    metrics["finetuned"] = True
    metrics["method"] = "fairprune_dp"

    save_metrics(metrics, out_path)
    print(f"  [DONE] FairPrune {dataset_name} eps={eps} sp={sp} seed={seed}: "
          f"OA={metrics['overall_accuracy']:.4f}, WGA={metrics['worst_group_accuracy']:.4f}")

    del pruned, model
    torch.cuda.empty_cache()


def run_ablation(dataset_name, eps, sp, seed, num_classes, train_ds, val_ds, test_ds):
    """Run ablation: magnitude vs global Fisher vs mean Fisher vs FairPrune-DP."""
    model_path = os.path.join(RESULTS_BASE, dataset_name, "dp_only",
                               f"model_eps{eps}_seed{seed}.pt")
    out_dir = os.path.join(RESULTS_BASE, dataset_name, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"  [SKIP] No DP model: {model_path}")
        return

    set_seed(seed)
    model = load_model(model_path, num_classes)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    train_loader = make_loader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    sparsity = sp / 100.0

    methods = {
        "magnitude": lambda m, s: magnitude_prune(m, s, keep_masks=True),
        "global_fisher": lambda m, s: fisher_prune(m, s, val_loader, DEVICE, n_samples=1000, keep_masks=True),
        "mean_fisher": lambda m, s: mean_fisher_prune(m, s, val_loader, DEVICE, n_samples=1000, keep_masks=True),
        "fairprune_dp": lambda m, s: fairprune_dp(m, s, val_loader, DEVICE, n_samples=2000, keep_masks=True),
    }

    for method_name, prune_fn in methods.items():
        out_path = os.path.join(out_dir, f"ablation_{method_name}_eps{eps}_sp{sp}_seed{seed}.json")
        if os.path.exists(out_path):
            print(f"  [SKIP] {out_path} exists")
            continue

        set_seed(seed)
        pruned = prune_fn(model, sparsity)
        pruned = finetune_with_masks(pruned, train_loader, {"ft_lr": 0.001, "ft_epochs": 5}, DEVICE)
        pruned = finalize_pruning(pruned)

        metrics = evaluate_model(pruned, test_loader, DEVICE)
        metrics["seed"] = seed
        metrics["epsilon"] = eps
        metrics["sparsity"] = sparsity
        metrics["actual_sparsity"] = get_sparsity(pruned)
        metrics["method"] = method_name

        save_metrics(metrics, out_path)
        print(f"  [DONE] Ablation {method_name} {dataset_name} eps={eps} sp={sp} seed={seed}: "
              f"OA={metrics['overall_accuracy']:.4f}, WGA={metrics['worst_group_accuracy']:.4f}")

        del pruned
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()


def main():
    print(f"Device: {DEVICE}")
    start_time = time.time()

    # =============================================
    # Part 1: UTKFace dp_comp for eps=4,8
    # =============================================
    print("\n" + "="*60)
    print("PART 1: UTKFace DP+Comp (eps=4,8)")
    print("="*60)

    # Load UTKFace data once
    print("Loading UTKFace dataset...")
    utk_train, utk_val, utk_test, utk_stats = get_dataset("utkface", seed=42)
    utk_num_classes = utk_stats["num_classes"]

    for eps in [4, 8]:
        for sp in SPARSITIES:
            for seed in SEEDS:
                run_dp_comp_prune("utkface", eps, sp, seed, utk_num_classes,
                                  utk_train, utk_test)

    # Also fill in missing eps=1 seeds (789, 1024)
    for sp in SPARSITIES:
        for seed in SEEDS:
            run_dp_comp_prune("utkface", 1, sp, seed, utk_num_classes,
                              utk_train, utk_test)

    print(f"\nPart 1 done in {time.time()-start_time:.0f}s")

    # =============================================
    # Part 2: FairPrune-DP on both datasets
    # =============================================
    print("\n" + "="*60)
    print("PART 2: FairPrune-DP")
    print("="*60)

    # UTKFace FairPrune
    for eps in EPSILONS:
        for sp in SPARSITIES:
            for seed in SEEDS:
                run_fairprune("utkface", eps, sp, seed, utk_num_classes,
                             utk_train, utk_val, utk_test)

    # Load CIFAR-10
    print("\nLoading CIFAR-10 dataset...")
    c10_train, c10_val, c10_test, c10_stats = get_dataset("cifar10", seed=42)
    c10_num_classes = c10_stats["num_classes"]

    # CIFAR-10 FairPrune
    for eps in EPSILONS:
        for sp in SPARSITIES:
            for seed in SEEDS:
                run_fairprune("cifar10", eps, sp, seed, c10_num_classes,
                             c10_train, c10_val, c10_test)

    print(f"\nPart 2 done in {time.time()-start_time:.0f}s")

    # =============================================
    # Part 3: Ablation study (CIFAR-10 eps=4, sp=70)
    # =============================================
    print("\n" + "="*60)
    print("PART 3: Ablation (CIFAR-10 eps=4, sp=70 & UTKFace eps=4, sp=70)")
    print("="*60)

    for seed in SEEDS:
        run_ablation("cifar10", 4, 70, seed, c10_num_classes,
                     c10_train, c10_val, c10_test)

    for seed in SEEDS:
        run_ablation("utkface", 4, 70, seed, utk_num_classes,
                     utk_train, utk_val, utk_test)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
