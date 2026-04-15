#!/usr/bin/env python3
"""Run missing FairPrune-DP experiments for CIFAR-10 and ablation studies."""

import sys, os, json, torch
import numpy as np

# Add shared to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from shared.models import get_model
from shared.data_loader import get_dataset, make_loader
from shared.compression import fairprune_dp, magnitude_prune, fisher_prune, mean_fisher_prune, finalize_pruning, get_sparsity
from shared.training import finetune_with_masks
from shared.metrics import evaluate_model

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SEEDS = [42, 123, 456, 789, 1024]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def run_fairprune_cifar10():
    """Run FairPrune-DP on CIFAR-10 DP-trained models."""
    dataset_name = "cifar10"
    num_classes = 10
    out_dir = os.path.join(RESULTS_DIR, dataset_name, "fairprune_dp")
    os.makedirs(out_dir, exist_ok=True)

    for eps in [1, 4, 8]:
        for sp_pct in [50, 70, 90]:
            sparsity = sp_pct / 100.0
            for seed in SEEDS:
                out_file = os.path.join(out_dir, f"metrics_eps{eps}_sp{sp_pct}_ft_seed{seed}.json")
                if os.path.exists(out_file):
                    print(f"  SKIP {dataset_name} fairprune eps={eps} sp={sp_pct} seed={seed} (exists)")
                    continue

                print(f"  Running {dataset_name} fairprune eps={eps} sp={sp_pct} seed={seed}...")

                # Load DP-trained model
                model_path = os.path.join(RESULTS_DIR, dataset_name, "dp_only", f"model_eps{eps}_seed{seed}.pt")
                if not os.path.exists(model_path):
                    print(f"    SKIP: model not found: {model_path}")
                    continue

                torch.manual_seed(seed)
                np.random.seed(seed)

                model = get_model("resnet18", num_classes)
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

                # Load data
                train_ds, val_ds, test_ds, stats = get_dataset(dataset_name, seed=42)
                val_loader = make_loader(val_ds, batch_size=256, shuffle=False, num_workers=0)
                train_loader = make_loader(train_ds, batch_size=256, shuffle=True, num_workers=0)
                test_loader = make_loader(test_ds, batch_size=256, shuffle=False, num_workers=0)

                # Apply FairPrune-DP
                pruned_model = fairprune_dp(model, sparsity, val_loader, DEVICE, n_samples=2000, alpha=0.5)

                # Fine-tune
                ft_config = {"ft_lr": 0.001, "ft_epochs": 5}
                pruned_model = finetune_with_masks(pruned_model, train_loader, ft_config, DEVICE)
                pruned_model = finalize_pruning(pruned_model)

                # Evaluate
                actual_sparsity = get_sparsity(pruned_model)
                metrics = evaluate_model(pruned_model, test_loader, DEVICE)
                # Remove bulky per-sample data
                for k in ["per_sample_losses", "per_sample_labels", "per_sample_subgroups", "per_sample_correct"]:
                    metrics.pop(k, None)
                metrics.update({
                    "seed": seed, "epsilon": eps, "sparsity": sparsity,
                    "actual_sparsity": actual_sparsity, "finetuned": True,
                })

                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"    Done: OA={metrics['overall_accuracy']:.4f} WGA={metrics['worst_group_accuracy']:.4f} Gap={metrics['accuracy_gap']:.4f}")

                del model, pruned_model
                torch.cuda.empty_cache()


def run_ablation_cifar10():
    """Run ablation: magnitude vs fisher vs mean-fisher vs fairprune on CIFAR-10 eps=4 sp=70%."""
    dataset_name = "cifar10"
    num_classes = 10
    eps = 4
    sp_pct = 70
    sparsity = 0.7
    out_dir = os.path.join(RESULTS_DIR, dataset_name, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    for seed in SEEDS:
        model_path = os.path.join(RESULTS_DIR, dataset_name, "dp_only", f"model_eps{eps}_seed{seed}.pt")
        if not os.path.exists(model_path):
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)
        base_model = get_model("resnet18", num_classes)
        base_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

        train_ds, val_ds, test_ds, stats = get_dataset(dataset_name, seed=42)
        val_loader = make_loader(val_ds, batch_size=256, shuffle=False, num_workers=0)
        train_loader = make_loader(train_ds, batch_size=256, shuffle=True, num_workers=0)
        test_loader = make_loader(test_ds, batch_size=256, shuffle=False, num_workers=0)
        ft_config = {"ft_lr": 0.001, "ft_epochs": 5}

        methods = {
            "magnitude": lambda m: magnitude_prune(m, sparsity, keep_masks=True),
            "global_fisher": lambda m: fisher_prune(m, sparsity, val_loader, DEVICE, n_samples=1000, keep_masks=True),
            "mean_fisher": lambda m: mean_fisher_prune(m, sparsity, val_loader, DEVICE, n_samples=1000, keep_masks=True),
            "fairprune_dp": lambda m: fairprune_dp(m, sparsity, val_loader, DEVICE, n_samples=2000, alpha=0.5, keep_masks=True),
        }

        for method_name, prune_fn in methods.items():
            out_file = os.path.join(out_dir, f"ablation_{method_name}_eps{eps}_sp{sp_pct}_seed{seed}.json")
            if os.path.exists(out_file):
                print(f"  SKIP ablation {method_name} seed={seed} (exists)")
                continue

            print(f"  Running ablation {method_name} seed={seed}...")
            pruned = prune_fn(base_model)
            pruned = finetune_with_masks(pruned, train_loader, ft_config, DEVICE)
            pruned = finalize_pruning(pruned)
            actual_sp = get_sparsity(pruned)
            metrics = evaluate_model(pruned, test_loader, DEVICE)
            for k in ["per_sample_losses", "per_sample_labels", "per_sample_subgroups", "per_sample_correct"]:
                metrics.pop(k, None)
            metrics.update({
                "seed": seed, "epsilon": eps, "sparsity": sparsity,
                "actual_sparsity": actual_sp, "method": method_name, "finetuned": True,
            })
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"    Done: OA={metrics['overall_accuracy']:.4f} WGA={metrics['worst_group_accuracy']:.4f} Gap={metrics['accuracy_gap']:.4f}")
            del pruned
            torch.cuda.empty_cache()

        del base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("Running missing FairPrune-DP experiments for CIFAR-10")
    print("=" * 60)
    run_fairprune_cifar10()

    print()
    print("=" * 60)
    print("Running ablation study on CIFAR-10")
    print("=" * 60)
    run_ablation_cifar10()

    print("\nAll done!")
