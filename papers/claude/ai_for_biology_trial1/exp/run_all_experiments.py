"""Run all experiments: baselines, main method, and ablations."""
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")
import torch

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from exp.shared.utils import SEEDS, set_seed, save_results
from exp.shared.data_loader import EmbeddingDataset
from exp.shared.train import run_flat_supcon, run_joint_hierarchical, run_currec
from exp.shared.eval import evaluate_model

RESULTS_DIR = os.path.join(BASE_DIR, "exp", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_datasets():
    """Load all datasets."""
    print("Loading datasets...")
    train_ds = EmbeddingDataset("train")
    new392_ds = EmbeddingDataset("new392")
    price149_ds = EmbeddingDataset("price149")

    test_datasets = {"new392": new392_ds, "price149": price149_ds}

    # Load rare classes
    data_dir = os.path.join(BASE_DIR, "exp", "data")
    with open(os.path.join(data_dir, "dataset_stats.json")) as f:
        stats = json.load(f)
    rare_classes_str = set(stats.get("rare_l4_classes", []))

    print(f"  Train: {train_ds.n_samples} samples, {train_ds.n_classes('ec_l4')} L4 classes")
    print(f"  New-392: {new392_ds.n_samples} samples")
    print(f"  Price-149: {price149_ds.n_samples} samples")
    print(f"  Rare L4 classes: {len(rare_classes_str)}")

    return train_ds, test_datasets, rare_classes_str


def run_baselines(train_ds, test_datasets, rare_classes_str):
    """Run baseline experiments."""
    # 1. Flat SupCon
    print("\n" + "="*60)
    print("FLAT SUPCON BASELINE")
    print("="*60)

    for seed in SEEDS:
        result_path = os.path.join(RESULTS_DIR, f"flat_supcon_seed{seed}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Seed {seed}: Already done, skipping.")
            continue
        print(f"\n  Seed {seed}:")
        start = time.time()
        results, _ = run_flat_supcon(
            seed, train_ds, test_datasets, rare_classes_str,
            num_epochs=90, device=DEVICE
        )
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}, "
              f"Price-149 F1={results['price149']['macro_f1']:.4f}")

    # 2. Joint Hierarchical
    print("\n" + "="*60)
    print("JOINT HIERARCHICAL SUPCON BASELINE")
    print("="*60)

    for seed in SEEDS:
        result_path = os.path.join(RESULTS_DIR, f"joint_hierarchical_seed{seed}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Seed {seed}: Already done, skipping.")
            continue
        print(f"\n  Seed {seed}:")
        start = time.time()
        results, _ = run_joint_hierarchical(
            seed, train_ds, test_datasets, rare_classes_str,
            num_epochs=90, device=DEVICE
        )
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}, "
              f"Price-149 F1={results['price149']['macro_f1']:.4f}")


def run_main_experiment(train_ds, test_datasets, rare_classes_str):
    """Run CurrEC main experiment."""
    print("\n" + "="*60)
    print("CURREC (PROPOSED METHOD)")
    print("="*60)

    for seed in SEEDS:
        result_path = os.path.join(RESULTS_DIR, f"currec_seed{seed}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Seed {seed}: Already done, skipping.")
            continue
        print(f"\n  Seed {seed}:")
        start = time.time()
        results, _ = run_currec(
            seed, train_ds, test_datasets, rare_classes_str,
            consistency_weight=0.5, temp_schedule=True, device=DEVICE
        )
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}, "
              f"Price-149 F1={results['price149']['macro_f1']:.4f}")


def run_ablations(train_ds, test_datasets, rare_classes_str):
    """Run ablation experiments."""

    tau_base = 0.1
    gamma = 0.85

    # 1. Reverse curriculum
    print("\n" + "="*60)
    print("ABLATION: REVERSE CURRICULUM (fine-to-coarse)")
    print("="*60)

    for seed in SEEDS:
        result_path = os.path.join(RESULTS_DIR, f"reverse_curriculum_seed{seed}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Seed {seed}: Already done, skipping.")
            continue
        print(f"\n  Seed {seed}:")
        start = time.time()
        # Reverse order: L4 -> L3 -> L2 -> L1
        phase_config = [
            {"level": "ec_l4", "epochs": 40, "lr": 5e-4, "temp": tau_base * gamma**3},
            {"level": "ec_l3", "epochs": 20, "lr": 1e-4, "temp": tau_base * gamma**2},
            {"level": "ec_l2", "epochs": 15, "lr": 5e-5, "temp": tau_base * gamma**1},
            {"level": "ec_l1", "epochs": 15, "lr": 1e-5, "temp": tau_base},
        ]
        results, _ = run_currec(
            seed, train_ds, test_datasets, rare_classes_str,
            phase_config=phase_config, consistency_weight=0.5, device=DEVICE
        )
        results["method"] = "reverse_curriculum"
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}")

    # 2. Random-order curriculum
    print("\n" + "="*60)
    print("ABLATION: RANDOM ORDER CURRICULUM (L3->L1->L4->L2)")
    print("="*60)

    for seed in SEEDS:
        result_path = os.path.join(RESULTS_DIR, f"random_order_seed{seed}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Seed {seed}: Already done, skipping.")
            continue
        print(f"\n  Seed {seed}:")
        start = time.time()
        phase_config = [
            {"level": "ec_l3", "epochs": 20, "lr": 5e-4, "temp": tau_base},
            {"level": "ec_l1", "epochs": 15, "lr": 3e-4, "temp": tau_base},
            {"level": "ec_l4", "epochs": 40, "lr": 1e-4, "temp": tau_base},
            {"level": "ec_l2", "epochs": 15, "lr": 5e-5, "temp": tau_base},
        ]
        results, _ = run_currec(
            seed, train_ds, test_datasets, rare_classes_str,
            phase_config=phase_config, consistency_weight=0.5, device=DEVICE
        )
        results["method"] = "random_order"
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}")

    # 3. No consistency regularization (lambda=0)
    print("\n" + "="*60)
    print("ABLATION: NO CONSISTENCY (lambda=0)")
    print("="*60)

    for seed in SEEDS:
        result_path = os.path.join(RESULTS_DIR, f"no_consistency_seed{seed}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Seed {seed}: Already done, skipping.")
            continue
        print(f"\n  Seed {seed}:")
        start = time.time()
        results, _ = run_currec(
            seed, train_ds, test_datasets, rare_classes_str,
            consistency_weight=0.0, temp_schedule=True, device=DEVICE
        )
        results["method"] = "no_consistency"
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}")

    # 4. Lambda sweep (seed=42 only)
    print("\n" + "="*60)
    print("ABLATION: LAMBDA SWEEP")
    print("="*60)

    for lam in [0.1, 0.25, 1.0]:
        result_path = os.path.join(RESULTS_DIR, f"lambda_sweep_lambda{lam}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Lambda {lam}: Already done, skipping.")
            continue
        print(f"\n  Lambda {lam} (seed=42):")
        start = time.time()
        results, _ = run_currec(
            42, train_ds, test_datasets, rare_classes_str,
            consistency_weight=lam, temp_schedule=True, device=DEVICE
        )
        results["method"] = f"lambda_sweep_{lam}"
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}")

    # 5. No temperature scheduling (fixed temp)
    print("\n" + "="*60)
    print("ABLATION: NO TEMPERATURE SCHEDULE")
    print("="*60)

    for seed in SEEDS:
        result_path = os.path.join(RESULTS_DIR, f"no_temp_schedule_seed{seed}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Seed {seed}: Already done, skipping.")
            continue
        print(f"\n  Seed {seed}:")
        start = time.time()
        results, _ = run_currec(
            seed, train_ds, test_datasets, rare_classes_str,
            consistency_weight=0.5, temp_schedule=False, device=DEVICE
        )
        results["method"] = "no_temp_schedule"
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}")

    # 6. Two-phase curriculum
    print("\n" + "="*60)
    print("ABLATION: TWO-PHASE CURRICULUM (L2 -> L4)")
    print("="*60)

    for seed in SEEDS:
        result_path = os.path.join(RESULTS_DIR, f"two_phase_seed{seed}_results.json")
        if os.path.exists(result_path):
            print(f"\n  Seed {seed}: Already done, skipping.")
            continue
        print(f"\n  Seed {seed}:")
        start = time.time()
        phase_config = [
            {"level": "ec_l2", "epochs": 45, "lr": 5e-4, "temp": tau_base},
            {"level": "ec_l4", "epochs": 45, "lr": 5e-5, "temp": tau_base * gamma},
        ]
        results, _ = run_currec(
            seed, train_ds, test_datasets, rare_classes_str,
            phase_config=phase_config, consistency_weight=0.5, device=DEVICE
        )
        results["method"] = "two_phase"
        results["runtime_seconds"] = time.time() - start
        save_results(results, result_path)
        print(f"  Done in {results['runtime_seconds']:.0f}s. "
              f"New-392 F1={results['new392']['macro_f1']:.4f}")


if __name__ == "__main__":
    # Load data
    train_ds, test_datasets, rare_classes_str = load_datasets()

    # Run all experiments
    total_start = time.time()

    run_baselines(train_ds, test_datasets, rare_classes_str)
    run_main_experiment(train_ds, test_datasets, rare_classes_str)
    run_ablations(train_ds, test_datasets, rare_classes_str)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time/3600:.2f} hours")
    print(f"{'='*60}")
