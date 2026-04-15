"""
Baseline 1: Standard Flow Matching with uniform weighting.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pathlib import Path
import argparse

from shared.data_loader import get_dataloader, download_kitti360_sample
from shared.models import VelocityNetwork
from shared.trainer import FlowMatchingTrainer
from shared.utils import set_seed, save_results, Timer, get_device


def main(seed=42, epochs=70, batch_size=32):
    """Run baseline uniform experiment."""
    set_seed(seed)
    
    exp_name = f"baseline_uniform_seed{seed}"
    output_dir = Path("outputs") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "log.txt"
    
    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    log(f"=== {exp_name} ===")
    log(f"Seed: {seed}, Epochs: {epochs}, Batch Size: {batch_size}")
    
    device = get_device()
    log(f"Device: {device}")
    
    # Data preparation
    timer = Timer()
    timer.start()
    
    data_dir = "data/kitti360"
    if not Path(data_dir).exists() or len(list(Path(data_dir).rglob("*.pt"))) == 0:
        log("Downloading/preparing data...")
        download_kitti360_sample(data_dir)
    
    log("Loading data...")
    train_loader = get_dataloader(data_dir, split='train', batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(data_dir, split='val', batch_size=batch_size, shuffle=False)
    val_dataset = train_loader.dataset if val_loader is None else val_loader.dataset
    
    data_time = timer.get_elapsed()
    log(f"Data loading took {data_time:.1f}s")
    
    # Model
    model = VelocityNetwork(
        point_dim=3,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        use_distance_conditioning=False,  # No distance conditioning for baseline
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Trainer
    trainer = FlowMatchingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=2e-4,
        weight_decay=1e-4,
        weighting_type='uniform',
    )
    
    # Train
    log("Starting training...")
    timer.start()
    train_losses, val_losses = trainer.train(epochs, save_dir=output_dir)
    train_time = timer.stop()
    
    log(f"Training took {train_time / 60:.1f} minutes")
    
    # Evaluate
    log("Evaluating...")
    metrics = trainer.evaluate(val_dataset, num_eval_samples=200)
    
    for k, v in metrics.items():
        log(f"  {k}: {v:.6f}")
    
    # Save results
    results = {
        "experiment": exp_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_params": num_params,
        "train_time_minutes": train_time / 60,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "metrics": metrics,
        "config": {
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 4,
            "use_distance_conditioning": False,
            "weighting_type": "uniform",
        }
    }
    
    results_path = output_dir / "results.json"
    save_results(results, results_path)
    log(f"Results saved to {results_path}")
    
    # Also save to aggregated results
    agg_results_path = Path("outputs/results") / f"{exp_name}.json"
    agg_results_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, agg_results_path)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    main(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
