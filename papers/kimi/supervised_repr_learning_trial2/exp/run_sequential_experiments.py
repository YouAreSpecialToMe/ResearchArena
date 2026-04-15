#!/usr/bin/env python3
"""
Sequential experiment runner for LASER-SCL.
Runs experiments one at a time to ensure completion.
"""
import subprocess
import json
import os
import time
from datetime import datetime

# Configuration
DATASET = "cifar100"
NOISE_RATE = 0.4
EPOCHS = 25
NUM_WORKERS = 2

# Experiments to run: (method, seed)
EXPERIMENTS = [
    ("supcon", 42),
    ("supcon", 123),
    ("supcon", 456),
    ("supcon_lr", 42),
    ("supcon_lr", 123),
    ("supcon_lr", 456),
    ("laser_scl", 42),
    ("laser_scl", 123),
    ("laser_scl", 456),
]

def run_experiment(method, seed):
    """Run a single experiment and wait for completion."""
    result_file = f"results/{method}_{DATASET}_n{int(NOISE_RATE*100)}_s{seed}.json"
    
    # Check if already completed
    if os.path.exists(result_file):
        print(f"[{datetime.now()}] {method} seed {seed} already exists, skipping")
        return True
    
    print(f"\n[{datetime.now()}] Starting {method} seed {seed}...")
    start_time = time.time()
    
    cmd = [
        "python", "exp/shared/train.py",
        "--dataset", DATASET,
        "--noise_rate", str(NOISE_RATE),
        "--method", method,
        "--epochs", str(EPOCHS),
        "--seed", str(seed),
        "--num_workers", str(NUM_WORKERS),
        "--save_dir", "results"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = (time.time() - start_time) / 60
        
        if result.returncode == 0 and os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            acc = data.get('final_accuracy', 0)
            print(f"[{datetime.now()}] ✓ {method} seed {seed} completed in {elapsed:.1f}min - Acc: {acc:.2f}%")
            return True
        else:
            print(f"[{datetime.now()}] ✗ {method} seed {seed} failed (return code: {result.returncode})")
            print(f"STDERR: {result.stderr[-500:] if result.stderr else 'None'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now()}] ✗ {method} seed {seed} timed out")
        return False
    except Exception as e:
        print(f"[{datetime.now()}] ✗ {method} seed {seed} error: {e}")
        return False

def main():
    print("="*60)
    print("LASER-SCL Sequential Experiment Runner")
    print(f"Started: {datetime.now()}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Config: {DATASET}, {NOISE_RATE*100}% noise, {EPOCHS} epochs")
    print("="*60)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    completed = 0
    failed = 0
    
    for i, (method, seed) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] ", end="")
        if run_experiment(method, seed):
            completed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"Completed: {completed}/{len(EXPERIMENTS)}")
    print(f"Failed: {failed}/{len(EXPERIMENTS)}")
    print(f"Finished: {datetime.now()}")
    print("="*60)
    
    # Run analysis if we have results
    if completed > 0:
        print("\nRunning analysis...")
        subprocess.run(["python", "analyze_results.py"])

if __name__ == "__main__":
    main()
