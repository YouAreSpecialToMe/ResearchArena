"""
Master script for re-running SpecCheck experiments.
Addresses all issues from self-review:
1. Fix circular labeling with NLI model
2. Complete Qwen experiments
3. Add multi-seed SpecCheck evaluation
4. Re-evaluate everything with fixed labels
5. Generate figures and results.json
"""
import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(name, script):
    print(f"\n{'#'*70}")
    print(f"# STEP: {name}")
    print(f"{'#'*70}\n")
    start = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, script)],
        cwd=os.path.dirname(BASE_DIR),
        capture_output=False,
    )
    elapsed = time.time() - start
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  [{status}] {name} completed in {elapsed/60:.1f} minutes\n")
    return result.returncode == 0


def main():
    total_start = time.time()

    steps = [
        ("Step 1: NLI-based relabeling (CPU)", "nli_labeling.py"),
        ("Step 2: Complete Qwen experiments (GPU)", "complete_qwen.py"),
        ("Step 3: Multi-seed SpecCheck (GPU)", "multiseed_speccheck.py"),
        ("Step 4: Re-evaluate + figures + results.json", "rerun_eval_and_figures.py"),
    ]

    results = {}
    for name, script in steps:
        ok = run_step(name, script)
        results[name] = ok
        if not ok and "Qwen" not in name:
            # Non-critical failures: continue anyway
            print(f"  WARNING: {name} failed, continuing...")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL STEPS COMPLETED in {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    for name, ok in results.items():
        print(f"  {'OK' if ok else 'FAIL'}: {name}")


if __name__ == "__main__":
    main()
