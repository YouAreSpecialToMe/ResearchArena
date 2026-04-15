"""Resume SCD experiments from where run_v2.py crashed.

Baselines (CD, Pseudo-Huber, Rectified Flow) already completed and saved.
This script re-precomputes teacher targets, then runs:
- SCD main (3 seeds)
- 3 ablation studies (seed=42)
- Aggregation + figure generation
"""
import os
import sys
import json
import time
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.set_float32_matmul_precision('high')

WORKSPACE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE))

# Import everything from run_v2
from exp.run_v2 import (
    set_seed, load_cifar10_to_gpu, UNet, MODEL_KWARGS, DEVICE,
    NUM_FREQ_BANDS, SEEDS, DISTILL_STEPS, NUM_FID_SAMPLES,
    TEACHER_TRAIN_STEPS, TEACHER_ODE_STEPS, SCD_LF_STEPS, SCD_HF_STEPS,
    create_fft_frequency_masks, precompute_teacher_targets,
    train_distillation, evaluate_method,
    aggregate_results, generate_figures,
)


def load_saved_results(method, seeds=SEEDS):
    """Load previously saved per-seed results."""
    results = {}
    for seed in seeds:
        p = WORKSPACE / 'exp' / method / f'results_seed{seed}.json'
        if p.exists():
            with open(p) as f:
                results[seed] = json.load(f)
    return results


def main():
    sys.stdout.reconfigure(line_buffering=True)
    overall_start = time.time()

    print("SCD Experiment v2 — RESUME (SCD + ablations + aggregation)")
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Load completed baseline results
    all_results = {}
    for method in ['cd_baseline', 'cd_pseudohuber', 'rectflow_baseline']:
        r = load_saved_results(method)
        if len(r) == len(SEEDS):
            print(f"  Loaded {method}: {len(r)} seeds")
            all_results[method] = r
        else:
            print(f"  WARNING: {method} only has {len(r)}/{len(SEEDS)} seeds!")
            all_results[method] = r

    # Load teacher and precompute targets
    teacher_ckpt = WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt'
    assert teacher_ckpt.exists(), f"Teacher checkpoint not found: {teacher_ckpt}"

    data = load_cifar10_to_gpu()
    real_images = data[:10000].cpu()

    teacher = UNet(**MODEL_KWARGS).to(DEVICE)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE, weights_only=True))
    precomputed = precompute_teacher_targets(teacher, data, num_pairs=50000, seed=42)
    del teacher; torch.cuda.empty_cache()
    del data; torch.cuda.empty_cache()

    masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)

    # SCD main (3 seeds)
    print(f"\n{'='*60}\nSPECTRAL CONSISTENCY DISTILLATION (3 seeds)\n{'='*60}")
    scd_config = {'weights': [1.0, 1.5, 2.5, 4.0], 'progressive': True,
                  'fixed_weights': False, 'use_composite_target': True}
    scd_results = {}
    for seed in SEEDS:
        # Check if already done
        rp = WORKSPACE / 'exp' / 'scd_main' / f'results_seed{seed}.json'
        if rp.exists():
            with open(rp) as f:
                prev = json.load(f)
            if prev.get('fid_1step') is not None:
                print(f"\n  scd_main seed={seed} already done, skipping.", flush=True)
                scd_results[seed] = prev
                continue

        model, tt = train_distillation(precomputed, 'scd_main', seed,
                                        use_spectral=True, spectral_config=scd_config,
                                        teacher_ckpt_path=teacher_ckpt, masks=masks)
        results = evaluate_method(model, [1, 2, 4], NUM_FID_SAMPLES, real_images, seed)
        results['train_time_min'] = tt; results['train_steps'] = DISTILL_STEPS
        scd_results[seed] = results
        with open(WORKSPACE / 'exp' / 'scd_main' / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
        del model; torch.cuda.empty_cache()
    all_results['scd_main'] = scd_results

    # Ablations (seed=42 only)
    print(f"\n{'='*60}\nABLATION STUDIES (seed=42)\n{'='*60}")
    ablation_configs = {
        'ablation_no_spectral_weight': {
            'weights': [1.0, 1.0, 1.0, 1.0], 'progressive': False,
            'fixed_weights': True, 'use_composite_target': True},
        'ablation_no_adaptive_teacher': {
            'weights': [1.0, 1.5, 2.5, 4.0], 'progressive': True,
            'fixed_weights': False, 'use_composite_target': False},
        'ablation_no_progressive': {
            'weights': [1.0, 1.5, 2.5, 4.0], 'progressive': False,
            'fixed_weights': True, 'use_composite_target': True},
    }
    ablation_results = {}
    for abl_name, abl_config in ablation_configs.items():
        rp = WORKSPACE / 'exp' / abl_name / 'results.json'
        if rp.exists():
            with open(rp) as f:
                prev = json.load(f)
            if prev.get('fid_1step') is not None:
                print(f"\n  {abl_name} already done, skipping.", flush=True)
                ablation_results[abl_name] = prev
                continue

        print(f"\n  --- {abl_name} ---")
        model, tt = train_distillation(precomputed, abl_name, seed=42,
                                        use_spectral=True, spectral_config=abl_config,
                                        teacher_ckpt_path=teacher_ckpt, masks=masks)
        results = evaluate_method(model, [1, 2, 4], NUM_FID_SAMPLES, real_images, 42)
        results['train_time_min'] = tt; results['train_steps'] = DISTILL_STEPS
        ablation_results[abl_name] = results
        (WORKSPACE / 'exp' / abl_name).mkdir(parents=True, exist_ok=True)
        with open(WORKSPACE / 'exp' / abl_name / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        del model; torch.cuda.empty_cache()
    all_results['ablations'] = ablation_results

    # Aggregate
    aggregated = aggregate_results(all_results)
    with open(WORKSPACE / 'results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Figures
    print("\nGenerating figures...", flush=True)
    generate_figures(aggregated, all_results)

    total_hours = (time.time() - overall_start) / 3600
    print(f"\n{'='*60}\nALL DONE. Total: {total_hours:.2f} hours\n{'='*60}")


if __name__ == '__main__':
    main()
