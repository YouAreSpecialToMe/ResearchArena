#!/usr/bin/env python3
"""
Priority follow-up: runs SCD + PH(1 seed) + ablations after CD baseline is done.
Reuses all infrastructure from run_final.py.
"""
import os, sys, json, time, copy, math, shutil, tempfile, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.set_float32_matmul_precision('high')

WORKSPACE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE))

# Import everything from run_final
from exp.run_final import (
    DEVICE, NUM_FREQ_BANDS, SEEDS, MODEL_KWARGS, DISTILL_STEPS, DISTILL_BATCH,
    NUM_FID_SAMPLES, TEACHER_STEPS_PER_BAND, TEACHER_STEPS_UNIFORM, SCD_WEIGHTS,
    EXP_DIR, FIG_DIR,
    set_seed, create_fft_frequency_masks, per_band_mse_eval,
    generate_samples, compute_fid, evaluate_model,
    train_distillation, aggregate_and_save, generate_figures, generate_qualitative,
    UNet
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(str(WORKSPACE / 'experiment_priority.log'), mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def main():
    log.info(f"Priority follow-up: SCD + PH(1 seed) + ablations")
    log.info(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")

    # Load CIFAR-10
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=str(WORKSPACE / 'data'), train=True,
                          download=True, transform=transform)
    real_images = torch.stack([ds[i][0] for i in range(min(NUM_FID_SAMPLES, len(ds)))]).to(DEVICE)
    masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)

    # Load precomputed targets
    cache_path = EXP_DIR / 'teacher' / 'precomputed_targets_full.pt'
    precomputed = torch.load(str(cache_path), map_location='cpu', weights_only=True)
    log.info(f"Loaded precomputed targets: {precomputed['num_pairs']} pairs")

    all_results = {}

    # Load existing results
    for method_dir, method_key in [
        ('rectflow_baseline_v2', 'rectflow_baseline'),
        ('cd_baseline_v2', 'cd_baseline'),
    ]:
        method_results = {}
        for seed in SEEDS:
            rpath = EXP_DIR / method_dir / f'results_seed{seed}.json'
            if rpath.exists():
                method_results[seed] = json.load(open(str(rpath)))
        if method_results:
            all_results[method_key] = method_results
            fids = [method_results[s]['1_step']['fid'] for s in SEEDS if s in method_results]
            log.info(f"Loaded {method_key}: 1-step FID = {np.mean(fids):.2f} +/- {np.std(fids):.2f}")

    # -----------------------------------------------------------------------
    # 1. SCD Main - Adaptive Teacher (3 seeds) - HIGHEST PRIORITY
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nSCD MAIN - Adaptive Teacher (3 seeds)\n{'='*60}")
    scd_results = {}
    for seed in SEEDS:
        rpath = EXP_DIR / 'scd_adaptive_v2' / f'results_seed{seed}.json'
        if rpath.exists():
            scd_results[seed] = json.load(open(str(rpath)))
            log.info(f"  SCD seed={seed} already done, loaded from file")
            continue
        model, train_time = train_distillation(
            precomputed, 'scd_adaptive_v2', seed, masks,
            spectral_weights=SCD_WEIGHTS,
            adaptive_teacher=True)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(rpath), 'w'), indent=2)
        scd_results[seed] = res
        del model; torch.cuda.empty_cache()
    all_results['scd_main'] = scd_results

    # -----------------------------------------------------------------------
    # 2. CD Pseudo-Huber (1 seed) - lower priority
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nCD PSEUDO-HUBER (seed=42 only)\n{'='*60}")
    ph_results = {}
    for seed in [42]:
        rpath = EXP_DIR / 'cd_pseudohuber_v2' / f'results_seed{seed}.json'
        if rpath.exists():
            ph_results[seed] = json.load(open(str(rpath)))
            log.info(f"  PH seed={seed} already done")
            continue
        model, train_time = train_distillation(
            precomputed, 'cd_pseudohuber_v2', seed, masks,
            use_pseudohuber=True)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(rpath), 'w'), indent=2)
        ph_results[seed] = res
        del model; torch.cuda.empty_cache()
    # Try remaining seeds if time allows
    for seed in [43, 44]:
        rpath = EXP_DIR / 'cd_pseudohuber_v2' / f'results_seed{seed}.json'
        if rpath.exists():
            ph_results[seed] = json.load(open(str(rpath)))
            continue
        model, train_time = train_distillation(
            precomputed, 'cd_pseudohuber_v2', seed, masks,
            use_pseudohuber=True)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(rpath), 'w'), indent=2)
        ph_results[seed] = res
        del model; torch.cuda.empty_cache()
    all_results['cd_pseudohuber'] = ph_results

    # -----------------------------------------------------------------------
    # 3. Ablation: no adaptive teacher (seed 42)
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nABLATION: SCD no adaptive teacher\n{'='*60}")
    rpath = EXP_DIR / 'ablation_no_adaptive_v2' / f'results_seed42.json'
    if rpath.exists():
        abl_res = json.load(open(str(rpath)))
        log.info("  Already done")
    else:
        model, train_time = train_distillation(
            precomputed, 'ablation_no_adaptive_v2', 42, masks,
            spectral_weights=SCD_WEIGHTS,
            adaptive_teacher=False)
        abl_res = evaluate_model(model, real_images, masks, seed=42)
        abl_res['train_time_min'] = train_time
        abl_res['train_steps'] = DISTILL_STEPS
        (EXP_DIR / 'ablation_no_adaptive_v2').mkdir(parents=True, exist_ok=True)
        json.dump(abl_res, open(str(rpath), 'w'), indent=2)
        del model; torch.cuda.empty_cache()
    all_results['ablation_no_adaptive'] = {42: abl_res}

    # -----------------------------------------------------------------------
    # 4. Ablation: uniform weights + adaptive teacher (seed 42)
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nABLATION: SCD uniform weights\n{'='*60}")
    rpath = EXP_DIR / 'ablation_uniform_weights_v2' / f'results_seed42.json'
    if rpath.exists():
        abl_res = json.load(open(str(rpath)))
        log.info("  Already done")
    else:
        model, train_time = train_distillation(
            precomputed, 'ablation_uniform_weights_v2', 42, masks,
            spectral_weights=[1.0, 1.0, 1.0, 1.0],
            adaptive_teacher=True)
        abl_res = evaluate_model(model, real_images, masks, seed=42)
        abl_res['train_time_min'] = train_time
        abl_res['train_steps'] = DISTILL_STEPS
        (EXP_DIR / 'ablation_uniform_weights_v2').mkdir(parents=True, exist_ok=True)
        json.dump(abl_res, open(str(rpath), 'w'), indent=2)
        del model; torch.cuda.empty_cache()
    all_results['ablation_uniform_weights'] = {42: abl_res}

    # -----------------------------------------------------------------------
    # 5. Aggregate + Figures
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nAGGREGATING RESULTS\n{'='*60}")
    aggregate_and_save(all_results)

    log.info(f"\n{'='*60}\nGENERATING FIGURES\n{'='*60}")
    generate_figures(all_results)
    generate_qualitative(all_results)

    log.info(f"\n{'='*60}\nALL PRIORITY EXPERIMENTS COMPLETE\n{'='*60}")


if __name__ == '__main__':
    main()
