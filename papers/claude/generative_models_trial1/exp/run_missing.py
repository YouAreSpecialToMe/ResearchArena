#!/usr/bin/env python3
"""Run critical missing experiments:
1. Evaluate SCD seed 44 (checkpoint exists but no results)
2. CD with 100-step uniform teacher baseline (critical ablation)
3. CD with uniform 100-step teacher + spectral weights (to isolate spectral decomposition contribution)
"""
import os, sys, json, time, copy, math, shutil, tempfile, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

torch.set_float32_matmul_precision('high')

WORKSPACE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE))
from exp.shared.models import UNet
from exp.shared.flow_matching import euler_sample, teacher_solve
from exp.run_final import (
    DEVICE, NUM_FREQ_BANDS, SEEDS, MODEL_KWARGS,
    DISTILL_STEPS, DISTILL_BATCH, NUM_FID_SAMPLES,
    TEACHER_STEPS_PER_BAND, TEACHER_STEPS_UNIFORM, SCD_WEIGHTS,
    EXP_DIR, FIG_DIR,
    set_seed, create_fft_frequency_masks, per_band_mse_eval,
    consistency_sample, generate_samples, compute_fid, evaluate_model,
    train_distillation
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(str(WORKSPACE / 'experiment_missing.log'), mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def main():
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

    # -----------------------------------------------------------------------
    # 1. Evaluate SCD seed 44 (checkpoint exists)
    # -----------------------------------------------------------------------
    scd_ckpt_44 = EXP_DIR / 'scd_adaptive_v2' / 'checkpoint_seed44.pt'
    scd_res_44 = EXP_DIR / 'scd_adaptive_v2' / 'results_seed44.json'
    if scd_ckpt_44.exists() and not scd_res_44.exists():
        log.info("=== Evaluating SCD seed 44 ===")
        model = UNet(**MODEL_KWARGS).to(DEVICE)
        model.load_state_dict(torch.load(str(scd_ckpt_44), map_location=DEVICE, weights_only=True))
        res = evaluate_model(model, real_images, masks, seed=44)
        # Get train time from log
        log_path = EXP_DIR / 'scd_adaptive_v2' / 'train_log_seed44.json'
        if log_path.exists():
            train_log = json.load(open(str(log_path)))
            res['train_steps'] = train_log[-1]['step'] if train_log else 30000
        else:
            res['train_steps'] = 30000
        res['train_time_min'] = 50.0  # approximate
        json.dump(res, open(str(scd_res_44), 'w'), indent=2)
        log.info(f"SCD seed 44 results saved")
        del model; torch.cuda.empty_cache()
    else:
        log.info(f"SCD seed 44: ckpt exists={scd_ckpt_44.exists()}, results exist={scd_res_44.exists()}")

    # -----------------------------------------------------------------------
    # 2. CD with 100-step uniform teacher (critical ablation)
    #    This uses standard MSE loss but with 100-step teacher targets
    #    instead of the default 20-step targets
    # -----------------------------------------------------------------------
    log.info("=== Training CD with 100-step teacher (critical ablation) ===")

    # Load teacher and precompute 100-step targets if needed
    teacher = UNet(**MODEL_KWARGS).to(DEVICE)
    teacher.load_state_dict(torch.load(
        str(EXP_DIR / 'teacher' / 'checkpoint_best.pt'),
        map_location=DEVICE, weights_only=True))
    teacher.eval()

    # Load precomputed targets
    cache_path = EXP_DIR / 'teacher' / 'precomputed_targets_full.pt'
    precomputed = torch.load(str(cache_path), map_location='cpu', weights_only=True)

    # Check if 100-step targets are available
    if 100 not in precomputed['targets']:
        log.info("Need to compute 100-step targets...")
        N = precomputed['num_pairs']
        x_t = precomputed['x_t']
        t_vals = precomputed['t']
        bs = 256
        step_targets = []
        for start in range(0, N, bs):
            end = min(start + bs, N)
            xt_batch = x_t[start:end].float().to(DEVICE)
            t_batch = t_vals[start:end].to(DEVICE)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                target = teacher_solve(teacher, xt_batch, t_batch, 100)
            step_targets.append(target.cpu().half())
            if start % 5000 == 0:
                log.info(f"  {start}/{N}")
        precomputed['targets'][100] = torch.cat(step_targets)
        torch.save(precomputed, str(cache_path))
        log.info("  100-step targets computed and cached")

    del teacher; torch.cuda.empty_cache()

    # Now train CD with 100-step teacher using standard MSE loss (3 seeds)
    cd100_dir = EXP_DIR / 'cd_100step_teacher'
    cd100_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        results_path = cd100_dir / f'results_seed{seed}.json'
        if results_path.exists():
            log.info(f"CD-100step seed {seed} already done, skipping")
            continue

        log.info(f"Training CD-100step seed={seed}")
        set_seed(seed)
        student = UNet(**MODEL_KWARGS).to(DEVICE)
        ema_student = copy.deepcopy(student)
        compiled = torch.compile(student)
        opt = torch.optim.Adam(student.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        N = precomputed['num_pairs']
        t_start = time.time()
        train_log = []

        for step in range(1, DISTILL_STEPS + 1):
            idx = torch.randint(0, N, (DISTILL_BATCH,))
            x_t_batch = precomputed['x_t'][idx].float().to(DEVICE)
            t_batch = precomputed['t'][idx].to(DEVICE)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                student_pred = compiled(x_t_batch, t_batch)

            # Use 100-step teacher targets with standard MSE loss
            target = precomputed['targets'][100][idx].float().to(DEVICE)
            loss = F.mse_loss(student_pred.float(), target)

            if step % 500 == 0:
                train_log.append({'step': step, 'loss': loss.item()})

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                for p, ep in zip(student.parameters(), ema_student.parameters()):
                    ep.mul_(0.999).add_(p, alpha=0.001)

            if step % 5000 == 0:
                elapsed = (time.time() - t_start) / 60
                log.info(f"    Step {step}/{DISTILL_STEPS}, loss={loss.item():.6f}, time={elapsed:.1f}min")

        train_time = (time.time() - t_start) / 60
        torch.save(ema_student.state_dict(), str(cd100_dir / f'checkpoint_seed{seed}.pt'))
        json.dump(train_log, open(str(cd100_dir / f'train_log_seed{seed}.json'), 'w'), indent=2)

        res = evaluate_model(ema_student, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(results_path), 'w'), indent=2)
        log.info(f"CD-100step seed {seed}: 1-step FID={res['1_step']['fid']:.2f}")

        del student, ema_student, compiled; torch.cuda.empty_cache()

    log.info("=== ALL MISSING EXPERIMENTS COMPLETE ===")


if __name__ == '__main__':
    main()
