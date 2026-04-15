"""Diagnose why SCD produces FID ~500 despite loss converging.

Tests:
1. Composite target quality — does it look like real data?
2. Spectral loss equivalence — does uniform-weight spectral = MSE?
3. Quick train: SCD spectral loss vs MSE on composite target vs MSE on 20-step target
"""
import sys
import time
import json
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

from exp.run_v2 import (
    set_seed, load_cifar10_to_gpu, UNet, MODEL_KWARGS, DEVICE,
    NUM_FREQ_BANDS, DISTILL_STEPS, DISTILL_BATCH, NUM_FID_SAMPLES,
    TEACHER_ODE_STEPS, SCD_LF_STEPS, SCD_HF_STEPS,
    create_fft_frequency_masks, build_composite_target_batch,
    spectral_loss_normalized, precompute_teacher_targets,
    ema_update, generate_samples_consistency, compute_fid_from_samples,
)

sys.stdout.reconfigure(line_buffering=True)

# Load teacher and precompute
teacher_ckpt = WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt'
data = load_cifar10_to_gpu()
real_images = data[:10000].cpu()

teacher = UNet(**MODEL_KWARGS).to(DEVICE)
teacher.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE, weights_only=True))
precomputed = precompute_teacher_targets(teacher, data, num_pairs=50000, seed=42)
del teacher, data; torch.cuda.empty_cache()

masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)

# ============ Test 1: Composite target quality ============
print("\n=== TEST 1: Composite target quality ===")
idx = torch.arange(1000)
target_lf = precomputed['targets'][SCD_LF_STEPS][idx].float().to(DEVICE)
target_hf = precomputed['targets'][TEACHER_ODE_STEPS][idx].float().to(DEVICE)
composite = build_composite_target_batch(target_lf, target_hf, masks)

# Check value ranges
print(f"  target_hf range: [{target_hf.min():.3f}, {target_hf.max():.3f}], mean: {target_hf.mean():.3f}")
print(f"  target_lf range: [{target_lf.min():.3f}, {target_lf.max():.3f}], mean: {target_lf.mean():.3f}")
print(f"  composite range: [{composite.min():.3f}, {composite.max():.3f}], mean: {composite.mean():.3f}")

# Check how different they are
diff_lf_hf = F.mse_loss(target_lf, target_hf)
diff_comp_hf = F.mse_loss(composite, target_hf)
diff_comp_lf = F.mse_loss(composite, target_lf)
print(f"  MSE(lf, hf): {diff_lf_hf:.6f}")
print(f"  MSE(composite, hf): {diff_comp_hf:.6f}")
print(f"  MSE(composite, lf): {diff_comp_lf:.6f}")

# Check FID of composite targets themselves
print("  Computing FID of composite targets...")
comp_for_fid = composite[:NUM_FID_SAMPLES].cpu() if len(composite) >= NUM_FID_SAMPLES else composite.cpu()
comp_fid = compute_fid_from_samples(comp_for_fid)
print(f"  Composite FID: {comp_fid:.2f}")

hf_for_fid = target_hf[:NUM_FID_SAMPLES].cpu() if len(target_hf) >= NUM_FID_SAMPLES else target_hf.cpu()
hf_fid = compute_fid_from_samples(hf_for_fid)
print(f"  20-step target FID: {hf_fid:.2f}")

del target_lf, target_hf, composite; torch.cuda.empty_cache()

# ============ Test 2: Loss equivalence ============
print("\n=== TEST 2: Spectral loss with uniform weights == MSE? ===")
idx = torch.arange(100)
x = precomputed['targets'][TEACHER_ODE_STEPS][idx].float().to(DEVICE)
y = precomputed['targets'][SCD_LF_STEPS][idx].float().to(DEVICE)
mse = F.mse_loss(x, y)
spec_loss, _ = spectral_loss_normalized(x, y, masks, [1.0, 1.0, 1.0, 1.0])
print(f"  MSE: {mse.item():.8f}")
print(f"  Spectral (uniform): {spec_loss.item():.8f}")
print(f"  Ratio: {spec_loss.item() / mse.item():.6f}")
del x, y; torch.cuda.empty_cache()

# ============ Test 3: Quick training comparison ============
print("\n=== TEST 3: Quick 5000-step training comparison ===")
QUICK_STEPS = 5000
QUICK_FID_SAMPLES = 10000

def quick_train(method_name, use_spectral_loss, use_composite_target, weights=None):
    print(f"\n  Training: {method_name}", flush=True)
    set_seed(42)
    student = UNet(**MODEL_KWARGS).to(DEVICE)
    student.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE, weights_only=True))
    ema_student = copy.deepcopy(student)
    compiled_student = torch.compile(student)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    N = precomputed['num_pairs']
    t0 = time.time()

    for step in range(1, QUICK_STEPS + 1):
        idx = torch.randint(0, N, (256,))
        x_t = precomputed['x_t'][idx].float().to(DEVICE)
        t_vals = precomputed['t'][idx].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            pred = compiled_student(x_t, t_vals)

            # Get target
            target_hf = precomputed['targets'][TEACHER_ODE_STEPS][idx].float().to(DEVICE)
            if use_composite_target:
                target_lf = precomputed['targets'][SCD_LF_STEPS][idx].float().to(DEVICE)
                target = build_composite_target_batch(target_lf, target_hf, masks)
            else:
                target = target_hf

            if use_spectral_loss:
                w = weights or [1.0, 1.0, 1.0, 1.0]
                loss, _ = spectral_loss_normalized(pred, target, masks, w)
            else:
                loss = F.mse_loss(pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        ema_update(ema_student, student, decay=0.999)

        if step % 1000 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"    Step {step}/{QUICK_STEPS}, Loss: {loss.item():.4f}, Time: {elapsed:.1f}min", flush=True)

    # Eval
    samples = generate_samples_consistency(ema_student, QUICK_FID_SAMPLES, num_steps=1, seed=42)
    fid1 = compute_fid_from_samples(samples)
    samples = generate_samples_consistency(ema_student, QUICK_FID_SAMPLES, num_steps=2, seed=42)
    fid2 = compute_fid_from_samples(samples)
    print(f"    {method_name}: 1-step FID={fid1:.2f}, 2-step FID={fid2:.2f}", flush=True)

    del student, ema_student, compiled_student; torch.cuda.empty_cache()
    return fid1, fid2

# A) CD baseline (MSE on 20-step target) — should work
fid_a = quick_train("CD_baseline (MSE, 20-step target)",
                     use_spectral_loss=False, use_composite_target=False)

# B) MSE on composite target — tests if composite target is the problem
fid_b = quick_train("MSE_composite (MSE, composite target)",
                     use_spectral_loss=False, use_composite_target=True)

# C) Spectral uniform on 20-step target — tests if spectral loss is the problem
fid_c = quick_train("Spectral_uniform (spectral[1,1,1,1], 20-step target)",
                     use_spectral_loss=True, use_composite_target=False,
                     weights=[1.0, 1.0, 1.0, 1.0])

# D) Spectral weighted on composite — the full SCD
fid_d = quick_train("SCD_full (spectral[1,1.5,2.5,4], composite target)",
                     use_spectral_loss=True, use_composite_target=True,
                     weights=[1.0, 1.5, 2.5, 4.0])

# E) Spectral uniform on composite — isolate weighting effect
fid_e = quick_train("SCD_uniform (spectral[1,1,1,1], composite target)",
                     use_spectral_loss=True, use_composite_target=True,
                     weights=[1.0, 1.0, 1.0, 1.0])

print("\n=== SUMMARY ===")
print(f"  A) CD baseline:        1-step={fid_a[0]:.2f}, 2-step={fid_a[1]:.2f}")
print(f"  B) MSE+composite:      1-step={fid_b[0]:.2f}, 2-step={fid_b[1]:.2f}")
print(f"  C) Spectral+20step:    1-step={fid_c[0]:.2f}, 2-step={fid_c[1]:.2f}")
print(f"  D) SCD full:           1-step={fid_d[0]:.2f}, 2-step={fid_d[1]:.2f}")
print(f"  E) SCD uniform:        1-step={fid_e[0]:.2f}, 2-step={fid_e[1]:.2f}")
print("\nDiagnosis complete.")
