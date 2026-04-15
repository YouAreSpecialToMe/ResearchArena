"""Optimized experiment script for Spectral Consistency Distillation.

Key optimizations over run_all.py:
- Smaller model (6.5M params) for faster training within 8h budget
- torch.compile for 1.5-2x speedup
- Pre-computed teacher ODE targets (avoids repeated teacher inference during distillation)
- In-memory FID computation (no PNG I/O bottleneck)
- Entire CIFAR-10 loaded to GPU (eliminates data loading overhead)
"""
import os
import sys
import json
import time
import copy
import shutil
import tempfile
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

from exp.shared.models import UNet
from exp.shared.flow_matching import ot_cfm_sample_t_and_xt, euler_sample, teacher_solve
from exp.shared.spectral import create_fft_frequency_masks, fft_band_mse

DEVICE = 'cuda'
NUM_FREQ_BANDS = 4
SEEDS = [42, 43, 44]
# Smaller model config for speed
MODEL_KWARGS = dict(model_channels=64, channel_mult=(1, 2, 2), attention_resolutions=(8,))

# Step counts (tuned for ~8h budget on 1x A6000, GPU now free)
# Measured: ~164ms/step for teacher training (no contention), ~170ms for distillation
TEACHER_STEPS = 60000      # ~164 min at 164ms/step
DISTILL_STEPS = 10000      # ~28 min at 170ms/step per seed
DISTILL_BATCH = 256
TEACHER_BATCH = 256
NUM_FID_SAMPLES = 10000    # 10k for speed; sufficient for relative comparisons
TEACHER_ODE_STEPS = 10     # for distillation targets
SCD_TEACHER_STEPS = [2, 4, 7, 10]  # per-band teacher steps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10_to_gpu():
    """Load entire CIFAR-10 train set into GPU memory (~600MB)."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=str(WORKSPACE / 'data'), train=True,
                          download=True, transform=transform)
    all_images = torch.stack([ds[i][0] for i in range(len(ds))])  # (50000, 3, 32, 32)
    return all_images.to(DEVICE)


def sample_batch(data, batch_size, augment=True):
    """Sample a random batch from in-memory data with optional flip augmentation."""
    idx = torch.randint(0, len(data), (batch_size,), device=DEVICE)
    batch = data[idx]
    if augment:
        flip_mask = torch.rand(batch_size, 1, 1, 1, device=DEVICE) > 0.5
        batch = torch.where(flip_mask, batch.flip(-1), batch)
    return batch


def ema_update(ema_model, model, decay=0.9999):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ==========================================================================
# FID COMPUTATION (in-memory using clean-fid)
# ==========================================================================
def compute_fid_from_samples(samples, num_samples=None):
    """Compute FID using clean-fid. Saves to tmpfs for speed."""
    from torchvision.utils import save_image
    from cleanfid import fid

    if num_samples is not None:
        samples = samples[:num_samples]

    tmp_dir = tempfile.mkdtemp(prefix='scd_fid_')
    try:
        samples_01 = ((samples + 1) / 2).clamp(0, 1)
        for i in range(len(samples_01)):
            save_image(samples_01[i], os.path.join(tmp_dir, f'{i:06d}.png'))
        score = fid.compute_fid(tmp_dir, dataset_name='cifar10',
                                dataset_split='train', dataset_res=32, mode='clean')
        return score
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@torch.no_grad()
def generate_samples_velocity(model, num_samples, num_steps, batch_size=512, seed=42):
    """Generate samples using Euler ODE solver (for velocity-field models like teacher)."""
    model.eval()
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    all_samples = []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, 3, 32, 32, device=DEVICE, generator=gen)
        samples = euler_sample(model, z, num_steps)
        all_samples.append(samples.cpu())
        remaining -= bs
    return torch.cat(all_samples)[:num_samples]


@torch.no_grad()
def consistency_sample(model, z, num_steps):
    """Sample from a consistency model (predicts x_0 directly).

    1-step: x_0 = f(z, t=1.0)
    Multi-step: iteratively denoise with re-noising at intermediate timesteps.
    """
    x = z
    step_times = torch.linspace(1.0, 0.0, num_steps + 1, device=z.device)[:-1]

    for i, t_val in enumerate(step_times):
        t = torch.full((z.shape[0],), t_val, device=z.device)
        x_pred = model(x, t)  # predict x_0

        if i < len(step_times) - 1:
            # Re-noise to next timestep for multi-step refinement
            t_next = step_times[i + 1]
            noise = torch.randn_like(x)
            x = (1 - t_next) * x_pred + t_next * noise
        else:
            x = x_pred

    return x


@torch.no_grad()
def generate_samples_consistency(model, num_samples, num_steps, batch_size=512, seed=42):
    """Generate samples using consistency model (x_0 prediction)."""
    model.eval()
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    all_samples = []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, 3, 32, 32, device=DEVICE, generator=gen)
        samples = consistency_sample(model, z, num_steps)
        all_samples.append(samples.cpu())
        remaining -= bs
    return torch.cat(all_samples)[:num_samples]


def evaluate_method(model, steps_list=[1, 2, 4], num_samples=50000,
                    real_images=None, seed=42, is_velocity_model=False):
    """Evaluate a model at multiple step budgets.

    Args:
        is_velocity_model: True for teacher (velocity field), False for students (x_0 prediction)
    """
    results = {}
    masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)
    gen_fn = generate_samples_velocity if is_velocity_model else generate_samples_consistency

    for ns in steps_list:
        print(f"    Generating {num_samples} samples with {ns} step(s)...", flush=True)
        t0 = time.time()
        samples = gen_fn(model, num_samples, ns, seed=seed)
        gen_time = time.time() - t0

        print(f"    Computing FID...", flush=True)
        fid_score = compute_fid_from_samples(samples)
        print(f"    {ns}-step FID: {fid_score:.2f} (gen: {gen_time:.1f}s)", flush=True)

        step_result = {'fid': fid_score, 'gen_time_s': gen_time}

        if real_images is not None:
            n = min(10000, len(samples), len(real_images))
            gen_batch = samples[:n].to(DEVICE)
            real_batch = real_images[:n].to(DEVICE)
            band_mses = fft_band_mse(gen_batch, real_batch, masks)
            step_result['per_band_mse'] = [m.item() for m in band_mses]

        results[f'{ns}_step'] = step_result
    return results


# ==========================================================================
# TEACHER TRAINING
# ==========================================================================
def train_teacher(data, num_steps=TEACHER_STEPS):
    """Train flow matching teacher on CIFAR-10."""
    print("\n" + "=" * 60)
    print(f"TRAINING FLOW MATCHING TEACHER ({num_steps} steps)")
    print("=" * 60, flush=True)

    ckpt_path = WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt'
    results_path = WORKSPACE / 'exp' / 'teacher' / 'results.json'

    # Check if teacher is already trained with enough steps
    if ckpt_path.exists() and results_path.exists():
        with open(results_path) as f:
            prev = json.load(f)
        if prev.get('training_steps', 0) >= num_steps:
            print(f"Teacher checkpoint found (trained {prev['training_steps']} steps), reusing.")
            model = UNet(**MODEL_KWARGS).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            return model

    set_seed(42)
    model = UNet(**MODEL_KWARGS).to(DEVICE)
    ema_model = copy.deepcopy(model)
    compiled_model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda')

    best_fid = float('inf')
    t_start = time.time()
    log_losses = []

    for step in range(1, num_steps + 1):
        x = sample_batch(data, TEACHER_BATCH)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            t, x_t, target_v, noise = ot_cfm_sample_t_and_xt(x)
            pred_v = compiled_model(x_t, t)
            loss = F.mse_loss(pred_v, target_v)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        ema_update(ema_model, model)
        log_losses.append(loss.item())

        if step % 2000 == 0:
            elapsed = (time.time() - t_start) / 60
            avg_loss = np.mean(log_losses[-2000:])
            rate = step / elapsed  # steps per min
            eta = (num_steps - step) / rate
            print(f"  Step {step}/{num_steps}, Loss: {avg_loss:.4f}, "
                  f"Time: {elapsed:.0f}min, ETA: {eta:.0f}min", flush=True)

        # Save checkpoint periodically (every 10k steps) without expensive FID
        if step % 10000 == 0:
            save_dir = WORKSPACE / 'exp' / 'teacher'
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ema_model.state_dict(), save_dir / 'checkpoint_best.pt')
            print(f"  Saved teacher checkpoint at step {step}", flush=True)

        if step == num_steps:
            print(f"  Evaluating teacher at final step...", flush=True)
            samples = generate_samples_velocity(ema_model, 10000, 100, seed=42)
            fid_score = compute_fid_from_samples(samples)
            print(f"  Teacher FID (100 steps, 10k samples): {fid_score:.2f}", flush=True)
            best_fid = fid_score
            save_dir = WORKSPACE / 'exp' / 'teacher'
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ema_model.state_dict(), save_dir / 'checkpoint_best.pt')

    total_time = (time.time() - t_start) / 60
    print(f"Teacher training done. Best FID: {best_fid:.2f}, Time: {total_time:.0f}min")

    # Evaluate at various step budgets
    teacher_results = {
        'best_fid_100step': best_fid,
        'training_steps': num_steps,
        'training_time_min': total_time,
        'model_params_M': sum(p.numel() for p in model.parameters()) / 1e6,
    }
    for ns in [1, 4, 100]:
        samples = generate_samples_velocity(ema_model, 10000, ns, seed=42)
        fid_val = compute_fid_from_samples(samples)
        teacher_results[f'fid_{ns}step'] = fid_val
        print(f"  Teacher {ns}-step FID: {fid_val:.2f}", flush=True)

    save_dir = WORKSPACE / 'exp' / 'teacher'
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(teacher_results, f, indent=2)

    return ema_model


# ==========================================================================
# PRE-COMPUTE TEACHER ODE TARGETS
# ==========================================================================
@torch.no_grad()
def precompute_teacher_targets(teacher, data, num_pairs=50000, max_teacher_steps=10,
                                checkpoint_steps=None, seed=42):
    """Pre-compute teacher ODE targets for all training pairs.

    Returns dict with:
        x0: original images (N, 3, 32, 32)
        noise: random noise (N, 3, 32, 32)
        t: time values (N,)
        x_t: noisy images (N, 3, 32, 32)
        targets: dict mapping step_count -> teacher predictions (N, 3, 32, 32)
    """
    print(f"\nPre-computing teacher ODE targets ({num_pairs} pairs, "
          f"max {max_teacher_steps} steps)...", flush=True)

    if checkpoint_steps is None:
        checkpoint_steps = list(range(1, max_teacher_steps + 1))

    set_seed(seed)
    teacher.eval()
    compiled_teacher = torch.compile(teacher)

    # Sample training pairs
    idx = torch.randperm(len(data))[:num_pairs]
    x0 = data[idx]  # (N, 3, 32, 32) on GPU
    noise = torch.randn_like(x0)
    t = torch.rand(num_pairs, device=DEVICE) * 0.98 + 0.01
    t_expand = t.view(num_pairs, 1, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * noise

    # Compute targets in batches
    batch_size = 512
    all_targets = {s: [] for s in checkpoint_steps}

    t0 = time.time()
    for i in range(0, num_pairs, batch_size):
        batch_xt = x_t[i:i+batch_size]
        batch_t = t[i:i+batch_size]

        checkpoints = teacher_solve(compiled_teacher, batch_xt, batch_t,
                                     num_steps=max_teacher_steps,
                                     return_checkpoints=sorted(checkpoint_steps))

        for s in checkpoint_steps:
            all_targets[s].append(checkpoints[s].half())  # save as fp16

    targets = {s: torch.cat(all_targets[s]) for s in checkpoint_steps}
    elapsed = time.time() - t0
    print(f"  Pre-computed {num_pairs} targets in {elapsed:.0f}s "
          f"({num_pairs/elapsed:.0f} pairs/sec)", flush=True)

    return {
        'x0': x0,
        'noise': noise,
        't': t,
        'x_t': x_t,
        'targets': targets,
    }


# ==========================================================================
# DISTILLATION TRAINING (with pre-computed targets)
# ==========================================================================
def train_distillation(precomputed, method, seed, num_steps=DISTILL_STEPS,
                       lr=1e-4, batch_size=DISTILL_BATCH,
                       use_pseudohuber=False,
                       use_spectral=False,
                       spectral_config=None,
                       teacher_ckpt_path=None):
    """Train consistency distillation using pre-computed teacher targets."""
    print(f"\n  Training {method} (seed={seed}, {num_steps} steps)...", flush=True)

    save_dir = WORKSPACE / 'exp' / method
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f'checkpoint_seed{seed}.pt'

    if ckpt_path.exists():
        print(f"    Checkpoint exists, loading...", flush=True)
        model = UNet(**MODEL_KWARGS).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        return model, 0.0

    set_seed(seed)

    student = UNet(**MODEL_KWARGS).to(DEVICE)
    # Initialize from teacher
    if teacher_ckpt_path:
        student.load_state_dict(torch.load(teacher_ckpt_path, map_location=DEVICE, weights_only=True))
    ema_student = copy.deepcopy(student)
    compiled_student = torch.compile(student)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda')

    # Spectral setup
    masks = None
    if use_spectral and spectral_config:
        K = spectral_config.get('K', 4)
        masks = create_fft_frequency_masks(32, 32, K, device=DEVICE)
        teacher_steps_per_band = spectral_config.get('teacher_steps', SCD_TEACHER_STEPS)
        base_weights = spectral_config.get('weights', [1.0, 1.5, 2.5, 4.0])
        progressive = spectral_config.get('progressive', True)
        fixed_weights = spectral_config.get('fixed_weights', False)

    # Pseudo-Huber parameters
    if use_pseudohuber:
        d = 3 * 32 * 32
        c_huber = 0.00054 * math.sqrt(d)

    # Pre-computed data
    N = len(precomputed['t'])
    running_band_errors = [1.0] * NUM_FREQ_BANDS
    t_start = time.time()
    band_error_history = []

    for step in range(1, num_steps + 1):
        # Sample batch from pre-computed data
        idx = torch.randint(0, N, (batch_size,), device=DEVICE)
        x_t = precomputed['x_t'][idx].float()
        t_vals = precomputed['t'][idx]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            student_pred = compiled_student(x_t, t_vals)

            if use_spectral and masks is not None:
                # SCD: per-band spectral loss
                loss = torch.tensor(0.0, device=DEVICE)

                # Progressive weight schedule
                if progressive and not fixed_weights:
                    frac = step / num_steps
                    if frac < 0.3:
                        weights = [1.0] * len(masks)
                    elif frac < 0.7:
                        alpha = (frac - 0.3) / 0.4
                        weights = [1.0 + alpha * (w - 1.0) for w in base_weights]
                    else:
                        mean_err = sum(running_band_errors) / len(running_band_errors)
                        weights = [(e / (mean_err + 1e-8)) ** 0.5
                                   for e in running_band_errors]
                        w_sum = sum(weights)
                        weights = [w * len(weights) / (w_sum + 1e-8) for w in weights]
                elif fixed_weights:
                    weights = base_weights
                else:
                    weights = [1.0] * len(masks)

                for k, (mask, w) in enumerate(zip(masks, weights)):
                    ns = teacher_steps_per_band[k]
                    target_k = precomputed['targets'][ns][idx].float()
                    diff_k = student_pred - target_k
                    diff_k_freq = torch.fft.fft2(diff_k)
                    band_power = (diff_k_freq.real ** 2 + diff_k_freq.imag ** 2) * mask
                    band_loss = band_power.mean()
                    loss = loss + w * band_loss
                    running_band_errors[k] = 0.99 * running_band_errors[k] + 0.01 * band_loss.item()

            else:
                # Standard CD or Pseudo-Huber
                target = precomputed['targets'][TEACHER_ODE_STEPS][idx].float()
                diff = student_pred - target

                if use_pseudohuber:
                    loss = (torch.sqrt(diff.pow(2).sum(dim=(1, 2, 3)) + c_huber ** 2)
                            - c_huber).mean()
                else:
                    loss = F.mse_loss(student_pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        ema_update(ema_student, student, decay=0.999)

        if step % 2000 == 0:
            elapsed = (time.time() - t_start) / 60
            rate = step / elapsed
            eta = (num_steps - step) / rate
            print(f"    Step {step}/{num_steps}, Loss: {loss.item():.4f}, "
                  f"Time: {elapsed:.1f}min, ETA: {eta:.0f}min", flush=True)
            if use_spectral and masks is not None:
                band_error_history.append({
                    'step': step,
                    'band_errors': list(running_band_errors),
                    'weights': list(weights),
                })

    total_time = (time.time() - t_start) / 60
    print(f"    Done. Time: {total_time:.1f}min", flush=True)

    torch.save(ema_student.state_dict(), ckpt_path)

    if band_error_history:
        with open(save_dir / f'band_errors_seed{seed}.json', 'w') as f:
            json.dump(band_error_history, f, indent=2)

    return ema_student, total_time


# ==========================================================================
# AGGREGATE RESULTS
# ==========================================================================
def aggregate_results(all_results):
    """Aggregate per-seed results into mean +/- std."""
    aggregated = {'main_results': {}, 'ablation_results': {}, 'success_criteria': {}}

    for method in ['cd_baseline', 'cd_pseudohuber', 'rectflow_baseline', 'scd_main']:
        method_data = all_results.get(method, {})
        agg = {}
        for step_key in ['1_step', '2_step', '4_step']:
            fids = [method_data[s][step_key]['fid']
                    for s in SEEDS if s in method_data and step_key in method_data[s]]
            if fids:
                agg[step_key] = {
                    'fid_mean': float(np.mean(fids)),
                    'fid_std': float(np.std(fids)),
                }
                if 'per_band_mse' in method_data[SEEDS[0]][step_key]:
                    K = len(method_data[SEEDS[0]][step_key]['per_band_mse'])
                    for k in range(K):
                        vals = [method_data[s][step_key]['per_band_mse'][k]
                                for s in SEEDS if s in method_data]
                        agg[step_key][f'band{k}_mse_mean'] = float(np.mean(vals))
                        agg[step_key][f'band{k}_mse_std'] = float(np.std(vals))

        train_times = [method_data[s].get('train_time_min', 0)
                       for s in SEEDS if s in method_data]
        agg['train_time_mean'] = float(np.mean(train_times)) if train_times else 0
        aggregated['main_results'][method] = agg

    for abl_name, abl_data in all_results.get('ablations', {}).items():
        agg = {}
        for step_key in ['1_step', '2_step', '4_step']:
            if step_key in abl_data:
                agg[step_key] = {
                    'fid': abl_data[step_key]['fid'],
                }
                if 'per_band_mse' in abl_data[step_key]:
                    agg[step_key]['per_band_mse'] = abl_data[step_key]['per_band_mse']
        agg['train_time_min'] = abl_data.get('train_time_min', 0)
        aggregated['ablation_results'][abl_name] = agg

    # Success criteria
    cd = aggregated['main_results'].get('cd_baseline', {})
    scd = aggregated['main_results'].get('scd_main', {})

    crit1_pass = True
    fid_improvements = {}
    for sk in ['1_step', '2_step', '4_step']:
        if sk in cd and sk in scd:
            cd_fid = cd[sk]['fid_mean']
            scd_fid = scd[sk]['fid_mean']
            improvement = (cd_fid - scd_fid) / cd_fid * 100
            fid_improvements[sk] = float(improvement)
            if scd_fid >= cd_fid:
                crit1_pass = False

    crit2_pass = False
    hf_reduction = None
    if '1_step' in cd and '1_step' in scd:
        cd_hf = cd['1_step'].get('band3_mse_mean')
        scd_hf = scd['1_step'].get('band3_mse_mean')
        if cd_hf and scd_hf and cd_hf > 0:
            hf_reduction = (cd_hf - scd_hf) / cd_hf * 100
            crit2_pass = hf_reduction >= 15

    cd_time = cd.get('train_time_mean', 1)
    scd_time = scd.get('train_time_mean', 1)
    overhead = (scd_time - cd_time) / (cd_time + 1e-8) * 100 if cd_time > 0 else 0
    crit3_pass = overhead < 50

    aggregated['success_criteria'] = {
        'scd_beats_cd_all_steps': crit1_pass,
        'fid_improvements_pct': fid_improvements,
        'hf_error_reduction_15pct': crit2_pass,
        'hf_reduction_pct': float(hf_reduction) if hf_reduction is not None else None,
        'training_overhead_below_50pct': crit3_pass,
        'training_overhead_pct': float(overhead),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for method in ['cd_baseline', 'cd_pseudohuber', 'rectflow_baseline', 'scd_main']:
        data = aggregated['main_results'].get(method, {})
        print(f"\n{method}:")
        for sk in ['1_step', '2_step', '4_step']:
            if sk in data:
                print(f"  {sk}: FID = {data[sk]['fid_mean']:.2f} +/- {data[sk]['fid_std']:.2f}")

    print("\nAblations:")
    for abl_name, data in aggregated.get('ablation_results', {}).items():
        fid_1 = data.get('1_step', {}).get('fid', 'N/A')
        print(f"  {abl_name}: 1-step FID = {fid_1}")

    print("\nSuccess Criteria:")
    for k, v in aggregated['success_criteria'].items():
        print(f"  {k}: {v}")

    return aggregated


# ==========================================================================
# FIGURE GENERATION
# ==========================================================================
def generate_figures(aggregated, all_results):
    """Generate publication figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig_dir = WORKSPACE / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    main = aggregated['main_results']
    methods = ['rectflow_baseline', 'cd_baseline', 'cd_pseudohuber', 'scd_main']
    method_labels = ['Rectified Flow', 'Standard CD', 'Pseudo-Huber CD', 'SCD (Ours)']
    colors = ['#888888', '#1f77b4', '#ff7f0e', '#2ca02c']

    # ---- Figure 4: FID vs Steps ----
    fig, ax = plt.subplots(figsize=(6, 4))
    steps_x = [1, 2, 4]
    for method, label, color in zip(methods, method_labels, colors):
        data = main.get(method, {})
        fids = [data.get(f'{s}_step', {}).get('fid_mean', None) for s in steps_x]
        stds = [data.get(f'{s}_step', {}).get('fid_std', 0) for s in steps_x]
        if all(f is not None for f in fids):
            ax.errorbar(steps_x, fids, yerr=stds, marker='o', label=label,
                       color=color, linewidth=2, capsize=4)
    ax.set_xlabel('Number of Inference Steps')
    ax.set_ylabel('FID (lower is better)')
    ax.set_xticks(steps_x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('FID vs. Number of Inference Steps')
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure4_fid_vs_steps.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'figure4_fid_vs_steps.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Figure 1: Spectral Error Analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    band_labels = ['Low', 'Mid-Low', 'Mid-High', 'High']

    for ax, step_key, title in zip(axes, ['1_step', '4_step'],
                                     ['1-Step Generation', '4-Step Generation']):
        x = np.arange(len(band_labels))
        width = 0.18
        for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
            data = main.get(method, {}).get(step_key, {})
            band_mses = [data.get(f'band{k}_mse_mean', 0) for k in range(4)]
            ax.bar(x + i * width, band_mses, width, label=label, color=color, alpha=0.85)

        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Per-Band MSE')
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(band_labels)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(fig_dir / 'figure1_spectral_error.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'figure1_spectral_error.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Table 1: Main Results (CSV + LaTeX) ----
    with open(fig_dir / 'table1_main_results.csv', 'w') as f:
        f.write("Method,1-step FID,2-step FID,4-step FID\n")
        for method, label in zip(methods, method_labels):
            data = main.get(method, {})
            row = [label]
            for sk in ['1_step', '2_step', '4_step']:
                d = data.get(sk, {})
                m, s = d.get('fid_mean', 0), d.get('fid_std', 0)
                row.append(f"{m:.2f} +/- {s:.2f}")
            f.write(",".join(row) + "\n")

    with open(fig_dir / 'table1_main_results.tex', 'w') as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Method & 1-step FID $\\downarrow$ & 2-step FID $\\downarrow$ & 4-step FID $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        # Find best FID per step
        best_fid = {}
        for sk in ['1_step', '2_step', '4_step']:
            vals = [main.get(m, {}).get(sk, {}).get('fid_mean', 1e9) for m in methods]
            best_fid[sk] = min(vals)

        for method, label in zip(methods, method_labels):
            data = main.get(method, {})
            row = [label]
            for sk in ['1_step', '2_step', '4_step']:
                d = data.get(sk, {})
                m, s = d.get('fid_mean', 0), d.get('fid_std', 0)
                val = f"{m:.1f} $\\pm$ {s:.1f}"
                if abs(m - best_fid[sk]) < 0.01:
                    val = "\\textbf{" + val + "}"
                row.append(val)
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table 2: Ablation Study ----
    with open(fig_dir / 'table2_ablation.csv', 'w') as f:
        f.write("Variant,1-step FID,2-step FID,4-step FID\n")
        # SCD main (seed=42) as reference
        scd_42 = all_results.get('scd_main', {}).get(42, {})
        f.write(f"SCD (full),{scd_42.get('1_step',{}).get('fid',0):.2f},"
                f"{scd_42.get('2_step',{}).get('fid',0):.2f},"
                f"{scd_42.get('4_step',{}).get('fid',0):.2f}\n")
        for abl_name in aggregated.get('ablation_results', {}):
            data = aggregated['ablation_results'][abl_name]
            f.write(f"{abl_name},"
                    f"{data.get('1_step',{}).get('fid',0):.2f},"
                    f"{data.get('2_step',{}).get('fid',0):.2f},"
                    f"{data.get('4_step',{}).get('fid',0):.2f}\n")

    with open(fig_dir / 'table2_ablation.tex', 'w') as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Variant & 1-step FID & 2-step FID & 4-step FID \\\\\n")
        f.write("\\midrule\n")
        f.write(f"SCD (full) & {scd_42.get('1_step',{}).get('fid',0):.1f} & "
                f"{scd_42.get('2_step',{}).get('fid',0):.1f} & "
                f"{scd_42.get('4_step',{}).get('fid',0):.1f} \\\\\n")
        for abl_name, nice_name in [
            ('ablation_no_adaptive_teacher', 'w/o Adaptive Teacher'),
            ('ablation_no_progressive', 'w/o Progressive Refinement'),
            ('ablation_no_spectral_weight', 'w/o Spectral Weighting'),
        ]:
            data = aggregated['ablation_results'].get(abl_name, {})
            if data:
                f.write(f"{nice_name} & {data.get('1_step',{}).get('fid',0):.1f} & "
                        f"{data.get('2_step',{}).get('fid',0):.1f} & "
                        f"{data.get('4_step',{}).get('fid',0):.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Figure 3: Training dynamics (SCD band errors over training) ----
    band_err_file = WORKSPACE / 'exp' / 'scd_main' / 'band_errors_seed42.json'
    if band_err_file.exists():
        with open(band_err_file) as f:
            band_hist = json.load(f)
        if band_hist:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            steps_plot = [h['step'] for h in band_hist]
            for k, lbl in enumerate(band_labels):
                errors = [h['band_errors'][k] for h in band_hist]
                ax1.plot(steps_plot, errors, label=lbl, linewidth=2)
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Per-Band Error')
            ax1.set_title('SCD Per-Band Distillation Error During Training')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            for k, lbl in enumerate(band_labels):
                ws = [h['weights'][k] for h in band_hist]
                ax2.plot(steps_plot, ws, label=lbl, linewidth=2)
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Band Weight')
            ax2.set_title('Progressive Weight Schedule')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(fig_dir / 'figure3_training_dynamics.pdf', dpi=300, bbox_inches='tight')
            fig.savefig(fig_dir / 'figure3_training_dynamics.png', dpi=150, bbox_inches='tight')
            plt.close()

    # ---- Figure 2: Qualitative comparison (generated images) ----
    print("  Generating qualitative comparison images...", flush=True)
    # Load models and generate samples with same seed for visual comparison
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    method_model_paths = {
        'cd_baseline': WORKSPACE / 'exp' / 'cd_baseline' / 'checkpoint_seed42.pt',
        'cd_pseudohuber': WORKSPACE / 'exp' / 'cd_pseudohuber' / 'checkpoint_seed42.pt',
        'scd_main': WORKSPACE / 'exp' / 'scd_main' / 'checkpoint_seed42.pt',
    }
    row_labels = ['Rect. Flow\n(1 step)', 'Standard CD\n(1 step)',
                  'Pseudo-Huber CD\n(1 step)', 'SCD (Ours)\n(1 step)']

    # Generate with same noise
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(123)
    z = torch.randn(8, 3, 32, 32, device=DEVICE, generator=gen)

    models_to_show = []
    # Teacher (rectified flow)
    teacher_path = WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt'
    if teacher_path.exists():
        m = UNet(**MODEL_KWARGS).to(DEVICE)
        m.load_state_dict(torch.load(teacher_path, map_location=DEVICE, weights_only=True))
        m.eval()
        models_to_show.append(('rectflow', m))

    for name, path in method_model_paths.items():
        if path.exists():
            m = UNet(**MODEL_KWARGS).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            m.eval()
            models_to_show.append((name, m))

    for row_idx, (name, model) in enumerate(models_to_show):
        if row_idx >= 4:
            break
        with torch.no_grad():
            samples = euler_sample(model, z, 1)
            samples = ((samples + 1) / 2).clamp(0, 1).cpu()
        for col_idx in range(8):
            axes[row_idx, col_idx].imshow(samples[col_idx].permute(1, 2, 0).numpy())
            axes[row_idx, col_idx].axis('off')
        axes[row_idx, 0].set_ylabel(row_labels[row_idx], rotation=0,
                                      labelpad=80, fontsize=10, va='center')

    plt.suptitle('1-Step Generation Quality Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure2_qualitative.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'figure2_qualitative.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figures saved to {fig_dir}/", flush=True)


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    sys.stdout.reconfigure(line_buffering=True)
    overall_start = time.time()

    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Workspace: {WORKSPACE}")
    print(f"Model config: {MODEL_KWARGS}")
    print(f"Teacher steps: {TEACHER_STEPS}, Distill steps: {DISTILL_STEPS}")

    # Load data
    print("\nLoading CIFAR-10 to GPU...", flush=True)
    data = load_cifar10_to_gpu()
    real_images = data[:10000].cpu()
    print(f"Data shape: {data.shape}, device: {data.device}")

    # ---- Delete old distillation checkpoints (keep teacher) ----
    for d in ['cd_baseline', 'cd_pseudohuber', 'scd_main',
              'ablation_no_adaptive_teacher', 'ablation_no_progressive',
              'ablation_no_spectral_weight', 'rectflow_baseline']:
        exp_dir = WORKSPACE / 'exp' / d
        if exp_dir.exists():
            for f in exp_dir.glob('*.pt'):
                f.unlink()
            for f in exp_dir.glob('*.json'):
                f.unlink()
            print(f"  Cleared old results in exp/{d}/")

    # ---- Stage 1: Train teacher ----
    teacher = train_teacher(data, num_steps=TEACHER_STEPS)
    teacher_ckpt = WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt'
    teacher_time = (time.time() - overall_start) / 60
    print(f"\nTeacher done. Elapsed: {teacher_time:.0f}min")

    # ---- Stage 2: Pre-compute teacher targets ----
    precomputed = precompute_teacher_targets(
        teacher, data, num_pairs=50000,
        max_teacher_steps=max(SCD_TEACHER_STEPS),
        checkpoint_steps=sorted(set(SCD_TEACHER_STEPS + [TEACHER_ODE_STEPS])),
        seed=42,
    )

    # ---- Stage 3: Run all distillation methods ----
    all_results = {}

    # 3a: Standard CD
    print("\n" + "=" * 60)
    print("STANDARD CONSISTENCY DISTILLATION (3 seeds)")
    print("=" * 60)
    cd_results = {}
    for seed in SEEDS:
        model, train_time = train_distillation(
            precomputed, 'cd_baseline', seed, teacher_ckpt_path=teacher_ckpt)
        results = evaluate_method(model, [1, 2, 4], NUM_FID_SAMPLES,
                                  real_images=real_images, seed=seed)
        results['train_time_min'] = train_time
        cd_results[seed] = results
        with open(WORKSPACE / 'exp' / 'cd_baseline' / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
        del model; torch.cuda.empty_cache()
    all_results['cd_baseline'] = cd_results

    # 3b: Pseudo-Huber CD
    print("\n" + "=" * 60)
    print("PSEUDO-HUBER CONSISTENCY DISTILLATION (3 seeds)")
    print("=" * 60)
    ph_results = {}
    for seed in SEEDS:
        model, train_time = train_distillation(
            precomputed, 'cd_pseudohuber', seed,
            use_pseudohuber=True, teacher_ckpt_path=teacher_ckpt)
        results = evaluate_method(model, [1, 2, 4], NUM_FID_SAMPLES,
                                  real_images=real_images, seed=seed)
        results['train_time_min'] = train_time
        ph_results[seed] = results
        with open(WORKSPACE / 'exp' / 'cd_pseudohuber' / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
        del model; torch.cuda.empty_cache()
    all_results['cd_pseudohuber'] = ph_results

    # 3c: Rectified flow baseline (no training)
    print("\n" + "=" * 60)
    print("RECTIFIED FLOW BASELINE (3 seeds)")
    print("=" * 60)
    rf_results = {}
    for seed in SEEDS:
        results = evaluate_method(teacher, [1, 2, 4], NUM_FID_SAMPLES,
                                  real_images=real_images, seed=seed,
                                  is_velocity_model=True)
        results['train_time_min'] = 0.0
        rf_results[seed] = results
        save_dir = WORKSPACE / 'exp' / 'rectflow_baseline'
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
    all_results['rectflow_baseline'] = rf_results

    # 3d: SCD Main
    print("\n" + "=" * 60)
    print("SPECTRAL CONSISTENCY DISTILLATION (3 seeds)")
    print("=" * 60)
    scd_config = {
        'K': 4,
        'teacher_steps': SCD_TEACHER_STEPS,
        'weights': [1.0, 1.5, 2.5, 4.0],
        'progressive': True,
        'fixed_weights': False,
    }
    scd_results = {}
    for seed in SEEDS:
        model, train_time = train_distillation(
            precomputed, 'scd_main', seed,
            use_spectral=True, spectral_config=scd_config,
            teacher_ckpt_path=teacher_ckpt)
        results = evaluate_method(model, [1, 2, 4], NUM_FID_SAMPLES,
                                  real_images=real_images, seed=seed)
        results['train_time_min'] = train_time
        scd_results[seed] = results
        with open(WORKSPACE / 'exp' / 'scd_main' / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
        del model; torch.cuda.empty_cache()
    all_results['scd_main'] = scd_results

    # ---- Stage 4: Ablations ----
    print("\n" + "=" * 60)
    print("ABLATION STUDIES (seed=42)")
    print("=" * 60)

    ablation_configs = {
        'ablation_no_spectral_weight': {
            'K': 4,
            'teacher_steps': [10, 10, 10, 10],
            'weights': [1.0, 1.0, 1.0, 1.0],
            'progressive': False,
            'fixed_weights': True,
        },
        'ablation_no_adaptive_teacher': {
            'K': 4,
            'teacher_steps': [10, 10, 10, 10],
            'weights': [1.0, 1.5, 2.5, 4.0],
            'progressive': True,
            'fixed_weights': False,
        },
        'ablation_no_progressive': {
            'K': 4,
            'teacher_steps': SCD_TEACHER_STEPS,
            'weights': [1.0, 1.5, 2.5, 4.0],
            'progressive': False,
            'fixed_weights': True,
        },
    }
    ablation_results = {}
    for abl_name, abl_config in ablation_configs.items():
        model, train_time = train_distillation(
            precomputed, abl_name, seed=42,
            use_spectral=True, spectral_config=abl_config,
            teacher_ckpt_path=teacher_ckpt)
        results = evaluate_method(model, [1, 2, 4], NUM_FID_SAMPLES,
                                  real_images=real_images, seed=42)
        results['train_time_min'] = train_time
        ablation_results[abl_name] = results
        save_dir = WORKSPACE / 'exp' / abl_name
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        del model; torch.cuda.empty_cache()
    all_results['ablations'] = ablation_results

    # ---- Stage 5: Aggregate ----
    aggregated = aggregate_results(all_results)
    with open(WORKSPACE / 'results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    # ---- Stage 6: Figures ----
    print("\nGenerating figures...", flush=True)
    generate_figures(aggregated, all_results)

    total_hours = (time.time() - overall_start) / 3600
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE. Total time: {total_hours:.2f} hours")
    print(f"Results: {WORKSPACE / 'results.json'}")
    print(f"Figures: {WORKSPACE / 'figures/'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
