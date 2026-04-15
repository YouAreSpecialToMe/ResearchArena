"""Master experiment script for Spectral Consistency Distillation.

Runs all experiments sequentially:
1. Train teacher flow matching model
2. Standard CD baseline (3 seeds)
3. Pseudo-Huber CD baseline (3 seeds)
4. Rectified flow baseline (3 seeds)
5. SCD main method (3 seeds)
6. Ablation studies (1 seed each)
7. Aggregate results
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from collections import defaultdict

# Setup paths
WORKSPACE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE))

from exp.shared.models import UNet
from exp.shared.flow_matching import ot_cfm_sample_t_and_xt, euler_sample, teacher_solve
from exp.shared.spectral import create_fft_frequency_masks, fft_band_mse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FREQ_BANDS = 4
SEEDS = [42, 43, 44]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cifar10_loader(batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=str(WORKSPACE / 'data'), train=train,
                          download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=4, pin_memory=True, drop_last=train)


def get_real_images(num=10000):
    """Get real CIFAR-10 images for spectral comparison."""
    loader = get_cifar10_loader(batch_size=256, train=True)
    imgs = []
    for x, _ in loader:
        imgs.append(x)
        if len(imgs) * 256 >= num:
            break
    return torch.cat(imgs)[:num]


def ema_update(ema_model, model, decay=0.9999):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


@torch.no_grad()
def generate_samples(model, num_samples, num_steps, batch_size=512,
                     device='cuda', seed=42):
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    all_samples = []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, 3, 32, 32, device=device, generator=gen)
        samples = euler_sample(model, z, num_steps)
        all_samples.append(samples.cpu())
        remaining -= bs
    return torch.cat(all_samples)[:num_samples]


def compute_fid(samples, tmp_base='/tmp/scd_fid'):
    """Compute FID for generated samples vs CIFAR-10."""
    from cleanfid import fid
    from torchvision.utils import save_image

    tmp_dir = tempfile.mkdtemp(prefix='scd_fid_')
    try:
        samples_01 = (samples + 1) / 2
        samples_01 = samples_01.clamp(0, 1)
        for i in range(len(samples_01)):
            save_image(samples_01[i], os.path.join(tmp_dir, f'{i:06d}.png'))
        score = fid.compute_fid(tmp_dir, dataset_name='cifar10',
                                dataset_split='train', dataset_res=32, mode='clean')
        return score
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_method(model, steps_list=[1, 2, 4], num_samples=10000,
                    real_images=None, seed=42):
    """Evaluate a model at multiple step budgets."""
    results = {}
    masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)

    for ns in steps_list:
        print(f"    Generating {num_samples} samples with {ns} step(s)...")
        t0 = time.time()
        samples = generate_samples(model, num_samples, ns, seed=seed)
        gen_time = time.time() - t0

        print(f"    Computing FID...")
        fid_score = compute_fid(samples)
        print(f"    {ns}-step FID: {fid_score:.2f} (gen time: {gen_time:.1f}s)")

        step_result = {'fid': fid_score, 'gen_time_s': gen_time}

        # Per-band spectral MSE
        if real_images is not None:
            n = min(10000, len(samples), len(real_images))
            gen_batch = samples[:n].to(DEVICE)
            real_batch = real_images[:n].to(DEVICE)
            band_mses = fft_band_mse(gen_batch, real_batch, masks)
            step_result['per_band_mse'] = [m.item() for m in band_mses]

        results[f'{ns}_step'] = step_result

    return results


# ============================================================================
# TEACHER TRAINING
# ============================================================================
def train_teacher(num_steps=20000, lr=2e-4, batch_size=128, eval_every=10000):
    """Train flow matching teacher on CIFAR-10."""
    print("\n" + "=" * 60)
    print("STAGE 1: Training Flow Matching Teacher")
    print("=" * 60)

    ckpt_path = WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt'
    if ckpt_path.exists():
        print(f"Teacher checkpoint found at {ckpt_path}, skipping training.")
        model = UNet().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        return model

    set_seed(42)
    model = UNet().to(DEVICE)
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda')
    loader = get_cifar10_loader(batch_size=batch_size)

    best_fid = float('inf')
    step = 0
    t_start = time.time()

    while step < num_steps:
        for x, _ in loader:
            if step >= num_steps:
                break

            x = x.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                t, x_t, target_v, noise = ot_cfm_sample_t_and_xt(x)
                pred_v = model(x_t, t)
                loss = F.mse_loss(pred_v, target_v)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ema_update(ema_model, model)
            step += 1

            if step % 500 == 0:
                elapsed = time.time() - t_start
                print(f"  Step {step}/{num_steps}, Loss: {loss.item():.4f}, "
                      f"Time: {elapsed / 60:.1f}min", flush=True)

            if step % eval_every == 0 or step == num_steps:
                print(f"  Evaluating teacher at step {step}...")
                samples = generate_samples(ema_model, 5000, 100, seed=42)
                fid_score = compute_fid(samples)
                print(f"  Teacher FID (100 steps, 10k samples): {fid_score:.2f}")
                if fid_score < best_fid:
                    best_fid = fid_score
                    save_dir = WORKSPACE / 'exp' / 'teacher'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(ema_model.state_dict(), save_dir / 'checkpoint_best.pt')
                    print(f"  Saved best teacher (FID={fid_score:.2f})")

    total_time = (time.time() - t_start) / 60
    print(f"Teacher training complete. Best FID: {best_fid:.2f}, Time: {total_time:.1f}min")

    # Save teacher results
    teacher_results = {
        'best_fid_100step': best_fid,
        'training_steps': num_steps,
        'training_time_min': total_time,
    }

    # Evaluate at different step budgets
    print("Evaluating teacher at various step counts...")
    for ns in [1, 4, 50, 100]:
        samples = generate_samples(ema_model, 5000, ns, seed=42)
        fid_score = compute_fid(samples)
        teacher_results[f'fid_{ns}step'] = fid_score
        print(f"  Teacher {ns}-step FID: {fid_score:.2f}")

    with open(WORKSPACE / 'exp' / 'teacher' / 'results.json', 'w') as f:
        json.dump(teacher_results, f, indent=2)

    return ema_model


# ============================================================================
# CONSISTENCY DISTILLATION
# ============================================================================
def train_consistency_distillation(teacher, method='cd_baseline', seed=42,
                                   num_steps=4000, lr=1e-4, batch_size=128,
                                   use_pseudohuber=False,
                                   use_spectral=False,
                                   spectral_config=None):
    """Train consistency distillation student.

    Args:
        teacher: pretrained teacher model (frozen)
        method: experiment name
        seed: random seed
        use_pseudohuber: use Pseudo-Huber loss instead of MSE
        use_spectral: use SCD spectral loss
        spectral_config: dict with SCD parameters
    """
    set_seed(seed)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = UNet().to(DEVICE)
    # Initialize student from teacher
    student.load_state_dict(teacher.state_dict())
    ema_student = copy.deepcopy(student)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda')
    loader = get_cifar10_loader(batch_size=batch_size)

    # Spectral setup
    masks = None
    if use_spectral and spectral_config:
        masks = create_fft_frequency_masks(32, 32, spectral_config.get('K', 4),
                                           device=DEVICE)
        teacher_steps_per_band = spectral_config.get('teacher_steps',
                                                      [10, 20, 50, 100])
        base_weights = spectral_config.get('weights', [1.0, 1.5, 2.5, 4.0])
        progressive = spectral_config.get('progressive', True)
        fixed_weights = spectral_config.get('fixed_weights', False)

    # Pseudo-Huber parameters
    if use_pseudohuber:
        d = 3 * 32 * 32
        c_huber = 0.00054 * math.sqrt(d)

    step = 0
    running_band_errors = [0.0] * NUM_FREQ_BANDS
    t_start = time.time()

    while step < num_steps:
        for x, _ in loader:
            if step >= num_steps:
                break
            x = x.to(DEVICE)
            B = x.shape[0]
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Sample t and construct x_t
                noise = torch.randn_like(x)
                t = torch.rand(B, device=DEVICE) * 0.98 + 0.01  # t in [0.01, 0.99]
                t_expand = t.view(B, 1, 1, 1)
                x_t = (1 - t_expand) * x + t_expand * noise

                # Student prediction: map x_t -> x_0
                student_pred = student(x_t, t)

                if use_spectral and spectral_config:
                    # SCD: spectral loss with per-band teacher supervision
                    loss = torch.tensor(0.0, device=DEVICE)

                    # Determine weights based on training phase
                    if progressive and not fixed_weights:
                        if step < num_steps * 0.3:
                            # Phase 1: uniform
                            weights = [1.0] * len(masks)
                        elif step < num_steps * 0.7:
                            # Phase 2: linearly ramp up HF weights
                            alpha = (step - num_steps * 0.3) / (num_steps * 0.4)
                            weights = [1.0 + alpha * (w - 1.0) for w in base_weights]
                        else:
                            # Phase 3: error-driven
                            if sum(running_band_errors) > 0:
                                mean_err = sum(running_band_errors) / len(running_band_errors)
                                weights = [(e / (mean_err + 1e-8)) ** 0.5
                                           for e in running_band_errors]
                                # Normalize
                                w_sum = sum(weights)
                                weights = [w * len(weights) / (w_sum + 1e-8)
                                           for w in weights]
                            else:
                                weights = base_weights
                    elif fixed_weights:
                        weights = base_weights
                    else:
                        weights = [1.0] * len(masks)

                    # Get teacher targets using single ODE solve with checkpoints
                    with torch.no_grad():
                        max_steps = max(teacher_steps_per_band)
                        checkpoints = teacher_solve(
                            teacher, x_t, t, num_steps=max_steps,
                            return_checkpoints=sorted(set(teacher_steps_per_band)))

                    # Compute per-band loss
                    for k, (mask, w) in enumerate(zip(masks, weights)):
                        ns = teacher_steps_per_band[k]
                        target_k = checkpoints[ns]
                        diff_k = student_pred - target_k
                        diff_k_freq = torch.fft.fft2(diff_k)
                        band_power = (diff_k_freq.real ** 2 + diff_k_freq.imag ** 2) * mask
                        band_loss = band_power.mean()
                        loss = loss + w * band_loss

                        # Update running error
                        running_band_errors[k] = 0.99 * running_band_errors[k] + 0.01 * band_loss.item()

                else:
                    # Standard CD or Pseudo-Huber: single teacher target
                    with torch.no_grad():
                        teacher_target = teacher_solve(teacher, x_t, t, num_steps=5)

                    diff = student_pred - teacher_target

                    if use_pseudohuber:
                        # Pseudo-Huber loss
                        loss = (torch.sqrt(diff.pow(2).sum(dim=(1, 2, 3)) + c_huber ** 2)
                                - c_huber).mean()
                    else:
                        # Standard MSE
                        loss = F.mse_loss(student_pred, teacher_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ema_update(ema_student, student, decay=0.999)
            step += 1

            if step % 1000 == 0:
                elapsed = (time.time() - t_start) / 60
                print(f"  [{method} seed={seed}] Step {step}/{num_steps}, "
                      f"Loss: {loss.item():.4f}, Time: {elapsed:.1f}min", flush=True)

    total_time = (time.time() - t_start) / 60
    print(f"  [{method} seed={seed}] Training complete. Time: {total_time:.1f}min")

    # Save checkpoint
    save_dir = WORKSPACE / 'exp' / method
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ema_student.state_dict(), save_dir / f'checkpoint_seed{seed}.pt')

    return ema_student, total_time


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Workspace: {WORKSPACE}")

    overall_start = time.time()

    # Download CIFAR-10
    print("\nDownloading CIFAR-10...")
    get_cifar10_loader(batch_size=2)  # trigger download
    real_images = get_real_images(10000)
    print(f"Real images for spectral analysis: {real_images.shape}")

    # ---- Stage 1: Train teacher ----
    teacher = train_teacher(num_steps=20000)

    # ---- Stage 2: Baselines and main method ----
    all_results = {}

    # --- 2a: Standard CD (3 seeds) ---
    print("\n" + "=" * 60)
    print("STAGE 2a: Standard Consistency Distillation (3 seeds)")
    print("=" * 60)
    cd_results_per_seed = {}
    for seed in SEEDS:
        print(f"\n  --- CD Baseline, seed={seed} ---")
        ckpt = WORKSPACE / 'exp' / 'cd_baseline' / f'checkpoint_seed{seed}.pt'
        if ckpt.exists():
            model = UNet().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            train_time = 0.0
        else:
            model, train_time = train_consistency_distillation(
                teacher, method='cd_baseline', seed=seed)
        results = evaluate_method(model, [1, 2, 4], num_samples=10000,
                                  real_images=real_images, seed=seed)
        results['train_time_min'] = train_time
        cd_results_per_seed[seed] = results
        with open(WORKSPACE / 'exp' / 'cd_baseline' / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
    all_results['cd_baseline'] = cd_results_per_seed

    # --- 2b: Pseudo-Huber CD (3 seeds) ---
    print("\n" + "=" * 60)
    print("STAGE 2b: Pseudo-Huber Consistency Distillation (3 seeds)")
    print("=" * 60)
    ph_results_per_seed = {}
    for seed in SEEDS:
        print(f"\n  --- Pseudo-Huber CD, seed={seed} ---")
        ckpt = WORKSPACE / 'exp' / 'cd_pseudohuber' / f'checkpoint_seed{seed}.pt'
        if ckpt.exists():
            model = UNet().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            train_time = 0.0
        else:
            model, train_time = train_consistency_distillation(
                teacher, method='cd_pseudohuber', seed=seed,
                use_pseudohuber=True)
        results = evaluate_method(model, [1, 2, 4], num_samples=10000,
                                  real_images=real_images, seed=seed)
        results['train_time_min'] = train_time
        ph_results_per_seed[seed] = results
        with open(WORKSPACE / 'exp' / 'cd_pseudohuber' / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
    all_results['cd_pseudohuber'] = ph_results_per_seed

    # --- 2c: Rectified flow baseline (3 seeds) ---
    print("\n" + "=" * 60)
    print("STAGE 2c: Rectified Flow Few-Step Baseline (3 seeds)")
    print("=" * 60)
    rf_results_per_seed = {}
    for seed in SEEDS:
        print(f"\n  --- Rectified Flow, seed={seed} ---")
        results = evaluate_method(teacher, [1, 2, 4], num_samples=10000,
                                  real_images=real_images, seed=seed)
        results['train_time_min'] = 0.0
        rf_results_per_seed[seed] = results
        save_dir = WORKSPACE / 'exp' / 'rectflow_baseline'
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
    all_results['rectflow_baseline'] = rf_results_per_seed

    # --- 2d: SCD Main Method (3 seeds) ---
    print("\n" + "=" * 60)
    print("STAGE 2d: Spectral Consistency Distillation - Main (3 seeds)")
    print("=" * 60)
    scd_config = {
        'K': 4,
        'teacher_steps': [2, 4, 7, 10],
        'weights': [1.0, 1.5, 2.5, 4.0],
        'progressive': True,
        'fixed_weights': False,
    }
    scd_results_per_seed = {}
    for seed in SEEDS:
        print(f"\n  --- SCD Main, seed={seed} ---")
        ckpt = WORKSPACE / 'exp' / 'scd_main' / f'checkpoint_seed{seed}.pt'
        if ckpt.exists():
            model = UNet().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            train_time = 0.0
        else:
            model, train_time = train_consistency_distillation(
                teacher, method='scd_main', seed=seed,
                use_spectral=True, spectral_config=scd_config)
        results = evaluate_method(model, [1, 2, 4], num_samples=10000,
                                  real_images=real_images, seed=seed)
        results['train_time_min'] = train_time
        scd_results_per_seed[seed] = results
        with open(WORKSPACE / 'exp' / 'scd_main' / f'results_seed{seed}.json', 'w') as f:
            json.dump(results, f, indent=2)
    all_results['scd_main'] = scd_results_per_seed

    # ---- Stage 3: Ablation studies (seed=42 only) ----
    print("\n" + "=" * 60)
    print("STAGE 3: Ablation Studies")
    print("=" * 60)

    ablation_configs = {
        'ablation_no_spectral_weight': {
            'K': 4,
            'teacher_steps': [10, 10, 10, 10],  # Uniform teacher steps
            'weights': [1.0, 1.0, 1.0, 1.0],  # Uniform weights (no spectral emphasis)
            'progressive': False,
            'fixed_weights': True,
        },
        'ablation_no_adaptive_teacher': {
            'K': 4,
            'teacher_steps': [5, 5, 5, 5],  # Same teacher steps for all bands
            'weights': [1.0, 1.5, 2.5, 4.0],
            'progressive': True,
            'fixed_weights': False,
        },
        'ablation_no_progressive': {
            'K': 4,
            'teacher_steps': [2, 4, 7, 10],
            'weights': [1.0, 1.5, 2.5, 4.0],
            'progressive': False,
            'fixed_weights': True,  # Fixed weights from start
        },
    }

    ablation_results = {}
    for abl_name, abl_config in ablation_configs.items():
        print(f"\n  --- Ablation: {abl_name} ---")
        ckpt = WORKSPACE / 'exp' / abl_name / f'checkpoint_seed42.pt'
        if ckpt.exists():
            model = UNet().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            train_time = 0.0
        else:
            model, train_time = train_consistency_distillation(
                teacher, method=abl_name, seed=42,
                use_spectral=True, spectral_config=abl_config)
        results = evaluate_method(model, [1, 2, 4], num_samples=10000,
                                  real_images=real_images, seed=42)
        results['train_time_min'] = train_time
        ablation_results[abl_name] = results
        save_dir = WORKSPACE / 'exp' / abl_name
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    all_results['ablations'] = ablation_results

    # ---- Stage 4: Aggregate results ----
    print("\n" + "=" * 60)
    print("STAGE 4: Aggregating Results")
    print("=" * 60)

    aggregated = aggregate_results(all_results)

    with open(WORKSPACE / 'results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    total_time = (time.time() - overall_start) / 3600
    print(f"\nAll experiments complete. Total time: {total_time:.2f} hours")
    print(f"Results saved to {WORKSPACE / 'results.json'}")

    return aggregated


def aggregate_results(all_results):
    """Aggregate per-seed results into mean +/- std."""
    aggregated = {'main_results': {}, 'ablation_results': {}, 'success_criteria': {}}

    # Main methods with 3 seeds
    for method in ['cd_baseline', 'cd_pseudohuber', 'rectflow_baseline', 'scd_main']:
        method_data = all_results.get(method, {})
        agg = {}
        for step_key in ['1_step', '2_step', '4_step']:
            fids = [method_data[s][step_key]['fid'] for s in SEEDS if s in method_data]
            if fids:
                agg[step_key] = {
                    'fid_mean': float(np.mean(fids)),
                    'fid_std': float(np.std(fids)),
                }
                # Aggregate per-band MSE if available
                if 'per_band_mse' in method_data[SEEDS[0]][step_key]:
                    K = len(method_data[SEEDS[0]][step_key]['per_band_mse'])
                    for k in range(K):
                        vals = [method_data[s][step_key]['per_band_mse'][k]
                                for s in SEEDS if s in method_data]
                        agg[step_key][f'band{k}_mse_mean'] = float(np.mean(vals))
                        agg[step_key][f'band{k}_mse_std'] = float(np.std(vals))

        train_times = [method_data[s].get('train_time_min', 0) for s in SEEDS if s in method_data]
        agg['train_time_mean'] = float(np.mean(train_times))
        aggregated['main_results'][method] = agg

    # Ablations (single seed)
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

    # Check success criteria
    cd_fids = aggregated['main_results'].get('cd_baseline', {})
    scd_fids = aggregated['main_results'].get('scd_main', {})

    # Criterion 1: SCD outperforms CD at all step budgets
    crit1_pass = True
    for sk in ['1_step', '2_step', '4_step']:
        if sk in cd_fids and sk in scd_fids:
            if scd_fids[sk]['fid_mean'] >= cd_fids[sk]['fid_mean']:
                crit1_pass = False

    # Criterion 2: HF error reduction >= 15%
    crit2_pass = False
    if '1_step' in cd_fids and '1_step' in scd_fids:
        cd_hf = cd_fids['1_step'].get('band3_mse_mean', None)
        scd_hf = scd_fids['1_step'].get('band3_mse_mean', None)
        if cd_hf and scd_hf and cd_hf > 0:
            hf_reduction = (cd_hf - scd_hf) / cd_hf * 100
            crit2_pass = hf_reduction >= 15
            aggregated['success_criteria']['hf_reduction_pct'] = float(hf_reduction)

    # Criterion 3: Training overhead < 50%
    cd_time = aggregated['main_results'].get('cd_baseline', {}).get('train_time_mean', 1)
    scd_time = aggregated['main_results'].get('scd_main', {}).get('train_time_mean', 1)
    overhead = (scd_time - cd_time) / (cd_time + 1e-8) * 100
    crit3_pass = overhead < 50

    aggregated['success_criteria'].update({
        'scd_beats_cd_all_steps': crit1_pass,
        'hf_error_reduction_15pct': crit2_pass,
        'training_overhead_below_50pct': crit3_pass,
        'training_overhead_pct': float(overhead),
    })

    # Print summary
    print("\n--- RESULTS SUMMARY ---")
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


if __name__ == '__main__':
    main()
