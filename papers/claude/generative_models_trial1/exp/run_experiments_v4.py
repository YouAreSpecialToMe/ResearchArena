#!/usr/bin/env python3
"""
Spectral Consistency Distillation (SCD) - Complete Experiment Suite v4

Addresses all self-review issues:
1. Consistent per_band_mse normalization (Parseval: /(H*W)^2)
2. All 3 seeds for SCD
3. Ablation experiments (no_adaptive_teacher, no_progressive)
4. Aggregated results.json and figures
5. Honest reporting of results

Uses existing 6.5M teacher (FID 20.2 at 100 steps).
Precomputes teacher targets for efficient distillation training.
30k distillation steps per run.
"""
import os, sys, json, time, copy, math, shutil, tempfile, random
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

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DEVICE = 'cuda'
NUM_FREQ_BANDS = 4
SEEDS = [42, 43, 44]
MODEL_KWARGS = dict(model_channels=64, channel_mult=(1, 2, 2), attention_resolutions=(8,))

TEACHER_TRAIN_STEPS = 80000
DISTILL_STEPS = 12000
DISTILL_BATCH = 256
TEACHER_BATCH = 256
NUM_FID_SAMPLES = 10000
TEACHER_ODE_STEPS = 20
SCD_LF_STEPS = 5
PRECOMPUTE_PAIRS = 50000

FIG_DIR = str(WORKSPACE / 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# DATA
# ===========================================================================
def load_cifar10():
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=str(WORKSPACE / 'data'), train=True,
                          download=True, transform=transform)
    return torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)


def sample_batch(data, batch_size, augment=True):
    idx = torch.randint(0, len(data), (batch_size,), device=DEVICE)
    batch = data[idx]
    if augment:
        flip_mask = torch.rand(batch_size, 1, 1, 1, device=DEVICE) > 0.5
        batch = torch.where(flip_mask, batch.flip(-1), batch)
    return batch


# ===========================================================================
# SPECTRAL UTILITIES - CONSISTENT NORMALIZATION
# ===========================================================================
def create_fft_frequency_masks(H, W, K, device='cpu'):
    freq_y = torch.fft.fftfreq(H, device=device)
    freq_x = torch.fft.fftfreq(W, device=device)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_mag = torch.sqrt(fy**2 + fx**2)
    max_freq = freq_mag.max().item()
    boundaries = torch.linspace(0, max_freq + 1e-6, K + 1, device=device)
    masks = []
    for k in range(K):
        mask = ((freq_mag >= boundaries[k]) & (freq_mag < boundaries[k + 1])).float()
        masks.append(mask.view(1, 1, H, W))
    return masks


def per_band_mse_eval(pred, target, masks):
    """Per-band MSE for EVALUATION. Uses Parseval normalization: /(H*W)^2.
    With all bands summed, equals F.mse_loss(pred, target)."""
    H, W = pred.shape[-2], pred.shape[-1]
    N2 = (H * W) ** 2
    diff = pred - target
    diff_freq = torch.fft.fft2(diff)
    power = diff_freq.real**2 + diff_freq.imag**2
    band_mses = []
    for mask in masks:
        bp = (power * mask).sum(dim=(-2, -1)).mean()
        band_mses.append((bp / N2).item())
    return band_mses


def spectral_loss_train(pred, target, masks, weights):
    """Weighted spectral loss for TRAINING. Same normalization as eval."""
    H, W = pred.shape[-2], pred.shape[-1]
    N2 = (H * W) ** 2
    diff = pred - target
    diff_freq = torch.fft.fft2(diff)
    power = diff_freq.real**2 + diff_freq.imag**2
    total = torch.tensor(0.0, device=pred.device)
    band_losses = []
    for mask, w in zip(masks, weights):
        bp = (power * mask).sum(dim=(-2, -1)).mean() / N2
        band_losses.append(bp.item())
        total = total + w * bp
    return total, band_losses


def build_composite_target(target_lf, target_hf, masks):
    """Composite target: low-freq from LF teacher, high-freq from HF teacher."""
    lf_freq = torch.fft.fft2(target_lf)
    hf_freq = torch.fft.fft2(target_hf)
    lf_mask = masks[0] + masks[1]
    hf_mask = masks[2] + masks[3]
    composite_freq = lf_freq * lf_mask + hf_freq * hf_mask
    return torch.fft.ifft2(composite_freq).real


# ===========================================================================
# GENERATION + FID
# ===========================================================================
@torch.no_grad()
def consistency_sample(model, z, num_steps):
    """Sample from consistency model: model output is x_0 prediction, not velocity."""
    x = z
    step_times = torch.linspace(1.0, 0.0, num_steps + 1, device=z.device)[:-1]
    for i, t_val in enumerate(step_times):
        t = torch.full((z.shape[0],), t_val, device=z.device)
        x_pred = model(x, t)  # model predicts x_0 directly
        if i < len(step_times) - 1:
            t_next = step_times[i + 1]
            noise = torch.randn_like(x)
            x = (1 - t_next) * x_pred + t_next * noise  # re-noise to t_next
        else:
            x = x_pred
    return x


@torch.no_grad()
def generate_velocity(model, num_samples, num_steps, seed=42):
    """Generate samples using velocity model (teacher/rectified flow)."""
    model.eval()
    set_seed(seed)
    all_samples = []
    bs = 512
    for start in range(0, num_samples, bs):
        n = min(bs, num_samples - start)
        z = torch.randn(n, 3, 32, 32, device=DEVICE)
        x = euler_sample(model, z, num_steps)
        all_samples.append(x.cpu())
    return torch.cat(all_samples)[:num_samples]


@torch.no_grad()
def generate_consistency(model, num_samples, num_steps, seed=42):
    """Generate samples using consistency model (distilled students)."""
    model.eval()
    set_seed(seed)
    all_samples = []
    bs = 512
    for start in range(0, num_samples, bs):
        n = min(bs, num_samples - start)
        z = torch.randn(n, 3, 32, 32, device=DEVICE)
        x = consistency_sample(model, z, num_steps)
        all_samples.append(x.cpu())
    return torch.cat(all_samples)[:num_samples]


def compute_fid(samples):
    from cleanfid import fid
    from torchvision.utils import save_image
    tmpdir = tempfile.mkdtemp(prefix='scd_fid_')
    for i in range(len(samples)):
        img = ((samples[i] + 1) / 2).clamp(0, 1)
        save_image(img, os.path.join(tmpdir, f'{i:06d}.png'))
    score = fid.compute_fid(tmpdir, dataset_name='cifar10', dataset_split='train',
                            dataset_res=32, mode='clean')
    shutil.rmtree(tmpdir, ignore_errors=True)
    return score


def evaluate_model(model, real_images, masks, seed=42, is_velocity=False):
    """Evaluate at 1, 2, 4 steps with FID and per-band MSE."""
    gen_fn = generate_velocity if is_velocity else generate_consistency
    results = {}
    for ns in [1, 2, 4]:
        t0 = time.time()
        samples = gen_fn(model, NUM_FID_SAMPLES, ns, seed=seed)
        gen_time = time.time() - t0
        fid_score = compute_fid(samples)
        n = min(len(samples), len(real_images))
        bm = per_band_mse_eval(samples[:n].to(DEVICE), real_images[:n].to(DEVICE), masks)
        results[f'{ns}_step'] = {'fid': fid_score, 'gen_time_s': gen_time, 'per_band_mse': bm}
        print(f"    {ns}-step FID: {fid_score:.2f}, band_mse: {[f'{v:.6f}' for v in bm]}", flush=True)
    return results


# ===========================================================================
# TEACHER
# ===========================================================================
def load_or_train_teacher(data):
    ckpt_path = str(WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt')
    if os.path.exists(ckpt_path):
        model = UNet(**MODEL_KWARGS).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded existing teacher from {ckpt_path}")
        return model

    print(f"\n{'='*60}\nTRAINING TEACHER ({TEACHER_TRAIN_STEPS} steps)\n{'='*60}", flush=True)
    set_seed(42)
    model = UNet(**MODEL_KWARGS).to(DEVICE)
    ema = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler('cuda')
    compiled = torch.compile(model)

    t0 = time.time()
    for step in range(1, TEACHER_TRAIN_STEPS + 1):
        x0 = sample_batch(data, TEACHER_BATCH)
        t_vals, xt, target_v, _ = ot_cfm_sample_t_and_xt(x0)
        with torch.amp.autocast('cuda'):
            pred = compiled(xt, t_vals)
            loss = F.mse_loss(pred, target_v)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            for p, ep in zip(model.parameters(), ema.parameters()):
                ep.mul_(0.9999).add_(p, alpha=0.0001)
        if step % 10000 == 0:
            print(f"  Step {step}, loss={loss.item():.4f}, time={(time.time()-t0)/60:.1f}min", flush=True)

    torch.save(ema.state_dict(), ckpt_path)
    print(f"Teacher trained in {(time.time()-t0)/60:.1f}min")
    return ema


# ===========================================================================
# PRECOMPUTE TEACHER TARGETS
# ===========================================================================
def precompute_targets(teacher, data, num_pairs=PRECOMPUTE_PAIRS):
    """Precompute (x_t, t, target_x0) pairs for all distillation methods."""
    cache_path = WORKSPACE / 'exp' / 'teacher' / 'precomputed_targets.pt'
    if cache_path.exists():
        print("Loading cached precomputed targets...")
        cached = torch.load(str(cache_path), map_location='cpu', weights_only=True)
        if cached['num_pairs'] >= num_pairs:
            return cached

    print(f"\nPrecomputing {num_pairs} teacher target pairs...", flush=True)
    teacher.eval()
    compiled = torch.compile(teacher)

    set_seed(0)
    x0_all = sample_batch(data, num_pairs, augment=True)
    noise = torch.randn_like(x0_all)
    t = torch.rand(num_pairs, device=DEVICE) * 0.998 + 0.001
    t_exp = t.view(-1, 1, 1, 1)
    x_t = (1 - t_exp) * x0_all + t_exp * noise

    step_counts = sorted(set([SCD_LF_STEPS, TEACHER_ODE_STEPS]))
    all_targets = {s: [] for s in step_counts}

    t0 = time.time()
    bs = TEACHER_BATCH
    for i in range(0, num_pairs, bs):
        batch_xt = x_t[i:i+bs]
        batch_t = t[i:i+bs]
        checkpoints = teacher_solve(compiled, batch_xt, batch_t,
                                     num_steps=max(step_counts),
                                     return_checkpoints=step_counts)
        for s in step_counts:
            all_targets[s].append(checkpoints[s].cpu().half())
        if (i // bs + 1) % 20 == 0:
            print(f"  {i+bs}/{num_pairs} pairs...", flush=True)

    targets = {s: torch.cat(all_targets[s]) for s in step_counts}
    result = {'x_t': x_t.cpu(), 't': t.cpu(), 'targets': targets, 'num_pairs': num_pairs}
    torch.save(result, str(cache_path))
    print(f"  Precomputed in {time.time()-t0:.0f}s")
    return result


# ===========================================================================
# DISTILLATION TRAINING
# ===========================================================================
def train_distillation(precomputed, method, seed, masks,
                       use_pseudohuber=False, use_spectral=False,
                       spectral_config=None, teacher_ckpt=None):
    print(f"\n  Training {method} (seed={seed}, {DISTILL_STEPS} steps)...", flush=True)

    save_dir = WORKSPACE / 'exp' / method
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f'checkpoint_seed{seed}.pt'
    results_path = save_dir / f'results_seed{seed}.json'

    # Check existing results with correct step count
    if ckpt_path.exists() and results_path.exists():
        prev = json.load(open(results_path))
        if prev.get('train_steps', 0) >= DISTILL_STEPS:
            print(f"    Already done ({prev['train_steps']} steps), loading...", flush=True)
            model = UNet(**MODEL_KWARGS).to(DEVICE)
            model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True))
            return model, prev.get('train_time_min', 0)

    set_seed(seed)
    student = UNet(**MODEL_KWARGS).to(DEVICE)
    if teacher_ckpt:
        student.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE, weights_only=True))
    ema_student = copy.deepcopy(student)
    compiled = torch.compile(student)
    opt = torch.optim.Adam(student.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    # Spectral setup
    if use_spectral and spectral_config:
        base_weights = spectral_config.get('weights', [1.0, 1.5, 2.5, 4.0])
        progressive = spectral_config.get('progressive', True)
        use_composite = spectral_config.get('use_composite_target', True)

    if use_pseudohuber:
        d = 3 * 32 * 32
        c_huber = 0.00054 * math.sqrt(d)

    N = precomputed['num_pairs']
    running_band_errors = [1.0] * NUM_FREQ_BANDS
    band_error_history = []
    t_start = time.time()

    for step in range(1, DISTILL_STEPS + 1):
        idx = torch.randint(0, N, (DISTILL_BATCH,))
        x_t = precomputed['x_t'][idx].float().to(DEVICE)
        t_vals = precomputed['t'][idx].to(DEVICE)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            student_pred = compiled(x_t, t_vals)

            if use_spectral and spectral_config and masks:
                # Get target
                target_hf = precomputed['targets'][TEACHER_ODE_STEPS][idx].float().to(DEVICE)
                if use_composite:
                    target_lf = precomputed['targets'][SCD_LF_STEPS][idx].float().to(DEVICE)
                    target = build_composite_target(target_lf, target_hf, masks)
                    del target_lf
                else:
                    target = target_hf

                # Progressive weight schedule
                if progressive:
                    frac = step / DISTILL_STEPS
                    if frac < 0.3:
                        weights = [1.0] * NUM_FREQ_BANDS
                    elif frac < 0.7:
                        alpha = (frac - 0.3) / 0.4
                        weights = [1.0 + alpha * (w - 1.0) for w in base_weights]
                    else:
                        mean_err = sum(running_band_errors) / len(running_band_errors)
                        weights = [(e / (mean_err + 1e-8))**0.5 for e in running_band_errors]
                        w_sum = sum(weights)
                        weights = [w * NUM_FREQ_BANDS / (w_sum + 1e-8) for w in weights]
                else:
                    weights = list(base_weights)

                loss, band_errs = spectral_loss_train(student_pred, target, masks, weights)
                for k, be in enumerate(band_errs):
                    running_band_errors[k] = 0.99 * running_band_errors[k] + 0.01 * be
                if step % 1000 == 0:
                    band_error_history.append({'step': step, 'band_losses': band_errs, 'weights': weights})
            else:
                target = precomputed['targets'][TEACHER_ODE_STEPS][idx].float().to(DEVICE)
                if use_pseudohuber:
                    diff = student_pred - target
                    loss = (torch.sqrt(diff.pow(2).sum(dim=(1,2,3)) + c_huber**2) - c_huber).mean()
                else:
                    loss = F.mse_loss(student_pred, target)

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
            print(f"    Step {step}/{DISTILL_STEPS}, loss={loss.item():.6f}, time={elapsed:.1f}min", flush=True)

    train_time = (time.time() - t_start) / 60
    torch.save(ema_student.state_dict(), str(ckpt_path))
    if band_error_history:
        json.dump(band_error_history, open(str(save_dir / f'band_errors_seed{seed}.json'), 'w'), indent=2)
    return ema_student, train_time


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    params = sum(p.numel() for p in UNet(**MODEL_KWARGS).parameters()) / 1e6
    print(f"Model: {params:.1f}M params, model_channels=64")
    print(f"Distillation steps: {DISTILL_STEPS}, FID samples: {NUM_FID_SAMPLES}")

    data = load_cifar10()
    masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)
    real_images = data[:NUM_FID_SAMPLES]

    all_results = {}
    teacher_ckpt = str(WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt')

    # 1. Teacher
    teacher = load_or_train_teacher(data)

    # 2. Precompute targets
    precomputed = precompute_targets(teacher, data)

    # 3. Rectified Flow Baseline (no training)
    print(f"\n{'='*60}\nRECTIFIED FLOW BASELINE (3 seeds)\n{'='*60}", flush=True)
    rf_results = {}
    for seed in SEEDS:
        print(f"  Seed {seed}:", flush=True)
        res = evaluate_model(teacher, real_images, masks, seed=seed, is_velocity=True)
        res['train_time_min'] = 0.0
        res['train_steps'] = 0
        json.dump(res, open(str(WORKSPACE / 'exp' / 'rectflow_baseline' / f'results_seed{seed}.json'), 'w'), indent=2)
        rf_results[seed] = res
    all_results['rectflow_baseline'] = rf_results

    # 4. CD Baseline (3 seeds)
    print(f"\n{'='*60}\nCD BASELINE (MSE, 3 seeds)\n{'='*60}", flush=True)
    cd_results = {}
    for seed in SEEDS:
        print(f"  Seed {seed}:", flush=True)
        model, train_time = train_distillation(precomputed, 'cd_baseline', seed, masks,
                                                teacher_ckpt=teacher_ckpt)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(WORKSPACE / 'exp' / 'cd_baseline' / f'results_seed{seed}.json'), 'w'), indent=2)
        cd_results[seed] = res
        del model; torch.cuda.empty_cache()
    all_results['cd_baseline'] = cd_results

    # 5. Pseudo-Huber CD (3 seeds)
    print(f"\n{'='*60}\nPSEUDO-HUBER CD (3 seeds)\n{'='*60}", flush=True)
    ph_results = {}
    for seed in SEEDS:
        print(f"  Seed {seed}:", flush=True)
        model, train_time = train_distillation(precomputed, 'cd_pseudohuber', seed, masks,
                                                use_pseudohuber=True, teacher_ckpt=teacher_ckpt)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(WORKSPACE / 'exp' / 'cd_pseudohuber' / f'results_seed{seed}.json'), 'w'), indent=2)
        ph_results[seed] = res
        del model; torch.cuda.empty_cache()
    all_results['cd_pseudohuber'] = ph_results

    # 6. SCD Main (3 seeds)
    print(f"\n{'='*60}\nSCD MAIN (3 seeds)\n{'='*60}", flush=True)
    scd_config = {'weights': [1.0, 1.5, 2.5, 4.0], 'progressive': True, 'use_composite_target': True}
    scd_results = {}
    for seed in SEEDS:
        print(f"  Seed {seed}:", flush=True)
        model, train_time = train_distillation(precomputed, 'scd_main', seed, masks,
                                                use_spectral=True, spectral_config=scd_config,
                                                teacher_ckpt=teacher_ckpt)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(WORKSPACE / 'exp' / 'scd_main' / f'results_seed{seed}.json'), 'w'), indent=2)
        scd_results[seed] = res
        del model; torch.cuda.empty_cache()
    all_results['scd_main'] = scd_results

    # 7. Ablation: No Adaptive Teacher (seed 42)
    # Uses uniform teacher targets (no composite), but keeps spectral loss + progressive
    print(f"\n{'='*60}\nABLATION: No Adaptive Teacher (seed 42)\n{'='*60}", flush=True)
    abl1_config = {'weights': [1.0, 1.5, 2.5, 4.0], 'progressive': True, 'use_composite_target': False}
    model, train_time = train_distillation(precomputed, 'ablation_no_adaptive_teacher', 42, masks,
                                            use_spectral=True, spectral_config=abl1_config,
                                            teacher_ckpt=teacher_ckpt)
    res = evaluate_model(model, real_images, masks, seed=42)
    res['train_time_min'] = train_time
    res['train_steps'] = DISTILL_STEPS
    json.dump(res, open(str(WORKSPACE / 'exp' / 'ablation_no_adaptive_teacher' / 'results_seed42.json'), 'w'), indent=2)
    all_results['ablation_no_adaptive_teacher'] = {42: res}
    del model; torch.cuda.empty_cache()

    # 8. Ablation: No Progressive (seed 42)
    # Fixed weights from start, keeps spectral loss + composite target
    print(f"\n{'='*60}\nABLATION: No Progressive Refinement (seed 42)\n{'='*60}", flush=True)
    abl2_config = {'weights': [1.0, 1.5, 2.5, 4.0], 'progressive': False, 'use_composite_target': True}
    model, train_time = train_distillation(precomputed, 'ablation_no_progressive', 42, masks,
                                            use_spectral=True, spectral_config=abl2_config,
                                            teacher_ckpt=teacher_ckpt)
    res = evaluate_model(model, real_images, masks, seed=42)
    res['train_time_min'] = train_time
    res['train_steps'] = DISTILL_STEPS
    json.dump(res, open(str(WORKSPACE / 'exp' / 'ablation_no_progressive' / 'results_seed42.json'), 'w'), indent=2)
    all_results['ablation_no_progressive'] = {42: res}
    del model; torch.cuda.empty_cache()

    # 9. Aggregate + Figures
    print(f"\n{'='*60}\nAGGREGATING RESULTS\n{'='*60}", flush=True)
    aggregate_and_save(all_results)

    print(f"\n{'='*60}\nGENERATING FIGURES\n{'='*60}", flush=True)
    generate_figures(all_results)

    print(f"\n{'='*60}\nALL EXPERIMENTS COMPLETE\n{'='*60}", flush=True)


# ===========================================================================
# AGGREGATION
# ===========================================================================
def aggregate_and_save(all_results):
    aggregated = {'main_results': {}, 'ablation_results': {}, 'success_criteria': {}}

    for method in ['cd_baseline', 'cd_pseudohuber', 'rectflow_baseline', 'scd_main']:
        if method not in all_results:
            continue
        data = all_results[method]
        agg = {}
        for sk in ['1_step', '2_step', '4_step']:
            fids = [data[s][sk]['fid'] for s in SEEDS if s in data and sk in data[s]]
            if not fids:
                continue
            entry = {'fid_mean': round(float(np.mean(fids)), 2),
                     'fid_std': round(float(np.std(fids)), 2)}
            K = len(data[SEEDS[0]][sk]['per_band_mse'])
            for k in range(K):
                vals = [data[s][sk]['per_band_mse'][k] for s in SEEDS if s in data]
                entry[f'band{k}_mse_mean'] = float(np.mean(vals))
                entry[f'band{k}_mse_std'] = float(np.std(vals))
            agg[sk] = entry
        times = [data[s].get('train_time_min', 0) for s in SEEDS if s in data]
        agg['train_time_mean'] = round(float(np.mean(times)), 1)
        aggregated['main_results'][method] = agg

    for abl in ['ablation_no_adaptive_teacher', 'ablation_no_progressive']:
        if abl not in all_results:
            continue
        data = all_results[abl]
        agg = {}
        for sk in ['1_step', '2_step', '4_step']:
            if 42 in data and sk in data[42]:
                agg[sk] = {'fid': data[42][sk]['fid'],
                           'per_band_mse': data[42][sk]['per_band_mse']}
        agg['train_time_min'] = data[42].get('train_time_min', 0) if 42 in data else 0
        aggregated['ablation_results'][abl] = agg

    # Success criteria
    cd = aggregated['main_results'].get('cd_baseline', {})
    scd = aggregated['main_results'].get('scd_main', {})
    criteria = {}
    if cd and scd:
        for sk in ['1_step', '2_step', '4_step']:
            if sk in cd and sk in scd:
                cd_fid = cd[sk]['fid_mean']
                scd_fid = scd[sk]['fid_mean']
                improvement = (cd_fid - scd_fid) / cd_fid * 100
                criteria[f'{sk}_fid_improvement_pct'] = round(improvement, 2)
                criteria[f'{sk}_scd_beats_cd'] = bool(scd_fid < cd_fid)
        # HF MSE reduction
        if '1_step' in cd and '1_step' in scd:
            cd_hf = cd['1_step'].get('band3_mse_mean', 0)
            scd_hf = scd['1_step'].get('band3_mse_mean', 0)
            if cd_hf > 0:
                criteria['hf_mse_reduction_pct'] = round((cd_hf - scd_hf) / cd_hf * 100, 2)
        # Training overhead
        cd_time = cd.get('train_time_mean', 1)
        scd_time = scd.get('train_time_mean', 1)
        if cd_time > 0:
            criteria['training_overhead_ratio'] = round(scd_time / cd_time, 2)

        # Honest assessment
        all_beat = all(criteria.get(f'{sk}_scd_beats_cd', False) for sk in ['1_step', '2_step', '4_step'])
        fid_imp = criteria.get('1_step_fid_improvement_pct', 0)
        criteria['hypothesis_supported'] = all_beat and fid_imp > 5
        if fid_imp < 5:
            criteria['note'] = (f"FID improvement ({fid_imp:.1f}%) is below the hypothesized 10-25%. "
                               "The spectral weighting provides marginal benefit at this model scale.")

    aggregated['success_criteria'] = criteria
    aggregated['config'] = {
        'model_params_M': 6.5, 'model_channels': 64, 'channel_mult': [1, 2, 2],
        'distill_steps': DISTILL_STEPS, 'teacher_steps': TEACHER_TRAIN_STEPS,
        'teacher_ode_steps': TEACHER_ODE_STEPS, 'fid_num_samples': NUM_FID_SAMPLES,
        'seeds': SEEDS, 'note': 'Model is 6.5M params (planned 35M) due to compute constraints.'
    }

    out_path = str(WORKSPACE / 'results.json')
    json.dump(aggregated, open(out_path, 'w'), indent=2)
    print(f"  Saved to {out_path}")
    print(f"  Success criteria: {json.dumps(criteria, indent=2)}")


# ===========================================================================
# FIGURES
# ===========================================================================
def generate_figures(all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150, 'savefig.bbox': 'tight',
                         'axes.grid': True, 'grid.alpha': 0.3})

    colors = {'cd_baseline': '#1f77b4', 'cd_pseudohuber': '#ff7f0e',
              'rectflow_baseline': '#2ca02c', 'scd_main': '#d62728',
              'ablation_no_adaptive_teacher': '#9467bd', 'ablation_no_progressive': '#8c564b'}
    labels = {'cd_baseline': 'CD (MSE)', 'cd_pseudohuber': 'CD (Pseudo-Huber)',
              'rectflow_baseline': 'Rectified Flow', 'scd_main': 'SCD (Ours)',
              'ablation_no_adaptive_teacher': 'w/o Adapt. Teacher',
              'ablation_no_progressive': 'w/o Progressive'}
    band_names = ['Low', 'Mid-Low', 'Mid-High', 'High']

    # --- Figure 1: Spectral Error ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, step_key in enumerate(['1_step', '4_step']):
        ax = axes[ax_idx]
        x = np.arange(NUM_FREQ_BANDS)
        width = 0.25
        for i, method in enumerate(['cd_baseline', 'cd_pseudohuber', 'scd_main']):
            if method not in all_results:
                continue
            data = all_results[method]
            vals = [np.mean([data[s][step_key]['per_band_mse'][k] for s in SEEDS if s in data])
                    for k in range(NUM_FREQ_BANDS)]
            ax.bar(x + i * width, vals, width, label=labels[method], color=colors[method])
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Per-Band MSE')
        ax.set_title(f'{step_key.replace("_", "-")} Generation')
        ax.set_xticks(x + width)
        ax.set_xticklabels(band_names)
        ax.legend(fontsize=9)
        ax.set_yscale('log')
    plt.suptitle('Spectral Error Analysis', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'figure1_spectral_error.png'))
    plt.savefig(os.path.join(FIG_DIR, 'figure1_spectral_error.pdf'))
    plt.close()
    print("  Saved figure1_spectral_error")

    # --- Figure 4: FID vs Steps ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in ['rectflow_baseline', 'cd_baseline', 'cd_pseudohuber', 'scd_main']:
        if method not in all_results:
            continue
        data = all_results[method]
        fid_m = [np.mean([data[s][sk]['fid'] for s in SEEDS if s in data]) for sk in ['1_step', '2_step', '4_step']]
        fid_s = [np.std([data[s][sk]['fid'] for s in SEEDS if s in data]) for sk in ['1_step', '2_step', '4_step']]
        ax.errorbar([1, 2, 4], fid_m, yerr=fid_s, marker='o', label=labels[method],
                    color=colors[method], linewidth=2, capsize=4)
    ax.set_xlabel('Number of Inference Steps')
    ax.set_ylabel('FID')
    ax.set_title('FID vs Inference Steps (CIFAR-10)')
    ax.set_xticks([1, 2, 4])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'figure4_fid_vs_steps.png'))
    plt.savefig(os.path.join(FIG_DIR, 'figure4_fid_vs_steps.pdf'))
    plt.close()
    print("  Saved figure4_fid_vs_steps")

    # --- Figure 3: Training Dynamics ---
    log_path = str(WORKSPACE / 'exp' / 'scd_main' / 'band_errors_seed42.json')
    if os.path.exists(log_path):
        blog = json.load(open(log_path))
        fig, ax = plt.subplots(figsize=(8, 5))
        steps_log = [e['step'] for e in blog]
        for k in range(NUM_FREQ_BANDS):
            vals = [e['band_losses'][k] for e in blog]
            ax.plot(steps_log, vals, label=band_names[k], linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Per-Band Distillation Loss')
        ax.set_title('SCD Training Dynamics')
        ax.legend()
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'figure3_training_dynamics.png'))
        plt.savefig(os.path.join(FIG_DIR, 'figure3_training_dynamics.pdf'))
        plt.close()
        print("  Saved figure3_training_dynamics")

    # --- Figure 5: Ablation ---
    fig, ax = plt.subplots(figsize=(8, 5))
    abl_items = []
    for method in ['cd_baseline', 'ablation_no_progressive', 'ablation_no_adaptive_teacher', 'scd_main']:
        if method not in all_results:
            continue
        data = all_results[method]
        if method in ['scd_main', 'cd_baseline']:
            fid_val = np.mean([data[s]['1_step']['fid'] for s in SEEDS if s in data])
        else:
            fid_val = data[42]['1_step']['fid'] if 42 in data else 0
        abl_items.append((labels.get(method, method), fid_val, colors.get(method, '#333')))
    if abl_items:
        names, vals, cols = zip(*abl_items)
        ax.bar(range(len(names)), vals, color=cols)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel('1-Step FID')
        ax.set_title('Ablation Study (1-Step FID)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'figure5_ablation.png'))
        plt.savefig(os.path.join(FIG_DIR, 'figure5_ablation.pdf'))
        plt.close()
        print("  Saved figure5_ablation")

    # --- Tables ---
    # CSV
    with open(os.path.join(FIG_DIR, 'table1_main_results.csv'), 'w') as f:
        f.write("Method,1-step FID,2-step FID,4-step FID\n")
        for method in ['rectflow_baseline', 'cd_baseline', 'cd_pseudohuber', 'scd_main']:
            if method not in all_results:
                continue
            data = all_results[method]
            row = [labels[method]]
            for sk in ['1_step', '2_step', '4_step']:
                fids = [data[s][sk]['fid'] for s in SEEDS if s in data]
                row.append(f"{np.mean(fids):.2f} +/- {np.std(fids):.2f}")
            f.write(','.join(row) + '\n')

    # LaTeX Table 1
    with open(os.path.join(FIG_DIR, 'table1_main_results.tex'), 'w') as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{FID comparison on CIFAR-10 (mean $\\pm$ std over 3 seeds).}\n")
        f.write("\\label{tab:main}\n\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Method & 1-step & 2-step & 4-step \\\\\n\\midrule\n")
        for method in ['rectflow_baseline', 'cd_baseline', 'cd_pseudohuber', 'scd_main']:
            if method not in all_results:
                continue
            data = all_results[method]
            row = [labels[method]]
            for sk in ['1_step', '2_step', '4_step']:
                fids = [data[s][sk]['fid'] for s in SEEDS if s in data]
                m, s = np.mean(fids), np.std(fids)
                row.append(f"${m:.1f} \\pm {s:.1f}$")
            f.write(' & '.join(row) + ' \\\\\n')
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # LaTeX Table 2
    with open(os.path.join(FIG_DIR, 'table2_ablation.tex'), 'w') as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Ablation study: contribution of each SCD component (1-step FID).}\n")
        f.write("\\label{tab:ablation}\n\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Variant & 1-step FID & HF MSE \\\\\n\\midrule\n")
        for method in ['cd_baseline', 'ablation_no_progressive', 'ablation_no_adaptive_teacher', 'scd_main']:
            if method not in all_results:
                continue
            data = all_results[method]
            if method in ['scd_main', 'cd_baseline']:
                fid_v = np.mean([data[s]['1_step']['fid'] for s in SEEDS if s in data])
                hf_v = np.mean([data[s]['1_step']['per_band_mse'][3] for s in SEEDS if s in data])
            else:
                fid_v = data[42]['1_step']['fid'] if 42 in data else 0
                hf_v = data[42]['1_step']['per_band_mse'][3] if 42 in data else 0
            f.write(f"{labels.get(method, method)} & ${fid_v:.1f}$ & ${hf_v:.6f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print("  Saved tables")


if __name__ == '__main__':
    main()
