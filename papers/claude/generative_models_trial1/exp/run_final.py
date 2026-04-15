#!/usr/bin/env python3
"""
SCD Experiments - Final version.

Key fixes from previous runs:
1. Proper frequency-adaptive teacher supervision (different teacher targets per band)
2. Longer training (30k steps, up from 12k)
3. 50k FID samples for evaluation
4. Honest reporting of results (negative if negative)
5. Complete ablation studies
6. Training loss logging throughout

Architecture: 6.5M param U-Net (model_channels=64, channel_mult=(1,2,2))
Teacher: pretrained flow matching model (FID 20.2 at 100 steps)
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

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DEVICE = 'cuda'
NUM_FREQ_BANDS = 4
SEEDS = [42, 43, 44]
MODEL_KWARGS = dict(model_channels=64, channel_mult=(1, 2, 2), attention_resolutions=(8,))

DISTILL_STEPS = 30000
DISTILL_BATCH = 256
NUM_FID_SAMPLES = 50000
# Teacher ODE steps per frequency band (key innovation: adaptive teacher)
TEACHER_STEPS_PER_BAND = [10, 20, 50, 100]  # low -> high frequency
TEACHER_STEPS_UNIFORM = 20  # for baseline and non-adaptive ablation
SCD_WEIGHTS = [1.0, 1.5, 2.5, 4.0]

EXP_DIR = WORKSPACE / 'exp'
FIG_DIR = WORKSPACE / 'figures'
os.makedirs(str(FIG_DIR), exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(str(WORKSPACE / 'experiment.log'), mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# SPECTRAL UTILITIES (FFT-based)
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
    """Per-band MSE using Parseval's theorem. Normalized by H*W."""
    H, W = pred.shape[-2], pred.shape[-1]
    HW = H * W
    diff = pred - target
    diff_freq = torch.fft.fft2(diff)
    power = diff_freq.real**2 + diff_freq.imag**2
    return [float((power * m).sum(dim=(-2, -1)).mean() / HW) for m in masks]


def spectral_loss_with_adaptive_targets(student_pred, band_targets, masks, weights):
    """SCD loss with per-band targets from different teacher step counts.

    Args:
        student_pred: student output (B, C, H, W)
        band_targets: dict mapping band_idx -> target tensor from that band's teacher
        masks: list of K frequency band masks
        weights: list of K scalar weights
    Returns:
        total_loss, list of per-band losses
    """
    H, W = student_pred.shape[-2], student_pred.shape[-1]
    HW = H * W
    total = torch.tensor(0.0, device=student_pred.device)
    band_losses = []

    pred_freq = torch.fft.fft2(student_pred.float())

    for k, (mask, w) in enumerate(zip(masks, weights)):
        target_k = band_targets[k]
        target_freq = torch.fft.fft2(target_k.float())

        diff_freq = pred_freq - target_freq
        power = diff_freq.real**2 + diff_freq.imag**2
        band_loss = (power * mask).sum(dim=(-2, -1)).mean() / HW
        band_losses.append(band_loss.item())
        total = total + w * band_loss

    return total, band_losses


def spectral_loss_uniform_target(student_pred, target, masks, weights):
    """SCD loss with same target for all bands (uniform teacher steps)."""
    H, W = student_pred.shape[-2], student_pred.shape[-1]
    HW = H * W
    diff = student_pred.float() - target.float()
    diff_freq = torch.fft.fft2(diff)
    power = diff_freq.real**2 + diff_freq.imag**2
    total = torch.tensor(0.0, device=student_pred.device)
    band_losses = []
    for mask, w in zip(masks, weights):
        bl = (power * mask).sum(dim=(-2, -1)).mean() / HW
        band_losses.append(bl.item())
        total = total + w * bl
    return total, band_losses


# ===========================================================================
# GENERATION + FID
# ===========================================================================
@torch.no_grad()
def consistency_sample(model, z, num_steps):
    x = z
    step_times = torch.linspace(1.0, 0.0, num_steps + 1, device=z.device)[:-1]
    for i, t_val in enumerate(step_times):
        t = torch.full((z.shape[0],), t_val, device=z.device)
        x_pred = model(x, t)
        if i < len(step_times) - 1:
            t_next = step_times[i + 1]
            noise = torch.randn_like(x)
            x = (1 - t_next) * x_pred + t_next * noise
        else:
            x = x_pred
    return x


@torch.no_grad()
def generate_samples(model, num_samples, num_steps, seed=42, is_velocity=False):
    model.eval()
    set_seed(seed)
    all_samples = []
    bs = 512
    for start in range(0, num_samples, bs):
        n = min(bs, num_samples - start)
        z = torch.randn(n, 3, 32, 32, device=DEVICE)
        if is_velocity:
            x = euler_sample(model, z, num_steps)
        else:
            x = consistency_sample(model, z, num_steps)
        all_samples.append(x.cpu())
    return torch.cat(all_samples)[:num_samples]


def compute_fid(samples):
    from cleanfid import fid
    from torchvision.utils import save_image
    tmpdir = tempfile.mkdtemp(prefix='scd_fid_')
    try:
        for i in range(len(samples)):
            img = ((samples[i] + 1) / 2).clamp(0, 1)
            save_image(img, os.path.join(tmpdir, f'{i:06d}.png'))
        score = fid.compute_fid(tmpdir, dataset_name='cifar10', dataset_split='train',
                                dataset_res=32, mode='clean')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return score


def evaluate_model(model, real_images, masks, seed=42, is_velocity=False):
    results = {}
    for ns in [1, 2, 4]:
        t0 = time.time()
        samples = generate_samples(model, NUM_FID_SAMPLES, ns, seed=seed, is_velocity=is_velocity)
        gen_time = time.time() - t0
        fid_score = compute_fid(samples)
        n = min(len(samples), len(real_images))
        bm = per_band_mse_eval(samples[:n].to(DEVICE), real_images[:n].to(DEVICE), masks)
        results[f'{ns}_step'] = {'fid': fid_score, 'gen_time_s': gen_time, 'per_band_mse': bm}
        log.info(f"    {ns}-step FID: {fid_score:.2f}, band_mse: {[f'{v:.6f}' for v in bm]}")
    return results


# ===========================================================================
# PRECOMPUTE TEACHER TARGETS
# ===========================================================================
def precompute_targets(teacher, dataset, step_counts, cache_path):
    """Precompute teacher ODE targets at multiple step counts."""
    if cache_path.exists():
        log.info(f"Loading cached targets from {cache_path}")
        return torch.load(str(cache_path), map_location='cpu', weights_only=True)

    log.info(f"Precomputing targets at step counts {step_counts}...")
    teacher.eval()
    N = len(dataset)

    # Sample x_t and t for all data points
    set_seed(0)
    all_x0 = torch.stack([dataset[i][0] for i in range(N)])
    noise = torch.randn_like(all_x0)
    t_vals = torch.rand(N)
    t_expand = t_vals.view(N, 1, 1, 1)
    x_t = (1 - t_expand) * all_x0 + t_expand * noise

    targets = {}
    bs = 256
    for num_steps in step_counts:
        log.info(f"  Computing {num_steps}-step targets...")
        step_targets = []
        t0 = time.time()
        for start in range(0, N, bs):
            end = min(start + bs, N)
            xt_batch = x_t[start:end].to(DEVICE)
            t_batch = t_vals[start:end].to(DEVICE)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                target = teacher_solve(teacher, xt_batch, t_batch, num_steps)
            step_targets.append(target.cpu().half())
        targets[num_steps] = torch.cat(step_targets)
        elapsed = time.time() - t0
        log.info(f"    Done in {elapsed:.1f}s")

    result = {'x_t': x_t.half(), 't': t_vals, 'targets': targets, 'num_pairs': N}
    torch.save(result, str(cache_path))
    log.info(f"  Saved to {cache_path}")
    return result


# ===========================================================================
# DISTILLATION TRAINING
# ===========================================================================
def train_distillation(precomputed, method, seed, masks,
                       use_pseudohuber=False,
                       spectral_weights=None,
                       adaptive_teacher=False,
                       progressive_schedule=None):
    """Train a consistency distillation model.

    Args:
        spectral_weights: per-band weights for spectral loss (None = standard MSE)
        adaptive_teacher: if True, use different teacher targets per band
        progressive_schedule: dict with 'uniform_frac', 'ramp_frac' for progressive weights
    """
    save_dir = EXP_DIR / method
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f'checkpoint_seed{seed}.pt'
    results_path = save_dir / f'results_seed{seed}.json'

    log.info(f"\n  Training {method} (seed={seed}, {DISTILL_STEPS} steps)...")
    set_seed(seed)
    student = UNet(**MODEL_KWARGS).to(DEVICE)
    ema_student = copy.deepcopy(student)
    compiled = torch.compile(student)
    opt = torch.optim.Adam(student.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    if use_pseudohuber:
        d = 3 * 32 * 32
        c_huber = 0.00054 * math.sqrt(d)

    N = precomputed['num_pairs']
    train_log = []
    t_start = time.time()

    for step in range(1, DISTILL_STEPS + 1):
        idx = torch.randint(0, N, (DISTILL_BATCH,))
        x_t = precomputed['x_t'][idx].float().to(DEVICE)
        t_vals = precomputed['t'][idx].to(DEVICE)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            student_pred = compiled(x_t, t_vals)

        if spectral_weights is not None:
            # Determine current weights
            if progressive_schedule:
                frac = step / DISTILL_STEPS
                uf = progressive_schedule.get('uniform_frac', 0.3)
                rf = progressive_schedule.get('ramp_frac', 0.4)
                if frac < uf:
                    weights = [1.0] * NUM_FREQ_BANDS
                elif frac < uf + rf:
                    alpha = (frac - uf) / rf
                    weights = [1.0 + alpha * (w - 1.0) for w in spectral_weights]
                else:
                    weights = list(spectral_weights)
            else:
                weights = list(spectral_weights)

            if adaptive_teacher:
                # Key innovation: different teacher targets per frequency band
                band_targets = {}
                for k in range(NUM_FREQ_BANDS):
                    teacher_steps = TEACHER_STEPS_PER_BAND[k]
                    band_targets[k] = precomputed['targets'][teacher_steps][idx].float().to(DEVICE)
                loss, band_errs = spectral_loss_with_adaptive_targets(
                    student_pred.float(), band_targets, masks, weights)
            else:
                target = precomputed['targets'][TEACHER_STEPS_UNIFORM][idx].float().to(DEVICE)
                loss, band_errs = spectral_loss_uniform_target(
                    student_pred.float(), target, masks, weights)

            if step % 500 == 0:
                train_log.append({
                    'step': step, 'loss': loss.item(),
                    'band_losses': band_errs, 'weights': weights
                })
        else:
            target = precomputed['targets'][TEACHER_STEPS_UNIFORM][idx].float().to(DEVICE)
            if use_pseudohuber:
                diff = student_pred.float() - target
                loss = (torch.sqrt(diff.pow(2).sum(dim=(1, 2, 3)) + c_huber**2) - c_huber).mean()
            else:
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
    torch.save(ema_student.state_dict(), str(ckpt_path))

    # Save training log
    log_path = save_dir / f'train_log_seed{seed}.json'
    json.dump(train_log, open(str(log_path), 'w'), indent=2)
    log.info(f"  Training done in {train_time:.1f} min. Saved to {ckpt_path}")

    return ema_student, train_time


# ===========================================================================
# MAIN EXPERIMENT RUNNER
# ===========================================================================
def main():
    log.info(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    params_m = sum(p.numel() for p in UNet(**MODEL_KWARGS).parameters()) / 1e6
    log.info(f"Model: {params_m:.1f}M params (model_channels=64, channel_mult=(1,2,2))")
    log.info(f"Config: {DISTILL_STEPS} steps, batch {DISTILL_BATCH}, {NUM_FID_SAMPLES} FID samples")
    log.info(f"SCD weights: {SCD_WEIGHTS}")
    log.info(f"Adaptive teacher steps: {TEACHER_STEPS_PER_BAND}")

    # Load CIFAR-10
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=str(WORKSPACE / 'data'), train=True,
                          download=True, transform=transform)

    # Load teacher
    teacher = UNet(**MODEL_KWARGS).to(DEVICE)
    teacher.load_state_dict(torch.load(
        str(EXP_DIR / 'teacher' / 'checkpoint_best.pt'),
        map_location=DEVICE, weights_only=True))
    teacher.eval()
    log.info("Loaded teacher model")

    # Precompute targets at all needed step counts
    needed_steps = sorted(set(TEACHER_STEPS_PER_BAND + [TEACHER_STEPS_UNIFORM]))
    cache_path = EXP_DIR / 'teacher' / 'precomputed_targets_full.pt'
    precomputed = precompute_targets(teacher, ds, needed_steps, cache_path)

    del teacher
    torch.cuda.empty_cache()

    # Load real images for per-band MSE evaluation
    real_images = torch.stack([ds[i][0] for i in range(min(NUM_FID_SAMPLES, len(ds)))]).to(DEVICE)
    masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)

    all_results = {}

    # -----------------------------------------------------------------------
    # 1. Rectified Flow baseline (evaluation only, no training)
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nRECTIFIED FLOW BASELINE (eval only)\n{'='*60}")
    rf_teacher = UNet(**MODEL_KWARGS).to(DEVICE)
    rf_teacher.load_state_dict(torch.load(
        str(EXP_DIR / 'teacher' / 'checkpoint_best.pt'),
        map_location=DEVICE, weights_only=True))
    rf_results = {}
    for seed in SEEDS:
        log.info(f"  RF eval seed={seed}")
        res = evaluate_model(rf_teacher, real_images, masks, seed=seed, is_velocity=True)
        res['train_time_min'] = 0
        res['train_steps'] = 0
        rf_results[seed] = res
        rdir = EXP_DIR / 'rectflow_baseline_v2'
        rdir.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(str(rdir / f'results_seed{seed}.json'), 'w'), indent=2)
    all_results['rectflow_baseline'] = rf_results
    del rf_teacher
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 2. CD Baseline - Standard MSE (3 seeds)
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nCD BASELINE (MSE, 3 seeds)\n{'='*60}")
    cd_results = {}
    for seed in SEEDS:
        model, train_time = train_distillation(
            precomputed, 'cd_baseline_v2', seed, masks)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(EXP_DIR / 'cd_baseline_v2' / f'results_seed{seed}.json'), 'w'), indent=2)
        cd_results[seed] = res
        del model; torch.cuda.empty_cache()
    all_results['cd_baseline'] = cd_results

    # -----------------------------------------------------------------------
    # 3. CD Pseudo-Huber Baseline (3 seeds)
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nCD PSEUDO-HUBER (3 seeds)\n{'='*60}")
    ph_results = {}
    for seed in SEEDS:
        model, train_time = train_distillation(
            precomputed, 'cd_pseudohuber_v2', seed, masks,
            use_pseudohuber=True)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(EXP_DIR / 'cd_pseudohuber_v2' / f'results_seed{seed}.json'), 'w'), indent=2)
        ph_results[seed] = res
        del model; torch.cuda.empty_cache()
    all_results['cd_pseudohuber'] = ph_results

    # -----------------------------------------------------------------------
    # 4. SCD Main - Adaptive Teacher + Spectral Weights (3 seeds)
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nSCD MAIN - Adaptive Teacher (3 seeds)\n{'='*60}")
    scd_results = {}
    for seed in SEEDS:
        model, train_time = train_distillation(
            precomputed, 'scd_adaptive_v2', seed, masks,
            spectral_weights=SCD_WEIGHTS,
            adaptive_teacher=True)
        res = evaluate_model(model, real_images, masks, seed=seed)
        res['train_time_min'] = train_time
        res['train_steps'] = DISTILL_STEPS
        json.dump(res, open(str(EXP_DIR / 'scd_adaptive_v2' / f'results_seed{seed}.json'), 'w'), indent=2)
        scd_results[seed] = res
        del model; torch.cuda.empty_cache()
    all_results['scd_main'] = scd_results

    # -----------------------------------------------------------------------
    # 5. Ablation: SCD without adaptive teacher (uniform 20-step targets)
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nABLATION: SCD no adaptive teacher\n{'='*60}")
    model, train_time = train_distillation(
        precomputed, 'ablation_no_adaptive_v2', 42, masks,
        spectral_weights=SCD_WEIGHTS,
        adaptive_teacher=False)
    res = evaluate_model(model, real_images, masks, seed=42)
    res['train_time_min'] = train_time
    res['train_steps'] = DISTILL_STEPS
    json.dump(res, open(str(EXP_DIR / 'ablation_no_adaptive_v2' / f'results_seed42.json'), 'w'), indent=2)
    all_results['ablation_no_adaptive'] = {42: res}
    del model; torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 6. Ablation: SCD without progressive refinement (fixed weights from start)
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nABLATION: SCD no progressive\n{'='*60}")
    model, train_time = train_distillation(
        precomputed, 'ablation_no_progressive_v2', 42, masks,
        spectral_weights=SCD_WEIGHTS,
        adaptive_teacher=True)  # has adaptive teacher but no progressive schedule
    res = evaluate_model(model, real_images, masks, seed=42)
    res['train_time_min'] = train_time
    res['train_steps'] = DISTILL_STEPS
    json.dump(res, open(str(EXP_DIR / 'ablation_no_progressive_v2' / f'results_seed42.json'), 'w'), indent=2)
    all_results['ablation_no_progressive'] = {42: res}
    del model; torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 7. Ablation: SCD with progressive schedule + adaptive teacher
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nABLATION: SCD with progressive schedule\n{'='*60}")
    model, train_time = train_distillation(
        precomputed, 'ablation_progressive_v2', 42, masks,
        spectral_weights=SCD_WEIGHTS,
        adaptive_teacher=True,
        progressive_schedule={'uniform_frac': 0.3, 'ramp_frac': 0.4})
    res = evaluate_model(model, real_images, masks, seed=42)
    res['train_time_min'] = train_time
    res['train_steps'] = DISTILL_STEPS
    json.dump(res, open(str(EXP_DIR / 'ablation_progressive_v2' / f'results_seed42.json'), 'w'), indent=2)
    all_results['ablation_progressive'] = {42: res}
    del model; torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 8. Ablation: SCD uniform spectral weights (all bands weight 1.0) + adaptive teacher
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nABLATION: SCD uniform weights\n{'='*60}")
    model, train_time = train_distillation(
        precomputed, 'ablation_uniform_weights_v2', 42, masks,
        spectral_weights=[1.0, 1.0, 1.0, 1.0],
        adaptive_teacher=True)
    res = evaluate_model(model, real_images, masks, seed=42)
    res['train_time_min'] = train_time
    res['train_steps'] = DISTILL_STEPS
    json.dump(res, open(str(EXP_DIR / 'ablation_uniform_weights_v2' / f'results_seed42.json'), 'w'), indent=2)
    all_results['ablation_uniform_weights'] = {42: res}
    del model; torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 9. Aggregate + Figures
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}\nAGGREGATING RESULTS\n{'='*60}")
    aggregate_and_save(all_results)

    log.info(f"\n{'='*60}\nGENERATING FIGURES\n{'='*60}")
    generate_figures(all_results)

    # Generate qualitative comparison
    generate_qualitative(all_results)

    log.info(f"\n{'='*60}\nALL EXPERIMENTS COMPLETE\n{'='*60}")


# ===========================================================================
# AGGREGATION
# ===========================================================================
def aggregate_and_save(all_results):
    aggregated = {'main_results': {}, 'ablation_results': {}, 'success_criteria': {}}

    # Main results (3 seeds)
    for method in ['cd_baseline', 'cd_pseudohuber', 'rectflow_baseline', 'scd_main']:
        if method not in all_results:
            continue
        data = all_results[method]
        agg = {}
        for sk in ['1_step', '2_step', '4_step']:
            fids = [data[s][sk]['fid'] for s in SEEDS if s in data and sk in data[s]]
            if not fids:
                continue
            entry = {
                'fid_mean': round(float(np.mean(fids)), 2),
                'fid_std': round(float(np.std(fids)), 2)
            }
            K = len(data[SEEDS[0]][sk]['per_band_mse'])
            for k in range(K):
                vals = [data[s][sk]['per_band_mse'][k] for s in SEEDS if s in data]
                entry[f'band{k}_mse_mean'] = float(np.mean(vals))
                entry[f'band{k}_mse_std'] = float(np.std(vals))
            agg[sk] = entry
        times = [data[s].get('train_time_min', 0) for s in SEEDS if s in data]
        agg['train_time_mean'] = round(float(np.mean(times)), 1)
        aggregated['main_results'][method] = agg

    # Ablation results (seed 42 only)
    for abl_key in ['ablation_no_adaptive', 'ablation_no_progressive',
                     'ablation_progressive', 'ablation_uniform_weights']:
        if abl_key not in all_results:
            continue
        data = all_results[abl_key]
        agg = {}
        for sk in ['1_step', '2_step', '4_step']:
            if 42 in data and sk in data[42]:
                agg[sk] = {
                    'fid': round(data[42][sk]['fid'], 2),
                    'per_band_mse': data[42][sk]['per_band_mse']
                }
        agg['train_time_min'] = round(data[42].get('train_time_min', 0), 1)
        aggregated['ablation_results'][abl_key] = agg

    # Success criteria - honest evaluation
    cd = aggregated['main_results'].get('cd_baseline', {})
    scd = aggregated['main_results'].get('scd_main', {})
    criteria = {}

    if cd and scd:
        scd_beats_all = True
        for sk in ['1_step', '2_step', '4_step']:
            if sk in cd and sk in scd:
                cd_fid = cd[sk]['fid_mean']
                scd_fid = scd[sk]['fid_mean']
                improvement = (cd_fid - scd_fid) / cd_fid * 100
                criteria[f'{sk}_fid_improvement_pct'] = round(improvement, 2)
                criteria[f'{sk}_scd_beats_cd'] = bool(scd_fid < cd_fid)
                if scd_fid >= cd_fid:
                    scd_beats_all = False

        # HF error comparison
        if '1_step' in cd and '1_step' in scd:
            cd_hf = cd['1_step'].get('band3_mse_mean', 0)
            scd_hf = scd['1_step'].get('band3_mse_mean', 0)
            if cd_hf > 0:
                criteria['hf_mse_reduction_pct'] = round((cd_hf - scd_hf) / cd_hf * 100, 2)

        # Training overhead
        cd_time = cd.get('train_time_mean', 1)
        scd_time = scd.get('train_time_mean', 1)
        if cd_time > 0:
            criteria['training_overhead_pct'] = round((scd_time - cd_time) / cd_time * 100, 1)

        criteria['scd_beats_cd_all_steps'] = scd_beats_all

        fid_imp = criteria.get('1_step_fid_improvement_pct', 0)
        hf_red = criteria.get('hf_mse_reduction_pct', 0)

        if scd_beats_all and fid_imp > 10:
            criteria['hypothesis_supported'] = True
            criteria['summary'] = (
                f"SCD outperforms CD at all step budgets. "
                f"1-step FID improvement: {fid_imp:.1f}%. "
                f"HF MSE reduction: {hf_red:.1f}%.")
        elif scd_beats_all and fid_imp > 0:
            criteria['hypothesis_supported'] = 'partially'
            criteria['summary'] = (
                f"SCD shows modest improvements over CD. "
                f"1-step FID improvement: {fid_imp:.1f}% (below hypothesized 10-25%). "
                f"HF MSE reduction: {hf_red:.1f}%.")
        else:
            criteria['hypothesis_supported'] = False
            criteria['summary'] = (
                f"SCD does NOT outperform CD. "
                f"1-step FID change: {fid_imp:.1f}% (negative means SCD is worse). "
                f"The spectral weighting approach does not improve distillation "
                f"at this model scale ({sum(p.numel() for p in UNet(**MODEL_KWARGS).parameters())/1e6:.1f}M params). "
                f"Possible reasons: (1) model capacity too small to benefit from "
                f"frequency-specific supervision, (2) spectral loss reweighting does not "
                f"change the optimal student prediction when the model is capacity-limited, "
                f"(3) the benefit of adaptive teacher supervision is diminished when "
                f"teacher quality is already limited (FID 20.2 at 100 steps).")

    aggregated['success_criteria'] = criteria
    aggregated['config'] = {
        'model_params_M': round(sum(p.numel() for p in UNet(**MODEL_KWARGS).parameters()) / 1e6, 2),
        'model_channels': MODEL_KWARGS['model_channels'],
        'channel_mult': list(MODEL_KWARGS['channel_mult']),
        'distill_steps': DISTILL_STEPS,
        'teacher_ode_steps_uniform': TEACHER_STEPS_UNIFORM,
        'teacher_ode_steps_adaptive': TEACHER_STEPS_PER_BAND,
        'fid_num_samples': NUM_FID_SAMPLES,
        'seeds': SEEDS,
        'scd_weights': SCD_WEIGHTS,
        'dataset': 'CIFAR-10 (32x32)',
    }

    out_path = str(WORKSPACE / 'results.json')
    json.dump(aggregated, open(out_path, 'w'), indent=2)
    log.info(f"  Saved aggregated results to {out_path}")
    log.info(f"  Success criteria: {json.dumps(criteria, indent=2)}")


# ===========================================================================
# FIGURES
# ===========================================================================
def generate_figures(all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12, 'figure.dpi': 150, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.3, 'font.family': 'serif'
    })

    colors = {
        'cd_baseline': '#1f77b4', 'cd_pseudohuber': '#ff7f0e',
        'rectflow_baseline': '#2ca02c', 'scd_main': '#d62728',
    }
    labels = {
        'cd_baseline': 'CD (MSE)', 'cd_pseudohuber': 'CD (Pseudo-Huber)',
        'rectflow_baseline': 'Rectified Flow', 'scd_main': 'SCD (Ours)',
    }
    band_names = ['Low', 'Mid-Low', 'Mid-High', 'High']

    # --- Figure 1: Spectral Error Analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (step_key, title) in enumerate([('1_step', '1-Step Generation'),
                                                  ('4_step', '4-Step Generation')]):
        ax = axes[ax_idx]
        x = np.arange(NUM_FREQ_BANDS)
        width = 0.22
        methods_to_plot = ['cd_baseline', 'cd_pseudohuber', 'scd_main']
        for i, method in enumerate(methods_to_plot):
            if method not in all_results:
                continue
            data = all_results[method]
            means = [np.mean([data[s][step_key]['per_band_mse'][k]
                             for s in SEEDS if s in data])
                     for k in range(NUM_FREQ_BANDS)]
            stds = [np.std([data[s][step_key]['per_band_mse'][k]
                           for s in SEEDS if s in data])
                    for k in range(NUM_FREQ_BANDS)]
            offset = (i - len(methods_to_plot)/2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=labels.get(method, method),
                   color=colors.get(method, '#333'), capsize=3, alpha=0.85)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Per-Band MSE')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(band_names)
        ax.legend(fontsize=9)
        ax.set_yscale('log')
    plt.suptitle('Spectral Error Analysis (CIFAR-10)', y=1.02, fontsize=14)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(str(FIG_DIR / f'figure1_spectral_error.{ext}'))
    plt.close()
    log.info("  Saved figure1_spectral_error")

    # --- Figure 2: FID vs Steps ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in ['rectflow_baseline', 'cd_baseline', 'cd_pseudohuber', 'scd_main']:
        if method not in all_results:
            continue
        data = all_results[method]
        steps_list = [1, 2, 4]
        fid_m = [np.mean([data[s][sk]['fid'] for s in SEEDS if s in data])
                 for sk in ['1_step', '2_step', '4_step']]
        fid_s = [np.std([data[s][sk]['fid'] for s in SEEDS if s in data])
                 for sk in ['1_step', '2_step', '4_step']]
        if method == 'rectflow_baseline' and fid_m[0] > 200:
            ax.errorbar(steps_list[1:], fid_m[1:], yerr=fid_s[1:], marker='o',
                        label=labels[method], color=colors[method], linewidth=2,
                        capsize=4, linestyle='--')
            ax.annotate(f'RF 1-step: {fid_m[0]:.0f}', xy=(1, fid_m[1]),
                       fontsize=8, ha='center', va='bottom')
        else:
            ax.errorbar(steps_list, fid_m, yerr=fid_s, marker='o',
                        label=labels[method], color=colors[method],
                        linewidth=2, capsize=4)
    ax.set_xlabel('Number of Inference Steps')
    ax.set_ylabel('FID (lower is better)')
    ax.set_title('FID vs Inference Steps (CIFAR-10)')
    ax.set_xticks([1, 2, 4])
    ax.legend()
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(str(FIG_DIR / f'figure2_fid_vs_steps.{ext}'))
    plt.close()
    log.info("  Saved figure2_fid_vs_steps")

    # --- Figure 3: Training Dynamics ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: training loss curves for all methods
    ax = axes[0]
    for method_dir, label, color in [
        ('cd_baseline_v2', 'CD (MSE)', '#1f77b4'),
        ('cd_pseudohuber_v2', 'CD (Pseudo-Huber)', '#ff7f0e'),
        ('scd_adaptive_v2', 'SCD (Ours)', '#d62728')]:
        log_path = EXP_DIR / method_dir / 'train_log_seed42.json'
        if log_path.exists():
            tlog = json.load(open(str(log_path)))
            steps = [e['step'] for e in tlog]
            losses = [e['loss'] for e in tlog]
            ax.plot(steps, losses, label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Right: per-band distillation error for SCD
    ax = axes[1]
    scd_log_path = EXP_DIR / 'scd_adaptive_v2' / 'train_log_seed42.json'
    if scd_log_path.exists():
        tlog = json.load(open(str(scd_log_path)))
        entries_with_bands = [e for e in tlog if 'band_losses' in e]
        if entries_with_bands:
            steps = [e['step'] for e in entries_with_bands]
            for k in range(NUM_FREQ_BANDS):
                vals = [e['band_losses'][k] for e in entries_with_bands]
                ax.plot(steps, vals, label=band_names[k], linewidth=1.5)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Per-Band Distillation Loss')
            ax.set_title('SCD Per-Band Training Dynamics')
            ax.legend(fontsize=9)
            ax.set_yscale('log')
    plt.suptitle('Training Dynamics', y=1.02, fontsize=14)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(str(FIG_DIR / f'figure3_training_dynamics.{ext}'))
    plt.close()
    log.info("  Saved figure3_training_dynamics")

    # --- Figure 4: Ablation Study ---
    fig, ax = plt.subplots(figsize=(10, 6))
    abl_items = []
    abl_colors = ['#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#d62728']

    abl_methods = [
        ('cd_baseline', 'CD Baseline (MSE)'),
        ('ablation_uniform_weights', 'SCD (uniform w, adaptive T)'),
        ('ablation_no_adaptive', 'SCD (weighted, uniform T)'),
        ('ablation_no_progressive', 'SCD (no progressive)'),
        ('ablation_progressive', 'SCD (progressive)'),
        ('scd_main', 'SCD Full (Ours)'),
    ]
    for method_key, label in abl_methods:
        if method_key not in all_results:
            continue
        data = all_results[method_key]
        if method_key in ['scd_main', 'cd_baseline']:
            fids = {sk: np.mean([data[s][sk]['fid'] for s in SEEDS if s in data])
                    for sk in ['1_step', '4_step']}
        else:
            fids = {sk: data[42][sk]['fid'] if 42 in data else 0
                    for sk in ['1_step', '4_step']}
        abl_items.append((label, fids['1_step'], fids['4_step']))

    if abl_items:
        names, fids1, fids4 = zip(*abl_items)
        x = np.arange(len(names))
        width = 0.35
        bars1 = ax.bar(x - width/2, fids1, width, label='1-step FID',
                       color=abl_colors[:len(names)], alpha=0.7)
        bars4 = ax.bar(x + width/2, fids4, width, label='4-step FID',
                       color=abl_colors[:len(names)], alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('FID (lower is better)')
        ax.set_title('Ablation Study')
        ax.legend()
        for bar, val in zip(bars1, fids1):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(str(FIG_DIR / f'figure4_ablation.{ext}'))
    plt.close()
    log.info("  Saved figure4_ablation")

    # --- Tables ---
    # Table 1: Main results
    with open(str(FIG_DIR / 'table1_main_results.csv'), 'w') as f:
        f.write("Method,1-step FID,2-step FID,4-step FID,Train Time (min)\n")
        for method in ['rectflow_baseline', 'cd_baseline', 'cd_pseudohuber', 'scd_main']:
            if method not in all_results:
                continue
            data = all_results[method]
            row = [labels.get(method, method)]
            for sk in ['1_step', '2_step', '4_step']:
                fids = [data[s][sk]['fid'] for s in SEEDS if s in data]
                row.append(f"{np.mean(fids):.2f} +/- {np.std(fids):.2f}")
            times = [data[s].get('train_time_min', 0) for s in SEEDS if s in data]
            row.append(f"{np.mean(times):.1f}")
            f.write(','.join(row) + '\n')

    with open(str(FIG_DIR / 'table1_main_results.tex'), 'w') as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{FID comparison on CIFAR-10 (mean $\\pm$ std over 3 seeds). "
                "Lower is better.}\n")
        f.write("\\label{tab:main}\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & 1-step & 2-step & 4-step & Train (min) \\\\\n\\midrule\n")
        for method in ['rectflow_baseline', 'cd_baseline', 'cd_pseudohuber', 'scd_main']:
            if method not in all_results:
                continue
            data = all_results[method]
            row = [labels.get(method, method)]
            best_fids = {}
            for sk in ['1_step', '2_step', '4_step']:
                fids = [data[s][sk]['fid'] for s in SEEDS if s in data]
                best_fids[sk] = np.mean(fids)
                row.append(f"${np.mean(fids):.1f} \\pm {np.std(fids):.1f}$")
            times = [data[s].get('train_time_min', 0) for s in SEEDS if s in data]
            row.append(f"${np.mean(times):.0f}$")
            f.write(' & '.join(row) + ' \\\\\n')
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Table 2: Ablation
    with open(str(FIG_DIR / 'table2_ablation.tex'), 'w') as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Ablation study on CIFAR-10 (seed=42). "
                "Spectral weights, adaptive teacher, and progressive schedule.}\n")
        f.write("\\label{tab:ablation}\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Variant & Spec. Wt. & Adap. T & 1-step FID & 4-step FID \\\\\n\\midrule\n")
        abl_rows = [
            ('cd_baseline', 'CD Baseline', 'No', 'No'),
            ('ablation_uniform_weights', 'Uniform Wt + Adap. T', 'Uniform', 'Yes'),
            ('ablation_no_adaptive', 'Weighted + Unif. T', 'Yes', 'No'),
            ('ablation_no_progressive', 'SCD (no progressive)', 'Yes', 'Yes'),
            ('ablation_progressive', 'SCD + Progressive', 'Yes', 'Yes'),
            ('scd_main', 'SCD Full (Ours)', 'Yes', 'Yes'),
        ]
        for key, name, sw, at in abl_rows:
            if key not in all_results:
                continue
            data = all_results[key]
            if key in ['scd_main', 'cd_baseline']:
                f1 = np.mean([data[s]['1_step']['fid'] for s in SEEDS if s in data])
                f4 = np.mean([data[s]['4_step']['fid'] for s in SEEDS if s in data])
            else:
                f1 = data[42]['1_step']['fid'] if 42 in data else 0
                f4 = data[42]['4_step']['fid'] if 42 in data else 0
            f.write(f"{name} & {sw} & {at} & ${f1:.1f}$ & ${f4:.1f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    log.info("  Saved tables")


def generate_qualitative(all_results):
    """Generate qualitative comparison of generated samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = [
        ('cd_baseline_v2', 'CD (MSE)', False),
        ('cd_pseudohuber_v2', 'CD (Pseudo-Huber)', False),
        ('scd_adaptive_v2', 'SCD (Ours)', False),
    ]

    fig, axes = plt.subplots(len(methods), 8, figsize=(16, 6))
    set_seed(42)
    z = torch.randn(8, 3, 32, 32, device=DEVICE)

    for row, (method_dir, label, is_vel) in enumerate(methods):
        ckpt_path = EXP_DIR / method_dir / 'checkpoint_seed42.pt'
        if not ckpt_path.exists():
            continue
        model = UNet(**MODEL_KWARGS).to(DEVICE)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True))
        model.eval()

        with torch.no_grad():
            samples = consistency_sample(model, z, 1)

        for col in range(8):
            img = ((samples[col] + 1) / 2).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=10, rotation=90, labelpad=10)
        del model

    plt.suptitle('1-Step Generated Samples (CIFAR-10)', fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(str(FIG_DIR / f'figure5_qualitative.{ext}'))
    plt.close()
    log.info("  Saved figure5_qualitative")


if __name__ == '__main__':
    main()
