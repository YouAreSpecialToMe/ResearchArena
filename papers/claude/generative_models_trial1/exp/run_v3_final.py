"""SCD Experiment v3 — final corrected version.

Fixes from v2 diagnostic:
1. Spectral loss normalization: divide by (H*W)^2 not H*W (was 1024x too large)
2. Drop composite targets (they produce FID ~500 artifacts)
3. SCD = spectral-weighted distillation loss on standard 20-step teacher target

Baselines (CD, Pseudo-Huber, Rectified Flow) reused from v2.
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

from exp.shared.models import UNet
from exp.shared.flow_matching import ot_cfm_sample_t_and_xt, euler_sample, teacher_solve

DEVICE = 'cuda'
NUM_FREQ_BANDS = 4
SEEDS = [42, 43, 44]
MODEL_KWARGS = dict(model_channels=64, channel_mult=(1, 2, 2), attention_resolutions=(8,))

DISTILL_STEPS = 20000
DISTILL_BATCH = 256
NUM_FID_SAMPLES = 10000
TEACHER_ODE_STEPS = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cifar10_to_gpu():
    from torchvision import datasets, transforms
    ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True,
                          transform=transforms.ToTensor())
    data = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)
    return data * 2 - 1


def sample_batch(data, batch_size, augment=True):
    idx = torch.randint(0, len(data), (batch_size,))
    batch = data[idx]
    if augment:
        if torch.rand(1).item() > 0.5:
            batch = batch.flip(-1)
    return batch


def ema_update(ema_model, model, decay=0.9999):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ===========================================================================
# Frequency-domain utilities (FIXED normalization)
# ===========================================================================
def create_fft_frequency_masks(H, W, K, device='cpu'):
    freq_y = torch.fft.fftfreq(H, device=device)
    freq_x = torch.fft.fftfreq(W, device=device)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_mag = torch.sqrt(fy ** 2 + fx ** 2)
    max_freq = freq_mag.max().item()
    boundaries = torch.linspace(0, max_freq + 1e-6, K + 1, device=device)
    masks = []
    for k in range(K):
        mask = ((freq_mag >= boundaries[k]) & (freq_mag < boundaries[k + 1])).float()
        masks.append(mask.view(1, 1, H, W))
    return masks


def spectral_loss_normalized(pred, target, masks, weights):
    """Weighted per-band spectral loss, CORRECTLY normalized to pixel-MSE scale.

    Parseval's theorem for unnormalized FFT:
        sum_k |FFT(x)[k]|^2 = H*W * sum_n |x[n]|^2

    So to get MSE = mean_n |x[n]|^2 = sum_n |x[n]|^2 / (H*W),
    we need: sum_k |FFT(x)[k]|^2 / (H*W)^2

    With uniform weights [1,1,1,1] this equals F.mse_loss(pred, target).
    """
    H, W = pred.shape[-2], pred.shape[-1]
    HW2 = (H * W) ** 2  # correct normalization
    diff = pred - target
    diff_freq = torch.fft.fft2(diff)
    power = diff_freq.real ** 2 + diff_freq.imag ** 2

    total_loss = torch.tensor(0.0, device=pred.device)
    band_losses = []
    for mask, w in zip(masks, weights):
        band_power = (power * mask).sum(dim=(-2, -1)).mean()  # mean over batch,channels
        band_mse = band_power / HW2
        band_losses.append(band_mse.item())
        total_loss = total_loss + w * band_mse
    return total_loss, band_losses


def fft_band_mse(pred, target, masks):
    H, W = pred.shape[-2], pred.shape[-1]
    HW2 = (H * W) ** 2
    diff = pred - target
    diff_freq = torch.fft.fft2(diff)
    power = diff_freq.real ** 2 + diff_freq.imag ** 2
    band_mses = []
    for mask in masks:
        band_power = (power * mask).sum(dim=(-2, -1)).mean()
        band_mses.append(band_power / HW2)
    return band_mses


# ===========================================================================
# FID computation
# ===========================================================================
def compute_fid_from_samples(samples, num_samples=None):
    import tempfile
    from cleanfid import fid
    from torchvision.utils import save_image
    tmpdir = tempfile.mkdtemp(prefix='scd_fid_')
    n = num_samples or len(samples)
    for i in range(n):
        img = ((samples[i] + 1) / 2).clamp(0, 1)
        save_image(img, os.path.join(tmpdir, f'{i:06d}.png'))
    score = fid.compute_fid(tmpdir, dataset_name='cifar10', dataset_split='train', dataset_res=32, mode='clean')
    import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
    return score


# ===========================================================================
# Sampling
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
def generate_samples_consistency(model, num_samples, num_steps, batch_size=512, seed=42):
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


@torch.no_grad()
def generate_samples_velocity(model, num_samples, num_steps, batch_size=512, seed=42):
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


def evaluate_method(model, steps_list, num_samples, real_images, seed, is_velocity_model=False):
    results = {}
    masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)
    gen_fn = generate_samples_velocity if is_velocity_model else generate_samples_consistency
    for ns in steps_list:
        print(f"    Eval {ns}-step ({num_samples} samples)...", flush=True)
        t0 = time.time()
        samples = gen_fn(model, num_samples, ns, seed=seed)
        gen_time = time.time() - t0
        fid_score = compute_fid_from_samples(samples)
        print(f"    {ns}-step FID: {fid_score:.2f} ({gen_time:.1f}s)", flush=True)
        step_result = {'fid': fid_score, 'gen_time_s': gen_time}
        if real_images is not None:
            n = min(10000, len(samples), len(real_images))
            gen_batch = samples[:n].to(DEVICE)
            real_batch = real_images[:n].to(DEVICE)
            band_mses = fft_band_mse(gen_batch, real_batch, masks)
            step_result['per_band_mse'] = [m.item() for m in band_mses]
        results[f'{ns}_step'] = step_result
    return results


# ===========================================================================
# PRE-COMPUTE TEACHER TARGETS (CPU storage)
# ===========================================================================
def precompute_teacher_targets(teacher, data, num_pairs=50000, seed=42):
    """Pre-compute teacher targets at TEACHER_ODE_STEPS. Stored on CPU."""
    print(f"\nPre-computing teacher targets ({num_pairs} pairs, steps={TEACHER_ODE_STEPS})...", flush=True)
    set_seed(seed)
    teacher.eval()
    compiled_teacher = torch.compile(teacher)

    idx = torch.randperm(len(data))[:num_pairs]
    x0 = data[idx]
    noise = torch.randn_like(x0)
    t = torch.rand(num_pairs, device=DEVICE) * 0.98 + 0.01
    t_expand = t.view(num_pairs, 1, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * noise

    batch_size = 512
    all_targets = []

    t0 = time.time()
    for i in range(0, num_pairs, batch_size):
        batch_xt = x_t[i:i+batch_size]
        batch_t = t[i:i+batch_size]
        target = teacher_solve(compiled_teacher, batch_xt, batch_t, num_steps=TEACHER_ODE_STEPS)
        all_targets.append(target.cpu().half())

    targets = torch.cat(all_targets)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({num_pairs/elapsed:.0f} pairs/sec)", flush=True)

    return {
        'x_t': x_t.cpu(),
        't': t.cpu(),
        'targets': targets,
        'num_pairs': num_pairs,
    }


# ===========================================================================
# DISTILLATION TRAINING
# ===========================================================================
def train_distillation(precomputed, method, seed, num_steps=DISTILL_STEPS,
                       lr=1e-4, batch_size=DISTILL_BATCH,
                       use_pseudohuber=False,
                       use_spectral=False,
                       spectral_config=None,
                       teacher_ckpt_path=None,
                       masks=None):
    print(f"\n  Training {method} (seed={seed}, {num_steps} steps)...", flush=True)

    save_dir = WORKSPACE / 'exp' / method
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f'checkpoint_seed{seed}.pt'
    results_path = save_dir / f'results_seed{seed}.json'

    # Resume from checkpoint if training already completed
    if ckpt_path.exists():
        print(f"    Found checkpoint, loading (skip training)...", flush=True)
        model = UNet(**MODEL_KWARGS).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        return model, 0.0

    set_seed(seed)
    student = UNet(**MODEL_KWARGS).to(DEVICE)
    if teacher_ckpt_path:
        student.load_state_dict(torch.load(teacher_ckpt_path, map_location=DEVICE, weights_only=True))
    ema_student = copy.deepcopy(student)
    compiled_student = torch.compile(student)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda')

    # Spectral config
    if use_spectral and spectral_config and masks is not None:
        base_weights = spectral_config.get('weights', [1.0, 1.5, 2.5, 4.0])
        progressive = spectral_config.get('progressive', True)
        fixed_weights = spectral_config.get('fixed_weights', False)

    if use_pseudohuber:
        d = 3 * 32 * 32
        c_huber = 0.00054 * math.sqrt(d)

    N = precomputed['num_pairs']
    running_band_errors = [1.0] * NUM_FREQ_BANDS
    t_start = time.time()
    band_error_history = []

    for step in range(1, num_steps + 1):
        idx = torch.randint(0, N, (batch_size,))
        x_t = precomputed['x_t'][idx].float().to(DEVICE)
        t_vals = precomputed['t'][idx].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            student_pred = compiled_student(x_t, t_vals)
            target = precomputed['targets'][idx].float().to(DEVICE)

            if use_spectral and masks is not None:
                # Progressive weight schedule
                if progressive and not fixed_weights:
                    frac = step / num_steps
                    if frac < 0.3:
                        weights = [1.0] * NUM_FREQ_BANDS
                    elif frac < 0.7:
                        alpha = (frac - 0.3) / 0.4
                        weights = [1.0 + alpha * (w - 1.0) for w in base_weights]
                    else:
                        mean_err = sum(running_band_errors) / len(running_band_errors)
                        weights = [(e / (mean_err + 1e-8)) ** 0.5
                                   for e in running_band_errors]
                        w_sum = sum(weights)
                        weights = [w * NUM_FREQ_BANDS / (w_sum + 1e-8) for w in weights]
                elif fixed_weights:
                    weights = list(base_weights)
                else:
                    weights = [1.0] * NUM_FREQ_BANDS

                loss, band_errs = spectral_loss_normalized(student_pred, target, masks, weights)
                for k, be in enumerate(band_errs):
                    running_band_errors[k] = 0.99 * running_band_errors[k] + 0.01 * be
            else:
                diff = student_pred - target
                if use_pseudohuber:
                    loss = (torch.sqrt(diff.pow(2).sum(dim=(1, 2, 3)) + c_huber ** 2)
                            - c_huber).mean()
                else:
                    loss = F.mse_loss(student_pred, target)

            del target

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        ema_update(ema_student, student, decay=0.999)

        if step % 2000 == 0:
            elapsed = (time.time() - t_start) / 60
            rate = step / elapsed if elapsed > 0 else 1
            eta = (num_steps - step) / rate
            msg = f"    Step {step}/{num_steps}, Loss: {loss.item():.6f}, Time: {elapsed:.1f}min, ETA: {eta:.0f}min"
            if use_spectral and masks is not None:
                msg += f"\n      Bands: {[f'{e:.4f}' for e in running_band_errors]}, W: {[f'{w:.2f}' for w in weights]}"
                band_error_history.append({'step': step, 'band_errors': list(running_band_errors), 'weights': list(weights)})
            print(msg, flush=True)

    total_time = (time.time() - t_start) / 60
    print(f"    Done. Time: {total_time:.1f}min", flush=True)
    torch.save(ema_student.state_dict(), ckpt_path)

    if band_error_history:
        with open(save_dir / f'band_errors_seed{seed}.json', 'w') as f:
            json.dump(band_error_history, f, indent=2)

    return ema_student, total_time


# ===========================================================================
# AGGREGATE + FIGURES
# ===========================================================================
def aggregate_results(all_results):
    aggregated = {'main_results': {}, 'ablation_results': {}, 'success_criteria': {}}

    for method in ['cd_baseline', 'cd_pseudohuber', 'rectflow_baseline', 'scd_main']:
        method_data = all_results.get(method, {})
        agg = {}
        for step_key in ['1_step', '2_step', '4_step']:
            fids = [method_data[s][step_key]['fid']
                    for s in SEEDS if s in method_data and step_key in method_data[s]]
            if fids:
                agg[step_key] = {'fid_mean': float(np.mean(fids)), 'fid_std': float(np.std(fids))}
                if 'per_band_mse' in method_data[SEEDS[0]][step_key]:
                    K = len(method_data[SEEDS[0]][step_key]['per_band_mse'])
                    for k in range(K):
                        vals = [method_data[s][step_key]['per_band_mse'][k]
                                for s in SEEDS if s in method_data]
                        agg[step_key][f'band{k}_mse_mean'] = float(np.mean(vals))
                        agg[step_key][f'band{k}_mse_std'] = float(np.std(vals))
        train_times = [method_data[s].get('train_time_min', 0) for s in SEEDS if s in method_data]
        agg['train_time_mean'] = float(np.mean(train_times)) if train_times else 0
        aggregated['main_results'][method] = agg

    for abl_name, abl_data in all_results.get('ablations', {}).items():
        agg = {}
        for step_key in ['1_step', '2_step', '4_step']:
            if step_key in abl_data:
                agg[step_key] = {'fid': abl_data[step_key]['fid']}
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
        cd_hf = sum(cd['1_step'].get(f'band{k}_mse_mean', 0) for k in [2, 3])
        scd_hf = sum(scd['1_step'].get(f'band{k}_mse_mean', 0) for k in [2, 3])
        if cd_hf > 0:
            hf_reduction = (cd_hf - scd_hf) / cd_hf * 100
            crit2_pass = hf_reduction >= 15

    cd_time = cd.get('train_time_mean', 1)
    scd_time = scd.get('train_time_mean', 1)
    overhead = (scd_time - cd_time) / (cd_time + 1e-8) * 100 if cd_time > 0 else 0

    aggregated['success_criteria'] = {
        'scd_beats_cd_all_steps': crit1_pass,
        'fid_improvements_pct': fid_improvements,
        'hf_error_reduction_15pct': crit2_pass,
        'hf_reduction_pct': float(hf_reduction) if hf_reduction is not None else None,
        'training_overhead_below_50pct': overhead < 50,
        'training_overhead_pct': float(overhead),
    }

    print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
    for method in ['cd_baseline', 'cd_pseudohuber', 'rectflow_baseline', 'scd_main']:
        data = aggregated['main_results'].get(method, {})
        print(f"\n{method}:")
        for sk in ['1_step', '2_step', '4_step']:
            if sk in data:
                print(f"  {sk}: FID = {data[sk]['fid_mean']:.2f} +/- {data[sk]['fid_std']:.2f}")
    print("\nAblations:")
    for abl_name, data in aggregated.get('ablation_results', {}).items():
        print(f"  {abl_name}: 1-step FID = {data.get('1_step',{}).get('fid','N/A')}")
    print("\nSuccess Criteria:")
    for k, v in aggregated['success_criteria'].items():
        print(f"  {k}: {v}")
    return aggregated


def generate_figures(aggregated, all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = WORKSPACE / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 10})

    main = aggregated['main_results']
    methods = ['rectflow_baseline', 'cd_baseline', 'cd_pseudohuber', 'scd_main']
    labels = ['Rectified Flow', 'Standard CD', 'Pseudo-Huber CD', 'SCD (Ours)']
    colors = ['#888888', '#1f77b4', '#ff7f0e', '#2ca02c']

    # Figure 4: FID vs Steps
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, label, color in zip(methods, labels, colors):
        data = main.get(method, {})
        fids = [data.get(f'{s}_step', {}).get('fid_mean') for s in [1, 2, 4]]
        stds = [data.get(f'{s}_step', {}).get('fid_std', 0) for s in [1, 2, 4]]
        if all(f is not None for f in fids):
            ax.errorbar([1, 2, 4], fids, yerr=stds, marker='o', label=label,
                       color=color, linewidth=2, capsize=4)
    ax.set_xlabel('Number of Inference Steps')
    ax.set_ylabel('FID (lower is better)')
    ax.set_xticks([1, 2, 4])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('FID vs. Number of Inference Steps')
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure4_fid_vs_steps.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'figure4_fid_vs_steps.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 1: Spectral Error Analysis
    band_names = ['Low', 'Mid-Low', 'Mid-High', 'High']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, sk, title in zip(axes, ['1_step', '4_step'], ['1-Step', '4-Step']):
        x = np.arange(4)
        width = 0.18
        for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
            data = main.get(method, {}).get(sk, {})
            vals = [data.get(f'band{k}_mse_mean', 0) for k in range(4)]
            ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Per-Band MSE')
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(band_names)
        ax.set_title(f'{title} Generation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure1_spectral_error.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'figure1_spectral_error.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Table 1
    with open(fig_dir / 'table1_main_results.csv', 'w') as f:
        f.write("Method,1-step FID,2-step FID,4-step FID\n")
        for method, label in zip(methods, labels):
            data = main.get(method, {})
            row = [label]
            for sk in ['1_step', '2_step', '4_step']:
                d = data.get(sk, {})
                row.append(f"{d.get('fid_mean',0):.2f} +/- {d.get('fid_std',0):.2f}")
            f.write(",".join(row) + "\n")

    with open(fig_dir / 'table1_main_results.tex', 'w') as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Method & 1-step FID $\\downarrow$ & 2-step FID $\\downarrow$ & 4-step FID $\\downarrow$ \\\\\n\\midrule\n")
        best = {}
        for sk in ['1_step', '2_step', '4_step']:
            best[sk] = min(main.get(m, {}).get(sk, {}).get('fid_mean', 1e9) for m in methods)
        for method, label in zip(methods, labels):
            data = main.get(method, {})
            row = [label]
            for sk in ['1_step', '2_step', '4_step']:
                d = data.get(sk, {})
                m, s = d.get('fid_mean', 0), d.get('fid_std', 0)
                val = f"{m:.1f} $\\pm$ {s:.1f}"
                if abs(m - best[sk]) < 0.01:
                    val = "\\textbf{" + val + "}"
                row.append(val)
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Table 2: Ablation
    scd_42 = all_results.get('scd_main', {}).get(42, {})
    with open(fig_dir / 'table2_ablation.csv', 'w') as f:
        f.write("Variant,1-step FID,2-step FID,4-step FID\n")
        f.write(f"SCD (full),{scd_42.get('1_step',{}).get('fid',0):.2f},"
                f"{scd_42.get('2_step',{}).get('fid',0):.2f},"
                f"{scd_42.get('4_step',{}).get('fid',0):.2f}\n")
        for abl_name in aggregated.get('ablation_results', {}):
            data = aggregated['ablation_results'][abl_name]
            f.write(f"{abl_name},{data.get('1_step',{}).get('fid',0):.2f},"
                    f"{data.get('2_step',{}).get('fid',0):.2f},"
                    f"{data.get('4_step',{}).get('fid',0):.2f}\n")

    with open(fig_dir / 'table2_ablation.tex', 'w') as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Variant & 1-step FID & 2-step FID & 4-step FID \\\\\n\\midrule\n")
        f.write(f"SCD (full) & {scd_42.get('1_step',{}).get('fid',0):.1f} & "
                f"{scd_42.get('2_step',{}).get('fid',0):.1f} & "
                f"{scd_42.get('4_step',{}).get('fid',0):.1f} \\\\\n")
        for abl_name, nice in [('ablation_no_progressive', 'w/o Progressive Refinement'),
                                ('ablation_no_spectral_weight', 'w/o Spectral Weighting (= CD)')]:
            data = aggregated['ablation_results'].get(abl_name, {})
            if data:
                f.write(f"{nice} & {data.get('1_step',{}).get('fid',0):.1f} & "
                        f"{data.get('2_step',{}).get('fid',0):.1f} & "
                        f"{data.get('4_step',{}).get('fid',0):.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Figure 3: Training dynamics
    band_err_file = WORKSPACE / 'exp' / 'scd_main' / 'band_errors_seed42.json'
    if band_err_file.exists():
        with open(band_err_file) as f:
            hist = json.load(f)
        if hist:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            steps_p = [h['step'] for h in hist]
            for k, lbl in enumerate(band_names):
                ax1.plot(steps_p, [h['band_errors'][k] for h in hist], label=lbl, linewidth=2)
                ax2.plot(steps_p, [h['weights'][k] for h in hist], label=lbl, linewidth=2)
            ax1.set_xlabel('Training Step'); ax1.set_ylabel('Per-Band Error')
            ax1.set_title('SCD Per-Band Error During Training'); ax1.legend(); ax1.grid(True, alpha=0.3)
            ax2.set_xlabel('Training Step'); ax2.set_ylabel('Band Weight')
            ax2.set_title('Progressive Weight Schedule'); ax2.legend(); ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(fig_dir / 'figure3_training_dynamics.pdf', dpi=300, bbox_inches='tight')
            fig.savefig(fig_dir / 'figure3_training_dynamics.png', dpi=150, bbox_inches='tight')
            plt.close()

    # Figure 2: Qualitative
    print("  Generating qualitative comparison...", flush=True)
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(123)
    z = torch.randn(8, 3, 32, 32, device=DEVICE, generator=gen)
    row_labels = ['Rect. Flow\n(1 step)', 'Standard CD\n(1 step)',
                  'Pseudo-Huber CD\n(1 step)', 'SCD (Ours)\n(1 step)']
    model_specs = [
        ('teacher', WORKSPACE / 'exp' / 'teacher' / 'checkpoint_best.pt', True),
        ('cd_baseline', WORKSPACE / 'exp' / 'cd_baseline' / 'checkpoint_seed42.pt', False),
        ('cd_pseudohuber', WORKSPACE / 'exp' / 'cd_pseudohuber' / 'checkpoint_seed42.pt', False),
        ('scd_main', WORKSPACE / 'exp' / 'scd_main' / 'checkpoint_seed42.pt', False),
    ]
    for row_idx, (name, path, is_vel) in enumerate(model_specs):
        if not path.exists() or row_idx >= 4:
            continue
        m = UNet(**MODEL_KWARGS).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.eval()
        with torch.no_grad():
            samp = euler_sample(m, z, 1) if is_vel else consistency_sample(m, z, 1)
            samp = ((samp + 1) / 2).clamp(0, 1).cpu()
        for c in range(8):
            axes[row_idx, c].imshow(samp[c].permute(1, 2, 0).numpy())
            axes[row_idx, c].axis('off')
        axes[row_idx, 0].set_ylabel(row_labels[row_idx], rotation=0,
                                      labelpad=80, fontsize=10, va='center')
        del m; torch.cuda.empty_cache()
    plt.suptitle('1-Step Generation Quality Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / 'figure2_qualitative.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'figure2_qualitative.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5: Ablation bar chart
    abl_data = aggregated.get('ablation_results', {})
    if abl_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = ['SCD\n(full)']
        fids = [scd_42.get('1_step', {}).get('fid', 0)]
        for abl_name, nice in [('ablation_no_progressive', 'w/o Progressive'),
                                ('ablation_no_spectral_weight', 'w/o Spectral\nWeighting')]:
            if abl_name in abl_data:
                names.append(nice)
                fids.append(abl_data[abl_name].get('1_step', {}).get('fid', 0))
        cd_fid = aggregated['main_results'].get('cd_baseline', {}).get('1_step', {}).get('fid_mean', 0)
        names.append('Standard CD')
        fids.append(cd_fid)
        bar_colors = ['#2ca02c'] + ['#ff7f0e'] * (len(names) - 2) + ['#1f77b4']
        bars = ax.bar(range(len(names)), fids, color=bar_colors, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel('1-Step FID (lower is better)')
        ax.set_title('Ablation Study')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, fid_v in zip(bars, fids):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{fid_v:.1f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        fig.savefig(fig_dir / 'figure5_ablation.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'figure5_ablation.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Figures saved to {fig_dir}/", flush=True)


# ===========================================================================
# MAIN
# ===========================================================================
def load_saved_results(method, seeds=SEEDS):
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

    print(f"SCD Experiment v3 (FINAL — fixed spectral loss, no composite targets)")
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Verify spectral loss correctness
    masks = create_fft_frequency_masks(32, 32, NUM_FREQ_BANDS, device=DEVICE)
    x = torch.randn(10, 3, 32, 32, device=DEVICE)
    y = torch.randn(10, 3, 32, 32, device=DEVICE)
    mse_ref = F.mse_loss(x, y)
    spec_val, _ = spectral_loss_normalized(x, y, masks, [1.0]*4)
    ratio = spec_val.item() / mse_ref.item()
    print(f"Spectral loss sanity check: MSE={mse_ref:.6f}, Spectral(uniform)={spec_val:.6f}, ratio={ratio:.4f}")
    assert abs(ratio - 1.0) < 0.01, f"Spectral loss not equal to MSE! ratio={ratio}"
    print("  PASS: spectral_loss(uniform) == MSE")

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
    assert teacher_ckpt.exists(), f"Teacher checkpoint not found"

    data = load_cifar10_to_gpu()
    real_images = data[:10000].cpu()

    teacher = UNet(**MODEL_KWARGS).to(DEVICE)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE, weights_only=True))
    precomputed = precompute_teacher_targets(teacher, data, num_pairs=50000, seed=42)
    del teacher; torch.cuda.empty_cache()
    del data; torch.cuda.empty_cache()

    # SCD main (3 seeds) — spectral-weighted loss on standard teacher targets
    print(f"\n{'='*60}\nSPECTRAL CONSISTENCY DISTILLATION (3 seeds)\n{'='*60}")
    scd_config = {'weights': [1.0, 1.5, 2.5, 4.0], 'progressive': True,
                  'fixed_weights': False}
    scd_results = {}
    for seed in SEEDS:
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
            'fixed_weights': True},  # uniform weights = standard MSE
        'ablation_no_progressive': {
            'weights': [1.0, 1.5, 2.5, 4.0], 'progressive': False,
            'fixed_weights': True},  # fixed weights, no progressive schedule
    }
    ablation_results = {}
    for abl_name, abl_config in ablation_configs.items():
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
