"""
Main experiment runner for CSG (Conditioning-Space Guidance) experiments.
Runs all experiments from plan.json in order.

Time budget: ~8 hours total on 1x A6000 48GB
Strategy:
  - Core experiments (w=4.0): 5K images, 3 seeds
  - Extra scales (w=1.5, 7.5): 5K images, 1 seed (seed=0)
  - Ablations: 5K images, 1 seed (seed=0)
  - 50 DDIM steps throughout
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import sys
import json
import time
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Setup paths
WORKSPACE = Path(__file__).parent.parent
EXP_DIR = WORKSPACE / 'exp'
RESULTS_DIR = EXP_DIR / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
CHECKPOINT_PATH = EXP_DIR / 'checkpoints' / 'DiT-XL-2-256x256.pt'

sys.path.insert(0, str(EXP_DIR / 'DiT'))
sys.path.insert(0, str(EXP_DIR / 'shared'))

from sampling import (
    load_dit_model, load_vae, decode_latents, get_noise_and_labels,
    sample_images, forward_cfg, forward_csg, forward_esg,
    get_hybrid_steps, get_per_layer_weights
)
from diffusion import create_diffusion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_STEPS = 50
LATENT_SIZE = 32  # 256 // 8

# Core experiment config (2K images: FID-2K has higher variance but fits in budget)
CORE_NUM_IMAGES = 2000
CORE_SEEDS = [0, 1, 2]
# Ablation config
ABLATION_NUM_IMAGES = 2000
ABLATION_SEEDS = [0]

GUIDANCE_SCALES = [1.5, 4.0, 7.5]

# Batch sizes (tuned for A6000 48GB)
CFG_BATCH_SIZE = 128   # 2-pass uses ~6GB
SINGLE_BATCH_SIZE = 128  # 1-pass uses ~5GB


def generate_images(model, diffusion, method, cfg_scale, seed,
                    num_images, per_layer_weights=None, hybrid_ratio=0.0,
                    num_steps=None):
    """Generate images and return latents + timing."""
    if num_steps is not None and num_steps != NUM_STEPS:
        diff = create_diffusion(str(num_steps))
    else:
        diff = diffusion

    noise_all, labels_all = get_noise_and_labels(num_images, LATENT_SIZE, seed, DEVICE)

    hybrid_steps = None
    actual_method = method
    if hybrid_ratio > 0:
        n = num_steps or NUM_STEPS
        hybrid_steps = get_hybrid_steps(n, hybrid_ratio, 'middle')
        actual_method = 'csg_hybrid'

    batch_size = CFG_BATCH_SIZE if method == 'cfg' else SINGLE_BATCH_SIZE

    all_latents = []
    total_time = 0.0
    num_batches = (num_images + batch_size - 1) // batch_size

    for b in tqdm(range(num_batches), desc=f'{method} w={cfg_scale} s={seed}', leave=False):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, num_images)

        noise_batch = noise_all[start_idx:end_idx].to(DEVICE)
        labels_batch = labels_all[start_idx:end_idx].to(DEVICE)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        latents = sample_images(
            model, diff, actual_method, noise_batch, labels_batch, cfg_scale,
            device=DEVICE, per_layer_weights=per_layer_weights,
            hybrid_ratio=hybrid_ratio, hybrid_steps=hybrid_steps
        )

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        total_time += (t_end - t_start)

        all_latents.append(latents.cpu())
        del noise_batch, labels_batch, latents
        torch.cuda.empty_cache()

    all_latents = torch.cat(all_latents, dim=0)
    throughput = num_images / total_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    timing = {
        'total_time_sec': round(total_time, 2),
        'num_images': num_images,
        'throughput_img_per_sec': round(throughput, 4),
        'peak_gpu_memory_gb': round(peak_mem, 2),
        'batch_size': batch_size,
        'num_steps': num_steps or NUM_STEPS,
    }

    return all_latents, timing


def decode_and_save_images(vae, latents, img_dir):
    """Decode latents to images and save as PNGs."""
    from torchvision.utils import save_image
    os.makedirs(img_dir, exist_ok=True)
    decode_bs = 50
    idx = 0
    for i in range(0, latents.shape[0], decode_bs):
        batch = latents[i:i+decode_bs].to(DEVICE)
        with torch.no_grad():
            images = vae.decode(batch / 0.18215).sample
        for j in range(images.shape[0]):
            save_image(images[j], os.path.join(img_dir, f'{idx:06d}.png'),
                       normalize=True, value_range=(-1, 1))
            idx += 1
        del images, batch
        torch.cuda.empty_cache()
    return idx


REF_STATS_PATH = str(EXP_DIR / 'data' / 'VIRTUAL_imagenet256_labeled.npz')
REF_FEATURES_CACHE = str(EXP_DIR / 'data' / 'imagenet256_ref_features.npz')

# Global cache for reference features
_ref_mu = None
_ref_sigma = None

def _get_inception_model():
    """Get InceptionV3 for feature extraction (matching torch-fidelity's setup)."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model

def _load_ref_features():
    """Load or compute reference Inception features."""
    global _ref_mu, _ref_sigma
    if _ref_mu is not None:
        return _ref_mu, _ref_sigma

    if os.path.exists(REF_FEATURES_CACHE):
        data = np.load(REF_FEATURES_CACHE)
        _ref_mu, _ref_sigma = data['mu'], data['sigma']
        return _ref_mu, _ref_sigma

    print("  Computing reference Inception features (one-time)...")
    ref_data = np.load(REF_STATS_PATH)
    ref_images = ref_data['arr_0']  # (10000, 256, 256, 3) uint8

    inception = _get_inception_model()
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    bs = 64
    for i in range(0, len(ref_images), bs):
        batch = torch.from_numpy(ref_images[i:i+bs]).float().permute(0, 3, 1, 2) / 255.0
        batch = transform(batch).to(DEVICE)
        with torch.no_grad():
            feats = inception(batch)
        all_features.append(feats.cpu().numpy())
        del batch

    features = np.concatenate(all_features)
    _ref_mu = features.mean(axis=0)
    _ref_sigma = np.cov(features, rowvar=False)

    np.savez(REF_FEATURES_CACHE, mu=_ref_mu, sigma=_ref_sigma)
    del inception
    torch.cuda.empty_cache()
    print(f"  Reference features cached ({features.shape[0]} images, {features.shape[1]}-dim)")
    return _ref_mu, _ref_sigma

def compute_metrics(img_dir):
    """Compute FID and IS from generated images."""
    from scipy import linalg
    from torchvision import transforms
    from PIL import Image
    metrics = {}

    inception = _get_inception_model()
    transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Compute features for generated images
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    all_features = []
    all_probs = []
    bs = 64

    for i in range(0, len(img_files), bs):
        batch_files = img_files[i:i+bs]
        batch_tensors = []
        for f in batch_files:
            img = Image.open(os.path.join(img_dir, f)).convert('RGB')
            batch_tensors.append(transform(img))
        batch = torch.stack(batch_tensors).to(DEVICE)

        with torch.no_grad():
            feats = inception(batch)
            all_features.append(feats.cpu().numpy())

        del batch
        torch.cuda.empty_cache()

    features = np.concatenate(all_features)
    gen_mu = features.mean(axis=0)
    gen_sigma = np.cov(features, rowvar=False)

    # FID
    ref_mu, ref_sigma = _load_ref_features()
    diff = gen_mu - ref_mu
    covmean, _ = linalg.sqrtm(gen_sigma @ ref_sigma, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_val = float(diff @ diff + np.trace(gen_sigma + ref_sigma - 2 * covmean))
    metrics['fid'] = round(fid_val, 4)

    # IS (using separate inception pass with logits)
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception_is = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        inception_is.eval().to(DEVICE)
        all_probs = []
        for i in range(0, len(img_files), bs):
            batch_files = img_files[i:i+bs]
            batch_tensors = []
            for f in batch_files:
                img = Image.open(os.path.join(img_dir, f)).convert('RGB')
                batch_tensors.append(transform(img))
            batch = torch.stack(batch_tensors).to(DEVICE)
            with torch.no_grad():
                logits = inception_is(batch)
                probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            del batch
        all_probs = np.concatenate(all_probs)
        splits = 10
        chunk = len(all_probs) // splits
        scores = []
        for k in range(splits):
            part = all_probs[k*chunk:(k+1)*chunk]
            kl = part * (np.log(part + 1e-10) - np.log(part.mean(0, keepdims=True) + 1e-10))
            scores.append(float(np.exp(kl.sum(1).mean())))
        metrics['is_mean'] = round(float(np.mean(scores)), 4)
        metrics['is_std'] = round(float(np.std(scores)), 4)
        del inception_is
    except Exception as e:
        print(f"    IS failed: {e}")
        metrics['is_mean'] = float('nan')
        metrics['is_std'] = float('nan')

    del inception
    torch.cuda.empty_cache()
    return metrics


def run_experiment(model, diffusion, vae, method, cfg_scale, seed,
                   num_images, per_layer_weights=None, hybrid_ratio=0.0,
                   num_steps=None, exp_name=None):
    """Run a single experiment: generate, decode, compute metrics."""
    if exp_name is None:
        exp_name = f"{method}_w{cfg_scale}_seed{seed}"
    save_dir = str(RESULTS_DIR / exp_name)

    # Check if already done
    metrics_file = os.path.join(save_dir, 'metrics.json')
    if os.path.exists(metrics_file):
        print(f"  Skipping {exp_name} (already done)")
        with open(metrics_file) as f:
            return json.load(f)

    print(f"\n--- {exp_name} ({num_images} images, {num_steps or NUM_STEPS} steps) ---")

    # Generate latents
    latents, timing = generate_images(
        model, diffusion, method, cfg_scale, seed, num_images,
        per_layer_weights=per_layer_weights,
        hybrid_ratio=hybrid_ratio,
        num_steps=num_steps
    )
    print(f"  Generated in {timing['total_time_sec']:.1f}s ({timing['throughput_img_per_sec']:.2f} img/s)")

    # Decode and save
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, 'images')
    n_saved = decode_and_save_images(vae, latents, img_dir)
    del latents
    torch.cuda.empty_cache()
    print(f"  Decoded and saved {n_saved} images")

    # Compute metrics
    print(f"  Computing metrics...")
    metrics = compute_metrics(img_dir)
    metrics.update(timing)

    # Save metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Clean up images to save disk space (keep metrics)
    shutil.rmtree(img_dir, ignore_errors=True)

    fid_str = f"{metrics.get('fid', 'N/A')}" if not np.isnan(metrics.get('fid', float('nan'))) else "N/A"
    is_str = f"{metrics.get('is_mean', 'N/A')}" if not np.isnan(metrics.get('is_mean', float('nan'))) else "N/A"
    print(f"  FID: {fid_str}, IS: {is_str}")

    return metrics


def run_linearity_analysis(model, diffusion, seed=0):
    """Analyze CSG vs CFG approximation quality across timesteps and guidance scales."""
    print(f"\n{'='*60}")
    print("Running linearity analysis")
    print(f"{'='*60}")

    save_dir = RESULTS_DIR / 'linearity_analysis'
    results_file = save_dir / 'linearity_results.json'
    if results_file.exists():
        print("  Skipping (already done)")
        with open(results_file) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    noise_all, labels_all = get_noise_and_labels(256, LATENT_SIZE, seed, DEVICE)
    batch_size = 64

    timestep_map = diffusion.timestep_map if hasattr(diffusion, 'timestep_map') else list(range(1000))
    test_step_indices = list(range(0, NUM_STEPS, 5)) + [NUM_STEPS - 1]
    test_scales = [1.5, 2.0, 3.0, 4.0, 5.0, 7.5]

    results = {}

    for w in test_scales:
        for step_idx in test_step_indices:
            errors = []
            cosines = []

            for b_start in range(0, 256, batch_size):
                b_end = min(b_start + batch_size, 256)
                x = noise_all[b_start:b_end].to(DEVICE)
                y_cond = labels_all[b_start:b_end].to(DEVICE)
                y_uncond = torch.full_like(y_cond, 1000)

                t_actual = timestep_map[step_idx]
                t_tensor = torch.full((x.shape[0],), t_actual, device=DEVICE, dtype=torch.long)

                with torch.no_grad():
                    out_cfg = forward_cfg(model, x, t_tensor, y_cond, y_uncond, w)
                    out_csg = forward_csg(model, x, t_tensor, y_cond, y_uncond, w)

                    # Use noise prediction channels only
                    if out_cfg.shape[1] == 8:
                        out_cfg = out_cfg[:, :4]
                    if out_csg.shape[1] == 8:
                        out_csg = out_csg[:, :4]

                    diff = (out_csg - out_cfg).flatten(1)
                    norm_cfg = out_cfg.flatten(1).norm(dim=1)
                    rel_error = (diff.norm(dim=1) / (norm_cfg + 1e-8)).cpu().tolist()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        out_csg.flatten(1), out_cfg.flatten(1), dim=1
                    ).cpu().tolist()

                    errors.extend(rel_error)
                    cosines.extend(cos_sim)

                del x, out_cfg, out_csg
                torch.cuda.empty_cache()

            key = f"step{step_idx}_w{w}"
            results[key] = {
                'step_idx': step_idx,
                'guidance_scale': w,
                'mean_relative_error': round(float(np.mean(errors)), 6),
                'std_relative_error': round(float(np.std(errors)), 6),
                'mean_cosine_similarity': round(float(np.mean(cosines)), 6),
                'std_cosine_similarity': round(float(np.std(cosines)), 6),
            }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Linearity analysis saved ({len(results)} configs)")
    return results


def aggregate_results():
    """Aggregate all experimental results into summary."""
    all_results = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        metrics_file = d / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                all_results[d.name] = json.load(f)

    # Group by method+config (remove seed suffix)
    from collections import defaultdict
    groups = defaultdict(list)
    for name, metrics in all_results.items():
        parts = name.rsplit('_seed', 1)
        group_key = parts[0] if len(parts) == 2 else name
        groups[group_key].append(metrics)

    summary = {}
    for group_key, metrics_list in groups.items():
        def safe_stats(values):
            clean = [v for v in values if v is not None and not np.isnan(v)]
            if not clean:
                return float('nan'), 0.0
            return float(np.mean(clean)), float(np.std(clean)) if len(clean) > 1 else 0.0

        fids = [m.get('fid') for m in metrics_list]
        is_means = [m.get('is_mean') for m in metrics_list]
        throughputs = [m.get('throughput_img_per_sec') for m in metrics_list]
        times = [m.get('total_time_sec') for m in metrics_list]

        fid_mean, fid_std = safe_stats(fids)
        is_m, is_s = safe_stats(is_means)
        tp_mean, tp_std = safe_stats(throughputs)
        time_mean, _ = safe_stats(times)

        summary[group_key] = {
            'fid_mean': round(fid_mean, 4),
            'fid_std': round(fid_std, 4),
            'is_mean': round(is_m, 4),
            'is_std': round(is_s, 4),
            'throughput_mean': round(tp_mean, 4),
            'throughput_std': round(tp_std, 4),
            'time_mean_sec': round(time_mean, 2),
            'num_seeds': len(metrics_list),
        }

    # Compute speedups relative to CFG w=4.0
    cfg_key = 'cfg_w4.0'
    if cfg_key in summary and summary[cfg_key]['throughput_mean'] > 0:
        cfg_tp = summary[cfg_key]['throughput_mean']
        for key, stats in summary.items():
            if stats['throughput_mean'] > 0 and not np.isnan(stats['throughput_mean']):
                stats['speedup_vs_cfg'] = round(stats['throughput_mean'] / cfg_tp, 4)

    # Evaluate success criteria
    success = evaluate_success_criteria(summary)

    final_results = {
        'summary': summary,
        'success_criteria': success,
        'config': {
            'core_num_images': CORE_NUM_IMAGES,
            'ablation_num_images': ABLATION_NUM_IMAGES,
            'num_steps': NUM_STEPS,
            'core_seeds': CORE_SEEDS,
            'ablation_seeds': ABLATION_SEEDS,
            'guidance_scales': GUIDANCE_SCALES,
            'model': 'DiT-XL/2',
            'image_size': 256,
        }
    }

    # Save
    with open(WORKSPACE / 'results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print table
    print(f"\n{'='*95}")
    print(f"{'Method':<35} {'FID':>12} {'IS':>12} {'Throughput':>14} {'Speedup':>10}")
    print(f"{'-'*95}")
    for key in sorted(summary.keys()):
        s = summary[key]
        fid_str = f"{s['fid_mean']:.2f}+/-{s['fid_std']:.2f}" if not np.isnan(s['fid_mean']) else "N/A"
        is_str = f"{s['is_mean']:.2f}" if not np.isnan(s['is_mean']) else "N/A"
        tp_str = f"{s['throughput_mean']:.2f}" if not np.isnan(s['throughput_mean']) else "N/A"
        sp_str = f"{s.get('speedup_vs_cfg', float('nan')):.2f}x" if not np.isnan(s.get('speedup_vs_cfg', float('nan'))) else "-"
        print(f"{key:<35} {fid_str:>12} {is_str:>12} {tp_str:>14} {sp_str:>10}")
    print(f"{'='*95}")

    return final_results


def evaluate_success_criteria(summary):
    """Evaluate pre-defined success criteria from idea.json."""
    results = {}

    cfg_w4 = summary.get('cfg_w4.0', {})
    csg_w4 = summary.get('csg_w4.0', {})

    # 1: CSG FID within 10% of CFG at w=4.0
    if not np.isnan(cfg_w4.get('fid_mean', float('nan'))) and not np.isnan(csg_w4.get('fid_mean', float('nan'))):
        ratio = csg_w4['fid_mean'] / cfg_w4['fid_mean'] if cfg_w4['fid_mean'] > 0 else float('nan')
        results['csg_fid_within_10pct_of_cfg'] = {
            'met': ratio <= 1.10 if not np.isnan(ratio) else False,
            'cfg_fid': cfg_w4['fid_mean'],
            'csg_fid': csg_w4['fid_mean'],
            'ratio': round(ratio, 4) if not np.isnan(ratio) else None,
        }

    # 2: CSG speedup >= 1.7x
    if 'speedup_vs_cfg' in csg_w4:
        results['csg_speedup_ge_1_7x'] = {
            'met': csg_w4['speedup_vs_cfg'] >= 1.7,
            'speedup': csg_w4['speedup_vs_cfg'],
        }

    # 3: CSG-H 20% within 5% FID of CFG
    csg_h20 = summary.get('csg_h_20pct', {})
    if not np.isnan(cfg_w4.get('fid_mean', float('nan'))) and not np.isnan(csg_h20.get('fid_mean', float('nan'))):
        ratio = csg_h20['fid_mean'] / cfg_w4['fid_mean'] if cfg_w4['fid_mean'] > 0 else float('nan')
        results['csg_h_20pct_within_5pct_fid'] = {
            'met': ratio <= 1.05 if not np.isnan(ratio) else False,
            'cfg_fid': cfg_w4['fid_mean'],
            'csg_h_fid': csg_h20['fid_mean'],
            'ratio': round(ratio, 4) if not np.isnan(ratio) else None,
        }

    # 4: Per-layer guidance improvement
    pl_fids = {}
    for sched in ['uniform', 'decreasing', 'increasing', 'bell']:
        key = f'csg_pl_{sched}'
        if key in summary and not np.isnan(summary[key]['fid_mean']):
            pl_fids[sched] = summary[key]['fid_mean']

    if pl_fids:
        best = min(pl_fids, key=pl_fids.get)
        results['per_layer_improvement'] = {
            'met': best != 'uniform' and pl_fids[best] < pl_fids.get('uniform', float('inf')),
            'fids': pl_fids,
            'best_schedule': best,
        }

    return results


def main():
    start_time = time.time()
    print("=" * 70)
    print("CSG Experiment Suite - Conditioning-Space Guidance for DiT")
    print("=" * 70)

    # ========================
    # Phase 1: Load model
    # ========================
    print("\n[Phase 1] Loading model and VAE...")
    model = load_dit_model(str(CHECKPOINT_PATH), DEVICE)
    vae = load_vae(DEVICE)
    diffusion = create_diffusion(str(NUM_STEPS))
    print(f"  Model: DiT-XL/2, Device: {DEVICE}, Steps: {NUM_STEPS}")

    all_results = {}

    # ========================
    # Phase 2: Core experiments at w=4.0 (3 seeds, 5K images)
    # ========================
    print(f"\n{'='*70}")
    print(f"[Phase 2] Core experiments (w=4.0, {CORE_NUM_IMAGES} images, {len(CORE_SEEDS)} seeds)")
    print(f"{'='*70}")

    for method in ['cfg', 'no_guidance', 'esg', 'csg']:
        w = 1.0 if method == 'no_guidance' else 4.0
        for seed in CORE_SEEDS:
            name = f"{method}_w{w}_seed{seed}" if method != 'no_guidance' else f"no_guidance_seed{seed}"
            metrics = run_experiment(model, diffusion, vae, method, w, seed,
                                    CORE_NUM_IMAGES, exp_name=name)
            all_results[name] = metrics

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Phase 2 complete. Elapsed: {elapsed:.1f} min")

    # ========================
    # Phase 3: Extra guidance scales (1 seed, 5K images)
    # ========================
    print(f"\n{'='*70}")
    print(f"[Phase 3] Extra guidance scales (w=1.5,7.5, seed=0, {ABLATION_NUM_IMAGES} images)")
    print(f"{'='*70}")

    for method in ['cfg', 'esg', 'csg']:
        for w in [1.5, 7.5]:
            for seed in ABLATION_SEEDS:
                name = f"{method}_w{w}_seed{seed}"
                metrics = run_experiment(model, diffusion, vae, method, w, seed,
                                        ABLATION_NUM_IMAGES, exp_name=name)
                all_results[name] = metrics

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Phase 3 complete. Elapsed: {elapsed:.1f} min")

    # ========================
    # Phase 4: Linearity analysis
    # ========================
    linearity = run_linearity_analysis(model, diffusion, seed=0)

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Phase 4 complete. Elapsed: {elapsed:.1f} min")

    # ========================
    # Phase 5: CSG-PL per-layer guidance ablation (1 seed, 5K images)
    # ========================
    print(f"\n{'='*70}")
    print(f"[Phase 5] CSG-PL per-layer ablation (4 schedules, seed=0)")
    print(f"{'='*70}")

    for schedule in ['uniform', 'decreasing', 'increasing', 'bell']:
        weights = get_per_layer_weights(schedule, num_layers=28, mean_w=4.0)
        for seed in ABLATION_SEEDS:
            name = f"csg_pl_{schedule}_seed{seed}"
            metrics = run_experiment(model, diffusion, vae, 'csg', 4.0, seed,
                                    ABLATION_NUM_IMAGES,
                                    per_layer_weights=weights, exp_name=name)
            all_results[name] = metrics

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Phase 5 complete. Elapsed: {elapsed:.1f} min")

    # ========================
    # Phase 6: CSG-H hybrid (1 seed, 5K images)
    # ========================
    print(f"\n{'='*70}")
    print(f"[Phase 6] CSG-H hybrid (10%,20%,30% CFG steps, seed=0)")
    print(f"{'='*70}")

    for ratio in [0.1, 0.2, 0.3]:
        for seed in ABLATION_SEEDS:
            name = f"csg_h_{int(ratio*100)}pct_seed{seed}"
            metrics = run_experiment(model, diffusion, vae, 'csg', 4.0, seed,
                                    ABLATION_NUM_IMAGES,
                                    hybrid_ratio=ratio, exp_name=name)
            all_results[name] = metrics

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Phase 6 complete. Elapsed: {elapsed:.1f} min")

    # ========================
    # Phase 7: Steps ablation (seed=0, 5K images)
    # ========================
    print(f"\n{'='*70}")
    print(f"[Phase 7] Steps ablation (25,100 steps, seed=0)")
    print(f"{'='*70}")

    for n_steps in [25, 100]:
        for method in ['cfg', 'csg']:
            name = f"steps_{method}_n{n_steps}_seed0"
            metrics = run_experiment(model, diffusion, vae, method, 4.0, 0,
                                    ABLATION_NUM_IMAGES,
                                    num_steps=n_steps, exp_name=name)
            all_results[name] = metrics

    elapsed = (time.time() - start_time) / 60
    print(f"\n  Phase 7 complete. Elapsed: {elapsed:.1f} min")

    # ========================
    # Phase 8: Aggregate
    # ========================
    print(f"\n{'='*70}")
    print("[Phase 8] Aggregating results")
    print(f"{'='*70}")

    final_results = aggregate_results()

    total_time = (time.time() - start_time) / 60
    print(f"\nTotal experiment time: {total_time:.1f} min ({total_time/60:.1f} hours)")
    print("All experiments complete!")


if __name__ == '__main__':
    main()
