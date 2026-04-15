"""
Focused experiment runner - runs only the essential experiments.
Generates 2000 images per config, computes FID and IS.
Saves images in memory (no PNG I/O) and computes features on-the-fly.

Time budget: ~6 hours remaining
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import linalg

WORKSPACE = Path(__file__).parent.parent
EXP_DIR = WORKSPACE / 'exp'
RESULTS_DIR = EXP_DIR / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
CHECKPOINT_PATH = EXP_DIR / 'checkpoints' / 'DiT-XL-2-256x256.pt'
REF_STATS_PATH = EXP_DIR / 'data' / 'VIRTUAL_imagenet256_labeled.npz'
REF_FEATURES_CACHE = EXP_DIR / 'data' / 'imagenet256_ref_features.npz'

sys.path.insert(0, str(EXP_DIR / 'DiT'))
sys.path.insert(0, str(EXP_DIR / 'shared'))

from sampling import (
    load_dit_model, load_vae, sample_images, get_noise_and_labels,
    forward_cfg, forward_csg, forward_esg,
    get_hybrid_steps, get_per_layer_weights
)
from diffusion import create_diffusion

DEVICE = 'cuda'
NUM_IMAGES = 2000
NUM_STEPS = 50
LATENT_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============ Inception feature extraction ============

_inception_model = None
_inception_is_model = None
_ref_mu = None
_ref_sigma = None


def get_inception():
    global _inception_model
    if _inception_model is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        _inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        _inception_model.fc = torch.nn.Identity()
        _inception_model.eval().to(DEVICE)
    return _inception_model


def get_inception_is():
    global _inception_is_model
    if _inception_is_model is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        _inception_is_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        _inception_is_model.eval().to(DEVICE)
    return _inception_is_model


def load_ref_features():
    global _ref_mu, _ref_sigma
    if _ref_mu is not None:
        return _ref_mu, _ref_sigma

    if REF_FEATURES_CACHE.exists():
        data = np.load(str(REF_FEATURES_CACHE))
        _ref_mu, _ref_sigma = data['mu'], data['sigma']
        print(f"  Loaded cached ref features: {_ref_mu.shape}")
        return _ref_mu, _ref_sigma

    print("  Computing reference Inception features (one-time, ~30s)...")
    from torchvision import transforms
    ref_data = np.load(str(REF_STATS_PATH))
    ref_images = ref_data['arr_0']

    inception = get_inception()
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
    features = np.concatenate(all_features)
    _ref_mu = features.mean(axis=0)
    _ref_sigma = np.cov(features, rowvar=False)
    np.savez(str(REF_FEATURES_CACHE), mu=_ref_mu, sigma=_ref_sigma)
    print(f"  Cached ref features: {features.shape}")
    return _ref_mu, _ref_sigma


def compute_fid_is_from_images(vae, latents):
    """Compute FID and IS directly from latents (no PNG saving)."""
    from torchvision import transforms

    inception = get_inception()
    inception_is = get_inception_is()

    resize_transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    all_probs = []
    decode_bs = 50

    for i in range(0, latents.shape[0], decode_bs):
        batch_lat = latents[i:i+decode_bs].to(DEVICE)
        with torch.no_grad():
            images = vae.decode(batch_lat / 0.18215).sample  # [-1, 1]
            # Convert to [0, 1] for Inception
            images_01 = (images + 1) / 2
            images_01 = images_01.clamp(0, 1)

            # Resize for Inception
            inp = resize_transform(images_01)

            # Features for FID
            feats = inception(inp)
            all_features.append(feats.cpu().numpy())

            # Logits for IS
            logits = inception_is(inp)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

        del batch_lat, images, images_01, inp
        torch.cuda.empty_cache()

    features = np.concatenate(all_features)
    all_probs = np.concatenate(all_probs)

    # FID
    gen_mu = features.mean(axis=0)
    gen_sigma = np.cov(features, rowvar=False)
    ref_mu, ref_sigma = load_ref_features()
    diff = gen_mu - ref_mu
    covmean, _ = linalg.sqrtm(gen_sigma @ ref_sigma, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_val = float(diff @ diff + np.trace(gen_sigma + ref_sigma - 2 * covmean))

    # IS
    splits = 10
    chunk = len(all_probs) // splits
    scores = []
    for k in range(splits):
        part = all_probs[k*chunk:(k+1)*chunk]
        kl = part * (np.log(part + 1e-10) - np.log(part.mean(0, keepdims=True) + 1e-10))
        scores.append(float(np.exp(kl.sum(1).mean())))

    return {
        'fid': round(fid_val, 4),
        'is_mean': round(float(np.mean(scores)), 4),
        'is_std': round(float(np.std(scores)), 4),
    }


def run_experiment(model, diffusion, vae, method, cfg_scale, seed,
                   per_layer_weights=None, hybrid_ratio=0.0,
                   num_steps=None, exp_name=None):
    """Run a single experiment: generate latents, compute FID/IS directly."""
    if exp_name is None:
        exp_name = f"{method}_w{cfg_scale}_seed{seed}"
    save_dir = RESULTS_DIR / exp_name
    metrics_file = save_dir / 'metrics.json'

    if metrics_file.exists():
        print(f"  Skipping {exp_name} (done)")
        with open(metrics_file) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n--- {exp_name} ---")

    if num_steps is not None and num_steps != NUM_STEPS:
        diff = create_diffusion(str(num_steps))
    else:
        diff = diffusion

    noise_all, labels_all = get_noise_and_labels(NUM_IMAGES, LATENT_SIZE, seed, DEVICE)

    hybrid_steps = None
    actual_method = method
    if hybrid_ratio > 0:
        n = num_steps or NUM_STEPS
        hybrid_steps = get_hybrid_steps(n, hybrid_ratio, 'middle')
        actual_method = 'csg_hybrid'

    batch_size = 128

    all_latents = []
    total_time = 0.0
    num_batches = (NUM_IMAGES + batch_size - 1) // batch_size

    for b in tqdm(range(num_batches), desc=exp_name, leave=False):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, NUM_IMAGES)

        noise_batch = noise_all[start_idx:end_idx].to(DEVICE)
        labels_batch = labels_all[start_idx:end_idx].to(DEVICE)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        latents = sample_images(
            model, diff, actual_method, noise_batch, labels_batch, cfg_scale,
            device=DEVICE, per_layer_weights=per_layer_weights,
            hybrid_ratio=hybrid_ratio, hybrid_steps=hybrid_steps
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        all_latents.append(latents.cpu())
        del noise_batch, labels_batch, latents
        torch.cuda.empty_cache()

    all_latents = torch.cat(all_latents, dim=0)
    throughput = NUM_IMAGES / total_time

    timing = {
        'total_time_sec': round(total_time, 2),
        'num_images': NUM_IMAGES,
        'throughput_img_per_sec': round(throughput, 4),
        'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        'batch_size': batch_size,
        'num_steps': num_steps or NUM_STEPS,
    }

    print(f"  Generated in {total_time:.1f}s ({throughput:.2f} img/s)")

    # Compute FID/IS directly from latents (skip PNG I/O)
    print(f"  Computing metrics...")
    metrics = compute_fid_is_from_images(vae, all_latents)
    metrics.update(timing)

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    del all_latents
    torch.cuda.empty_cache()

    print(f"  FID: {metrics['fid']:.2f}, IS: {metrics['is_mean']:.1f}, Speedup ref later")
    return metrics


def run_linearity_analysis(model, diffusion):
    """Quick linearity analysis."""
    save_dir = RESULTS_DIR / 'linearity_analysis'
    results_file = save_dir / 'linearity_results.json'
    if results_file.exists():
        print("  Linearity analysis already done")
        with open(results_file) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    print("\n--- Linearity Analysis ---")

    noise_all, labels_all = get_noise_and_labels(256, LATENT_SIZE, 0, DEVICE)
    timestep_map = diffusion.timestep_map
    test_steps = list(range(0, NUM_STEPS, 5)) + [NUM_STEPS - 1]
    test_scales = [1.5, 2.0, 3.0, 4.0, 5.0, 7.5]

    results = {}
    bs = 64

    for w in tqdm(test_scales, desc="linearity"):
        for step_idx in test_steps:
            errors, cosines = [], []
            for b in range(0, 256, bs):
                x = noise_all[b:b+bs].to(DEVICE)
                y_cond = labels_all[b:b+bs].to(DEVICE)
                y_uncond = torch.full_like(y_cond, 1000)
                t_tensor = torch.full((x.shape[0],), timestep_map[step_idx], device=DEVICE, dtype=torch.long)

                with torch.no_grad():
                    out_cfg = forward_cfg(model, x, t_tensor, y_cond, y_uncond, w)
                    out_csg = forward_csg(model, x, t_tensor, y_cond, y_uncond, w)
                    if out_cfg.shape[1] == 8:
                        out_cfg, out_csg = out_cfg[:, :4], out_csg[:, :4]

                    diff = (out_csg - out_cfg).flatten(1)
                    norm = out_cfg.flatten(1).norm(dim=1)
                    errors.extend((diff.norm(dim=1) / (norm + 1e-8)).cpu().tolist())
                    cosines.extend(torch.nn.functional.cosine_similarity(
                        out_csg.flatten(1), out_cfg.flatten(1), dim=1).cpu().tolist())

                del x, out_cfg, out_csg
                torch.cuda.empty_cache()

            results[f"step{step_idx}_w{w}"] = {
                'step_idx': step_idx,
                'guidance_scale': w,
                'mean_relative_error': round(float(np.mean(errors)), 6),
                'std_relative_error': round(float(np.std(errors)), 6),
                'mean_cosine_similarity': round(float(np.mean(cosines)), 6),
            }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Linearity analysis: {len(results)} configs")
    return results


def aggregate_and_save(all_results):
    """Aggregate results, evaluate criteria, save."""
    from collections import defaultdict

    groups = defaultdict(list)
    for name, metrics in all_results.items():
        parts = name.rsplit('_seed', 1)
        group_key = parts[0] if len(parts) == 2 else name
        groups[group_key].append(metrics)

    summary = {}
    for key, mlist in groups.items():
        def stats(vals):
            clean = [v for v in vals if v is not None and not np.isnan(v)]
            return (float(np.mean(clean)), float(np.std(clean)) if len(clean) > 1 else 0.0) if clean else (float('nan'), 0.0)

        fm, fs = stats([m.get('fid') for m in mlist])
        im, ist = stats([m.get('is_mean') for m in mlist])
        tm, ts = stats([m.get('throughput_img_per_sec') for m in mlist])

        summary[key] = {
            'fid_mean': round(fm, 4), 'fid_std': round(fs, 4),
            'is_mean': round(im, 4), 'is_std': round(ist, 4),
            'throughput_mean': round(tm, 4), 'throughput_std': round(ts, 4),
            'num_seeds': len(mlist),
        }

    # Speedups
    cfg_tp = summary.get('cfg_w4.0', {}).get('throughput_mean', 0)
    if cfg_tp > 0:
        for k, s in summary.items():
            if s['throughput_mean'] > 0 and not np.isnan(s['throughput_mean']):
                s['speedup_vs_cfg'] = round(s['throughput_mean'] / cfg_tp, 4)

    # Success criteria
    success = {}
    cfg_w4 = summary.get('cfg_w4.0', {})
    csg_w4 = summary.get('csg_w4.0', {})

    if not np.isnan(cfg_w4.get('fid_mean', float('nan'))) and not np.isnan(csg_w4.get('fid_mean', float('nan'))):
        ratio = csg_w4['fid_mean'] / cfg_w4['fid_mean']
        success['csg_fid_within_10pct'] = {
            'met': ratio <= 1.10,
            'cfg_fid': cfg_w4['fid_mean'], 'csg_fid': csg_w4['fid_mean'], 'ratio': round(ratio, 4)
        }

    if 'speedup_vs_cfg' in csg_w4:
        success['csg_speedup_ge_1_7x'] = {
            'met': csg_w4['speedup_vs_cfg'] >= 1.7,
            'speedup': csg_w4['speedup_vs_cfg']
        }

    csg_h20 = summary.get('csg_h_20pct', {})
    if not np.isnan(cfg_w4.get('fid_mean', float('nan'))) and not np.isnan(csg_h20.get('fid_mean', float('nan'))):
        ratio = csg_h20['fid_mean'] / cfg_w4['fid_mean']
        success['csg_h_20pct_within_5pct'] = {
            'met': ratio <= 1.05,
            'cfg_fid': cfg_w4['fid_mean'], 'csg_h_fid': csg_h20['fid_mean'], 'ratio': round(ratio, 4)
        }

    pl_fids = {}
    for sched in ['uniform', 'decreasing', 'increasing', 'bell']:
        k = f'csg_pl_{sched}'
        if k in summary and not np.isnan(summary[k]['fid_mean']):
            pl_fids[sched] = summary[k]['fid_mean']
    if pl_fids:
        best = min(pl_fids, key=pl_fids.get)
        success['per_layer_improvement'] = {
            'met': best != 'uniform' and pl_fids.get(best, float('inf')) < pl_fids.get('uniform', float('inf')),
            'fids': pl_fids, 'best_schedule': best
        }

    final = {
        'summary': summary,
        'success_criteria': success,
        'config': {
            'num_images': NUM_IMAGES, 'num_steps': NUM_STEPS,
            'model': 'DiT-XL/2', 'image_size': 256,
        }
    }

    with open(WORKSPACE / 'results.json', 'w') as f:
        json.dump(final, f, indent=2)
    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(final, f, indent=2)

    # Print table
    print(f"\n{'='*90}")
    print(f"{'Method':<30} {'FID':>12} {'IS':>10} {'Throughput':>12} {'Speedup':>8}")
    print(f"{'-'*90}")
    for k in sorted(summary.keys()):
        s = summary[k]
        fid_s = f"{s['fid_mean']:.1f}±{s['fid_std']:.1f}" if not np.isnan(s['fid_mean']) else "N/A"
        is_s = f"{s['is_mean']:.1f}" if not np.isnan(s['is_mean']) else "N/A"
        tp_s = f"{s['throughput_mean']:.2f}" if not np.isnan(s['throughput_mean']) else "N/A"
        sp_s = f"{s.get('speedup_vs_cfg', float('nan')):.2f}x" if not np.isnan(s.get('speedup_vs_cfg', float('nan'))) else "-"
        print(f"{k:<30} {fid_s:>12} {is_s:>10} {tp_s:>12} {sp_s:>8}")
    print(f"{'='*90}")

    return final


def main():
    start_time = time.time()
    print("=" * 60)
    print("CSG Focused Experiments")
    print("=" * 60)

    model = load_dit_model(str(CHECKPOINT_PATH), DEVICE)
    vae = load_vae(DEVICE)
    diffusion = create_diffusion(str(NUM_STEPS))

    # Pre-load ref features
    load_ref_features()

    all_results = {}

    # ===== CORE: CFG vs CSG vs baselines at w=4.0, 3 seeds =====
    print(f"\n{'='*60}")
    print("[Core] CFG, CSG, ESG, No-guidance at w=4.0 (3 seeds)")
    print(f"{'='*60}")

    for method in ['cfg', 'csg', 'esg', 'no_guidance']:
        w = 1.0 if method == 'no_guidance' else 4.0
        for seed in [0, 1, 2]:
            name = f"{method}_w{w}_seed{seed}" if method != 'no_guidance' else f"no_guidance_seed{seed}"
            m = run_experiment(model, diffusion, vae, method, w, seed, exp_name=name)
            all_results[name] = m

    elapsed = (time.time() - start_time) / 60
    print(f"\nCore done. Elapsed: {elapsed:.0f} min")

    # ===== Extra scales (1 seed) =====
    print(f"\n{'='*60}")
    print("[Scales] w=1.5 and w=7.5 (seed=0)")
    print(f"{'='*60}")

    for method in ['cfg', 'csg', 'esg']:
        for w in [1.5, 7.5]:
            name = f"{method}_w{w}_seed0"
            m = run_experiment(model, diffusion, vae, method, w, 0, exp_name=name)
            all_results[name] = m

    elapsed = (time.time() - start_time) / 60
    print(f"\nScales done. Elapsed: {elapsed:.0f} min")

    # ===== Linearity analysis =====
    run_linearity_analysis(model, diffusion)
    elapsed = (time.time() - start_time) / 60
    print(f"\nLinearity done. Elapsed: {elapsed:.0f} min")

    # ===== CSG-PL (1 seed) =====
    print(f"\n{'='*60}")
    print("[CSG-PL] Per-layer ablation (seed=0)")
    print(f"{'='*60}")

    for sched in ['uniform', 'decreasing', 'increasing', 'bell']:
        weights = get_per_layer_weights(sched, 28, 4.0)
        name = f"csg_pl_{sched}_seed0"
        m = run_experiment(model, diffusion, vae, 'csg', 4.0, 0,
                           per_layer_weights=weights, exp_name=name)
        all_results[name] = m

    elapsed = (time.time() - start_time) / 60
    print(f"\nCSG-PL done. Elapsed: {elapsed:.0f} min")

    # ===== CSG-H (1 seed) =====
    print(f"\n{'='*60}")
    print("[CSG-H] Hybrid (seed=0)")
    print(f"{'='*60}")

    for ratio in [0.1, 0.2, 0.3]:
        name = f"csg_h_{int(ratio*100)}pct_seed0"
        m = run_experiment(model, diffusion, vae, 'csg', 4.0, 0,
                           hybrid_ratio=ratio, exp_name=name)
        all_results[name] = m

    elapsed = (time.time() - start_time) / 60
    print(f"\nCSG-H done. Elapsed: {elapsed:.0f} min")

    # ===== Steps ablation (1 seed) =====
    remaining_hours = 8 - elapsed / 60
    if remaining_hours > 1.5:
        print(f"\n{'='*60}")
        print("[Steps] Steps ablation (seed=0)")
        print(f"{'='*60}")

        for n_steps in [25, 100]:
            for method in ['cfg', 'csg']:
                name = f"steps_{method}_n{n_steps}_seed0"
                m = run_experiment(model, diffusion, vae, method, 4.0, 0,
                                   num_steps=n_steps, exp_name=name)
                all_results[name] = m
    else:
        print(f"\nSkipping steps ablation ({remaining_hours:.1f}h remaining)")

    # ===== Aggregate =====
    print(f"\n{'='*60}")
    print("Aggregating results")
    print(f"{'='*60}")
    aggregate_and_save(all_results)

    total = (time.time() - start_time) / 60
    print(f"\nTotal time: {total:.0f} min ({total/60:.1f} hours)")


if __name__ == '__main__':
    main()
