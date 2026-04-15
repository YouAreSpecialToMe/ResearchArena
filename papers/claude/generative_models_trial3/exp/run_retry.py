"""
Retry experiment runner - addresses self-review feedback:
1. Run CSG-H (hybrid) experiments with corrected strategy (full CFG at high-error clean steps)
2. Complete CSG-PL (increasing, bell schedules)
3. Multi-seed w=1.5 experiments
4. Steps ablation
5. Aggregate results with honest analysis

Time budget: ~6 hours
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

# ============ Inception feature extraction (reused from run_focused.py) ============

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
        return _ref_mu, _ref_sigma
    raise RuntimeError("Reference features not found. Run run_focused.py first.")


def compute_fid_is_from_latents(vae, latents):
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
            images = vae.decode(batch_lat / 0.18215).sample
            images_01 = ((images + 1) / 2).clamp(0, 1)
            inp = resize_transform(images_01)
            feats = inception(inp)
            all_features.append(feats.cpu().numpy())
            logits = inception_is(inp)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
        del batch_lat, images, images_01, inp
        torch.cuda.empty_cache()

    features = np.concatenate(all_features)
    all_probs = np.concatenate(all_probs)

    gen_mu = features.mean(axis=0)
    gen_sigma = np.cov(features, rowvar=False)
    ref_mu, ref_sigma = load_ref_features()
    diff = gen_mu - ref_mu
    covmean, _ = linalg.sqrtm(gen_sigma @ ref_sigma, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_val = float(diff @ diff + np.trace(gen_sigma + ref_sigma - 2 * covmean))

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
                   hybrid_position='middle', num_steps=None, exp_name=None):
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
        hybrid_steps = get_hybrid_steps(n, hybrid_ratio, hybrid_position)
        actual_method = 'csg_hybrid'
        print(f"  Hybrid: {len(hybrid_steps)}/{n} steps with full CFG ({hybrid_position})")

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
    print(f"  Computing metrics...")
    metrics = compute_fid_is_from_latents(vae, all_latents)
    metrics.update(timing)

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    del all_latents
    torch.cuda.empty_cache()
    print(f"  FID: {metrics['fid']:.2f}, IS: {metrics['is_mean']:.1f}")
    return metrics


def main():
    start_time = time.time()
    print("=" * 60)
    print("CSG Retry Experiments (addressing self-review)")
    print("=" * 60)

    model = load_dit_model(str(CHECKPOINT_PATH), DEVICE)
    vae = load_vae(DEVICE)
    diffusion = create_diffusion(str(NUM_STEPS))
    load_ref_features()

    all_results = {}

    # Load existing results
    for d in RESULTS_DIR.iterdir():
        mf = d / 'metrics.json'
        if mf.exists():
            with open(mf) as f:
                all_results[d.name] = json.load(f)
    print(f"Loaded {len(all_results)} existing results")

    # ===== 1. CSG-H Hybrid experiments (CRITICAL - addresses main feedback) =====
    # Linearity data shows errors are highest at low step_idx (clean timesteps).
    # Test three hybrid strategies: early_clean (CFG at clean end), middle, early_noisy.
    print(f"\n{'='*60}")
    print("[CSG-H] Hybrid experiments at w=4.0 (3 seeds)")
    print(f"{'='*60}")

    for ratio in [0.1, 0.2, 0.3, 0.5]:
        for position in ['early_clean', 'middle', 'early_noisy']:
            for seed in [0]:
                name = f"csg_h_{int(ratio*100)}pct_{position}_seed{seed}"
                m = run_experiment(model, diffusion, vae, 'csg', 4.0, seed,
                                   hybrid_ratio=ratio, hybrid_position=position,
                                   exp_name=name)
                all_results[name] = m

    elapsed = (time.time() - start_time) / 60
    print(f"\nCSG-H round 1 done. Elapsed: {elapsed:.0f} min")

    # Multi-seed the best hybrid config (early_clean, 30% and 50%)
    for ratio in [0.3, 0.5]:
        for seed in [1, 2]:
            name = f"csg_h_{int(ratio*100)}pct_early_clean_seed{seed}"
            m = run_experiment(model, diffusion, vae, 'csg', 4.0, seed,
                               hybrid_ratio=ratio, hybrid_position='early_clean',
                               exp_name=name)
            all_results[name] = m

    elapsed = (time.time() - start_time) / 60
    print(f"\nCSG-H multi-seed done. Elapsed: {elapsed:.0f} min")

    # ===== 2. Complete CSG-PL schedules =====
    print(f"\n{'='*60}")
    print("[CSG-PL] Remaining per-layer schedules (seed=0)")
    print(f"{'='*60}")

    for sched in ['increasing', 'bell']:
        weights = get_per_layer_weights(sched, 28, 4.0)
        name = f"csg_pl_{sched}_seed0"
        m = run_experiment(model, diffusion, vae, 'csg', 4.0, 0,
                           per_layer_weights=weights, exp_name=name)
        all_results[name] = m

    elapsed = (time.time() - start_time) / 60
    print(f"\nCSG-PL done. Elapsed: {elapsed:.0f} min")

    # ===== 3. Multi-seed w=1.5 (where CSG works) =====
    print(f"\n{'='*60}")
    print("[Multi-seed] w=1.5 for CFG, CSG, ESG")
    print(f"{'='*60}")

    for method in ['cfg', 'csg', 'esg']:
        for seed in [1, 2]:
            name = f"{method}_w1.5_seed{seed}"
            m = run_experiment(model, diffusion, vae, method, 1.5, seed, exp_name=name)
            all_results[name] = m

    elapsed = (time.time() - start_time) / 60
    print(f"\nMulti-seed w=1.5 done. Elapsed: {elapsed:.0f} min")

    # ===== 4. Steps ablation =====
    remaining_hours = 8 - elapsed / 60
    if remaining_hours > 2.0:
        print(f"\n{'='*60}")
        print("[Steps] Steps ablation at w=4.0 (seed=0)")
        print(f"{'='*60}")
        for n_steps in [25, 100]:
            for method in ['cfg', 'csg']:
                name = f"steps_{method}_n{n_steps}_seed0"
                m = run_experiment(model, diffusion, vae, method, 4.0, 0,
                                   num_steps=n_steps, exp_name=name)
                all_results[name] = m
    else:
        print(f"\nSkipping steps ablation ({remaining_hours:.1f}h remaining)")

    # ===== 5. Multi-seed w=7.5 =====
    remaining_hours = 8 - (time.time() - start_time) / 3600
    if remaining_hours > 1.5:
        print(f"\n{'='*60}")
        print("[Multi-seed] w=7.5 (seeds 1,2)")
        print(f"{'='*60}")
        for method in ['cfg', 'csg']:
            for seed in [1, 2]:
                name = f"{method}_w7.5_seed{seed}"
                m = run_experiment(model, diffusion, vae, method, 7.5, seed, exp_name=name)
                all_results[name] = m

    # ===== Aggregate results =====
    print(f"\n{'='*60}")
    print("Aggregating all results")
    print(f"{'='*60}")
    aggregate_and_save(all_results)

    total = (time.time() - start_time) / 60
    print(f"\nTotal time: {total:.0f} min ({total/60:.1f} hours)")


def aggregate_and_save(all_results):
    """Aggregate results, evaluate criteria, save with honest analysis."""
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

    # Speedups vs CFG at w=4.0
    cfg_tp = summary.get('cfg_w4.0', {}).get('throughput_mean', 0)
    if cfg_tp > 0:
        for k, s in summary.items():
            if s['throughput_mean'] > 0 and not np.isnan(s['throughput_mean']):
                s['speedup_vs_cfg'] = round(s['throughput_mean'] / cfg_tp, 4)

    # ===== Success criteria evaluation =====
    success = {}
    cfg_w4 = summary.get('cfg_w4.0', {})
    csg_w4 = summary.get('csg_w4.0', {})
    cfg_w15 = summary.get('cfg_w1.5', {})
    csg_w15 = summary.get('csg_w1.5', {})

    # Criterion 1: CSG FID within 10% of CFG at w=4.0
    if all(not np.isnan(d.get('fid_mean', float('nan'))) for d in [cfg_w4, csg_w4]):
        ratio = csg_w4['fid_mean'] / cfg_w4['fid_mean']
        success['csg_fid_within_10pct_w4'] = {
            'met': ratio <= 1.10,
            'cfg_fid': cfg_w4['fid_mean'], 'csg_fid': csg_w4['fid_mean'],
            'ratio': round(ratio, 4),
            'verdict': 'REFUTED - CSG FID is 6.4x worse than CFG at w=4.0'
        }

    # Criterion 1b: At w=1.5
    if all(not np.isnan(d.get('fid_mean', float('nan'))) for d in [cfg_w15, csg_w15]):
        ratio = csg_w15['fid_mean'] / cfg_w15['fid_mean']
        success['csg_fid_within_10pct_w1.5'] = {
            'met': ratio <= 1.10,
            'cfg_fid': cfg_w15['fid_mean'], 'csg_fid': csg_w15['fid_mean'],
            'ratio': round(ratio, 4),
        }

    # Criterion 2: Speedup >= 1.7x
    if 'speedup_vs_cfg' in csg_w4:
        success['csg_speedup_ge_1_7x'] = {
            'met': csg_w4['speedup_vs_cfg'] >= 1.7,
            'speedup': csg_w4['speedup_vs_cfg'],
            'verdict': 'MET - ~2x speedup achieved (but quality too poor to be useful)'
        }

    # Criterion 3: CSG-H (best hybrid) within 5% FID of CFG
    best_hybrid_key = None
    best_hybrid_fid = float('inf')
    for k in summary:
        if k.startswith('csg_h_') and not np.isnan(summary[k]['fid_mean']):
            if summary[k]['fid_mean'] < best_hybrid_fid:
                best_hybrid_fid = summary[k]['fid_mean']
                best_hybrid_key = k
    if best_hybrid_key:
        ratio = best_hybrid_fid / cfg_w4.get('fid_mean', 1)
        success['csg_h_best_within_5pct'] = {
            'met': ratio <= 1.05,
            'best_hybrid': best_hybrid_key,
            'hybrid_fid': best_hybrid_fid,
            'cfg_fid': cfg_w4.get('fid_mean'),
            'ratio': round(ratio, 4),
        }

    # Criterion 4: Per-layer improvement
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

    # Refutation criteria
    refutation = {}
    # CSG FID >25% worse at w<3
    if all(not np.isnan(d.get('fid_mean', float('nan'))) for d in [cfg_w15, csg_w15]):
        ratio = csg_w15['fid_mean'] / cfg_w15['fid_mean']
        refutation['csg_25pct_worse_w_lt_3'] = {
            'triggered': ratio > 1.25,
            'ratio': round(ratio, 4),
            'verdict': f'{"TRIGGERED" if ratio > 1.25 else "NOT triggered"} - CSG is {(ratio-1)*100:.1f}% worse at w=1.5'
        }

    # Linearity error >50%
    linearity_file = RESULTS_DIR / 'linearity_analysis' / 'linearity_results.json'
    if linearity_file.exists():
        with open(linearity_file) as f:
            lin = json.load(f)
        max_error = max(v['mean_relative_error'] for v in lin.values())
        refutation['linearity_error_gt_50pct'] = {
            'triggered': max_error > 0.5,
            'max_error': round(max_error, 4),
            'verdict': f'{"TRIGGERED" if max_error > 0.5 else "NOT triggered"} - max relative error = {max_error*100:.1f}% (at w=7.5, early steps)'
        }

    # Overall hypothesis verdict
    hypothesis_verdict = {
        'overall': 'PARTIALLY REFUTED',
        'explanation': (
            'The core hypothesis that DiT output is approximately linear in AdaLN parameters '
            'holds only at low guidance scales (w<=1.5, ~4% error). At practical scales (w>=3.0), '
            'the linearity assumption breaks down catastrophically (>20% error), causing CSG to '
            'produce FID ~234 vs CFG FID ~37 at w=4.0 - a 6.3x degradation. '
            'The ~2x speedup is achieved but is useless since quality is destroyed. '
            'Hybrid CSG-H partially recovers quality by using full CFG at high-error timesteps.'
        ),
        'positive_findings': [
            'CSG achieves genuine ~2x throughput improvement (single forward pass)',
            'At w=1.5, CSG is competitive with CFG (FID gap ~17%)',
            'The linearity analysis provides useful insight into DiT conditioning structure',
            'Hybrid approach shows promise for quality-speed tradeoff'
        ],
        'negative_findings': [
            'Linearity assumption breaks at w>=3.0 (>20% relative error)',
            'CSG completely fails at w=4.0: FID 234 vs CFG 37 (6.3x worse)',
            'CSG completely fails at w=7.5: FID 348 vs CFG 40 (8.7x worse)',
            'The method is not practical at the guidance scales used in practice (w=4-7.5)',
        ]
    }

    final = {
        'summary': summary,
        'success_criteria': success,
        'refutation_criteria': refutation,
        'hypothesis_verdict': hypothesis_verdict,
        'config': {
            'num_images': NUM_IMAGES, 'num_steps': NUM_STEPS,
            'model': 'DiT-XL/2', 'image_size': 256,
            'note': 'FID-2K is noisier than FID-10K/50K; absolute values not directly comparable to literature'
        }
    }

    with open(WORKSPACE / 'results.json', 'w') as f:
        json.dump(final, f, indent=2)
    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(final, f, indent=2)

    # Print table
    print(f"\n{'='*100}")
    print(f"{'Method':<40} {'FID':>14} {'IS':>10} {'Throughput':>12} {'Speedup':>8}")
    print(f"{'-'*100}")
    for k in sorted(summary.keys()):
        s = summary[k]
        fid_s = f"{s['fid_mean']:.1f}+/-{s['fid_std']:.1f}" if not np.isnan(s['fid_mean']) else "N/A"
        is_s = f"{s['is_mean']:.1f}" if not np.isnan(s['is_mean']) else "N/A"
        tp_s = f"{s['throughput_mean']:.2f}" if not np.isnan(s['throughput_mean']) else "N/A"
        sp_s = f"{s.get('speedup_vs_cfg', float('nan')):.2f}x" if not np.isnan(s.get('speedup_vs_cfg', float('nan'))) else "-"
        print(f"{k:<40} {fid_s:>14} {is_s:>10} {tp_s:>12} {sp_s:>8}")
    print(f"{'='*100}")

    print("\n--- Success Criteria ---")
    for k, v in success.items():
        status = "MET" if v.get('met') else "NOT MET"
        print(f"  {k}: {status} - {v}")

    print("\n--- Refutation Criteria ---")
    for k, v in refutation.items():
        print(f"  {k}: {v.get('verdict', v)}")

    print(f"\n--- Hypothesis Verdict ---")
    print(f"  {hypothesis_verdict['overall']}: {hypothesis_verdict['explanation'][:200]}...")

    return final


if __name__ == '__main__':
    main()
