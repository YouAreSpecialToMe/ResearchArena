"""
Focused critical experiments only:
1. CSG-H 30%, 50% early_clean (seed 0) - can hybrid recover quality?
2. Multi-seed w=1.5 for CFG, CSG, ESG (seeds 1,2) - where CSG works
3. CSG-PL increasing (seed 0) - complete ablation
4. CSG-H 30% early_clean multi-seed (seeds 1,2)
Then aggregate + figures.
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
    get_hybrid_steps, get_per_layer_weights
)
from diffusion import create_diffusion

DEVICE = 'cuda'
NUM_IMAGES = 2000
NUM_STEPS = 50
LATENT_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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
    data = np.load(str(REF_FEATURES_CACHE))
    _ref_mu, _ref_sigma = data['mu'], data['sigma']
    return _ref_mu, _ref_sigma


def compute_fid_is(vae, latents):
    from torchvision import transforms
    inception = get_inception()
    inception_is = get_inception_is()
    resize = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    all_features, all_probs = [], []
    for i in range(0, latents.shape[0], 50):
        batch = latents[i:i+50].to(DEVICE)
        with torch.no_grad():
            imgs = vae.decode(batch / 0.18215).sample
            imgs = ((imgs + 1) / 2).clamp(0, 1)
            inp = resize(imgs)
            all_features.append(inception(inp).cpu().numpy())
            all_probs.append(torch.nn.functional.softmax(inception_is(inp), dim=1).cpu().numpy())
        del batch, imgs, inp
        torch.cuda.empty_cache()

    features = np.concatenate(all_features)
    probs = np.concatenate(all_probs)
    gen_mu, gen_sigma = features.mean(0), np.cov(features, rowvar=False)
    ref_mu, ref_sigma = load_ref_features()
    diff = gen_mu - ref_mu
    covmean, _ = linalg.sqrtm(gen_sigma @ ref_sigma, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(gen_sigma + ref_sigma - 2 * covmean))

    scores = []
    chunk = len(probs) // 10
    for k in range(10):
        p = probs[k*chunk:(k+1)*chunk]
        kl = p * (np.log(p + 1e-10) - np.log(p.mean(0, keepdims=True) + 1e-10))
        scores.append(float(np.exp(kl.sum(1).mean())))

    return {'fid': round(fid, 4), 'is_mean': round(float(np.mean(scores)), 4),
            'is_std': round(float(np.std(scores)), 4)}


def run_exp(model, diffusion, vae, method, cfg_scale, seed,
            per_layer_weights=None, hybrid_ratio=0.0, hybrid_position='middle',
            exp_name=None):
    if exp_name is None:
        exp_name = f"{method}_w{cfg_scale}_seed{seed}"
    save_dir = RESULTS_DIR / exp_name
    metrics_file = save_dir / 'metrics.json'

    if metrics_file.exists():
        print(f"  Skip {exp_name}")
        with open(metrics_file) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n--- {exp_name} ---")

    noise, labels = get_noise_and_labels(NUM_IMAGES, LATENT_SIZE, seed, DEVICE)

    hybrid_steps = None
    actual_method = method
    if hybrid_ratio > 0:
        hybrid_steps = get_hybrid_steps(NUM_STEPS, hybrid_ratio, hybrid_position)
        actual_method = 'csg_hybrid'
        print(f"  Hybrid: {len(hybrid_steps)}/{NUM_STEPS} full-CFG steps ({hybrid_position})")

    bs = 128
    all_lat = []
    t_total = 0.0

    for b in tqdm(range(0, NUM_IMAGES, bs), desc=exp_name, leave=False):
        n_batch = noise[b:b+bs].to(DEVICE)
        l_batch = labels[b:b+bs].to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lat = sample_images(model, diffusion, actual_method, n_batch, l_batch, cfg_scale,
                           device=DEVICE, per_layer_weights=per_layer_weights,
                           hybrid_ratio=hybrid_ratio, hybrid_steps=hybrid_steps)
        torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        all_lat.append(lat.cpu())
        del n_batch, l_batch, lat
        torch.cuda.empty_cache()

    all_lat = torch.cat(all_lat)
    tp = NUM_IMAGES / t_total
    print(f"  {t_total:.0f}s, {tp:.2f} img/s")

    metrics = compute_fid_is(vae, all_lat)
    metrics.update({
        'total_time_sec': round(t_total, 2), 'num_images': NUM_IMAGES,
        'throughput_img_per_sec': round(tp, 4),
        'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        'batch_size': bs, 'num_steps': NUM_STEPS,
    })

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    del all_lat
    torch.cuda.empty_cache()
    print(f"  FID={metrics['fid']:.1f} IS={metrics['is_mean']:.1f}")
    return metrics


def main():
    t0 = time.time()
    print("=" * 60)
    print("Critical experiments")
    print("=" * 60)

    model = load_dit_model(str(CHECKPOINT_PATH), DEVICE)
    vae = load_vae(DEVICE)
    diffusion = create_diffusion(str(NUM_STEPS))
    load_ref_features()

    R = {}  # results
    # Load existing
    for d in RESULTS_DIR.iterdir():
        mf = d / 'metrics.json'
        if mf.exists():
            with open(mf) as f:
                R[d.name] = json.load(f)
    print(f"Loaded {len(R)} existing results")

    # 1. CSG-H 30% and 50% early_clean (seed 0) - CRITICAL
    print("\n[1] Hybrid 30%/50% early_clean")
    for ratio in [0.3, 0.5]:
        name = f"csg_h_{int(ratio*100)}pct_early_clean_seed0"
        R[name] = run_exp(model, diffusion, vae, 'csg', 4.0, 0,
                          hybrid_ratio=ratio, hybrid_position='early_clean', exp_name=name)

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.0f} min")

    # 2. CSG-H 10% early_noisy for comparison
    print("\n[2] Hybrid 10% early_noisy")
    name = "csg_h_10pct_early_noisy_seed0"
    R[name] = run_exp(model, diffusion, vae, 'csg', 4.0, 0,
                      hybrid_ratio=0.1, hybrid_position='early_noisy', exp_name=name)

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.0f} min")

    # 3. Multi-seed w=1.5 (where CSG works)
    print("\n[3] Multi-seed w=1.5")
    for method in ['cfg', 'csg', 'esg']:
        for seed in [1, 2]:
            name = f"{method}_w1.5_seed{seed}"
            R[name] = run_exp(model, diffusion, vae, method, 1.5, seed, exp_name=name)

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.0f} min")

    # 4. CSG-PL increasing (complete ablation)
    print("\n[4] CSG-PL increasing")
    weights = get_per_layer_weights('increasing', 28, 4.0)
    name = "csg_pl_increasing_seed0"
    R[name] = run_exp(model, diffusion, vae, 'csg', 4.0, 0,
                      per_layer_weights=weights, exp_name=name)

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.0f} min")

    # 5. If time remains: CSG-H 30% early_clean multi-seed
    remaining_h = 8 - elapsed / 60
    if remaining_h > 1.5:
        print("\n[5] Multi-seed hybrid 30% early_clean")
        for seed in [1, 2]:
            name = f"csg_h_30pct_early_clean_seed{seed}"
            R[name] = run_exp(model, diffusion, vae, 'csg', 4.0, seed,
                              hybrid_ratio=0.3, hybrid_position='early_clean', exp_name=name)

    # 6. If time remains: steps ablation
    remaining_h = 8 - (time.time() - t0) / 3600
    if remaining_h > 1.0:
        print("\n[6] Steps ablation")
        for n_steps in [25]:
            for method in ['cfg', 'csg']:
                name = f"steps_{method}_n{n_steps}_seed0"
                diff2 = create_diffusion(str(n_steps))
                R[name] = run_exp(model, diff2, vae, method, 4.0, 0, exp_name=name)

    total = (time.time() - t0) / 60
    print(f"\nTotal: {total:.0f} min")
    print(f"Total results: {len(R)}")


if __name__ == '__main__':
    main()
