"""
Run multi-seed experiments for CSG-H 50% early_clean and generate qualitative images.
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
from PIL import Image

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
    get_hybrid_steps
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
            hybrid_ratio=0.0, hybrid_position='middle', exp_name=None):
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
                           device=DEVICE, hybrid_ratio=hybrid_ratio, hybrid_steps=hybrid_steps)
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


def generate_qualitative_grid(model, diffusion, vae):
    """Generate a qualitative comparison grid: CFG vs CSG at w=1.5 and w=4.0."""
    print("\n=== Generating qualitative comparison grid ===")

    # 8 diverse ImageNet classes
    class_indices = [1, 497, 980, 963, 207, 817, 323, 967]
    class_names = ['goldfish', 'church', 'volcano', 'pizza', 'golden_retriever',
                   'sports_car', 'monarch_butterfly', 'espresso']

    seed = 42
    torch.manual_seed(seed)
    noise = torch.randn(len(class_indices), 4, LATENT_SIZE, LATENT_SIZE).to(DEVICE)
    labels = torch.tensor(class_indices, device=DEVICE)

    methods_configs = [
        ('cfg', 1.5, 'CFG w=1.5'),
        ('csg', 1.5, 'CSG w=1.5'),
        ('cfg', 4.0, 'CFG w=4.0'),
        ('csg', 4.0, 'CSG w=4.0'),
    ]

    all_images = {}
    for method, w, label in methods_configs:
        print(f"  Generating {label}...")
        with torch.no_grad():
            latents = sample_images(model, diffusion, method, noise, labels, w, device=DEVICE)
            imgs = vae.decode(latents / 0.18215).sample
            imgs = ((imgs + 1) / 2).clamp(0, 1)
            all_images[label] = imgs.cpu()
        torch.cuda.empty_cache()

    # Create grid: 4 rows (methods) x 8 cols (classes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nrows = len(methods_configs)
    ncols = len(class_indices)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.5))

    for r, (method, w, label) in enumerate(methods_configs):
        for c in range(ncols):
            img = all_images[label][c].permute(1, 2, 0).numpy()
            axes[r, c].imshow(img)
            axes[r, c].axis('off')
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=11, rotation=0, labelpad=80, va='center')
            if r == 0:
                axes[r, c].set_title(class_names[c], fontsize=10)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'figure6_qualitative.pdf'), dpi=150, bbox_inches='tight')
    fig.savefig(str(FIGURES_DIR / 'figure6_qualitative.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved qualitative grid to figures/figure6_qualitative.pdf")


def controlled_throughput_measurement(model, diffusion, vae):
    """Controlled throughput measurement with warmup."""
    print("\n=== Controlled throughput measurement ===")

    seed = 0
    noise, labels = get_noise_and_labels(256, LATENT_SIZE, seed, DEVICE)

    results = {}
    for method in ['cfg', 'csg']:
        for w in [1.5, 4.0]:
            label = f"{method}_w{w}"
            # Warmup: 1 batch
            n_batch = noise[:64].to(DEVICE)
            l_batch = labels[:64].to(DEVICE)
            with torch.no_grad():
                _ = sample_images(model, diffusion, method, n_batch, l_batch, w, device=DEVICE)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Timed: 3 runs of 128 images each
            times = []
            for run in range(3):
                n_batch = noise[:128].to(DEVICE)
                l_batch = labels[:128].to(DEVICE)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = sample_images(model, diffusion, method, n_batch, l_batch, w, device=DEVICE)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(128.0 / (t1 - t0))
                del n_batch, l_batch
                torch.cuda.empty_cache()

            results[label] = {
                'throughput_runs': [round(t, 4) for t in times],
                'throughput_mean': round(np.mean(times), 4),
                'throughput_std': round(np.std(times), 4),
            }
            print(f"  {label}: {np.mean(times):.2f} +/- {np.std(times):.2f} img/s")

    # Compute speedup ratios
    for w in [1.5, 4.0]:
        cfg_tp = results[f'cfg_w{w}']['throughput_mean']
        csg_tp = results[f'csg_w{w}']['throughput_mean']
        results[f'speedup_w{w}'] = round(csg_tp / cfg_tp, 4)
        print(f"  Speedup at w={w}: {csg_tp / cfg_tp:.2f}x")

    with open(RESULTS_DIR / 'controlled_throughput.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved to results/controlled_throughput.json")
    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("Multi-seed + qualitative experiments")
    print("=" * 60)

    model = load_dit_model(str(CHECKPOINT_PATH), DEVICE)
    vae = load_vae(DEVICE)
    diffusion = create_diffusion(str(NUM_STEPS))
    load_ref_features()

    # 1. Multi-seed CSG-H 50% early_clean (seeds 1, 2)
    print("\n[1] CSG-H 50% early_clean multi-seed")
    for seed in [1, 2]:
        name = f"csg_h_50pct_early_clean_seed{seed}"
        run_exp(model, diffusion, vae, 'csg', 4.0, seed,
                hybrid_ratio=0.5, hybrid_position='early_clean', exp_name=name)

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.0f} min")

    # 2. Multi-seed CFG w=7.5 (seeds 1, 2)
    print("\n[2] CFG w=7.5 multi-seed")
    for seed in [1, 2]:
        name = f"cfg_w7.5_seed{seed}"
        run_exp(model, diffusion, vae, 'cfg', 7.5, seed, exp_name=name)

    elapsed = (time.time() - t0) / 60
    print(f"  Elapsed: {elapsed:.0f} min")

    # 3. Controlled throughput measurement
    controlled_throughput_measurement(model, diffusion, vae)

    # 4. Qualitative image grid
    generate_qualitative_grid(model, diffusion, vae)

    total = (time.time() - t0) / 60
    print(f"\nTotal: {total:.0f} min")


if __name__ == '__main__':
    main()
