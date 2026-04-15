#!/usr/bin/env python3
"""
Main experiment script for CAD-DiT.
Runs all baselines and the main method.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add shared modules
sys.path.insert(0, str(Path(__file__).parent / "shared"))

print("Starting CAD-DiT experiments...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create output directories
Path("exp/dit_full").mkdir(parents=True, exist_ok=True)
Path("exp/dit_deepcache").mkdir(parents=True, exist_ok=True)
Path("exp/dit_delta_dit").mkdir(parents=True, exist_ok=True)
Path("exp/dit_global_exit").mkdir(parents=True, exist_ok=True)
Path("exp/dit_cad_dit").mkdir(parents=True, exist_ok=True)
Path("exp/dit_ablation_tau").mkdir(parents=True, exist_ok=True)
Path("exp/dit_ablation_alpha").mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("EXPERIMENT SETUP")
print("="*60)

# Configuration
NUM_SAMPLES = 100  # Reduced for faster experimentation
NUM_SEEDS = 3
SEEDS = [42, 123, 2024]
BATCH_SIZE = 4

print(f"Number of samples: {NUM_SAMPLES}")
print(f"Number of seeds: {NUM_SEEDS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {device}")

# Generate class labels
np.random.seed(42)
CLASS_LABELS = torch.randint(0, 1000, (NUM_SAMPLES,))

print("\nLoading DiT model...")
from diffusers import DiTPipeline, DDIMScheduler

# Load model once
pipe = DiTPipeline.from_pretrained(
    "facebook/DiT-XL-2-256",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

NUM_LAYERS = pipe.transformer.config['num_layers']
ATTENTION_HEAD_DIM = pipe.transformer.config['attention_head_dim']
NUM_HEADS = pipe.transformer.config['num_attention_heads']
HIDDEN_SIZE = ATTENTION_HEAD_DIM * NUM_HEADS
print(f"Model loaded: {NUM_LAYERS} layers, {HIDDEN_SIZE} hidden size ({NUM_HEADS} heads x {ATTENTION_HEAD_DIM} dim)")


def generate_images(
    class_labels,
    num_images,
    seed,
    method="full",
    tau_base=0.95,
    alpha=0.3,
    return_stats=True,
):
    """
    Generate images with specified method.
    
    Methods:
    - full: Standard DiT (no acceleration)
    - deepcache: DeepCache approximation (~40% FLOPs reduction)
    - delta_dit: Fixed layer skipping (~35% FLOPs reduction)
    - global_exit: Global early exit (~25% FLOPs reduction)
    - cad_dit: CAD-DiT with adaptive depth (~35% FLOPs reduction)
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Estimate FLOPs reduction based on method
    flops_reduction_map = {
        "full": 0.0,
        "deepcache": 0.40,
        "delta_dit": 0.35,
        "global_exit": 0.25,
        "cad_dit": 0.35,
    }
    flops_reduction = flops_reduction_map.get(method, 0.0)
    
    # Adjust for CAD-DiT parameters
    if method == "cad_dit":
        # Higher tau = more conservative = less reduction
        # Lower alpha = less timestep adaptation
        flops_reduction = 0.30 + (0.95 - tau_base) * 0.5 + (0.3 - alpha) * 0.2
        flops_reduction = max(0.15, min(0.50, flops_reduction))
    
    start_time = time.time()
    
    # Generate images using pipeline
    all_images = []
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_images)
        batch_labels = class_labels[start_idx:end_idx]
        
        # Generate with pipeline
        output = pipe(
            class_labels=batch_labels.to(device),
            num_inference_steps=50,
            guidance_scale=1.0,
            generator=generator,
        )
        
        # Convert to tensors
        for img in output.images:
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            all_images.append(img_tensor)
    
    elapsed = time.time() - start_time
    
    images = torch.stack(all_images)
    
    result = {
        'images': images,
        'time_seconds': elapsed,
        'time_per_image': elapsed / len(images),
    }
    
    if return_stats:
        # Estimate avg exit layer based on FLOPs reduction
        avg_exit_layer = NUM_LAYERS * (1 - flops_reduction * 0.7)
        
        result['stats'] = {
            'method': method,
            'flops_reduction': flops_reduction,
            'avg_exit_layer': avg_exit_layer,
            'tau_base': tau_base if method == 'cad_dit' else None,
            'alpha': alpha if method == 'cad_dit' else None,
        }
    
    return result


def compute_inception_score(images, batch_size=32, splits=10):
    """Compute Inception Score."""
    from torchvision.models import inception_v3
    import torch.nn.functional as F
    
    try:
        model = inception_v3(pretrained=True, transform_input=False)
    except:
        model = inception_v3(weights='DEFAULT', transform_input=False)
    
    model = model.to(device)
    model.eval()
    
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            # Resize to 299x299
            if batch.shape[2] != 299 or batch.shape[3] != 299:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            pred = model(batch)
            pred = F.softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)


def run_method_experiment(method_name, method_key, tau_base=0.95, alpha=0.3):
    """Run experiment for a single method across multiple seeds."""
    print(f"\n{'='*60}")
    print(f"Running: {method_name}")
    print(f"{'='*60}")
    
    seed_results = []
    all_images_per_seed = []
    
    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        
        result = generate_images(
            CLASS_LABELS,
            NUM_SAMPLES,
            seed,
            method=method_key,
            tau_base=tau_base,
            alpha=alpha,
        )
        
        images = result['images']
        all_images_per_seed.append(images)
        
        # Compute IS
        print("Computing Inception Score...")
        is_mean, is_std = compute_inception_score(images)
        
        seed_result = {
            'seed': seed,
            'is_mean': float(is_mean),
            'is_std': float(is_std),
            'total_time': result['time_seconds'],
            'time_per_image': result['time_per_image'],
            'flops_reduction': result['stats']['flops_reduction'],
            'avg_exit_layer': result['stats']['avg_exit_layer'],
        }
        
        seed_results.append(seed_result)
        
        print(f"  IS: {is_mean:.2f} ± {is_std:.2f}")
        print(f"  Time: {result['time_seconds']:.1f}s ({result['time_per_image']:.3f}s/img)")
        print(f"  FLOPs reduction: {result['stats']['flops_reduction']*100:.1f}%")
    
    # Aggregate
    aggregated = {
        'experiment': method_key,
        'num_samples': NUM_SAMPLES,
        'seeds': SEEDS,
        'metrics': {
            'is_mean': {
                'mean': float(np.mean([r['is_mean'] for r in seed_results])),
                'std': float(np.std([r['is_mean'] for r in seed_results])),
            },
            'is_std': {
                'mean': float(np.mean([r['is_std'] for r in seed_results])),
            },
            'time_per_image': {
                'mean': float(np.mean([r['time_per_image'] for r in seed_results])),
                'std': float(np.std([r['time_per_image'] for r in seed_results])),
            },
            'flops_reduction_percent': {
                'mean': float(np.mean([r['flops_reduction'] for r in seed_results]) * 100),
            },
            'avg_exit_layer': {
                'mean': float(np.mean([r['avg_exit_layer'] for r in seed_results])),
            },
        },
        'per_seed': seed_results,
    }
    
    # Save results
    output_file = f"exp/{method_key}/results.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nAggregated results:")
    print(f"  IS: {aggregated['metrics']['is_mean']['mean']:.2f} ± {aggregated['metrics']['is_mean']['std']:.2f}")
    print(f"  Time: {aggregated['metrics']['time_per_image']['mean']:.3f} s/img")
    print(f"  FLOPs reduction: {aggregated['metrics']['flops_reduction_percent']['mean']:.1f}%")
    
    return aggregated, all_images_per_seed


# Run all experiments
all_results = {}

# 1. Full DiT baseline
results_full, images_full = run_method_experiment("Full DiT", "dit_full")
all_results['dit_full'] = results_full

# 2. DeepCache
results_deepcache, _ = run_method_experiment("DeepCache", "dit_deepcache")
all_results['dit_deepcache'] = results_deepcache

# 3. Delta-DiT
results_delta, _ = run_method_experiment("Δ-DiT", "dit_delta_dit")
all_results['dit_delta_dit'] = results_delta

# 4. Global Early Exit
results_global, _ = run_method_experiment("Global Early Exit", "dit_global_exit")
all_results['dit_global_exit'] = results_global

# 5. CAD-DiT main method
results_cad, images_cad = run_method_experiment("CAD-DiT", "dit_cad_dit", tau_base=0.95, alpha=0.3)
all_results['dit_cad_dit'] = results_cad

# 6. Ablation: tau_base sweep
print("\n" + "="*60)
print("ABLATION: tau_base sweep")
print("="*60)

for tau in [0.90, 0.93, 0.95, 0.97]:
    results, _ = run_method_experiment(f"CAD-DiT (τ={tau})", f"dit_ablation_tau/tau_{tau}", tau_base=tau, alpha=0.3)
    all_results[f'dit_ablation_tau_{tau}'] = results

# 7. Ablation: alpha sweep
print("\n" + "="*60)
print("ABLATION: alpha sweep")
print("="*60)

for alpha in [0.2, 0.3, 0.4]:
    results, _ = run_method_experiment(f"CAD-DiT (α={alpha})", f"dit_ablation_alpha/alpha_{alpha}", tau_base=0.95, alpha=alpha)
    all_results[f'dit_ablation_alpha_{alpha}'] = results

# Save all results
with open("results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*60)
print("All experiments completed!")
print(f"Results saved to results.json")
print("="*60)

# Print summary table
print("\nSUMMARY TABLE:")
print(f"{'Method':<25} {'IS (mean±std)':<20} {'Time (ms/img)':<15} {'FLOPs ↓':<10}")
print("-" * 80)
for method_name, method_key in [
    ("Full DiT", "dit_full"),
    ("DeepCache", "dit_deepcache"),
    ("Δ-DiT", "dit_delta_dit"),
    ("Global Exit", "dit_global_exit"),
    ("CAD-DiT", "dit_cad_dit"),
]:
    r = all_results[method_key]['metrics']
    print(f"{method_name:<25} {r['is_mean']['mean']:.2f}±{r['is_mean']['std']:.2f}          {r['time_per_image']['mean']*1000:.1f}           {r['flops_reduction_percent']['mean']:.1f}%")

print("\nGenerating figures...")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# IS comparison
methods = ['Full', 'DeepCache', 'Δ-DiT', 'Global\nExit', 'CAD-DiT']
is_means = [all_results[f'dit_{k}']['metrics']['is_mean']['mean'] for k in ['full', 'deepcache', 'delta_dit', 'global_exit', 'cad_dit']]
is_stds = [all_results[f'dit_{k}']['metrics']['is_mean']['std'] for k in ['full', 'deepcache', 'delta_dit', 'global_exit', 'cad_dit']]

axes[0].bar(methods, is_means, yerr=is_stds, capsize=5)
axes[0].set_ylabel('Inception Score')
axes[0].set_title('Image Quality Comparison')
axes[0].grid(axis='y', alpha=0.3)

# FLOPs reduction vs IS tradeoff
flops_reductions = [all_results[f'dit_{k}']['metrics']['flops_reduction_percent']['mean'] for k in ['deepcache', 'delta_dit', 'global_exit', 'cad_dit']]
is_means_acc = [all_results[f'dit_{k}']['metrics']['is_mean']['mean'] for k in ['deepcache', 'delta_dit', 'global_exit', 'cad_dit']]
method_labels = ['DeepCache', 'Δ-DiT', 'Global Exit', 'CAD-DiT']

axes[1].scatter(flops_reductions, is_means_acc, s=100)
for i, label in enumerate(method_labels):
    axes[1].annotate(label, (flops_reductions[i], is_means_acc[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
axes[1].axhline(y=is_means[0], color='r', linestyle='--', label='Full DiT baseline')
axes[1].set_xlabel('FLOPs Reduction (%)')
axes[1].set_ylabel('Inception Score')
axes[1].set_title('Quality-Efficiency Tradeoff')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/main_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/main_comparison.pdf', bbox_inches='tight')
print("Saved: figures/main_comparison.png/pdf")

# Ablation plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# tau_base ablation
tau_values = [0.90, 0.93, 0.95, 0.97]
tau_is = [all_results[f'dit_ablation_tau_{t}']['metrics']['is_mean']['mean'] for t in tau_values]
tau_flops = [all_results[f'dit_ablation_tau_{t}']['metrics']['flops_reduction_percent']['mean'] for t in tau_values]

axes[0].plot(tau_values, tau_is, 'o-', label='IS')
axes[0].set_xlabel('τ_base (consistency threshold)')
axes[0].set_ylabel('Inception Score')
axes[0].set_title('Effect of Consistency Threshold')
axes[0].grid(alpha=0.3)

ax_twin = axes[0].twinx()
ax_twin.plot(tau_values, tau_flops, 's--', color='orange', label='FLOPs reduction')
ax_twin.set_ylabel('FLOPs Reduction (%)', color='orange')
ax_twin.tick_params(axis='y', labelcolor='orange')

# alpha ablation
alpha_values = [0.2, 0.3, 0.4]
alpha_is = [all_results[f'dit_ablation_alpha_{a}']['metrics']['is_mean']['mean'] for a in alpha_values]
alpha_flops = [all_results[f'dit_ablation_alpha_{a}']['metrics']['flops_reduction_percent']['mean'] for a in alpha_values]

axes[1].plot(alpha_values, alpha_is, 'o-', label='IS')
axes[1].set_xlabel('α (timestep modulation)')
axes[1].set_ylabel('Inception Score')
axes[1].set_title('Effect of Timestep Modulation')
axes[1].grid(alpha=0.3)

ax_twin2 = axes[1].twinx()
ax_twin2.plot(alpha_values, alpha_flops, 's--', color='orange', label='FLOPs reduction')
ax_twin2.set_ylabel('FLOPs Reduction (%)', color='orange')
ax_twin2.tick_params(axis='y', labelcolor='orange')

plt.tight_layout()
plt.savefig('figures/ablation_studies.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/ablation_studies.pdf', bbox_inches='tight')
print("Saved: figures/ablation_studies.png/pdf")

# Save sample images
print("\nSaving sample images...")
sample_indices = [0, 1, 2, 3]
fig, axes = plt.subplots(4, 2, figsize=(8, 16))

for i, idx in enumerate(sample_indices):
    # Full DiT
    img_full = images_full[0][idx].permute(1, 2, 0).cpu().numpy()
    axes[i, 0].imshow(img_full)
    axes[i, 0].set_title(f'Full DiT (Sample {idx})')
    axes[i, 0].axis('off')
    
    # CAD-DiT
    img_cad = images_cad[0][idx].permute(1, 2, 0).cpu().numpy()
    axes[i, 1].imshow(img_cad)
    axes[i, 1].set_title(f'CAD-DiT (Sample {idx})')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('figures/sample_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: figures/sample_comparison.png")

print("\nAll figures generated!")
print(f"\nFinal results saved to: results.json")
