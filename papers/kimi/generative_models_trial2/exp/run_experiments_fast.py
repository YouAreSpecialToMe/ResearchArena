#!/usr/bin/env python3
"""Streamlined experiments for UALQ-Diff - optimized for 8-hour budget."""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Import shared modules
sys.path.insert(0, 'exp')
from shared.models import VAE, UNetModel, SimpleDiffusion
from shared.metrics import get_inception_model, extract_inception_features, calculate_fid
from shared.data_loader import denormalize, get_cifar10_loaders, get_cifar10_reference_images

# Set seeds for reproducibility
SEEDS = [42, 123, 456]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reduced epochs for faster completion
CONFIG = {
    'baseline_ldm': {'epochs_vae': 15, 'epochs_diff': 25, 'batch_size': 128},
    'baseline_cat': {'epochs_vae': 15, 'epochs_diff': 25, 'batch_size': 128},
    'ualq_diff': {'epochs_stage1': 10, 'epochs_stage2': 20, 'epochs_stage3': 5, 'batch_size': 128},
}


def train_vae_simple(vae, train_loader, epochs, lr=1e-3, seed=42):
    """Train VAE with simplified logging."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon, mean, logvar = vae(images)
            
            recon_loss = nn.functional.mse_loss(recon, images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / images.shape[0]
            loss = recon_loss + 0.001 * kl_loss
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  VAE Epoch {epoch+1}/{epochs}: Loss={epoch_loss/len(train_loader):.4f}')
    
    return vae


def train_diffusion_simple(diffusion, vae, train_loader, epochs, lr=2e-4, seed=42):
    """Train diffusion with simplified logging."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    vae.eval()
    diffusion.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(DEVICE)
            with torch.no_grad():
                mean, _ = vae.encode(images)
                latents = mean * 0.18215
            
            optimizer.zero_grad()
            loss = diffusion(latents)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Diff Epoch {epoch+1}/{epochs}: Loss={epoch_loss/len(train_loader):.4f}')
    
    return diffusion


@torch.no_grad()
def evaluate_model(vae, diffusion, num_samples=2000, batch_size=100, seed=42):
    """Evaluate model - simplified version."""
    vae.eval()
    diffusion.eval()
    torch.manual_seed(seed)
    
    all_samples = []
    start_time = time.time()
    peak_memory = 0
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    for i in range(num_batches):
        current_batch = min(batch_size, num_samples - i * batch_size)
        shape = (current_batch, 128, 8, 8)
        latents = diffusion.sample(shape, device=DEVICE, num_inference_steps=25)  # Reduced steps
        latents = latents / 0.18215
        samples = vae.decode(latents)
        all_samples.append(samples.cpu())
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(DEVICE))
    
    inference_time = time.time() - start_time
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    all_samples_denorm = torch.clamp(denormalize(all_samples), 0, 1)
    
    # Load reference images
    ref_dir = './data/cifar10_reference'
    from PIL import Image
    from torchvision import transforms
    
    ref_images = []
    ref_files = sorted([f for f in os.listdir(ref_dir) if f.endswith('.png')])[:num_samples]
    for f in ref_files:
        img = Image.open(os.path.join(ref_dir, f)).convert('RGB')
        ref_images.append(transforms.ToTensor()(img))
    ref_images = torch.stack(ref_images)
    
    # Calculate FID
    inception_model = get_inception_model()
    fake_features = extract_inception_features(all_samples_denorm, inception_model, device=DEVICE)
    real_features = extract_inception_features(ref_images, inception_model, device=DEVICE)
    fid = calculate_fid(real_features, fake_features)
    
    return {
        'fid': fid,
        'time_per_image_ms': inference_time / num_samples * 1000,
        'peak_memory_gb': peak_memory / (1024 ** 3),
    }


def run_baseline_ldm(seed):
    """Run Standard LDM baseline."""
    print(f"\n{'='*60}")
    print(f"Baseline LDM - Seed {seed}")
    print(f"{'='*60}")
    
    cfg = CONFIG['baseline_ldm']
    train_loader, _, _ = get_cifar10_loaders(batch_size=cfg['batch_size'])
    
    # Create models
    vae = VAE(in_channels=3, latent_channels=128, base_channels=64).to(DEVICE)
    unet = UNetModel(in_channels=128, model_channels=64, out_channels=128).to(DEVICE)
    diffusion = SimpleDiffusion(unet, num_timesteps=1000).to(DEVICE)
    
    # Train
    print("Training VAE...")
    vae = train_vae_simple(vae, train_loader, cfg['epochs_vae'], seed=seed)
    
    print("Training Diffusion...")
    diffusion = train_diffusion_simple(diffusion, vae, train_loader, cfg['epochs_diff'], seed=seed)
    
    # Evaluate
    print("Evaluating...")
    metrics = evaluate_model(vae, diffusion, num_samples=2000, seed=seed)
    metrics['seed'] = seed
    metrics['method'] = 'Standard LDM'
    
    # Save
    os.makedirs('exp/baseline_ldm', exist_ok=True)
    torch.save(vae.state_dict(), f'exp/baseline_ldm/vae_seed{seed}.pt')
    torch.save(diffusion.state_dict(), f'exp/baseline_ldm/diffusion_seed{seed}.pt')
    with open(f'exp/baseline_ldm/results_seed{seed}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results: FID={metrics['fid']:.2f}, Time={metrics['time_per_image_ms']:.1f}ms")
    return metrics


def run_baseline_cat(seed):
    """Run CAT-style adaptive tokenization baseline."""
    print(f"\n{'='*60}")
    print(f"Baseline CAT - Seed {seed}")
    print(f"{'='*60}")
    
    cfg = CONFIG['baseline_cat']
    train_loader, _, _ = get_cifar10_loaders(batch_size=cfg['batch_size'])
    
    # Import adaptive VAE
    sys.path.insert(0, 'exp/baseline_cat')
    from run import AdaptiveVAE, train_adaptive_vae, train_diffusion
    
    vae = AdaptiveVAE(in_channels=3, latent_channels=128, base_channels=64).to(DEVICE)
    unet = UNetModel(in_channels=128, model_channels=64, out_channels=128).to(DEVICE)
    diffusion = SimpleDiffusion(unet, num_timesteps=1000).to(DEVICE)
    
    print("Training Adaptive VAE...")
    vae = train_adaptive_vae(vae, train_loader, DEVICE, cfg['epochs_vae'], seed=seed)
    
    print("Training Diffusion...")
    diffusion = train_diffusion(diffusion, vae, train_loader, DEVICE, cfg['epochs_diff'], seed=seed)
    
    print("Evaluating...")
    metrics = evaluate_model(vae, diffusion, num_samples=2000, seed=seed)
    metrics['seed'] = seed
    metrics['method'] = 'CAT-Style'
    
    os.makedirs('exp/baseline_cat', exist_ok=True)
    torch.save(vae.state_dict(), f'exp/baseline_cat/vae_seed{seed}.pt')
    torch.save(diffusion.state_dict(), f'exp/baseline_cat/diffusion_seed{seed}.pt')
    with open(f'exp/baseline_cat/results_seed{seed}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results: FID={metrics['fid']:.2f}, Time={metrics['time_per_image_ms']:.1f}ms")
    return metrics


def run_ualq_diff(seed):
    """Run UALQ-Diff main experiment."""
    print(f"\n{'='*60}")
    print(f"UALQ-Diff - Seed {seed}")
    print(f"{'='*60}")
    
    cfg = CONFIG['ualq_diff']
    train_loader, _, _ = get_cifar10_loaders(batch_size=cfg['batch_size'])
    
    sys.path.insert(0, 'exp/ualq_diff')
    from run import AdaptiveVAE, ThreeDQuantizedUNet, train_stage1_warmup, train_stage2_joint, train_stage3_finetune
    
    vae = AdaptiveVAE(in_channels=3, latent_channels=128, base_channels=64).to(DEVICE)
    unet = UNetModel(in_channels=128, model_channels=64, out_channels=128).to(DEVICE)
    quantized_unet = ThreeDQuantizedUNet(unet, num_timesteps=1000)
    diffusion = SimpleDiffusion(quantized_unet, num_timesteps=1000).to(DEVICE)
    
    print("Stage 1: Warm-up...")
    vae = train_stage1_warmup(vae, train_loader, DEVICE, cfg['epochs_stage1'], seed=seed)
    
    print("Stage 2: Joint Training...")
    vae, diffusion = train_stage2_joint(vae, diffusion, train_loader, DEVICE, cfg['epochs_stage2'], seed=seed)
    
    print("Stage 3: Fine-tuning...")
    vae, diffusion = train_stage3_finetune(vae, diffusion, train_loader, DEVICE, cfg['epochs_stage3'], seed=seed)
    
    print("Evaluating...")
    metrics = evaluate_model(vae, diffusion, num_samples=2000, seed=seed)
    metrics['seed'] = seed
    metrics['method'] = 'UALQ-Diff'
    
    os.makedirs('exp/ualq_diff', exist_ok=True)
    torch.save(vae.state_dict(), f'exp/ualq_diff/vae_seed{seed}.pt')
    torch.save(diffusion.state_dict(), f'exp/ualq_diff/diffusion_seed{seed}.pt')
    with open(f'exp/ualq_diff/results_seed{seed}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results: FID={metrics['fid']:.2f}, Time={metrics['time_per_image_ms']:.1f}ms")
    return metrics


def aggregate_results():
    """Aggregate all results."""
    methods = ['baseline_ldm', 'baseline_cat', 'ualq_diff']
    all_results = {}
    
    for method in methods:
        method_results = []
        for seed in SEEDS:
            result_file = f'exp/{method}/results_seed{seed}.json'
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    method_results.append(json.load(f))
        
        if method_results:
            fid_values = [r['fid'] for r in method_results]
            time_values = [r['time_per_image_ms'] for r in method_results]
            mem_values = [r['peak_memory_gb'] for r in method_results]
            
            all_results[method] = {
                'name': method_results[0]['method'],
                'fid_mean': np.mean(fid_values),
                'fid_std': np.std(fid_values),
                'time_mean': np.mean(time_values),
                'time_std': np.std(time_values),
                'memory_mean': np.mean(mem_values),
                'memory_std': np.std(mem_values),
                'values': method_results
            }
    
    return all_results


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'FID':<20} {'Time (ms/img)':<20} {'Memory (GB)':<15}")
    print("-"*80)
    
    for method, data in results.items():
        fid_str = f"{data['fid_mean']:.2f} ± {data['fid_std']:.2f}"
        time_str = f"{data['time_mean']:.1f} ± {data['time_std']:.1f}"
        mem_str = f"{data['memory_mean']:.2f} ± {data['memory_std']:.2f}"
        print(f"{data['name']:<25} {fid_str:<20} {time_str:<20} {mem_str:<15}")
    
    print("="*80)
    
    # Check success criteria
    print("\nSUCCESS CRITERIA EVALUATION:")
    if 'baseline_ldm' in results and 'ualq_diff' in results:
        baseline_fid = results['baseline_ldm']['fid_mean']
        ualq_fid = results['ualq_diff']['fid_mean']
        fid_diff = ualq_fid - baseline_fid
        
        print(f"  1. FID within 0.5 of baseline: {fid_diff:.2f} {'✓ PASS' if fid_diff <= 0.5 else '✗ FAIL'}")
        
        baseline_time = results['baseline_ldm']['time_mean']
        ualq_time = results['ualq_diff']['time_mean']
        speedup = baseline_time / ualq_time if ualq_time > 0 else 0
        print(f"  2. Speedup >= 2x: {speedup:.2f}x {'✓ PASS' if speedup >= 2 else '✗ FAIL'}")
        
        baseline_mem = results['baseline_ldm']['memory_mean']
        ualq_mem = results['ualq_diff']['memory_mean']
        mem_reduction = (1 - ualq_mem / baseline_mem) * 100 if baseline_mem > 0 else 0
        print(f"  3. Memory reduction >= 50%: {mem_reduction:.1f}% {'✓ PASS' if mem_reduction >= 50 else '✗ FAIL'}")


def main():
    """Main execution."""
    total_start = time.time()
    
    print("="*80)
    print("UALQ-DIFF STREAMLINED EXPERIMENTS")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Config: {CONFIG}")
    
    # Prepare data
    print("\nPreparing data...")
    get_cifar10_reference_images(10000, save_dir='./data/cifar10_reference')
    
    # Run experiments for each seed
    for seed in SEEDS:
        try:
            run_baseline_ldm(seed)
        except Exception as e:
            print(f"Error in baseline_ldm seed {seed}: {e}")
        
        try:
            run_baseline_cat(seed)
        except Exception as e:
            print(f"Error in baseline_cat seed {seed}: {e}")
        
        try:
            run_ualq_diff(seed)
        except Exception as e:
            print(f"Error in ualq_diff seed {seed}: {e}")
    
    # Aggregate and print results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS...")
    results = aggregate_results()
    print_summary(results)
    
    # Save final results
    final_results = {
        'experiments': results,
        'total_runtime_minutes': (time.time() - total_start) / 60,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nTotal runtime: {final_results['total_runtime_minutes']:.1f} minutes")
    print("Results saved to: results.json")


if __name__ == '__main__':
    main()
