"""Baseline 1: Standard LDM with fixed VAE and full precision."""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# Add shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.models import VAE, UNetModel, SimpleDiffusion
from shared.metrics import compute_fid_from_tensors, get_inception_model, extract_inception_features, calculate_fid
from shared.data_loader import denormalize, get_cifar10_loaders, get_cifar10_reference_images


def train_vae(vae, train_loader, val_loader, device, epochs=30, lr=1e-3, seed=42):
    """Train VAE on reconstruction."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        pbar = tqdm(train_loader, desc=f'VAE Epoch {epoch+1}/{epochs}')
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            optimizer.zero_grad()
            recon, mean, logvar = vae(images)
            
            # Reconstruction loss (MSE)
            recon_loss = nn.functional.mse_loss(recon, images)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / images.shape[0]
            
            # Total loss
            loss = recon_loss + 0.001 * kl_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
        
        avg_loss = train_loss / len(train_loader)
        avg_recon = recon_loss_sum / len(train_loader)
        avg_kl = kl_loss_sum / len(train_loader)
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return vae


def train_diffusion(diffusion, vae, train_loader, device, epochs=50, lr=2e-4, seed=42):
    """Train diffusion model in latent space."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    vae.eval()
    diffusion.train()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Diff Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            # Encode to latent space
            with torch.no_grad():
                mean, _ = vae.encode(images)
                # Scale latents (common practice)
                latents = mean * 0.18215
            
            optimizer.zero_grad()
            loss = diffusion(latents)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return diffusion


@torch.no_grad()
def evaluate_model(vae, diffusion, num_samples=5000, batch_size=100, 
                   num_inference_steps=50, device='cuda', save_dir=None):
    """Evaluate model by generating samples and computing FID."""
    vae.eval()
    diffusion.eval()
    
    all_samples = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Time inference
    start_time = time.time()
    peak_memory = 0
    
    for i in tqdm(range(num_batches), desc='Generating samples'):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Sample from diffusion
        shape = (current_batch_size, 128, 8, 8)  # Latent shape for CIFAR-10 (32x32 -> 8x8)
        latents = diffusion.sample(shape, device=device, num_inference_steps=num_inference_steps)
        latents = latents / 0.18215
        
        # Decode to image space
        samples = vae.decode(latents)
        all_samples.append(samples.cpu())
        
        # Track peak memory
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
    
    inference_time = time.time() - start_time
    time_per_image = inference_time / num_samples * 1000  # ms
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    
    # Denormalize for FID calculation
    all_samples_denorm = denormalize(all_samples)
    all_samples_denorm = torch.clamp(all_samples_denorm, 0, 1)
    
    # Save generated images
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        from torchvision.utils import save_image
        for i, img in enumerate(all_samples_denorm):
            save_image(img, os.path.join(save_dir, f'gen_{i:05d}.png'))
    
    # Compute FID
    # Load reference images
    ref_dir = './data/cifar10_reference'
    if not os.path.exists(ref_dir) or len(os.listdir(ref_dir)) < 1000:
        print("Preparing reference images...")
        get_cifar10_reference_images(10000, save_dir=ref_dir)
    
    # Load reference images as tensors
    from PIL import Image
    ref_images = []
    ref_files = sorted([f for f in os.listdir(ref_dir) if f.endswith('.png')])[:num_samples]
    for f in ref_files:
        img = Image.open(os.path.join(ref_dir, f)).convert('RGB')
        img = transforms.ToTensor()(img)
        ref_images.append(img)
    ref_images = torch.stack(ref_images)
    
    # Calculate FID
    print("Computing FID...")
    inception_model = get_inception_model()
    fake_features = extract_inception_features(all_samples_denorm, inception_model, device=device)
    real_features = extract_inception_features(ref_images, inception_model, device=device)
    fid = calculate_fid(real_features, fake_features)
    
    # Calculate Inception Score
    print("Computing IS...")
    from shared.metrics import calculate_inception_score
    try:
        # Need to restore classifier for IS
        is_model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
        is_mean, is_std = calculate_inception_score(all_samples_denorm, is_model, device=device)
    except Exception as e:
        print(f"IS calculation failed: {e}")
        is_mean, is_std = 0.0, 0.0
    
    # Memory in GB
    peak_memory_gb = peak_memory / (1024 ** 3)
    
    return {
        'fid': fid,
        'is_mean': is_mean,
        'is_std': is_std,
        'time_per_image_ms': time_per_image,
        'peak_memory_gb': peak_memory_gb,
        'num_samples': num_samples,
        'num_inference_steps': num_inference_steps
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs_vae', type=int, default=30)
    parser.add_argument('--epochs_diff', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./exp/baseline_ldm')
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Loading CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    
    # Prepare reference images
    get_cifar10_reference_images(10000, save_dir='./data/cifar10_reference')
    
    # Create models
    print("Creating models...")
    vae = VAE(in_channels=3, latent_channels=128, base_channels=64).to(device)
    
    unet = UNetModel(
        in_channels=128,  # Latent channels
        model_channels=64,
        out_channels=128,
        num_res_blocks=2,
        attention_resolutions=(8,),
        dropout=0.1
    ).to(device)
    
    diffusion = SimpleDiffusion(unet, num_timesteps=1000).to(device)
    
    # Train VAE
    print("\n=== Training VAE ===")
    start_time = time.time()
    vae = train_vae(vae, train_loader, val_loader, device, 
                    epochs=args.epochs_vae, lr=1e-3, seed=args.seed)
    vae_time = time.time() - start_time
    
    # Save VAE
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(vae.state_dict(), os.path.join(args.output_dir, f'vae_seed{args.seed}.pt'))
    
    # Train Diffusion
    print("\n=== Training Diffusion ===")
    start_time = time.time()
    diffusion = train_diffusion(diffusion, vae, train_loader, device,
                               epochs=args.epochs_diff, lr=2e-4, seed=args.seed)
    diff_time = time.time() - start_time
    
    # Save diffusion
    torch.save(diffusion.state_dict(), os.path.join(args.output_dir, f'diffusion_seed{args.seed}.pt'))
    
    # Evaluate
    print("\n=== Evaluating ===")
    save_dir = os.path.join(args.output_dir, f'generated_seed{args.seed}')
    metrics = evaluate_model(vae, diffusion, num_samples=args.num_samples,
                            num_inference_steps=50, device=device, save_dir=save_dir)
    
    # Add training times
    metrics['vae_training_time_min'] = vae_time / 60
    metrics['diff_training_time_min'] = diff_time / 60
    metrics['total_training_time_min'] = (vae_time + diff_time) / 60
    metrics['seed'] = args.seed
    
    print("\n=== Results ===")
    print(f"FID: {metrics['fid']:.4f}")
    print(f"IS: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}")
    print(f"Time per image: {metrics['time_per_image_ms']:.2f} ms")
    print(f"Peak memory: {metrics['peak_memory_gb']:.2f} GB")
    
    # Save results
    results_file = os.path.join(args.output_dir, f'results_seed{args.seed}.json')
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == '__main__':
    main()
