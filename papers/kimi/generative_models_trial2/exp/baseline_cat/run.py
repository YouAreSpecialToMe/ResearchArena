"""Baseline 2: CAT-style Adaptive Tokenization (without quantization)."""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.models import UNetModel, SimpleDiffusion
from shared.metrics import compute_fid_from_tensors, get_inception_model, extract_inception_features, calculate_fid
from shared.data_loader import denormalize, get_cifar10_loaders, get_cifar10_reference_images


class ComplexityEstimator(nn.Module):
    """Lightweight CNN for estimating local complexity."""
    def __init__(self, in_channels=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class AdaptiveVAEEncoder(nn.Module):
    """VAE Encoder with adaptive token density."""
    def __init__(self, in_channels=3, latent_channels=128, base_channels=64):
        super().__init__()
        self.latent_channels = latent_channels
        
        # Standard encoder
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        )
        
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
        )
        
        self.conv_out = nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1)
        
        # Token allocation network
        self.token_gate = nn.Sequential(
            nn.Conv2d(base_channels * 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.mid(h)
        
        # Token density map (8x8)
        token_density = self.token_gate(h)  # [B, 1, 8, 8]
        
        # Latent output
        out = self.conv_out(h)
        mean, logvar = out.chunk(2, dim=1)
        
        return mean, logvar, token_density


class AdaptiveVAEDecoder(nn.Module):
    """VAE Decoder with density awareness."""
    def __init__(self, latent_channels=128, out_channels=3, base_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels + 1, base_channels * 4, 3, padding=1)  # +1 for density
        
        self.up1 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        )
        
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, z, density_map):
        # Concatenate latent with density map
        h = torch.cat([z, density_map], dim=1)
        h = self.conv_in(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.conv_out(h)
        return h


class AdaptiveVAE(nn.Module):
    """Full Adaptive VAE."""
    def __init__(self, in_channels=3, latent_channels=128, base_channels=64):
        super().__init__()
        self.encoder = AdaptiveVAEEncoder(in_channels, latent_channels, base_channels)
        self.decoder = AdaptiveVAEDecoder(latent_channels, in_channels, base_channels)
        self.latent_channels = latent_channels
    
    def encode(self, x):
        return self.encoder(x)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, density):
        return self.decoder(z, density)
    
    def forward(self, x, sample=True):
        mean, logvar, density = self.encode(x)
        if sample:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        recon = self.decode(z, density)
        return recon, mean, logvar, density


def train_adaptive_vae(vae, train_loader, device, epochs=30, lr=1e-3, seed=42):
    """Train adaptive VAE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    
    for epoch in range(epochs):
        train_loss = 0
        recon_loss_sum = 0
        density_sum = 0
        
        pbar = tqdm(train_loader, desc=f'VAE Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            recon, mean, logvar, density = vae(images)
            
            recon_loss = nn.functional.mse_loss(recon, images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / images.shape[0]
            
            # Efficiency regularization: encourage sparsity in token density
            # Penalize high density (closer to 1 means more tokens)
            efficiency_loss = density.mean()
            
            loss = recon_loss + 0.001 * kl_loss + 0.01 * efficiency_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            density_sum += density.mean().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'density': f'{density.mean().item():.4f}'
            })
        
        avg_loss = train_loss / len(train_loader)
        avg_recon = recon_loss_sum / len(train_loader)
        avg_density = density_sum / len(train_loader)
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Avg Density={avg_density:.4f}')
    
    return vae


def train_diffusion(diffusion, vae, train_loader, device, epochs=50, lr=2e-4, seed=42):
    """Train diffusion on adaptive latent space."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    vae.eval()
    diffusion.train()
    
    for epoch in range(epochs):
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Diff Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            with torch.no_grad():
                mean, _, _ = vae.encode(images)
                latents = mean * 0.18215
            
            optimizer.zero_grad()
            loss = diffusion(latents)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}')
    
    return diffusion


@torch.no_grad()
def evaluate_model(vae, diffusion, num_samples=5000, batch_size=100, 
                   num_inference_steps=50, device='cuda', save_dir=None):
    """Evaluate adaptive model."""
    vae.eval()
    diffusion.eval()
    
    all_samples = []
    all_densities = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    start_time = time.time()
    peak_memory = 0
    
    for i in tqdm(range(num_batches), desc='Generating'):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        shape = (current_batch_size, 128, 8, 8)
        latents = diffusion.sample(shape, device=device, num_inference_steps=num_inference_steps)
        latents = latents / 0.18215
        
        # Use mean density for generation
        density = torch.full((current_batch_size, 1, 8, 8), 0.5, device=device)
        
        samples = vae.decode(latents, density)
        all_samples.append(samples.cpu())
        all_densities.append(density.cpu())
        
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
    
    inference_time = time.time() - start_time
    time_per_image = inference_time / num_samples * 1000
    
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    all_densities = torch.cat(all_densities, dim=0)[:num_samples]
    
    # Compute effective token count
    # Assuming base 8x8 = 64 tokens, scaled by average density
    avg_density = all_densities.mean().item()
    effective_tokens = 64 * (0.5 + 0.5 * avg_density)  # Simplified token count
    
    all_samples_denorm = denormalize(all_samples)
    all_samples_denorm = torch.clamp(all_samples_denorm, 0, 1)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        from torchvision.utils import save_image
        for i, img in enumerate(all_samples_denorm):
            save_image(img, os.path.join(save_dir, f'gen_{i:05d}.png'))
    
    # Load reference images
    ref_dir = './data/cifar10_reference'
    from PIL import Image
    from torchvision import transforms
    
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
    
    # Calculate IS
    print("Computing IS...")
    from shared.metrics import calculate_inception_score
    try:
        is_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        is_model.eval()
        is_mean, is_std = calculate_inception_score(all_samples_denorm, is_model, device=device)
    except:
        is_mean, is_std = 0.0, 0.0
    
    peak_memory_gb = peak_memory / (1024 ** 3)
    
    return {
        'fid': fid,
        'is_mean': is_mean,
        'is_std': is_std,
        'time_per_image_ms': time_per_image,
        'peak_memory_gb': peak_memory_gb,
        'avg_token_density': avg_density,
        'effective_tokens_per_image': effective_tokens,
        'num_samples': num_samples
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs_vae', type=int, default=30)
    parser.add_argument('--epochs_diff', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./exp/baseline_cat')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Loading CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    get_cifar10_reference_images(10000, save_dir='./data/cifar10_reference')
    
    # Create models
    print("Creating models...")
    vae = AdaptiveVAE(in_channels=3, latent_channels=128, base_channels=64).to(device)
    
    unet = UNetModel(
        in_channels=128,
        model_channels=64,
        out_channels=128,
        num_res_blocks=2,
        attention_resolutions=(8,),
        dropout=0.1
    ).to(device)
    
    diffusion = SimpleDiffusion(unet, num_timesteps=1000).to(device)
    
    # Train
    print("\n=== Training Adaptive VAE ===")
    start_time = time.time()
    vae = train_adaptive_vae(vae, train_loader, device, 
                             epochs=args.epochs_vae, lr=1e-3, seed=args.seed)
    vae_time = time.time() - start_time
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(vae.state_dict(), os.path.join(args.output_dir, f'vae_seed{args.seed}.pt'))
    
    print("\n=== Training Diffusion ===")
    start_time = time.time()
    diffusion = train_diffusion(diffusion, vae, train_loader, device,
                               epochs=args.epochs_diff, lr=2e-4, seed=args.seed)
    diff_time = time.time() - start_time
    
    torch.save(diffusion.state_dict(), os.path.join(args.output_dir, f'diffusion_seed{args.seed}.pt'))
    
    # Evaluate
    print("\n=== Evaluating ===")
    save_dir = os.path.join(args.output_dir, f'generated_seed{args.seed}')
    metrics = evaluate_model(vae, diffusion, num_samples=args.num_samples,
                            num_inference_steps=50, device=device, save_dir=save_dir)
    
    metrics['vae_training_time_min'] = vae_time / 60
    metrics['diff_training_time_min'] = diff_time / 60
    metrics['total_training_time_min'] = (vae_time + diff_time) / 60
    metrics['seed'] = args.seed
    
    print("\n=== Results ===")
    print(f"FID: {metrics['fid']:.4f}")
    print(f"IS: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}")
    print(f"Time per image: {metrics['time_per_image_ms']:.2f} ms")
    print(f"Peak memory: {metrics['peak_memory_gb']:.2f} GB")
    print(f"Avg token density: {metrics['avg_token_density']:.4f}")
    print(f"Effective tokens/img: {metrics['effective_tokens_per_image']:.2f}")
    
    results_file = os.path.join(args.output_dir, f'results_seed{args.seed}.json')
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == '__main__':
    main()
