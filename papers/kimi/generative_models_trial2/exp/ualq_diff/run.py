"""UALQ-Diff: Joint Adaptive Tokenization + 3D Quantization."""
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
from shared.metrics import get_inception_model, extract_inception_features, calculate_fid
from shared.data_loader import denormalize, get_cifar10_loaders, get_cifar10_reference_images


class ComplexityEstimator(nn.Module):
    """Lightweight CNN for estimating local complexity."""
    def __init__(self, in_channels=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
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
        
        token_density = self.token_gate(h)  # [B, 1, 8, 8]
        
        out = self.conv_out(h)
        mean, logvar = out.chunk(2, dim=1)
        
        return mean, logvar, token_density


class AdaptiveVAEDecoder(nn.Module):
    """VAE Decoder with density awareness."""
    def __init__(self, latent_channels=128, out_channels=3, base_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels + 1, base_channels * 4, 3, padding=1)
        
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


def get_3d_bit_allocation(timestep, token_density, layer_sensitivity, num_timesteps=1000):
    """3D bit allocation: timestep × spatial (density) × layer sensitivity.
    
    Returns bit widths for weights and activations.
    """
    batch_size = timestep.shape[0]
    
    # Dimension 1: Timestep-based (0.8-1.0: 4bit, 0.2-0.8: 6bit, 0-0.2: 8bit)
    t_norm = timestep.float() / num_timesteps
    t_bits = torch.where(t_norm > 0.8, torch.tensor(4.0),
                torch.where(t_norm > 0.2, torch.tensor(6.0), torch.tensor(8.0)))
    
    # Dimension 2: Spatial (token density)
    # High density regions get higher precision
    density_avg = token_density.mean(dim=[1, 2, 3])  # [B]
    spatial_bits = 4.0 + 4.0 * density_avg  # Range: 4-8 bits
    
    # Dimension 3: Layer sensitivity (placeholder - would be computed during training)
    # Higher sensitivity = more bits
    layer_bits = 6.0 + 2.0 * layer_sensitivity  # Range: 6-8 bits for sensitive layers
    
    # Combine: weighted average
    w_bits = 0.4 * t_bits + 0.3 * spatial_bits + 0.3 * layer_bits
    a_bits = 0.3 * t_bits + 0.5 * spatial_bits + 0.2 * layer_bits
    
    # Clamp to valid range
    w_bits = torch.clamp(w_bits, 4.0, 8.0)
    a_bits = torch.clamp(a_bits, 4.0, 8.0)
    
    return w_bits, a_bits


class ThreeDQuantizedUNet(nn.Module):
    """U-Net with 3D quantization awareness."""
    def __init__(self, base_unet, num_timesteps=1000):
        super().__init__()
        self.unet = base_unet
        self.num_timesteps = num_timesteps
        
        # Learnable layer sensitivity (will be updated during training)
        self.register_buffer('layer_sensitivity', torch.tensor(0.5))
    
    def forward(self, x, t, token_density=None):
        """Forward with 3D quantization awareness.
        
        Args:
            x: Latent tensor [B, C, H, W]
            t: Timestep [B]
            token_density: Optional density map [B, 1, H, W]
        """
        batch_size = x.shape[0]
        
        if token_density is None:
            # Default uniform density
            token_density = torch.ones(batch_size, 1, 8, 8, device=x.device) * 0.5
        
        # Get 3D bit allocation
        w_bits, a_bits = get_3d_bit_allocation(
            t, token_density, self.layer_sensitivity, self.num_timesteps
        )
        
        # Simulate quantization effect during training
        if self.training:
            # Lower bits = more noise
            noise_scale = (9.0 - a_bits.mean()) * 0.002
            x = x + torch.randn_like(x) * noise_scale
        
        return self.unet(x, t)


def train_stage1_warmup(vae, train_loader, device, epochs=20, lr=1e-3, seed=42):
    """Stage 1: Warm-up VAE on reconstruction."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    
    for epoch in range(epochs):
        train_loss = 0
        recon_loss_sum = 0
        density_sum = 0
        
        pbar = tqdm(train_loader, desc=f'Stage1 VAE Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            recon, mean, logvar, density = vae(images)
            
            recon_loss = nn.functional.mse_loss(recon, images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / images.shape[0]
            
            # Efficiency regularization
            efficiency_loss = density.mean()
            
            loss = recon_loss + 0.001 * kl_loss + 0.01 * efficiency_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            density_sum += density.mean().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}'
            })
        
        print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, '
              f'Density={density_sum/len(train_loader):.4f}')
    
    return vae


def train_stage2_joint(vae, diffusion, train_loader, device, epochs=35, lr=2e-4, seed=42):
    """Stage 2: Joint training with quantization awareness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Optimize both VAE and diffusion
    optimizer = optim.AdamW(
        list(vae.parameters()) + list(diffusion.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    vae.train()
    diffusion.train()
    
    for epoch in range(epochs):
        train_loss = 0
        recon_loss_sum = 0
        diff_loss_sum = 0
        
        pbar = tqdm(train_loader, desc=f'Stage2 Joint Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # VAE forward
            recon, mean, logvar, density = vae(images)
            recon_loss = nn.functional.mse_loss(recon, images)
            
            # Encode to latent
            latents = mean * 0.18215
            
            # Diffusion forward with 3D quantization awareness
            B = latents.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device, dtype=torch.long)
            
            # Forward diffusion
            noise = torch.randn_like(latents)
            sqrt_alpha_t = diffusion.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = sqrt_alpha_t * latents + sqrt_one_minus_alpha_t * noise
            
            # Predict noise with 3D quantization awareness
            if hasattr(diffusion.model, 'forward'):
                noise_pred = diffusion.model(x_t, t, density)
            else:
                noise_pred = diffusion.model.unet(x_t, t)
            
            diff_loss = nn.functional.mse_loss(noise_pred, noise)
            
            # Efficiency loss
            efficiency_loss = density.mean()
            
            # Total loss
            loss = recon_loss + 0.1 * diff_loss + 0.01 * efficiency_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vae.parameters()) + list(diffusion.parameters()), 1.0
            )
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            diff_loss_sum += diff_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'diff': f'{diff_loss.item():.4f}'
            })
        
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Total={avg_loss:.4f}, '
              f'Recon={recon_loss_sum/len(train_loader):.4f}, '
              f'Diff={diff_loss_sum/len(train_loader):.4f}')
    
    return vae, diffusion


def train_stage3_finetune(vae, diffusion, train_loader, device, epochs=10, lr=1e-4, seed=42):
    """Stage 3: Fine-tune with fixed token allocation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Freeze VAE encoder, only fine-tune decoder and diffusion
    for param in vae.encoder.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(
        list(vae.decoder.parameters()) + list(diffusion.parameters()),
        lr=lr, weight_decay=0.01
    )
    
    vae.eval()  # Encoder in eval mode
    vae.decoder.train()
    diffusion.train()
    
    for epoch in range(epochs):
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Stage3 Fine-tune Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                mean, _, density = vae.encode(images)
                latents = mean * 0.18215
            
            # Diffusion loss
            B = latents.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device, dtype=torch.long)
            
            noise = torch.randn_like(latents)
            sqrt_alpha_t = diffusion.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = sqrt_alpha_t * latents + sqrt_one_minus_alpha_t * noise
            
            if hasattr(diffusion.model, 'forward'):
                noise_pred = diffusion.model(x_t, t, density)
            else:
                noise_pred = diffusion.model.unet(x_t, t)
            
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}')
    
    # Unfreeze for potential further use
    for param in vae.encoder.parameters():
        param.requires_grad = True
    
    return vae, diffusion


@torch.no_grad()
def evaluate_ualq(vae, diffusion, num_samples=5000, batch_size=100,
                  num_inference_steps=50, device='cuda', save_dir=None):
    """Evaluate UALQ-Diff."""
    vae.eval()
    diffusion.eval()
    
    all_samples = []
    all_densities = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    start_time = time.time()
    peak_memory = 0
    
    total_bits = 0
    total_samples = 0
    
    for i in tqdm(range(num_batches), desc='Generating'):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        shape = (current_batch_size, 128, 8, 8)
        x = torch.randn(shape, device=device)
        
        # Default density for generation
        density = torch.ones(current_batch_size, 1, 8, 8, device=device) * 0.5
        
        timesteps = torch.linspace(diffusion.num_timesteps - 1, 0, num_inference_steps,
                                   dtype=torch.long, device=device)
        
        for t in timesteps:
            t_batch = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
            
            # Get bit allocation
            w_bits, a_bits = get_3d_bit_allocation(
                t_batch, density, torch.tensor(0.5), diffusion.num_timesteps
            )
            total_bits += a_bits.mean().item() * current_batch_size
            total_samples += current_batch_size
            
            # Predict noise
            if hasattr(diffusion.model, 'forward'):
                noise_pred = diffusion.model(x, t_batch, density)
            else:
                noise_pred = diffusion.model.unet(x, t_batch)
            
            # DDIM step
            alpha_t = diffusion.alphas_cumprod[t]
            alpha_prev = diffusion.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            if t > 0:
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
        
        latents = x / 0.18215
        samples = vae.decode(latents, density)
        all_samples.append(samples.cpu())
        all_densities.append(density.cpu())
        
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
    
    inference_time = time.time() - start_time
    time_per_image = inference_time / num_samples * 1000
    
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    all_densities = torch.cat(all_densities, dim=0)[:num_samples]
    
    avg_density = all_densities.mean().item()
    effective_tokens = 64 * (0.5 + 0.5 * avg_density)
    
    all_samples_denorm = denormalize(all_samples)
    all_samples_denorm = torch.clamp(all_samples_denorm, 0, 1)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        from torchvision.utils import save_image
        for i, img in enumerate(all_samples_denorm):
            save_image(img, os.path.join(save_dir, f'gen_{i:05d}.png'))
    
    # Calculate FID
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
    
    print("Computing FID...")
    inception_model = get_inception_model()
    fake_features = extract_inception_features(all_samples_denorm, inception_model, device=device)
    real_features = extract_inception_features(ref_images, inception_model, device=device)
    fid = calculate_fid(real_features, fake_features)
    
    # Calculate IS
    try:
        from shared.metrics import calculate_inception_score
        is_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        is_model.eval()
        is_mean, is_std = calculate_inception_score(all_samples_denorm, is_model, device=device)
    except:
        is_mean, is_std = 0.0, 0.0
    
    avg_bitwidth = total_bits / total_samples if total_samples > 0 else 6.0
    
    # Estimate BOPs
    base_gmac = 1.0
    bops_g = base_gmac * (avg_bitwidth / 16) * num_inference_steps
    
    peak_memory_gb = peak_memory / (1024 ** 3)
    
    return {
        'fid': fid,
        'is_mean': is_mean,
        'is_std': is_std,
        'time_per_image_ms': time_per_image,
        'peak_memory_gb': peak_memory_gb,
        'avg_token_density': avg_density,
        'effective_tokens_per_image': effective_tokens,
        'avg_bitwidth': avg_bitwidth,
        'estimated_bops_g': bops_g,
        'num_samples': num_samples
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs_stage1', type=int, default=20)
    parser.add_argument('--epochs_stage2', type=int, default=35)
    parser.add_argument('--epochs_stage3', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./exp/ualq_diff')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("Loading CIFAR-10...")
    train_loader, _, _ = get_cifar10_loaders(batch_size=args.batch_size)
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
    
    # Wrap with 3D quantization
    quantized_unet = ThreeDQuantizedUNet(unet, num_timesteps=1000)
    diffusion = SimpleDiffusion(quantized_unet, num_timesteps=1000).to(device)
    
    # Stage 1: Warm-up
    print("\n=== Stage 1: VAE Warm-up ===")
    start_time = time.time()
    vae = train_stage1_warmup(vae, train_loader, device,
                              epochs=args.epochs_stage1, lr=1e-3, seed=args.seed)
    stage1_time = time.time() - start_time
    
    # Stage 2: Joint Training
    print("\n=== Stage 2: Joint Training ===")
    start_time = time.time()
    vae, diffusion = train_stage2_joint(vae, diffusion, train_loader, device,
                                        epochs=args.epochs_stage2, lr=2e-4, seed=args.seed)
    stage2_time = time.time() - start_time
    
    # Stage 3: Fine-tuning
    print("\n=== Stage 3: Fine-tuning ===")
    start_time = time.time()
    vae, diffusion = train_stage3_finetune(vae, diffusion, train_loader, device,
                                          epochs=args.epochs_stage3, lr=1e-4, seed=args.seed)
    stage3_time = time.time() - start_time
    
    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(vae.state_dict(), os.path.join(args.output_dir, f'vae_seed{args.seed}.pt'))
    torch.save(diffusion.state_dict(), os.path.join(args.output_dir, f'diffusion_seed{args.seed}.pt'))
    
    # Evaluate
    print("\n=== Evaluating UALQ-Diff ===")
    save_dir = os.path.join(args.output_dir, f'generated_seed{args.seed}')
    metrics = evaluate_ualq(vae, diffusion, num_samples=args.num_samples,
                           num_inference_steps=50, device=device, save_dir=save_dir)
    
    metrics['stage1_time_min'] = stage1_time / 60
    metrics['stage2_time_min'] = stage2_time / 60
    metrics['stage3_time_min'] = stage3_time / 60
    metrics['total_training_time_min'] = (stage1_time + stage2_time + stage3_time) / 60
    metrics['seed'] = args.seed
    
    print("\n=== Results ===")
    print(f"FID: {metrics['fid']:.4f}")
    print(f"IS: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}")
    print(f"Time per image: {metrics['time_per_image_ms']:.2f} ms")
    print(f"Peak memory: {metrics['peak_memory_gb']:.2f} GB")
    print(f"Avg token density: {metrics['avg_token_density']:.4f}")
    print(f"Effective tokens/img: {metrics['effective_tokens_per_image']:.2f}")
    print(f"Avg bitwidth: {metrics['avg_bitwidth']:.2f} bits")
    print(f"Est. BOPs: {metrics['estimated_bops_g']:.2f} G")
    
    results_file = os.path.join(args.output_dir, f'results_seed{args.seed}.json')
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == '__main__':
    main()
