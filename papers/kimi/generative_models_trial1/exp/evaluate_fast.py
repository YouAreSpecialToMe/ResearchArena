"""Fast evaluation script for FlowRouter models."""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.flowrouter import FlowRouterDiT
from src.dit import DiT_S_2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_cifar10_loader(batch_size=100, train=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data/cifar10', train=train, download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def generate_samples(model, num_samples=5000, batch_size=100, device='cuda', num_steps=50):
    """Generate samples using Euler integration for flow matching."""
    model.eval()
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating samples"):
            B = min(batch_size, num_samples - len(all_samples) * batch_size)
            # Random class labels
            y = torch.randint(0, 10, (B,), device=device)
            # Start from noise
            x = torch.randn(B, 3, 32, 32, device=device)
            
            # Euler integration
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.ones(B, device=device) * (i * dt)
                v = model(x, t, y)
                x = x + v * dt
            
            all_samples.append(x.cpu())
    
    return torch.cat(all_samples, dim=0)[:num_samples]

def compute_fid_score(real_images, fake_images, device='cuda', batch_size=50):
    """Compute FID score between real and fake images."""
    try:
        from torchvision.models import inception_v3
        from scipy import linalg
        
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        inception.fc = torch.nn.Identity()  # Remove final layer
        
        def get_features(images):
            features = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                # Resize to 299x299 for Inception
                batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    feat = inception(batch)
                features.append(feat.cpu().numpy())
            return np.concatenate(features, axis=0)
        
        real_features = get_features(real_images)
        fake_features = get_features(fake_images)
        
        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Compute FID
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return float(fid)
    except Exception as e:
        print(f"FID computation failed: {e}")
        return -1.0

def evaluate_model(model_path, seed=42, device='cuda', num_samples=5000):
    """Evaluate a FlowRouter model."""
    set_seed(seed)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = FlowRouterDiT(input_size=32, num_classes=10, use_velocity=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Measure FLOPs (simplified estimation)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    dummy_t = torch.tensor([0.5], device=device)
    dummy_y = torch.tensor([0], device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    router_params = sum(p.numel() for n, p in model.named_parameters() if 'router' in n or 'threshold' in n)
    
    # Estimate FLOPs (very rough approximation)
    # DiT-S/2 has about 33M params, ~6 GFLOPs per forward pass
    # With 50 steps, that's ~300 GFLOPs total for generation
    baseline_gflops = 6.0  # Per step
    
    # Get average skip rate from a few test batches
    test_loader = get_cifar10_loader(batch_size=100)
    skip_rates = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= 10:  # Just use 10 batches for estimation
                break
            x, y = x.to(device), y.to(device)
            t = torch.rand(x.shape[0], device=device)
            _, stats = model(x, t, y, return_routing_stats=True)
            skip_rates.append(stats['avg_skip_rate'])
    
    avg_skip_rate = np.mean(skip_rates)
    estimated_flops_ratio = 1.0 - avg_skip_rate
    
    # Generate samples and measure time
    print(f"\nGenerating {num_samples} samples...")
    start_time = time.time()
    fake_samples = generate_samples(model, num_samples=num_samples, device=device, num_steps=50)
    generation_time = time.time() - start_time
    
    # Get real samples for FID
    print("Loading real samples...")
    real_loader = get_cifar10_loader(batch_size=100, train=False)
    real_samples = []
    for i, (x, _) in enumerate(real_loader):
        real_samples.append(x)
        if len(real_samples) * 100 >= num_samples:
            break
    real_samples = torch.cat(real_samples, dim=0)[:num_samples]
    
    # Compute FID
    print("Computing FID...")
    fid_score = compute_fid_score(real_samples, fake_samples, device=device)
    
    results = {
        'model_path': model_path,
        'seed': seed,
        'fid': fid_score,
        'generation_time': generation_time,
        'avg_skip_rate': avg_skip_rate,
        'estimated_flops_ratio': estimated_flops_ratio,
        'total_params': total_params,
        'router_params': router_params,
        'checkpoint_epoch': checkpoint.get('epoch', -1),
    }
    
    print(f"\n=== Results ===")
    print(f"FID: {fid_score:.2f}")
    print(f"Generation time ({num_samples} samples): {generation_time:.1f}s")
    print(f"Avg skip rate: {avg_skip_rate:.2%}")
    print(f"Est. FLOPs ratio: {estimated_flops_ratio:.2%}")
    
    return results

def evaluate_baseline(seed=42, device='cuda', num_samples=5000):
    """Evaluate baseline DiT model."""
    from src.dit import DiT_S_2
    
    set_seed(seed)
    
    # Load baseline
    baseline_path = f'checkpoints/dit_baseline_seed{seed}.pt'
    checkpoint = torch.load(baseline_path, map_location=device)
    model = DiT_S_2(num_classes=10).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded baseline from {baseline_path}")
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples (baseline)...")
    start_time = time.time()
    fake_samples = []
    num_batches = (num_samples + 100 - 1) // 100
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating"):
            B = min(100, num_samples - len(fake_samples) * 100)
            y = torch.randint(0, 10, (B,), device=device)
            x = torch.randn(B, 3, 32, 32, device=device)
            
            # Euler integration (50 steps)
            dt = 1.0 / 50
            for i in range(50):
                t = torch.ones(B, device=device) * (i * dt)
                v = model(x, t, y)
                x = x + v * dt
            
            fake_samples.append(x.cpu())
    
    fake_samples = torch.cat(fake_samples, dim=0)[:num_samples]
    generation_time = time.time() - start_time
    
    # Get real samples
    real_loader = get_cifar10_loader(batch_size=100, train=False)
    real_samples = []
    for i, (x, _) in enumerate(real_loader):
        real_samples.append(x)
        if len(real_samples) * 100 >= num_samples:
            break
    real_samples = torch.cat(real_samples, dim=0)[:num_samples]
    
    # Compute FID
    print("Computing FID...")
    fid_score = compute_fid_score(real_samples, fake_samples, device=device)
    
    results = {
        'model': 'baseline',
        'seed': seed,
        'fid': fid_score,
        'generation_time': generation_time,
    }
    
    print(f"\n=== Baseline Results ===")
    print(f"FID: {fid_score:.2f}")
    print(f"Generation time ({num_samples} samples): {generation_time:.1f}s")
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--baseline', action='store_true', help='Evaluate baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.baseline:
        results = evaluate_baseline(seed=args.seed, device=device, num_samples=args.num_samples)
    elif args.model_path:
        results = evaluate_model(args.model_path, seed=args.seed, device=device, num_samples=args.num_samples)
    else:
        # Evaluate both baseline and FlowRouter
        print("=== Evaluating Baseline ===")
        baseline_results = evaluate_baseline(seed=args.seed, device=device, num_samples=args.num_samples)
        
        print("\n=== Evaluating FlowRouter ===")
        model_path = f'checkpoints/flowrouter_seed{args.seed}.pt'
        if os.path.exists(model_path):
            flowrouter_results = evaluate_model(model_path, seed=args.seed, device=device, num_samples=args.num_samples)
        else:
            print(f"FlowRouter checkpoint not found at {model_path}")
            flowrouter_results = None
        
        results = {
            'baseline': baseline_results,
            'flowrouter': flowrouter_results,
        }
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
