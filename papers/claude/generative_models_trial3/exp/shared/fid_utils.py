"""
FID and IS computation using ADM's precomputed reference statistics.
"""
import torch
import numpy as np
import os
from scipy import linalg
from pathlib import Path


def compute_fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref):
    """Compute FID between two sets of statistics."""
    diff = mu_gen - mu_ref
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_ref, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean)
    return float(fid)


def get_inception_model(device='cuda'):
    """Get InceptionV3 model for feature extraction."""
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def compute_inception_features(images_dir, device='cuda', batch_size=64):
    """Compute Inception features from a directory of PNG images."""
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model = get_inception_model(device)

    # Load all images
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    features_list = []

    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i:i+batch_size]
        batch_tensors = []
        for f in batch_files:
            img = Image.open(os.path.join(images_dir, f)).convert('RGB')
            batch_tensors.append(transform(img))

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = model(batch)
        features_list.append(feats.cpu().numpy())
        del batch
        torch.cuda.empty_cache()

    features = np.concatenate(features_list, axis=0)
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)

    del model
    torch.cuda.empty_cache()

    return mu, sigma, features


def compute_fid_against_imagenet(images_dir, ref_stats_path, device='cuda'):
    """Compute FID of generated images against ImageNet 256 reference."""
    # Load reference stats
    ref_data = np.load(ref_stats_path)
    mu_ref = ref_data['mu']
    sigma_ref = ref_data['sigma']

    # Compute features for generated images
    mu_gen, sigma_gen, _ = compute_inception_features(images_dir, device)

    fid = compute_fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)
    return fid


def compute_inception_score(images_dir, device='cuda', batch_size=64, splits=10):
    """Compute Inception Score from a directory of PNG images."""
    from torchvision.models import inception_v3
    from torchvision import transforms
    from PIL import Image

    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    all_probs = []

    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i:i+batch_size]
        batch_tensors = []
        for f in batch_files:
            img = Image.open(os.path.join(images_dir, f)).convert('RGB')
            batch_tensors.append(transform(img))

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        del batch
        torch.cuda.empty_cache()

    all_probs = np.concatenate(all_probs, axis=0)

    # Compute IS with splits
    scores = []
    chunk_size = len(all_probs) // splits
    for k in range(splits):
        part = all_probs[k * chunk_size: (k + 1) * chunk_size]
        kl = part * (np.log(part + 1e-10) - np.log(part.mean(axis=0, keepdims=True) + 1e-10))
        kl_sum = kl.sum(axis=1).mean()
        scores.append(np.exp(kl_sum))

    del model
    torch.cuda.empty_cache()

    return float(np.mean(scores)), float(np.std(scores))


def compute_all_metrics(images_dir, ref_stats_path, device='cuda'):
    """Compute FID and IS."""
    metrics = {}

    try:
        fid = compute_fid_against_imagenet(images_dir, ref_stats_path, device)
        metrics['fid'] = round(fid, 4)
        print(f"    FID: {fid:.2f}")
    except Exception as e:
        print(f"    FID error: {e}")
        metrics['fid'] = float('nan')

    try:
        is_mean, is_std = compute_inception_score(images_dir, device)
        metrics['is_mean'] = round(is_mean, 4)
        metrics['is_std'] = round(is_std, 4)
        print(f"    IS: {is_mean:.2f} +/- {is_std:.2f}")
    except Exception as e:
        print(f"    IS error: {e}")
        metrics['is_mean'] = float('nan')
        metrics['is_std'] = float('nan')

    return metrics
