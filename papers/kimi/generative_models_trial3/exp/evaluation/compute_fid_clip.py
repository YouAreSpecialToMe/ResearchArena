#!/usr/bin/env python3
"""
Compute FID and CLIP scores for generated images.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def load_images(directory: str, max_images: int = None) -> list:
    """Load all images from a directory."""
    image_files = sorted([
        f for f in os.listdir(directory)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if max_images:
        image_files = image_files[:max_images]
    
    images = []
    for f in image_files:
        try:
            img = Image.open(os.path.join(directory, f)).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return images


class InceptionFeatureExtractor:
    """Extract features from Inception v3 for FID computation."""
    
    def __init__(self, device='cuda'):
        self.device = device
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract(self, images: list, batch_size: int = 32) -> np.ndarray:
        """Extract features from images."""
        features = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
            
            feat = self.model(tensors)
            features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)


def compute_fid(mu1, sigma1, mu2, sigma2) -> float:
    """Compute FID between two distributions."""
    from scipy import linalg
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_fid_between_dirs(real_dir: str, fake_dir: str, device='cuda') -> dict:
    """Compute FID between two directories of images."""
    print(f"Loading images from {real_dir}...")
    real_images = load_images(real_dir)
    print(f"Loaded {len(real_images)} real images")
    
    print(f"Loading images from {fake_dir}...")
    fake_images = load_images(fake_dir)
    print(f"Loaded {len(fake_images)} generated images")
    
    # Extract features
    extractor = InceptionFeatureExtractor(device=device)
    
    print("Extracting features from real images...")
    real_features = extractor.extract(real_images)
    
    print("Extracting features from generated images...")
    fake_features = extractor.extract(fake_images)
    
    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute FID
    fid = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return {
        'fid': fid,
        'num_real': len(real_images),
        'num_fake': len(fake_images),
    }


class CLIPScore:
    """Compute CLIP score for text-image alignment."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            self.model = self.model.to(device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        except ImportError:
            print("open_clip not available, trying alternative...")
            # Fallback: don't compute CLIP
            self.model = None
    
    @torch.no_grad()
    def compute(self, images: list, texts: list, batch_size: int = 32) -> dict:
        """Compute CLIP score between images and texts."""
        if self.model is None:
            return {'clip_score': None}
        
        scores = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Computing CLIP"):
            batch_images = images[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            # Process images
            image_tensors = torch.stack([
                self.preprocess(img) for img in batch_images
            ]).to(self.device)
            
            # Process texts
            text_tokens = self.tokenizer(batch_texts).to(self.device)
            
            # Get features
            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features * text_features).sum(dim=-1)
            
            # Scale to 0-100
            scores.extend((similarity * 100).cpu().numpy().tolist())
        
        return {
            'clip_score_mean': float(np.mean(scores)),
            'clip_score_std': float(np.std(scores)),
            'clip_scores': scores,
        }


def compute_metrics_for_experiment(
    output_dir: str,
    methods: list,
    seeds: list,
    prompts_file: str,
    reference_dir: str = None,
):
    """Compute metrics for all methods."""
    print("="*60)
    print("COMPUTING METRICS")
    print("="*60)
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        
        method_results = {}
        
        for seed in seeds:
            image_dir = os.path.join(output_dir, f"{method}_seed{seed}")
            
            if not os.path.exists(image_dir):
                print(f"Directory not found: {image_dir}")
                continue
            
            print(f"\nSeed {seed}:")
            
            # Load images
            images = load_images(image_dir)
            if len(images) == 0:
                print(f"No images found in {image_dir}")
                continue
            
            # Get corresponding prompts
            texts = prompts[:len(images)]
            
            # Compute CLIP score
            clip_scorer = CLIPScore()
            clip_result = clip_scorer.compute(images, texts)
            
            print(f"  CLIP Score: {clip_result.get('clip_score_mean', 'N/A')}")
            
            method_results[f'seed_{seed}'] = clip_result
        
        results[method] = method_results
    
    # Save results
    output_file = os.path.join(output_dir, 'metrics_clip.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMetrics saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['baseline_50step', 'baseline_25step', 'vast_2x', 'vast_3x'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--prompts', type=str, required=True)
    parser.add_argument('--reference_dir', type=str, default=None)
    args = parser.parse_args()
    
    compute_metrics_for_experiment(
        args.output_dir,
        args.methods,
        args.seeds,
        args.prompts,
        args.reference_dir,
    )


if __name__ == '__main__':
    main()
