"""
Evaluation Metrics for VAST Experiments
Includes FID, CLIP Score, and utility functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os


class FIDMetric:
    """Compute FID (Fréchet Inception Distance) using Inception features."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.inception = self._load_inception()
        
    def _load_inception(self):
        """Load InceptionV3 for feature extraction."""
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        model.fc = nn.Identity()  # Remove classification layer
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        """Preprocess images for Inception."""
        # Inception expects 299x299 images
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensors = [transform(img.convert('RGB')) for img in images]
        return torch.stack(tensors)
    
    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract Inception features from images."""
        batch_size = 32
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = self.preprocess(batch).to(self.device)
            
            # Get features
            features = self.inception(tensors)
            all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def compute_statistics(self, features: np.ndarray) -> tuple:
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """Calculate FID between two distributions."""
        from scipy import linalg
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)
    
    def compute(self, generated_images: List[Image.Image], 
                reference_images: Optional[List[Image.Image]] = None) -> Dict[str, float]:
        """
        Compute FID score.
        
        If reference_images is None, returns statistics for generated images only.
        """
        gen_features = self.extract_features(generated_images)
        gen_mu, gen_sigma = self.compute_statistics(gen_features)
        
        if reference_images is not None:
            ref_features = self.extract_features(reference_images)
            ref_mu, ref_sigma = self.compute_statistics(ref_features)
            fid = self.calculate_fid(gen_mu, gen_sigma, ref_mu, ref_sigma)
            return {'fid': fid}
        else:
            # Return statistics for later comparison
            return {
                'mu': gen_mu.tolist(),
                'sigma': gen_sigma.tolist(),
                'num_samples': len(generated_images)
            }


class CLIPMetric:
    """Compute CLIP score for text-image alignment."""
    
    def __init__(self, model_name: str = "ViT-B-32", device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = self._load_clip(model_name)
        
    def _load_clip(self, model_name: str):
        """Load CLIP model."""
        import open_clip
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='openai'
        )
        model = model.to(self.device)
        model.eval()
        
        return model, preprocess
    
    @torch.no_grad()
    def compute(self, images: List[Image.Image], texts: List[str]) -> Dict[str, float]:
        """Compute CLIP score between images and texts."""
        # Preprocess images
        image_tensors = torch.stack([
            self.preprocess(img.convert('RGB')) for img in images
        ]).to(self.device)
        
        # Tokenize texts
        text_tokens = open_clip.tokenize(texts).to(self.device)
        
        # Get embeddings
        image_features = self.model.encode_image(image_tensors)
        text_features = self.model.encode_text(text_tokens)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity
        similarity = (image_features * text_features).sum(dim=-1)
        
        # Scale to 0-100 range (standard CLIP score convention)
        scores = similarity.cpu().numpy() * 100
        
        return {
            'clip_score_mean': float(np.mean(scores)),
            'clip_score_std': float(np.std(scores)),
            'clip_scores': scores.tolist(),
        }


class MetricsEvaluator:
    """Unified evaluator for all metrics."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fid_metric = FIDMetric(device=device)
        self.clip_metric = CLIPMetric(device=device)
    
    def evaluate(
        self,
        generated_images: List[Image.Image],
        prompts: List[str],
        reference_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all metrics.
        
        Args:
            generated_images: List of generated PIL images
            prompts: List of text prompts
            reference_images: Optional reference images for FID
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Compute FID
        print("Computing FID...")
        fid_result = self.fid_metric.compute(generated_images, reference_images)
        results.update(fid_result)
        
        # Compute CLIP score
        print("Computing CLIP score...")
        clip_result = self.clip_metric.compute(generated_images, prompts)
        results.update(clip_result)
        
        return results
    
    def evaluate_batch(
        self,
        results_dir: str,
        prompts_file: str,
        reference_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate metrics for a batch of images.
        
        Args:
            results_dir: Directory containing generated images
            prompts_file: JSON file with prompts
            reference_dir: Optional directory with reference images
            
        Returns:
            Dictionary with metrics
        """
        # Load prompts
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            prompts = data if isinstance(data, list) else data.get('prompts', [])
        
        # Load generated images
        image_files = sorted([
            f for f in os.listdir(results_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        generated_images = [
            Image.open(os.path.join(results_dir, f)) for f in image_files
        ]
        
        # Load reference images if provided
        reference_images = None
        if reference_dir is not None:
            ref_files = sorted([
                f for f in os.listdir(reference_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            reference_images = [
                Image.open(os.path.join(reference_dir, f)) for f in ref_files
            ]
        
        # Ensure prompts match images
        prompts = prompts[:len(generated_images)]
        
        return self.evaluate(generated_images, prompts, reference_images)


def aggregate_results_across_seeds(
    results_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate results from multiple seeds.
    
    Args:
        results_list: List of result dictionaries from different seeds
        
    Returns:
        Aggregated results with mean and std
    """
    if not results_list:
        return {}
    
    aggregated = {}
    
    # Collect all scalar metrics
    scalar_metrics = {}
    for result in results_list:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in scalar_metrics:
                    scalar_metrics[key] = []
                scalar_metrics[key].append(value)
    
    # Compute mean and std
    for key, values in scalar_metrics.items():
        arr = np.array(values)
        aggregated[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'values': values,
        }
    
    return aggregated


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
