"""
Extract representations from pre-trained models.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pickle

from shared.utils import set_seed, save_results, Timer
from shared.models import ResNetFeatureExtractor, ViTFeatureExtractor, GPT2FeatureExtractor, load_resnet18, load_vit_tiny, load_gpt2
from transformers import GPT2Tokenizer, ViTModel


def extract_vision_representations(
    model_name: str,
    dataset_name: str,
    n_samples: int = 5000,
    seed: int = 42,
    device: str = 'cuda'
):
    """Extract representations from vision models."""
    set_seed(seed)
    
    print(f"\nExtracting {model_name} representations from {dataset_name}...")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='data/vision', train=True, download=False, transform=transform
        )
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='data/vision', train=True, download=False, transform=transform
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Sample subset
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
    
    # Extract labels
    labels = [dataset.targets[i] for i in indices]
    labels = np.array(labels)
    
    # Load model and extract features
    if model_name == 'resnet18':
        model, _ = load_resnet18(device)
        extractor = ResNetFeatureExtractor(model)
        features = extractor.extract(loader)
    elif model_name == 'vit_tiny':
        model = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224').to(device)
        extractor = ViTFeatureExtractor(model)
        features = extractor.extract(loader)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"  Extracted features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Save
    results = {
        'features': features,
        'labels': labels,
        'indices': indices,
        'model': model_name,
        'dataset': dataset_name,
        'n_samples': n_samples
    }
    
    save_dir = f'results/representations/{model_name}_{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    np.savez(f'{save_dir}/seed{seed}.npz', **results)
    
    return results


def extract_language_representations(
    concept_type: str = 'days',
    seed: int = 42,
    device: str = 'cuda'
):
    """Extract representations from GPT-2 for cyclic concepts."""
    set_seed(seed)
    
    print(f"\nExtracting GPT-2 representations for {concept_type}...")
    
    # Load cyclic concepts
    with open('data/language/cyclic_concepts.pkl', 'rb') as f:
        data = pickle.load(f)
    
    prompts = data[concept_type]['prompts']
    labels = data[concept_type]['labels']
    
    # Load model
    model, _ = load_gpt2(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    extractor = GPT2FeatureExtractor(model, layer_idx=-1)
    features = extractor.extract_from_texts(prompts, tokenizer, batch_size=32)
    
    print(f"  Extracted features: {features.shape}")
    print(f"  Labels: {len(labels)}")
    
    # Save
    results = {
        'features': features,
        'labels': np.array(labels),
        'prompts': prompts,
        'model': 'gpt2',
        'concept_type': concept_type
    }
    
    save_dir = f'results/representations/gpt2_{concept_type}'
    os.makedirs(save_dir, exist_ok=True)
    np.savez(f'{save_dir}/seed{seed}.npz', **{k: v for k, v in results.items() if k != 'prompts'})
    
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Vision models
    for seed in [42, 123, 456]:
        for model in ['resnet18', 'vit_tiny']:
            for dataset in ['cifar10', 'cifar100']:
                try:
                    extract_vision_representations(
                        model, dataset, n_samples=5000, seed=seed, device=device
                    )
                except Exception as e:
                    print(f"Error extracting {model}/{dataset} (seed {seed}): {e}")
    
    # Language models
    for seed in [42, 123, 456]:
        for concept in ['days', 'months', 'numbers']:
            try:
                extract_language_representations(concept, seed=seed, device=device)
            except Exception as e:
                print(f"Error extracting {concept} (seed {seed}): {e}")
    
    print("\nRepresentation extraction complete!")


if __name__ == "__main__":
    main()
