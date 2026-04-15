"""
Robustness evaluation for all trained models.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from exp.shared.models import TopKSAE, JumpReLUSAE, DenoisingSAE, RobustSAE
from exp.shared.data_loader import get_art_science_prompts
from exp.shared.utils import save_results

def load_model(model_type, checkpoint_path, device='cuda'):
    """Load a trained model."""
    # Load checkpoint to get dimensions
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer dimensions from state dict
    d_model = checkpoint['model_state_dict']['W_enc'].shape[0]
    d_sae = checkpoint['model_state_dict']['W_enc'].shape[1]
    
    if model_type == 'topk':
        model = TopKSAE(d_model, d_sae, topk=32)
    elif model_type == 'jumprelu':
        model = JumpReLUSAE(d_model, d_sae)
    elif model_type == 'denoising':
        model = DenoisingSAE(d_model, d_sae, topk=32)
    elif model_type == 'robust':
        model = RobustSAE(d_model, d_sae, topk=32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def compute_robustness_metrics(model, prompts_a, prompts_b, tokenizer, 
                               llm_model, layer_idx, device='cuda', k=50):
    """
    Compute robustness metrics:
    1. Population overlap change
    2. Feature stability under perturbation
    """
    model.eval()
    
    # Get activations for both categories
    with torch.no_grad():
        tokens_a = tokenizer(prompts_a, return_tensors='pt', 
                            padding=True, truncation=True, max_length=128).to(device)
        tokens_b = tokenizer(prompts_b, return_tensors='pt',
                            padding=True, truncation=True, max_length=128).to(device)
        
        outputs_a = llm_model(**tokens_a, output_hidden_states=True)
        outputs_b = llm_model(**tokens_b, output_hidden_states=True)
        
        acts_a = outputs_a.hidden_states[layer_idx].mean(dim=1).float()
        acts_b = outputs_b.hidden_states[layer_idx].mean(dim=1).float()
        
        z_a, _ = model.encode(acts_a)
        z_b, _ = model.encode(acts_b)
        
        # Get top-k features
        _, topk_a = torch.topk(z_a, k, dim=-1)
        _, topk_b = torch.topk(z_b, k, dim=-1)
    
    # Compute baseline overlap (art vs science)
    baseline_overlaps = []
    for i in range(len(prompts_a)):
        set_a = set(topk_a[i].cpu().numpy())
        set_b = set(topk_b[i % len(prompts_b)].cpu().numpy())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union > 0 else 0
        baseline_overlaps.append(jaccard)
    
    baseline_overlap = np.mean(baseline_overlaps)
    
    # Simulate attack: Add science suffix to art prompts
    attack_suffix = " Scientific research shows empirical evidence."
    attacked_prompts = [p + attack_suffix for p in prompts_a]
    
    with torch.no_grad():
        tokens_attacked = tokenizer(attacked_prompts, return_tensors='pt',
                                   padding=True, truncation=True, max_length=128).to(device)
        outputs_attacked = llm_model(**tokens_attacked, output_hidden_states=True)
        acts_attacked = outputs_attacked.hidden_states[layer_idx].mean(dim=1).float()
        z_attacked, _ = model.encode(acts_attacked)
        _, topk_attacked = torch.topk(z_attacked, k, dim=-1)
    
    # Compute post-attack overlaps
    attacked_overlaps = []
    for i in range(len(prompts_a)):
        set_orig = set(topk_a[i].cpu().numpy())
        set_attacked = set(topk_attacked[i].cpu().numpy())
        intersection = len(set_orig & set_attacked)
        union = len(set_orig | set_attacked)
        jaccard = intersection / union if union > 0 else 0
        attacked_overlaps.append(jaccard)
    
    attacked_overlap = np.mean(attacked_overlaps)
    
    # Feature stability: measure representation distance
    with torch.no_grad():
        rep_dist = torch.norm(z_a - z_attacked, dim=1).mean().item()
        
        # L2 distance normalized by activation magnitude
        z_a_norm = torch.norm(z_a, dim=1)
        normalized_dist = (torch.norm(z_a - z_attacked, dim=1) / (z_a_norm + 1e-8)).mean().item()
    
    return {
        'baseline_overlap': baseline_overlap,
        'attacked_overlap': attacked_overlap,
        'overlap_change': abs(attacked_overlap - baseline_overlap),
        'representation_distance': rep_dist,
        'normalized_distance': normalized_dist,
        'feature_stability': 1.0 - normalized_dist  # Higher is better
    }

def evaluate_all_models(device='cuda'):
    # Load LLM for activation extraction
    model_name = "EleutherAI/pythia-70m-deduped"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        output_hidden_states=True
    )
    llm_model.eval()
    
    # Get layer idx (middle)
    layer_idx = 3
    
    # Get prompts
    art_prompts, science_prompts = get_art_science_prompts()
    
    # Models to evaluate
    models_to_eval = [
        ('topk', 'models/topk_baseline_seed42_best.pt'),
        ('jumprelu', 'models/jumprelu_baseline_seed42_best.pt'),
        ('denoising', 'models/denoising_baseline_seed42_best.pt'),
        ('robust', 'models/robustsae_full_seed42_best.pt'),
    ]
    
    results = {}
    
    for model_name_key, checkpoint_path in models_to_eval:
        print(f"\nEvaluating {model_name_key}...")
        try:
            model = load_model(model_name_key, checkpoint_path, device)
            metrics = compute_robustness_metrics(
                model, art_prompts, science_prompts, tokenizer,
                llm_model, layer_idx, device
            )
            results[model_name_key] = metrics
            print(f"  Overlap change: {metrics['overlap_change']:.4f}")
            print(f"  Feature stability: {metrics['feature_stability']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name_key] = {'error': str(e)}
    
    # Save results
    save_results(results, 'results/robustness_evaluation.json')
    print("\nResults saved to results/robustness_evaluation.json")
    
    return results

if __name__ == '__main__':
    evaluate_all_models()
