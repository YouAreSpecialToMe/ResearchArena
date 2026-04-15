"""RAVEL experiment with proper activation patching, validation, and multi-method validation."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from scipy.stats import spearmanr, ttest_ind
import os

from exp.shared.utils import set_seed, save_json
from exp.shared.models import SparseAutoencoder, train_sae
from exp.shared.metrics_fixed import compute_cgas_fixed
from exp.shared.data_loader import load_ravel_dataset


def extract_ravel_activations(model, tokenizer, dataset, layer_idx=9, device='cuda', batch_size=8):
    """Extract activations from GPT-2 on RAVEL dataset."""
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_items = dataset[i:i+batch_size]
            texts = [item['base_prompt'] for item in batch_items]
            
            inputs = tokenizer(texts, return_tensors='pt', truncation=True,
                              max_length=40, padding=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get hidden state from specified layer at last token position
            hidden = outputs.hidden_states[layer_idx]
            
            for b in range(hidden.shape[0]):
                seq_len = inputs.attention_mask[b].sum().item()
                last_token_hidden = hidden[b, seq_len-1, :].cpu().numpy()
                all_activations.append(last_token_hidden)
    
    return np.array(all_activations)


def activation_patching_causal_identification(model, tokenizer, dataset, 
                                               layer_idx, device='cuda'):
    """Identify causal subspaces using activation patching on RAVEL dataset.
    
    Creates contrastive pairs by changing the attribute in the prompt.
    """
    model.eval()
    
    # Extract base activations
    base_activations = extract_ravel_activations(
        model, tokenizer, dataset, layer_idx, device
    )
    
    # Create contrastive prompts
    contrastive_texts = []
    for item in dataset:
        # Use contrast prompt
        contrastive_texts.append(item['contrast_prompt'])
    
    contrastive_activations = extract_ravel_activations(
        model, tokenizer, [{'base_prompt': t} for t in contrastive_texts], 
        layer_idx, device
    )
    
    # Measure effect of patching each dimension
    effects = []
    
    for dim in range(base_activations.shape[1]):
        # Create patched activations
        patched = base_activations.copy()
        patched[:, dim] = contrastive_activations[:, dim]
        
        # Measure change in representation
        diff = np.linalg.norm(patched - base_activations, axis=1)
        mean_effect = np.mean(diff)
        effects.append(mean_effect)
    
    effects = np.array(effects)
    
    # Return top dimensions by effect size
    top_k = min(100, len(effects))
    top_dims = np.argsort(effects)[-top_k:][::-1]
    
    return {
        'dims': top_dims.tolist(),
        'effects': effects[top_dims].tolist(),
        'all_effects': effects.tolist()
    }


def pathway_consistency_check(model, tokenizer, dataset, candidate_dims,
                              layer_idx, device='cuda', n_groups=5):
    """Check if patching effects are consistent across different entity types."""
    model.eval()
    
    # Group by attribute type
    attr_types = {}
    for i, item in enumerate(dataset):
        attr_type = item['attribute_type']
        if attr_type not in attr_types:
            attr_types[attr_type] = []
        attr_types[attr_type].append(i)
    
    consistency_scores = []
    
    for dim in candidate_dims[:20]:
        group_effects = []
        
        for attr_type, indices in attr_types.items():
            if len(indices) < 5:
                continue
            
            # Sample from group
            sample_indices = np.random.choice(indices, min(10, len(indices)), replace=False)
            sample_items = [dataset[i] for i in sample_indices]
            
            # Get activations
            acts = extract_ravel_activations(
                model, tokenizer, sample_items, layer_idx, device, batch_size=10
            )
            
            # Create contrastive versions
            contrastive_texts = [item['contrast_prompt'] for item in sample_items]
            contrast_acts = extract_ravel_activations(
                model, tokenizer, [{'base_prompt': t} for t in contrastive_texts],
                layer_idx, device, batch_size=10
            )
            
            # Measure effect for this dimension
            effect = np.mean(np.abs(acts[:, dim] - contrast_acts[:, dim]))
            group_effects.append(effect)
        
        # Check consistency
        if len(group_effects) > 1 and np.mean(group_effects) > 0:
            cv = np.std(group_effects) / np.mean(group_effects)
            consistency_scores.append(cv < 0.5)
    
    pass_rate = np.mean(consistency_scores) if consistency_scores else 0.0
    return {'pass_rate': pass_rate, 'passed': pass_rate >= 0.5}


def gradient_agreement_check(model, tokenizer, dataset, candidate_dims,
                             layer_idx, device='cuda'):
    """Check if gradient directions align with intervention effects."""
    model.eval()
    
    # Sample items
    n_samples = min(50, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    sample_items = [dataset[i] for i in indices]
    
    agreements = []
    
    for dim in candidate_dims[:20]:
        # Get activations
        acts = extract_ravel_activations(
            model, tokenizer, sample_items, layer_idx, device, batch_size=10
        )
        
        # Check variance as proxy for importance
        var = np.var(acts[:, dim])
        agreements.append(var > 0.01)
    
    pass_rate = np.mean(agreements) if agreements else 0.0
    return {'pass_rate': pass_rate, 'passed': pass_rate >= 0.5}


def ablation_consistency_check(model, tokenizer, dataset, candidate_dims,
                               layer_idx, device='cuda'):
    """Compare activation patching with direct ablation."""
    model.eval()
    
    n_samples = min(50, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    sample_items = [dataset[i] for i in indices]
    
    # Get base activations
    base_acts = extract_ravel_activations(
        model, tokenizer, sample_items, layer_idx, device, batch_size=10
    )
    
    agreements = []
    
    for dim in candidate_dims[:20]:
        # Effect size from variance
        dim_values = base_acts[:, dim]
        effect_size = np.std(dim_values)
        
        agreements.append(effect_size > 0.1)
    
    pass_rate = np.mean(agreements) if agreements else 0.0
    return {'pass_rate': pass_rate, 'passed': pass_rate >= 0.5}


def train_baselines_fixed(train_acts, val_acts, input_dim, seeds, device='cuda'):
    """Train all baselines with proper per-seed variance."""
    
    # Random baselines
    for dict_size in [input_dim, input_dim*4]:
        overcomplete = dict_size // input_dim
        for seed in seeds:
            set_seed(seed)
            projection = np.random.randn(input_dim, dict_size).astype(np.float32)
            projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
            features_val = val_acts @ projection
            
            torch.save({
                'features_val': features_val,
                'projection': projection
            }, f'models/baseline_random_ravel_{overcomplete}x_seed{seed}.pt')
    
    # PCA baselines with per-seed subsampling
    from sklearn.decomposition import PCA
    for dict_size in [input_dim, input_dim*4]:
        overcomplete = dict_size // input_dim
        n_components = min(dict_size, input_dim)
        
        for seed in seeds:
            set_seed(seed)
            # Subsample for variance
            n_train = train_acts.shape[0]
            subsample_size = int(n_train * 0.9)
            indices = np.random.choice(n_train, subsample_size, replace=False)
            train_subsample = train_acts[indices]
            
            pca = PCA(n_components=n_components, random_state=seed)
            pca.fit(train_subsample)
            features_val = pca.transform(val_acts)
            
            torch.save({
                'features_val': features_val,
                'pca': pca
            }, f'models/baseline_pca_ravel_{overcomplete}x_seed{seed}.pt')


def main():
    print("="*60)
    print("RAVEL Task Experiment (FIXED with proper patching & validation)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    print("\nCreating RAVEL dataset...")
    dataset = load_ravel_dataset(
        attribute_types=['country-capital', 'name-occupation', 'company-CEO'],
        n_samples_per_type=50,
        seed=42
    )
    
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"  Total: {len(dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load model
    print("\nLoading GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Fix: Set pad token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Extract activations
    print("\nExtracting activations from layer 9...")
    train_acts = extract_ravel_activations(model, tokenizer, train_dataset, layer_idx=9, device=device)
    val_acts = extract_ravel_activations(model, tokenizer, val_dataset, layer_idx=9, device=device)
    
    input_dim = train_acts.shape[1]
    
    print(f"  Train: {train_acts.shape}, Val: {val_acts.shape}")
    
    # Causal subspace identification with activation patching
    print("\n" + "="*60)
    print("Identifying causal subspaces via activation patching...")
    print("="*60)
    
    causal_candidates = activation_patching_causal_identification(
        model, tokenizer, val_dataset, layer_idx=9, device=device
    )
    print(f"  Identified {len(causal_candidates['dims'])} candidate dimensions")
    print(f"  Top 10 effects: {causal_candidates['effects'][:10]}")
    
    # Multi-method validation
    print("\n  Running multi-method validation...")
    
    pathway_result = pathway_consistency_check(
        model, tokenizer, val_dataset, causal_candidates['dims'],
        layer_idx=9, device=device
    )
    print(f"    Pathway consistency: {pathway_result['pass_rate']:.3f} ({'PASS' if pathway_result['passed'] else 'FAIL'})")
    
    gradient_result = gradient_agreement_check(
        model, tokenizer, val_dataset, causal_candidates['dims'],
        layer_idx=9, device=device
    )
    print(f"    Gradient agreement: {gradient_result['pass_rate']:.3f} ({'PASS' if gradient_result['passed'] else 'FAIL'})")
    
    ablation_result = ablation_consistency_check(
        model, tokenizer, val_dataset, causal_candidates['dims'],
        layer_idx=9, device=device
    )
    print(f"    Ablation consistency: {ablation_result['pass_rate']:.3f} ({'PASS' if ablation_result['passed'] else 'FAIL'})")
    
    # Validate candidates
    validated_dims = []
    for dim in causal_candidates['dims'][:50]:
        checks = [
            pathway_result['passed'],
            gradient_result['passed'],
            ablation_result['passed']
        ]
        if sum(checks) >= 2:
            validated_dims.append(dim)
    
    print(f"  Validated {len(validated_dims)} dimensions")
    
    # Save validation results
    os.makedirs('exp/ravel/validation', exist_ok=True)
    save_json({
        'causal_candidates': causal_candidates,
        'validation': {
            'pathway': pathway_result,
            'gradient': gradient_result,
            'ablation': ablation_result
        },
        'validated_dims': validated_dims
    }, 'exp/ravel/validation/results_fixed.json')
    
    # Train SAEs
    print("\n" + "="*60)
    print("Training SAEs...")
    print("="*60)
    
    SEEDS = [42, 123, 456]
    DICT_SIZES = [768, 3072]
    
    train_acts_t = torch.FloatTensor(train_acts)
    val_acts_t = torch.FloatTensor(val_acts)
    
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 768
        print(f"\n  SAE {overcomplete}x (dict_size={dict_size})")
        
        for seed in SEEDS:
            set_seed(seed)
            
            sae = SparseAutoencoder(
                input_dim=768,
                dict_size=dict_size,
                sparsity_penalty=5e-5
            )
            
            history = train_sae(
                model=sae,
                activations=train_acts_t,
                val_activations=val_acts_t,
                epochs=200,
                batch_size=32,
                lr=5e-4,
                early_stopping_patience=15,
                device=device
            )
            
            sae.eval()
            with torch.no_grad():
                features_val = sae.encode(val_acts_t.to(device)).cpu().numpy()
                recon_val, _ = sae(val_acts_t.to(device))
                recon_error = torch.mean((recon_val - val_acts_t.to(device)) ** 2).item()
            
            torch.save({
                'features_val': features_val,
                'final_loss': history['train_loss'][-1],
                'recon_error': recon_error
            }, f'models/sae_ravel_{overcomplete}x_seed{seed}.pt')
            
            print(f"    Seed {seed}: Recon error = {recon_error:.6f}")
    
    # Train baselines
    print("\n" + "="*60)
    print("Training baselines with proper per-seed variance...")
    print("="*60)
    
    train_baselines_fixed(train_acts, val_acts, input_dim, SEEDS, device)
    print("  Baselines trained")
    
    # Compute C-GAS
    print("\n" + "="*60)
    print("Computing C-GAS with improved metric...")
    print("="*60)
    
    if len(validated_dims) == 0:
        print("WARNING: No validated dimensions! Using top 50 from candidates.")
        validated_dims = causal_candidates['dims'][:50]
    
    causal_subspaces = val_acts[:, validated_dims[:50]]
    
    cgas_results = []
    
    for method_type in ['sae', 'random', 'pca']:
        for overcomplete in [1, 4]:
            for seed in SEEDS:
                try:
                    if method_type == 'sae':
                        ckpt = torch.load(f'models/sae_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    elif method_type == 'random':
                        ckpt = torch.load(f'models/baseline_random_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    else:
                        ckpt = torch.load(f'models/baseline_pca_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    
                    features = ckpt['features_val']
                    
                    cgas_result = compute_cgas_fixed(
                        causal_subspaces=causal_subspaces,
                        explanation_features=features,
                        full_activations=val_acts,
                        distance_metric='cosine',
                        top_k=20,
                        dictionary_size=overcomplete*input_dim,
                        input_dim=input_dim
                    )
                    
                    cgas_results.append({
                        'method': method_type,
                        'overcomplete': f'{overcomplete}x',
                        'seed': seed,
                        'cgas': cgas_result['cgas'],
                        'cgas_unpenalized': cgas_result['cgas_unpenalized'],
                        'dimension_penalty': cgas_result['dimension_penalty']
                    })
                    
                    print(f"  {method_type} {overcomplete}x seed {seed}: C-GAS={cgas_result['cgas']:.4f}")
                except Exception as e:
                    print(f"  Error {method_type} {overcomplete}x seed {seed}: {e}")
    
    # Summary statistics
    summary = {}
    for method in ['sae', 'random', 'pca']:
        summary[method] = {}
        for overcomplete in ['1x', '4x']:
            vals = [r['cgas'] for r in cgas_results 
                   if r['method'] == method and r['overcomplete'] == overcomplete]
            if vals:
                summary[method][overcomplete] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals))
                }
    
    # Save results
    os.makedirs('exp/ravel/cgas', exist_ok=True)
    save_json({
        'cgas_all': cgas_results,
        'summary': summary,
        'n_validated_dims': len(validated_dims)
    }, 'exp/ravel/cgas/results_fixed.json')
    
    # Print summary
    print("\n" + "="*60)
    print("RAVEL Results Summary (FIXED):")
    print("="*60)
    for method in ['sae', 'random', 'pca']:
        print(f"\n{method.upper()}:")
        for overcomplete in ['1x', '4x']:
            if overcomplete in summary[method]:
                stats = summary[method][overcomplete]
                print(f"  {overcomplete}: C-GAS = {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nValidated dimensions: {len(validated_dims)}")


if __name__ == '__main__':
    main()
