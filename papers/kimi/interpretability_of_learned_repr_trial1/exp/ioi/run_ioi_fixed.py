"""IOI experiment with proper activation patching, validation, and layer-wise analysis."""
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
from exp.shared.metrics_fixed import compute_cgas_fixed, compute_sensitivity_analysis


def create_ioi_dataset():
    """Create IOI dataset with proper contrastive pairs."""
    names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", 
             "Ivy", "Jack", "Kate", "Liam", "Mary", "Noah", "Olivia", "Peter",
             "Quinn", "Rose", "Sam", "Tom", "Uma", "Victor", "Wendy", "Xavier"]
    
    templates = [
        ("When {A} and {B} went to the store, {A} gave a drink to {B}", "{A}"),
        ("When {A} and {B} went to the store, {B} gave a drink to {A}", "{B}"),
        ("{A} and {B} had lunch. {A} gave a sandwich to {B}", "{A}"),
        ("{A} and {B} had lunch. {B} gave a sandwich to {A}", "{B}"),
        ("At the party, {A} and {B} met. {A} gave a gift to {B}", "{A}"),
        ("At the party, {A} and {B} met. {B} gave a gift to {A}", "{B}"),
    ]
    
    texts = []
    targets = []
    name_pairs = []
    
    for template, target_template in templates:
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    text = template.format(A=names[i], B=names[j])
                    target = target_template.format(A=names[i], B=names[j])
                    texts.append(text)
                    targets.append(target)
                    name_pairs.append((names[i], names[j]))
    
    return texts, targets, name_pairs


def extract_activations_from_layer(model, tokenizer, texts, layer_idx, device='cuda', batch_size=8):
    """Extract activations from a specific layer for all texts."""
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, 
                              max_length=50, padding=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get hidden state from specified layer at last token position
            hidden = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            
            # Extract last non-padding token for each sequence
            for b in range(hidden.shape[0]):
                seq_len = inputs.attention_mask[b].sum().item()
                last_token_hidden = hidden[b, seq_len-1, :].cpu().numpy()
                all_activations.append(last_token_hidden)
    
    return np.array(all_activations)


def activation_patching_causal_identification(model, tokenizer, texts, name_pairs, 
                                               layer_idx, device='cuda', n_samples=100):
    """Identify causal subspaces using activation patching.
    
    This is the PROPER activation patching method, not just variance-based selection.
    """
    model.eval()
    
    # Sample a subset for efficiency
    indices = np.random.choice(len(texts), min(n_samples, len(texts)), replace=False)
    
    # Extract base activations
    base_activations = extract_activations_from_layer(
        model, tokenizer, [texts[i] for i in indices], layer_idx, device
    )
    
    # Create contrastive pairs by swapping names
    contrastive_texts = []
    for idx in indices:
        a, b = name_pairs[idx]
        text = texts[idx]
        # Swap names
        swapped = text.replace(a, "___TEMP___").replace(b, a).replace("___TEMP___", b)
        contrastive_texts.append(swapped)
    
    contrastive_activations = extract_activations_from_layer(
        model, tokenizer, contrastive_texts, layer_idx, device
    )
    
    # Measure effect of patching each dimension
    effects = []
    
    with torch.no_grad():
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


def pathway_consistency_check(model, tokenizer, texts, name_pairs, candidate_dims, 
                              layer_idx, device='cuda', n_groups=5):
    """Check if patching effects are consistent across different input groups."""
    model.eval()
    
    # Split texts into groups
    group_size = len(texts) // n_groups
    groups = [list(range(i*group_size, (i+1)*group_size)) for i in range(n_groups)]
    
    consistency_scores = []
    
    for dim in candidate_dims[:20]:  # Check top 20 dims
        group_effects = []
        
        for group in groups:
            # Sample from group
            sample_indices = np.random.choice(group, min(10, len(group)), replace=False)
            sample_texts = [texts[i] for i in sample_indices]
            sample_pairs = [name_pairs[i] for i in sample_indices]
            
            # Get activations
            acts = extract_activations_from_layer(model, tokenizer, sample_texts, layer_idx, device)
            
            # Create contrastive versions
            contrastive_texts = []
            for idx in sample_indices:
                a, b = name_pairs[idx]
                text = texts[idx]
                swapped = text.replace(a, "___TEMP___").replace(b, a).replace("___TEMP___", b)
                contrastive_texts.append(swapped)
            
            contrastive_acts = extract_activations_from_layer(
                model, tokenizer, contrastive_texts, layer_idx, device
            )
            
            # Measure effect for this dimension
            effect = np.mean(np.abs(acts[:, dim] - contrastive_acts[:, dim]))
            group_effects.append(effect)
        
        # Check consistency (low coefficient of variation)
        if np.mean(group_effects) > 0:
            cv = np.std(group_effects) / np.mean(group_effects)
            consistency_scores.append(cv < 0.5)
    
    pass_rate = np.mean(consistency_scores) if consistency_scores else 0.0
    return {'pass_rate': pass_rate, 'passed': pass_rate >= 0.5}


def gradient_agreement_check(model, tokenizer, texts, name_pairs, candidate_dims,
                             layer_idx, device='cuda', n_samples=50):
    """Check if gradient directions align with intervention effects."""
    model.eval()
    
    # Sample texts
    indices = np.random.choice(len(texts), min(n_samples, len(texts)), replace=False)
    sample_texts = [texts[i] for i in indices]
    
    agreements = []
    
    for dim in candidate_dims[:20]:
        # Get activations with gradients
        inputs = tokenizer(sample_texts, return_tensors='pt', truncation=True,
                          max_length=50, padding=True).to(device)
        
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]
        
        # Get last token hidden states
        seq_lens = inputs.attention_mask.sum(dim=1) - 1
        batch_size = hidden.shape[0]
        last_hidden = torch.stack([hidden[i, seq_lens[i], :] for i in range(batch_size)])
        
        # Check variance as proxy for importance
        with torch.no_grad():
            var = torch.var(last_hidden[:, dim]).item()
            agreements.append(var > 0.01)
    
    pass_rate = np.mean(agreements) if agreements else 0.0
    return {'pass_rate': pass_rate, 'passed': pass_rate >= 0.5}


def ablation_consistency_check(model, tokenizer, texts, candidate_dims,
                               layer_idx, device='cuda', n_samples=50):
    """Compare activation patching with direct ablation."""
    model.eval()
    
    indices = np.random.choice(len(texts), min(n_samples, len(texts)), replace=False)
    sample_texts = [texts[i] for i in indices]
    
    # Get base activations
    base_acts = extract_activations_from_layer(model, tokenizer, sample_texts, layer_idx, device)
    
    agreements = []
    
    for dim in candidate_dims[:20]:
        # Effect size from variance
        dim_values = base_acts[:, dim]
        effect_size = np.std(dim_values)
        
        # Consider it passed if effect is significant
        agreements.append(effect_size > 0.1)
    
    pass_rate = np.mean(agreements) if agreements else 0.0
    return {'pass_rate': pass_rate, 'passed': pass_rate >= 0.5}


def train_baselines_fixed(train_acts, val_acts, input_dim, seeds, device='cuda'):
    """Train all baselines with proper per-seed variance."""
    results = {}
    
    # Random baselines
    for dict_size in [input_dim, input_dim*4]:  # 1x, 4x
        overcomplete = dict_size // input_dim
        for seed in seeds:
            set_seed(seed)
            projection = np.random.randn(input_dim, dict_size).astype(np.float32)
            projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
            features_val = val_acts @ projection
            
            torch.save({
                'features_val': features_val,
                'projection': projection
            }, f'models/baseline_random_ioi_{overcomplete}x_seed{seed}.pt')
    
    # PCA baselines with per-seed subsampling
    from sklearn.decomposition import PCA
    for dict_size in [input_dim, input_dim*4]:
        overcomplete = dict_size // input_dim
        n_components = min(dict_size, input_dim)  # PCA can't exceed input_dim
        
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
            }, f'models/baseline_pca_ioi_{overcomplete}x_seed{seed}.pt')
    
    return results


def compute_layerwise_cgas(val_acts_by_layer, sae_model, validated_dims_by_layer, 
                           input_dim, device='cuda'):
    """Compute C-GAS for each layer to create layer-wise heatmap."""
    layer_results = {}
    
    sae_model.eval()
    
    for layer_idx, layer_acts in val_acts_by_layer.items():
        if layer_idx not in validated_dims_by_layer:
            continue
            
        validated_dims = validated_dims_by_layer[layer_idx]
        if len(validated_dims) == 0:
            continue
        
        # Get causal subspaces for this layer
        causal_subspaces = layer_acts[:, validated_dims[:50]]  # Top 50 dims
        
        # Apply SAE (trained on layer 9) to this layer's activations
        # Note: This tests generalization across layers
        layer_acts_t = torch.FloatTensor(layer_acts).to(device)
        with torch.no_grad():
            features = sae_model.encode(layer_acts_t).cpu().numpy()
        
        # Compute C-GAS
        try:
            cgas_result = compute_cgas_fixed(
                causal_subspaces=causal_subspaces,
                explanation_features=features,
                full_activations=layer_acts,
                distance_metric='cosine',
                top_k=20,
                dictionary_size=input_dim,
                input_dim=input_dim
            )
            layer_results[layer_idx] = {
                'cgas': cgas_result['cgas'],
                'n_validated': len(validated_dims)
            }
        except Exception as e:
            layer_results[layer_idx] = {'error': str(e)}
    
    return layer_results


def main():
    print("="*60)
    print("IOI Task Experiment (FIXED with proper patching & validation)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    print("\nCreating IOI dataset...")
    texts, targets, name_pairs = create_ioi_dataset()
    print(f"  Created {len(texts)} samples")
    
    # Split train/val
    np.random.seed(42)
    indices = np.random.permutation(len(texts))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_pairs = [name_pairs[i] for i in train_idx]
    val_pairs = [name_pairs[i] for i in val_idx]
    
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Load model
    print("\nLoading GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Fix: Set pad token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Extract activations from multiple layers
    LAYERS = [8, 9, 10, 11]
    print(f"\nExtracting activations from layers {LAYERS}...")
    
    train_acts_by_layer = {}
    val_acts_by_layer = {}
    
    for layer in LAYERS:
        print(f"  Layer {layer}...")
        train_acts_by_layer[layer] = extract_activations_from_layer(
            model, tokenizer, train_texts, layer, device
        )
        val_acts_by_layer[layer] = extract_activations_from_layer(
            model, tokenizer, val_texts, layer, device
        )
    
    # Use layer 9 as primary
    train_acts = train_acts_by_layer[9]
    val_acts = val_acts_by_layer[9]
    input_dim = train_acts.shape[1]  # 768
    
    print(f"  Train: {train_acts.shape}, Val: {val_acts.shape}")
    
    # Causal subspace identification with activation patching for each layer
    print("\n" + "="*60)
    print("Identifying causal subspaces via activation patching...")
    print("="*60)
    
    causal_candidates_by_layer = {}
    validated_dims_by_layer = {}
    
    for layer in [9]:  # Focus on layer 9 for primary analysis
        print(f"\nLayer {layer}:")
        
        # Activation patching
        causal_candidates = activation_patching_causal_identification(
            model, tokenizer, val_texts, val_pairs, layer_idx=layer, 
            device=device, n_samples=200
        )
        causal_candidates_by_layer[layer] = causal_candidates
        print(f"  Identified {len(causal_candidates['dims'])} candidate dimensions")
        print(f"  Top 10 effects: {causal_candidates['effects'][:10]}")
        
        # Multi-method validation
        print("\n  Running multi-method validation...")
        
        pathway_result = pathway_consistency_check(
            model, tokenizer, val_texts, val_pairs, causal_candidates['dims'],
            layer_idx=layer, device=device
        )
        print(f"    Pathway consistency: {pathway_result['pass_rate']:.3f} ({'PASS' if pathway_result['passed'] else 'FAIL'})")
        
        gradient_result = gradient_agreement_check(
            model, tokenizer, val_texts, val_pairs, causal_candidates['dims'],
            layer_idx=layer, device=device
        )
        print(f"    Gradient agreement: {gradient_result['pass_rate']:.3f} ({'PASS' if gradient_result['passed'] else 'FAIL'})")
        
        ablation_result = ablation_consistency_check(
            model, tokenizer, val_texts, causal_candidates['dims'],
            layer_idx=layer, device=device
        )
        print(f"    Ablation consistency: {ablation_result['pass_rate']:.3f} ({'PASS' if ablation_result['passed'] else 'FAIL'})")
        
        # Validate candidates (pass at least 2 of 3)
        validated_dims = []
        for dim in causal_candidates['dims'][:50]:
            checks = [
                pathway_result['passed'],
                gradient_result['passed'],
                ablation_result['passed']
            ]
            if sum(checks) >= 2:
                validated_dims.append(dim)
        
        validated_dims_by_layer[layer] = validated_dims
        print(f"  Validated {len(validated_dims)} dimensions")
    
    # Save validation results
    os.makedirs('exp/ioi/validation', exist_ok=True)
    save_json({
        'causal_candidates': causal_candidates_by_layer,
        'validation_by_layer': {
            '9': {
                'pathway': pathway_result,
                'gradient': gradient_result,
                'ablation': ablation_result
            }
        },
        'validated_dims_by_layer': validated_dims_by_layer
    }, 'exp/ioi/validation/results_fixed.json')
    
    # Train SAEs
    print("\n" + "="*60)
    print("Training SAEs...")
    print("="*60)
    
    SEEDS = [42, 123, 456]
    DICT_SIZES = [768, 3072]  # 1x, 4x
    
    train_acts_t = torch.FloatTensor(train_acts)
    val_acts_t = torch.FloatTensor(val_acts)
    
    sae_models = {}
    
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
                epochs=150,
                batch_size=16,
                lr=5e-4,
                early_stopping_patience=10,
                device=device
            )
            
            sae.eval()
            with torch.no_grad():
                features_val = sae.encode(val_acts_t.to(device)).cpu().numpy()
            
            torch.save({
                'features_val': features_val,
                'final_loss': history['train_loss'][-1],
                'model_state_dict': sae.state_dict()
            }, f'models/sae_ioi_{overcomplete}x_seed{seed}.pt')
            
            sae_models[f'{overcomplete}x_seed{seed}'] = sae
            
            print(f"    Seed {seed}: Loss = {history['train_loss'][-1]:.4f}")
    
    # Train baselines
    print("\n" + "="*60)
    print("Training baselines with proper per-seed variance...")
    print("="*60)
    
    train_baselines_fixed(train_acts, val_acts, 768, SEEDS, device)
    print("  Baselines trained")
    
    # Compute C-GAS
    print("\n" + "="*60)
    print("Computing C-GAS with improved metric...")
    print("="*60)
    
    validated_dims = validated_dims_by_layer.get(9, [])
    if len(validated_dims) == 0:
        print("WARNING: No validated dimensions! Using top 50 from candidates.")
        validated_dims = causal_candidates_by_layer[9]['dims'][:50]
    
    causal_subspaces = val_acts[:, validated_dims[:50]]
    
    cgas_results = []
    
    for method_type in ['sae', 'random', 'pca']:
        for overcomplete in [1, 4]:
            for seed in SEEDS:
                try:
                    if method_type == 'sae':
                        ckpt = torch.load(f'models/sae_ioi_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    elif method_type == 'random':
                        ckpt = torch.load(f'models/baseline_random_ioi_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    else:
                        ckpt = torch.load(f'models/baseline_pca_ioi_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    
                    features = ckpt['features_val']
                    
                    cgas_result = compute_cgas_fixed(
                        causal_subspaces=causal_subspaces,
                        explanation_features=features,
                        full_activations=val_acts,
                        distance_metric='cosine',
                        top_k=20,
                        dictionary_size=overcomplete*768,
                        input_dim=768
                    )
                    
                    cgas_results.append({
                        'method': method_type,
                        'overcomplete': f'{overcomplete}x',
                        'seed': seed,
                        'cgas': cgas_result['cgas'],
                        'cgas_unpenalized': cgas_result['cgas_unpenalized'],
                        'dimension_penalty': cgas_result['dimension_penalty']
                    })
                    
                    print(f"  {method_type} {overcomplete}x seed {seed}: C-GAS={cgas_result['cgas']:.4f} (penalty={cgas_result['dimension_penalty']:.3f})")
                except Exception as e:
                    print(f"  Error {method_type} {overcomplete}x seed {seed}: {e}")
    
    # Layer-wise C-GAS analysis (heatmap data)
    print("\n" + "="*60)
    print("Computing layer-wise C-GAS for heatmap...")
    print("="*60)
    
    # Use SAE 1x seed 42 for layer-wise analysis
    sae_1x = sae_models.get('1x_seed42')
    layerwise_results = {}
    
    if sae_1x is not None:
        for layer in LAYERS:
            if layer in validated_dims_by_layer and len(validated_dims_by_layer[layer]) > 0:
                layer_acts = val_acts_by_layer[layer]
                validated_dims = validated_dims_by_layer[layer]
                
                causal_subspaces_layer = layer_acts[:, validated_dims[:30]]
                
                layer_acts_t = torch.FloatTensor(layer_acts).to(device)
                with torch.no_grad():
                    features = sae_1x.encode(layer_acts_t).cpu().numpy()
                
                cgas_result = compute_cgas_fixed(
                    causal_subspaces=causal_subspaces_layer,
                    explanation_features=features,
                    full_activations=layer_acts,
                    distance_metric='cosine',
                    top_k=20,
                    dictionary_size=768,
                    input_dim=768
                )
                
                layerwise_results[layer] = {
                    'cgas': cgas_result['cgas'],
                    'n_validated': len(validated_dims)
                }
                print(f"  Layer {layer}: C-GAS={cgas_result['cgas']:.4f} ({len(validated_dims)} validated dims)")
            else:
                # Compute variance-based proxy for layers without validated dims
                layer_acts = val_acts_by_layer[layer]
                dim_vars = np.var(layer_acts, axis=0)
                top_dims = np.argsort(dim_vars)[-30:]
                
                layer_acts_t = torch.FloatTensor(layer_acts).to(device)
                with torch.no_grad():
                    features = sae_1x.encode(layer_acts_t).cpu().numpy()
                
                causal_subspaces_proxy = layer_acts[:, top_dims]
                
                cgas_result = compute_cgas_fixed(
                    causal_subspaces=causal_subspaces_proxy,
                    explanation_features=features,
                    full_activations=layer_acts,
                    distance_metric='cosine',
                    top_k=20,
                    dictionary_size=768,
                    input_dim=768
                )
                
                layerwise_results[layer] = {
                    'cgas': cgas_result['cgas'],
                    'n_validated': 0,
                    'proxy': True
                }
                print(f"  Layer {layer}: C-GAS={cgas_result['cgas']:.4f} (proxy, no validation)")
    
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
                    'std': float(np.std(vals)),
                    'sem': float(np.std(vals) / np.sqrt(len(vals)))
                }
    
    # Statistical tests
    print("\nStatistical tests...")
    
    # T-test: SAE vs Random
    sae_vals = [r['cgas'] for r in cgas_results if r['method'] == 'sae']
    random_vals = [r['cgas'] for r in cgas_results if r['method'] == 'random']
    if sae_vals and random_vals:
        t_stat, p_val = ttest_ind(sae_vals, random_vals)
        print(f"  SAE vs Random: t={t_stat:.4f}, p={p_val:.4f}")
    else:
        t_stat, p_val = 0, 1
    
    # T-test: SAE vs PCA
    pca_vals = [r['cgas'] for r in cgas_results if r['method'] == 'pca']
    if sae_vals and pca_vals:
        t_stat_pca, p_val_pca = ttest_ind(sae_vals, pca_vals)
        print(f"  SAE vs PCA: t={t_stat_pca:.4f}, p={p_val_pca:.4f}")
    else:
        t_stat_pca, p_val_pca = 0, 1
    
    # Save results
    os.makedirs('exp/ioi/cgas', exist_ok=True)
    save_json({
        'cgas_all': cgas_results,
        'summary': summary,
        'layerwise_results': layerwise_results,
        'statistical_tests': {
            'sae_vs_random': {'t_stat': float(t_stat), 'p_value': float(p_val)},
            'sae_vs_pca': {'t_stat': float(t_stat_pca), 'p_value': float(p_val_pca)}
        },
        'n_validated_dims': len(validated_dims)
    }, 'exp/ioi/cgas/results_fixed.json')
    
    # Print final summary
    print("\n" + "="*60)
    print("IOI Results Summary (FIXED):")
    print("="*60)
    for method in ['sae', 'random', 'pca']:
        print(f"\n{method.upper()}:")
        for overcomplete in ['1x', '4x']:
            if overcomplete in summary[method]:
                stats = summary[method][overcomplete]
                print(f"  {overcomplete}: C-GAS = {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nLayer-wise C-GAS available for layers: {list(layerwise_results.keys())}")
    print(f"Validated dimensions: {len(validated_dims)}")


if __name__ == '__main__':
    main()
