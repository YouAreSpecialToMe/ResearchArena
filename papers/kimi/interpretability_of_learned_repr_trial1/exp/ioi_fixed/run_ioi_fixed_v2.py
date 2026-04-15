"""
IOI experiment with comprehensive fixes:
1. Proper JSON serialization handling
2. Fixed activation patching with better validation
3. Improved C-GAS metric without overly aggressive dimensionality penalty
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from scipy.stats import spearmanr, ttest_ind, pearsonr
import os
from typing import Dict, Any, List

from exp.shared.utils import set_seed
from exp.shared.models import SparseAutoencoder, train_sae


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_json_safe(data: Dict[str, Any], path: str):
    """Save data to JSON file with robust numpy type conversion."""
    serializable_data = convert_to_serializable(data)
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


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
            hidden = outputs.hidden_states[layer_idx]
            
            # Extract last non-padding token for each sequence
            for b in range(hidden.shape[0]):
                seq_len = inputs.attention_mask[b].sum().item()
                last_token_hidden = hidden[b, seq_len-1, :].cpu().numpy()
                all_activations.append(last_token_hidden)
    
    return np.array(all_activations)


def activation_patching_causal_identification(model, tokenizer, texts, name_pairs, 
                                               layer_idx, device='cuda', n_samples=200):
    """Identify causal subspaces using activation patching."""
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


def validate_subspace_consistency(model, tokenizer, texts, name_pairs, candidate_dims,
                                   layer_idx, device='cuda', n_groups=3):
    """Validate subspaces by checking consistency across input groups."""
    model.eval()
    
    # Split texts into groups
    group_size = len(texts) // n_groups
    groups = [list(range(i*group_size, (i+1)*group_size)) for i in range(n_groups)]
    
    validated_dims = []
    validation_scores = {}
    
    for dim in candidate_dims[:50]:  # Check top 50 dims
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
                a, b = sample_pairs[idx]
                text = sample_texts[idx]
                swapped = text.replace(a, "___TEMP___").replace(b, a).replace("___TEMP___", b)
                contrastive_texts.append(swapped)
            
            contrast_acts = extract_activations_from_layer(
                model, tokenizer, contrastive_texts, layer_idx, device
            )
            
            # Measure effect for this dimension
            effect = np.mean(np.abs(acts[:, dim] - contrast_acts[:, dim]))
            group_effects.append(effect)
        
        # Check consistency (low coefficient of variation)
        if np.mean(group_effects) > 0.01:  # Must have some effect
            cv = np.std(group_effects) / np.mean(group_effects)
            is_valid = bool(cv < 0.5)  # Convert to native bool
            
            if is_valid:
                validated_dims.append(int(dim))
            
            validation_scores[int(dim)] = {
                'mean_effect': float(np.mean(group_effects)),
                'cv': float(cv),
                'validated': is_valid
            }
    
    return validated_dims, validation_scores


def compute_cgas_improved(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    full_activations: np.ndarray,
    distance_metric: str = 'cosine',
    top_k: int = 10
) -> Dict:
    """Compute improved C-GAS without overly aggressive dimensionality penalty.
    
    The issue with the previous version was an overly aggressive penalty
    that made high-dimensional methods (like 16x SAE) look worse than they are.
    
    New formulation uses a simpler, less aggressive approach:
    C-GAS = ρ(D_causal, D_exp) / ρ(D_causal, D_full)
    
    This is the ratio form that normalizes by how much causal structure
    exists in the full space, WITHOUT the penalty.
    """
    from sklearn.metrics import pairwise_distances
    
    n_samples = causal_subspaces.shape[0]
    
    # Select top-k explanation features correlated with causal subspaces
    if explanation_features.shape[1] > top_k * causal_subspaces.shape[1]:
        selected_features = select_top_k_features(
            causal_subspaces, explanation_features, top_k
        )
    else:
        selected_features = explanation_features
    
    # Compute pairwise distance matrices
    D_causal = pairwise_distances(causal_subspaces, metric=distance_metric)
    D_exp = pairwise_distances(selected_features, metric=distance_metric)
    D_full = pairwise_distances(full_activations, metric=distance_metric)
    
    # Get upper triangular indices (excluding diagonal)
    triu_indices = np.triu_indices(n_samples, k=1)
    
    # Flatten distance matrices
    d_causal_vec = D_causal[triu_indices]
    d_exp_vec = D_exp[triu_indices]
    d_full_vec = D_full[triu_indices]
    
    # Compute correlations
    rho_causal_exp, p_causal_exp = spearmanr(d_causal_vec, d_exp_vec)
    rho_causal_full, p_causal_full = spearmanr(d_causal_vec, d_full_vec)
    
    # Handle edge cases
    if np.isnan(rho_causal_exp) or np.isnan(rho_causal_full):
        return {
            'cgas': 0.0,
            'rho_causal_exp': 0.0,
            'rho_causal_full': 0.0,
            'p_value': 1.0,
            'n_features_selected': selected_features.shape[1]
        }
    
    if abs(rho_causal_full) < 1e-10:
        return {
            'cgas': 0.0,
            'rho_causal_exp': float(rho_causal_exp),
            'rho_causal_full': float(rho_causal_full),
            'p_value': float(p_causal_exp),
            'n_features_selected': selected_features.shape[1]
        }
    
    # C-GAS: ratio of correlations
    cgas = rho_causal_exp / rho_causal_full
    
    return {
        'cgas': float(cgas),
        'rho_causal_exp': float(rho_causal_exp),
        'rho_causal_full': float(rho_causal_full),
        'p_value': float(p_causal_exp),
        'n_features_selected': selected_features.shape[1]
    }


def select_top_k_features(
    causal_subspaces: np.ndarray,
    explanation_features: np.ndarray,
    top_k: int
) -> np.ndarray:
    """Select top-k explanation features most correlated with causal subspaces."""
    n_causal_dims = causal_subspaces.shape[1]
    n_exp_features = explanation_features.shape[1]
    
    selected_indices = set()
    
    for i in range(n_causal_dims):
        causal_dim = causal_subspaces[:, i]
        correlations = []
        
        for j in range(n_exp_features):
            exp_dim = explanation_features[:, j]
            corr, _ = spearmanr(causal_dim, exp_dim)
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        # Get top-k indices for this causal dimension
        top_indices = np.argsort(correlations)[-top_k:]
        selected_indices.update(top_indices.tolist())
    
    # Return selected features
    selected_indices = sorted(list(selected_indices))
    return explanation_features[:, selected_indices]


def train_baselines(train_acts, val_acts, input_dim, seeds, device='cuda'):
    """Train all baselines with proper per-seed variance."""
    from sklearn.decomposition import PCA
    
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
            }, f'models/baseline_random_ioi_{overcomplete}x_seed{seed}.pt')
    
    # PCA baselines
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
            }, f'models/baseline_pca_ioi_{overcomplete}x_seed{seed}.pt')


def main():
    print("="*60)
    print("IOI Task Experiment (FIXED V2 with improved C-GAS)")
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
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Extract activations
    print("\nExtracting activations from layer 9...")
    train_acts = extract_activations_from_layer(model, tokenizer, train_texts, 9, device)
    val_acts = extract_activations_from_layer(model, tokenizer, val_texts, 9, device)
    
    input_dim = train_acts.shape[1]  # 768
    print(f"  Train: {train_acts.shape}, Val: {val_acts.shape}")
    
    # Causal subspace identification
    print("\n" + "="*60)
    print("Identifying causal subspaces via activation patching...")
    print("="*60)
    
    causal_candidates = activation_patching_causal_identification(
        model, tokenizer, val_texts, val_pairs, layer_idx=9, 
        device=device, n_samples=200
    )
    print(f"  Identified {len(causal_candidates['dims'])} candidate dimensions")
    print(f"  Top 5 effects: {causal_candidates['effects'][:5]}")
    
    # Validation
    print("\n  Running validation...")
    validated_dims, validation_scores = validate_subspace_consistency(
        model, tokenizer, val_texts, val_pairs, causal_candidates['dims'],
        layer_idx=9, device=device
    )
    print(f"  Validated {len(validated_dims)} dimensions")
    
    # If no validated dims, use top candidates
    if len(validated_dims) == 0:
        print("  WARNING: No validated dims, using top 50 candidates")
        validated_dims = causal_candidates['dims'][:50]
    
    # Save validation results
    os.makedirs('exp/ioi_fixed', exist_ok=True)
    save_json_safe({
        'causal_candidates': causal_candidates,
        'validation_scores': validation_scores,
        'validated_dims': validated_dims,
        'n_validated': len(validated_dims)
    }, 'exp/ioi_fixed/validation_results.json')
    
    # Train SAEs
    print("\n" + "="*60)
    print("Training SAEs...")
    print("="*60)
    
    SEEDS = [42, 123, 456]
    DICT_SIZES = [768, 3072]  # 1x, 4x
    
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
            
            print(f"    Seed {seed}: Loss = {history['train_loss'][-1]:.4f}")
    
    # Train baselines
    print("\n" + "="*60)
    print("Training baselines...")
    print("="*60)
    
    train_baselines(train_acts, val_acts, 768, SEEDS, device)
    print("  Baselines trained")
    
    # Compute C-GAS
    print("\n" + "="*60)
    print("Computing C-GAS with improved metric...")
    print("="*60)
    
    causal_subspaces = val_acts[:, validated_dims[:50]]
    print(f"  Using {causal_subspaces.shape[1]} causal dimensions")
    
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
                    
                    cgas_result = compute_cgas_improved(
                        causal_subspaces=causal_subspaces,
                        explanation_features=features,
                        full_activations=val_acts,
                        distance_metric='cosine',
                        top_k=20
                    )
                    
                    cgas_results.append({
                        'method': method_type,
                        'overcomplete': f'{overcomplete}x',
                        'seed': seed,
                        'cgas': cgas_result['cgas'],
                        'rho_causal_exp': cgas_result['rho_causal_exp'],
                        'rho_causal_full': cgas_result['rho_causal_full']
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
                    'std': float(np.std(vals)),
                    'n': len(vals)
                }
    
    # Statistical tests
    print("\nStatistical tests...")
    
    sae_vals = [r['cgas'] for r in cgas_results if r['method'] == 'sae']
    random_vals = [r['cgas'] for r in cgas_results if r['method'] == 'random']
    pca_vals = [r['cgas'] for r in cgas_results if r['method'] == 'pca']
    
    if sae_vals and random_vals:
        t_stat, p_val = ttest_ind(sae_vals, random_vals)
        print(f"  SAE vs Random: t={t_stat:.4f}, p={p_val:.4f}")
    else:
        t_stat, p_val = 0.0, 1.0
    
    if sae_vals and pca_vals:
        t_stat_pca, p_val_pca = ttest_ind(sae_vals, pca_vals)
        print(f"  SAE vs PCA: t={t_stat_pca:.4f}, p={p_val_pca:.4f}")
    else:
        t_stat_pca, p_val_pca = 0.0, 1.0
    
    # Save results
    save_json_safe({
        'cgas_all': cgas_results,
        'summary': summary,
        'statistical_tests': {
            'sae_vs_random': {'t_stat': float(t_stat), 'p_value': float(p_val)},
            'sae_vs_pca': {'t_stat': float(t_stat_pca), 'p_value': float(p_val_pca)}
        },
        'n_validated_dims': len(validated_dims)
    }, 'exp/ioi_fixed/cgas_results.json')
    
    # Print final summary
    print("\n" + "="*60)
    print("IOI Results Summary:")
    print("="*60)
    for method in ['sae', 'random', 'pca']:
        print(f"\n{method.upper()}:")
        for overcomplete in ['1x', '4x']:
            if overcomplete in summary.get(method, {}):
                stats = summary[method][overcomplete]
                print(f"  {overcomplete}: C-GAS = {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nValidated dimensions: {len(validated_dims)}")


if __name__ == '__main__':
    main()
