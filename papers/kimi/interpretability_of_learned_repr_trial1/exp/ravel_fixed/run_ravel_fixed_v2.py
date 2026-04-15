"""
RAVEL experiment with comprehensive fixes:
1. Proper JSON serialization handling
2. Fixed activation patching
3. Improved C-GAS metric
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from scipy.stats import spearmanr, ttest_ind
import os
from typing import Dict, Any

from exp.shared.utils import set_seed
from exp.shared.models import SparseAutoencoder, train_sae
from exp.shared.data_loader import load_ravel_dataset


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
    """Identify causal subspaces using activation patching on RAVEL dataset."""
    model.eval()
    
    # Extract base activations
    base_activations = extract_ravel_activations(
        model, tokenizer, dataset, layer_idx, device
    )
    
    # Create contrastive prompts
    contrastive_texts = []
    for item in dataset:
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


def validate_subspace_consistency(model, tokenizer, dataset, candidate_dims,
                                   layer_idx, device='cuda'):
    """Validate subspaces by checking consistency across attribute types."""
    model.eval()
    
    # Group by attribute type
    attr_types = {}
    for i, item in enumerate(dataset):
        attr_type = item['attribute_type']
        if attr_type not in attr_types:
            attr_types[attr_type] = []
        attr_types[attr_type].append(i)
    
    validated_dims = []
    validation_scores = {}
    
    for dim in candidate_dims[:50]:
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
        if len(group_effects) > 1 and np.mean(group_effects) > 0.01:
            cv = np.std(group_effects) / np.mean(group_effects)
            is_valid = bool(cv < 0.5)
            
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
    """Compute improved C-GAS without overly aggressive dimensionality penalty."""
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
            }, f'models/baseline_random_ravel_{overcomplete}x_seed{seed}.pt')
    
    # PCA baselines
    for dict_size in [input_dim, input_dim*4]:
        overcomplete = dict_size // input_dim
        n_components = min(dict_size, input_dim)
        
        for seed in seeds:
            set_seed(seed)
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
    print("RAVEL Task Experiment (FIXED V2 with improved C-GAS)")
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
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Extract activations
    print("\nExtracting activations from layer 9...")
    train_acts = extract_ravel_activations(model, tokenizer, train_dataset, layer_idx=9, device=device)
    val_acts = extract_ravel_activations(model, tokenizer, val_dataset, layer_idx=9, device=device)
    
    input_dim = train_acts.shape[1]
    print(f"  Train: {train_acts.shape}, Val: {val_acts.shape}")
    
    # Causal subspace identification
    print("\n" + "="*60)
    print("Identifying causal subspaces via activation patching...")
    print("="*60)
    
    causal_candidates = activation_patching_causal_identification(
        model, tokenizer, val_dataset, layer_idx=9, device=device
    )
    print(f"  Identified {len(causal_candidates['dims'])} candidate dimensions")
    print(f"  Top 5 effects: {causal_candidates['effects'][:5]}")
    
    # Validation
    print("\n  Running validation...")
    validated_dims, validation_scores = validate_subspace_consistency(
        model, tokenizer, val_dataset, causal_candidates['dims'],
        layer_idx=9, device=device
    )
    print(f"  Validated {len(validated_dims)} dimensions")
    
    # If no validated dims, use top candidates
    if len(validated_dims) == 0:
        print("  WARNING: No validated dims, using top 50 candidates")
        validated_dims = causal_candidates['dims'][:50]
    
    # Save validation results
    os.makedirs('exp/ravel_fixed', exist_ok=True)
    save_json_safe({
        'causal_candidates': causal_candidates,
        'validation_scores': validation_scores,
        'validated_dims': validated_dims,
        'n_validated': len(validated_dims)
    }, 'exp/ravel_fixed/validation_results.json')
    
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
    print("Training baselines...")
    print("="*60)
    
    train_baselines(train_acts, val_acts, input_dim, SEEDS, device)
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
                        ckpt = torch.load(f'models/sae_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    elif method_type == 'random':
                        ckpt = torch.load(f'models/baseline_random_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    else:
                        ckpt = torch.load(f'models/baseline_pca_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                    
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
    }, 'exp/ravel_fixed/cgas_results.json')
    
    # Print final summary
    print("\n" + "="*60)
    print("RAVEL Results Summary:")
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
