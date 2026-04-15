"""Unified IOI experiment script."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
from transformer_lens import HookedTransformer
import json
import os

from exp.shared.utils import set_seed, save_json
from exp.shared.models import SparseAutoencoder, train_sae
from exp.shared.metrics import compute_cgas
from exp.shared.data_loader import create_ioi_templates

def extract_ioi_activations(model, dataset, layer_idx=9, device='cuda'):
    """Extract activations from GPT-2 on IOI dataset."""
    model.eval()
    activations_orig = []
    activations_contrast = []
    
    with torch.no_grad():
        for item in dataset:
            # Original sentence
            tokens_orig = model.to_tokens(item['sent_orig'])
            _, cache = model.run_with_cache(tokens_orig)
            act_orig = cache[f'blocks.{layer_idx}.hook_resid_post'][0, -1, :].cpu().numpy()
            activations_orig.append(act_orig)
            
            # Contrastive sentence
            tokens_contrast = model.to_tokens(item['sent_contrast'])
            _, cache = model.run_with_cache(tokens_contrast)
            act_contrast = cache[f'blocks.{layer_idx}.hook_resid_post'][0, -1, :].cpu().numpy()
            activations_contrast.append(act_contrast)
    
    return np.array(activations_orig), np.array(activations_contrast)

def identify_causal_subspaces_ioi(model, dataset, layer_idx=9, device='cuda'):
    """Identify causal subspaces for IOI using activation patching."""
    model.eval()
    
    # We'll focus on the position of the indirect object token
    n_samples = min(50, len(dataset))  # Use subset for efficiency
    d_model = model.cfg.d_model
    
    causal_effects = np.zeros(d_model)
    
    with torch.no_grad():
        for i in range(n_samples):
            item = dataset[i]
            
            # Get tokens
            tokens_orig = model.to_tokens(item['sent_orig'])
            tokens_contrast = model.to_tokens(item['sent_contrast'])
            
            # Get baseline output for IO token
            logits_orig, _ = model.run_with_cache(tokens_orig)
            io_token_id = model.to_tokens(item['io_token'])[0, 1].item()
            prob_orig = torch.softmax(logits_orig[0, -1, :], dim=-1)[io_token_id].item()
            
            # Try patching each dimension
            for dim_idx in range(0, d_model, 10):  # Sample every 10th dim for efficiency
                # Get cache from contrastive
                _, cache_contrast = model.run_with_cache(tokens_contrast)
                act_contrast = cache_contrast[f'blocks.{layer_idx}.hook_resid_post'][0, -1, :].clone()
                
                # Get cache from original
                _, cache_orig = model.run_with_cache(tokens_orig)
                act_orig = cache_orig[f'blocks.{layer_idx}.hook_resid_post'][0, -1, :].clone()
                
                # Patch this dimension
                act_patched = act_orig.clone()
                act_patched[dim_idx] = act_contrast[dim_idx]
                
                # Forward from this layer to output
                # Simplified: measure change in activation
                effect = torch.abs(act_patched - act_orig).mean().item()
                causal_effects[dim_idx] += effect
    
    # Normalize
    causal_effects /= n_samples
    
    # Get top dimensions
    top_dims = np.argsort(causal_effects)[-100:][::-1]
    
    return {
        'dims': top_dims.tolist(),
        'effects': causal_effects[top_dims].tolist(),
        'layer': layer_idx
    }

def main():
    print("="*60)
    print("IOI Task Experiment")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    print("\nCreating IOI dataset...")
    dataset = create_ioi_templates()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load GPT-2
    print("\nLoading GPT-2 Small...")
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    
    # Extract activations
    print("\nExtracting activations from layer 9...")
    train_acts_orig, train_acts_contrast = extract_ioi_activations(
        model, train_dataset[:100], layer_idx=9, device=device  # Use subset
    )
    val_acts_orig, val_acts_contrast = extract_ioi_activations(
        model, val_dataset[:30], layer_idx=9, device=device
    )
    
    # Combine orig and contrast
    train_acts = np.vstack([train_acts_orig, train_acts_contrast])
    val_acts = np.vstack([val_acts_orig, val_acts_contrast])
    
    print(f"  Train activations: {train_acts.shape}")
    print(f"  Val activations: {val_acts.shape}")
    
    # Convert to torch
    train_acts_t = torch.FloatTensor(train_acts)
    val_acts_t = torch.FloatTensor(val_acts)
    
    # Train SAEs
    print("\nTraining SAEs...")
    SEEDS = [42, 123, 456]
    DICT_SIZES = [768, 3072]  # 1x, 4x (skip 16x for time)
    
    sae_results = []
    
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
            
            # Evaluate
            sae.eval()
            with torch.no_grad():
                features_val = sae.encode(val_acts_t.to(device)).cpu().numpy()
                recon_val, _ = sae(val_acts_t.to(device))
                recon_error = torch.mean((recon_val - val_acts_t.to(device)) ** 2).item()
            
            result = {
                'seed': seed,
                'dict_size': dict_size,
                'overcomplete': overcomplete,
                'recon_error': recon_error,
                'final_loss': history['train_loss'][-1]
            }
            sae_results.append(result)
            
            # Save
            torch.save({
                'model_state_dict': sae.state_dict(),
                'features_val': features_val,
                'result': result
            }, f'models/sae_ioi_{overcomplete}x_seed{seed}.pt')
            
            print(f"    Seed {seed}: Recon error = {recon_error:.6f}")
    
    # Train baselines
    print("\nTraining baselines...")
    
    # Random baseline
    random_results = []
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 768
        for seed in SEEDS:
            set_seed(seed)
            projection = np.random.randn(768, dict_size).astype(np.float32)
            projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
            features_val = val_acts @ projection
            
            random_results.append({
                'seed': seed,
                'overcomplete': overcomplete,
                'features_val': features_val
            })
            
            torch.save({
                'projection': projection,
                'features_val': features_val
            }, f'models/baseline_random_ioi_{overcomplete}x_seed{seed}.pt')
    
    # PCA baseline
    from sklearn.decomposition import PCA
    pca_results = []
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 768
        for seed in SEEDS:
            set_seed(seed)
            pca = PCA(n_components=dict_size, random_state=seed)
            features_val = pca.fit_transform(train_acts)
            pca_results.append({
                'seed': seed,
                'overcomplete': overcomplete,
                'features_val': features_val
            })
            
            torch.save({
                'pca': pca,
                'features_val': features_val
            }, f'models/baseline_pca_ioi_{overcomplete}x_seed{seed}.pt')
    
    # Identify causal subspaces (simplified)
    print("\nIdentifying causal subspaces...")
    causal_info = identify_causal_subspaces_ioi(model, val_dataset[:20], layer_idx=9, device=device)
    
    # Compute C-GAS (simplified - use top dims as causal)
    print("\nComputing C-GAS scores...")
    
    cgas_results = []
    
    # Use top 50 dims as "causal"
    causal_dims = causal_info['dims'][:50]
    causal_subspaces = val_acts[:, causal_dims]
    
    for method_type in ['sae', 'random', 'pca']:
        for overcomplete in [1, 4]:
            for seed in SEEDS:
                # Load features
                if method_type == 'sae':
                    ckpt = torch.load(f'models/sae_ioi_{overcomplete}x_seed{seed}.pt', weights_only=False)
                elif method_type == 'random':
                    ckpt = torch.load(f'models/baseline_random_ioi_{overcomplete}x_seed{seed}.pt', weights_only=False)
                else:
                    ckpt = torch.load(f'models/baseline_pca_ioi_{overcomplete}x_seed{seed}.pt', weights_only=False)
                
                features = ckpt['features_val']
                
                # Compute C-GAS
                cgas, rho_ce, rho_cf = compute_cgas(
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
                    'cgas': float(cgas)
                })
    
    # Compute summary
    summary = {}
    for method in ['sae', 'random', 'pca']:
        summary[method] = {}
        for overcomplete in ['1x', '4x']:
            vals = [r['cgas'] for r in cgas_results 
                   if r['method'] == method and r['overcomplete'] == overcomplete]
            summary[method][overcomplete] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals))
            }
    
    # Save results
    save_json({
        'sae_training': sae_results,
        'cgas_all': cgas_results,
        'summary': summary,
        'causal_info': causal_info
    }, 'exp/ioi/cgas/results.json')
    
    print("\n" + "="*60)
    print("IOI Results Summary:")
    print("="*60)
    for method in ['sae', 'random', 'pca']:
        print(f"\n{method.upper()}:")
        for overcomplete in ['1x', '4x']:
            stats = summary[method][overcomplete]
            print(f"  {overcomplete}: C-GAS = {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\nResults saved to exp/ioi/cgas/results.json")

if __name__ == '__main__':
    main()
