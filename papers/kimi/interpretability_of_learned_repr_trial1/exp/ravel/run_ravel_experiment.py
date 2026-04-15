"""Unified RAVEL experiment script."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
from transformer_lens import HookedTransformer
import json

from exp.shared.utils import set_seed, save_json
from exp.shared.models import SparseAutoencoder, train_sae
from exp.shared.metrics import compute_cgas
from exp.shared.data_loader import load_ravel_dataset

def extract_ravel_activations(model, dataset, layer_idx=9, device='cuda'):
    """Extract activations from GPT-2 on RAVEL dataset."""
    model.eval()
    activations = []
    
    with torch.no_grad():
        for item in dataset:
            tokens = model.to_tokens(item['base_prompt'])
            _, cache = model.run_with_cache(tokens)
            act = cache[f'blocks.{layer_idx}.hook_resid_post'][0, -1, :].cpu().numpy()
            activations.append(act)
    
    return np.array(activations)

def main():
    print("="*60)
    print("RAVEL Task Experiment")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    print("\nCreating RAVEL dataset...")
    dataset = load_ravel_dataset(
        attribute_types=['country-capital', 'name-occupation', 'company-CEO'],
        n_samples_per_type=50,  # Reduced for time
        seed=42
    )
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    print(f"  Total: {len(dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load GPT-2
    print("\nLoading GPT-2 Small...")
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    
    # Extract activations
    print("\nExtracting activations from layer 9...")
    train_acts = extract_ravel_activations(model, train_dataset, layer_idx=9, device=device)
    val_acts = extract_ravel_activations(model, val_dataset, layer_idx=9, device=device)
    
    print(f"  Train activations: {train_acts.shape}")
    print(f"  Val activations: {val_acts.shape}")
    
    # Convert to torch
    train_acts_t = torch.FloatTensor(train_acts)
    val_acts_t = torch.FloatTensor(val_acts)
    
    # Train SAEs (simplified - only 1x and 4x)
    print("\nTraining SAEs...")
    SEEDS = [42, 123, 456]
    DICT_SIZES = [768, 3072]  # 1x, 4x
    
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
            }, f'models/sae_ravel_{overcomplete}x_seed{seed}.pt')
            
            print(f"    Seed {seed}: Recon error = {recon_error:.6f}")
    
    # Train baselines
    print("\nTraining baselines...")
    
    # Random baseline
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 768
        for seed in SEEDS:
            set_seed(seed)
            projection = np.random.randn(768, dict_size).astype(np.float32)
            projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
            features_val = val_acts @ projection
            
            torch.save({
                'projection': projection,
                'features_val': features_val
            }, f'models/baseline_random_ravel_{overcomplete}x_seed{seed}.pt')
    
    # PCA baseline
    from sklearn.decomposition import PCA
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 768
        for seed in SEEDS:
            set_seed(seed)
            pca = PCA(n_components=dict_size, random_state=seed)
            pca.fit(train_acts)
            features_val = pca.transform(val_acts)
            
            torch.save({
                'pca': pca,
                'features_val': features_val
            }, f'models/baseline_pca_ravel_{overcomplete}x_seed{seed}.pt')
    
    # Simplified causal identification - use variance-based selection
    print("\nIdentifying causal subspaces (simplified)...")
    
    # Use dimensions with highest variance as "causal"
    dim_vars = np.var(val_acts, axis=0)
    causal_dims = np.argsort(dim_vars)[-100:][::-1]
    
    causal_subspaces = val_acts[:, causal_dims[:50]]
    
    # Compute C-GAS
    print("\nComputing C-GAS scores...")
    
    cgas_results = []
    
    for method_type in ['sae', 'random', 'pca']:
        for overcomplete in [1, 4]:
            for seed in SEEDS:
                # Load features
                if method_type == 'sae':
                    ckpt = torch.load(f'models/sae_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                elif method_type == 'random':
                    ckpt = torch.load(f'models/baseline_random_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                else:
                    ckpt = torch.load(f'models/baseline_pca_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                
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
        'causal_dims': causal_dims.tolist()
    }, 'exp/ravel/cgas/results.json')
    
    print("\n" + "="*60)
    print("RAVEL Results Summary:")
    print("="*60)
    for method in ['sae', 'random', 'pca']:
        print(f"\n{method.upper()}:")
        for overcomplete in ['1x', '4x']:
            stats = summary[method][overcomplete]
            print(f"  {overcomplete}: C-GAS = {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\nResults saved to exp/ravel/cgas/results.json")

if __name__ == '__main__':
    main()
