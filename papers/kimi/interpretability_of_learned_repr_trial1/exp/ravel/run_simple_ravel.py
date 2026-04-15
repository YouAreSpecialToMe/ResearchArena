"""Simplified RAVEL experiment using transformers directly."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from exp.shared.utils import set_seed, save_json
from exp.shared.models import SparseAutoencoder, train_sae
from exp.shared.metrics import compute_cgas

def extract_activations(model, tokenizer, texts, layer_idx=9, device='cuda'):
    """Extract activations from a specific layer."""
    model.eval()
    activations = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=40).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx][0, -1, :].cpu().numpy()
            activations.append(hidden)
    
    return np.array(activations)

def main():
    print("="*60)
    print("RAVEL Task Experiment (Simplified)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create simplified factual recall dataset
    print("\nCreating RAVEL-style dataset...")
    facts = [
        "The capital of France is Paris",
        "The capital of Germany is Berlin",
        "The capital of Italy is Rome",
        "The capital of Spain is Madrid",
        "The capital of UK is London",
        "The capital of Japan is Tokyo",
        "The capital of China is Beijing",
        "The capital of Russia is Moscow",
        "The capital of India is New Delhi",
        "The capital of Brazil is Brasilia",
        "Einstein was a physicist",
        "Shakespeare was a writer",
        "Mozart was a composer",
        "Picasso was an artist",
        "Marie Curie was a scientist",
        "Newton was a scientist",
        "Beethoven was a composer",
        "Van Gogh was an artist",
        "Steve Jobs was the CEO of Apple",
        "Bill Gates founded Microsoft",
        "Elon Musk leads Tesla",
        "Jeff Bezos founded Amazon",
        "Mark Zuckerberg founded Facebook",
    ]
    
    # Create variations
    texts = []
    for fact in facts:
        texts.append(fact)
        # Add variations
        if "capital" in fact:
            country = fact.split("of ")[1].split(" is")[0]
            capital = fact.split("is ")[1]
            texts.append(f"{capital} is the capital city of {country}")
            texts.append(f"In {country}, the capital is {capital}")
    
    print(f"  Created {len(texts)} text samples")
    
    # Load GPT-2
    print("\nLoading GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Extract activations
    print("\nExtracting activations from layer 9...")
    # Split 80/20
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    train_acts = extract_activations(model, tokenizer, train_texts, layer_idx=9, device=device)
    val_acts = extract_activations(model, tokenizer, val_texts, layer_idx=9, device=device)
    
    print(f"  Train: {train_acts.shape}, Val: {val_acts.shape}")
    
    # Train SAEs
    print("\nTraining SAEs...")
    SEEDS = [42, 123, 456]
    DICT_SIZES = [768, 3072]  # 1x, 4x
    
    train_acts_t = torch.FloatTensor(train_acts)
    val_acts_t = torch.FloatTensor(val_acts)
    
    sae_results = []
    
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 768
        print(f"\n  SAE {overcomplete}x")
        
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
            
            result = {
                'seed': seed,
                'dict_size': dict_size,
                'overcomplete': overcomplete,
                'final_loss': history['train_loss'][-1]
            }
            sae_results.append(result)
            
            torch.save({
                'features_val': features_val,
                'result': result
            }, f'models/sae_ravel_{overcomplete}x_seed{seed}.pt')
            
            print(f"    Seed {seed}: Loss = {history['train_loss'][-1]:.4f}")
    
    # Random baseline
    print("\nGenerating random baselines...")
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
    
    # PCA baseline (limited by sample size)
    from sklearn.decomposition import PCA
    print("\nTraining PCA baselines...")
    max_components = min(train_acts.shape[0], train_acts.shape[1])
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 768
        n_components = min(dict_size, max_components)
        for seed in SEEDS:
            set_seed(seed)
            pca = PCA(n_components=n_components, random_state=seed)
            pca.fit(train_acts)
            features_val = pca.transform(val_acts)
            
            torch.save({
                'pca': pca,
                'features_val': features_val
            }, f'models/baseline_pca_ravel_{overcomplete}x_seed{seed}.pt')
    
    # Simplified causal identification
    print("\nIdentifying causal subspaces...")
    dim_vars = np.var(val_acts, axis=0)
    causal_dims = np.argsort(dim_vars)[-50:][::-1]
    causal_subspaces = val_acts[:, causal_dims]
    
    # Compute C-GAS
    print("\nComputing C-GAS...")
    cgas_results = []
    
    for method_type in ['sae', 'random', 'pca']:
        for overcomplete in [1, 4]:
            for seed in SEEDS:
                if method_type == 'sae':
                    ckpt = torch.load(f'models/sae_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                elif method_type == 'random':
                    ckpt = torch.load(f'models/baseline_random_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                else:
                    ckpt = torch.load(f'models/baseline_pca_ravel_{overcomplete}x_seed{seed}.pt', weights_only=False)
                
                features = ckpt['features_val']
                
                cgas, _, _ = compute_cgas(
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
    
    # Summary
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
    
    save_json({
        'cgas_all': cgas_results,
        'summary': summary
    }, 'exp/ravel/cgas/results.json')
    
    print("\n" + "="*60)
    print("RAVEL Results Summary:")
    print("="*60)
    for method in ['sae', 'random', 'pca']:
        print(f"\n{method.upper()}:")
        for overcomplete in ['1x', '4x']:
            stats = summary[method][overcomplete]
            print(f"  {overcomplete}: C-GAS = {stats['mean']:.4f} ± {stats['std']:.4f}")

if __name__ == '__main__':
    main()
