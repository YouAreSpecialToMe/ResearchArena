#!/usr/bin/env python3
"""
Feature Type Analysis
Analyze which feature types are amenable to linear probing vs. requiring non-linear methods.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Categorized concepts
feature_categories = {
    'linear': {
        'pos_tag_noun': 'syntactic',
        'pos_tag_verb': 'syntactic',
        'number_presence': 'syntactic',
        'question_mark': 'syntactic'
    },
    'semantic': {
        'positive_sentiment': 'semantic',
        'negative_sentiment': 'semantic',
        'person_entity': 'semantic',
        'location_entity': 'semantic'
    }
}

all_concepts = []
for cat in feature_categories.values():
    all_concepts.extend(list(cat.keys()))

layers = [3, 6, 9]
checkpoints = [16000, 64000, 143000]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_texts(n_samples=1200, seed=42):
    """Generate texts with concept labels."""
    np.random.seed(seed)
    
    templates = {
        'noun': [
            "The cat sits.", "A book exists.", "The table stands.",
            "Water flows.", "Time passes."
        ],
        'verb': [
            "He runs fast.", "She writes well.", "They jump high.",
            "We think deeply.", "It moves quickly."
        ],
        'number': [
            "The value is 42.", "Count to 100.", "Year 2024.",
            "Price: $50.", "Temperature 72 degrees."
        ],
        'question': [
            "What is this?", "How are you?", "Why now?",
            "When did it happen?", "Who came first?"
        ],
        'positive': [
            "I love this!", "Wonderful experience!", "Great success!",
            "Amazing quality!", "Excellent work!"
        ],
        'negative': [
            "I hate this.", "Terrible experience.", "Major failure.",
            "Poor quality.", "Bad outcome."
        ],
        'person': [
            "Einstein said.", "Marie Curie discovered.", "Newton found.",
            "Tesla invented.", "Darwin observed."
        ],
        'location': [
            "In Paris, we...", "London is...", "Tokyo has...",
            "Berlin was...", "Sydney offers..."
        ]
    }
    
    texts = []
    labels = {c: [] for c in all_concepts}
    
    samples_per = n_samples // 8
    
    for _ in range(samples_per):
        # Noun-heavy texts
        texts.append(np.random.choice(templates['noun']))
        for c in all_concepts:
            labels[c].append(1 if c == 'pos_tag_noun' else 0)
        
        # Verb-heavy texts
        texts.append(np.random.choice(templates['verb']))
        for c in all_concepts:
            labels[c].append(1 if c == 'pos_tag_verb' else 0)
        
        # Number texts
        texts.append(np.random.choice(templates['number']))
        for c in all_concepts:
            labels[c].append(1 if c == 'number_presence' else 0)
        
        # Question texts
        texts.append(np.random.choice(templates['question']))
        for c in all_concepts:
            labels[c].append(1 if c == 'question_mark' else 0)
        
        # Positive sentiment
        texts.append(np.random.choice(templates['positive']))
        for c in all_concepts:
            labels[c].append(1 if c == 'positive_sentiment' else 0)
        
        # Negative sentiment
        texts.append(np.random.choice(templates['negative']))
        for c in all_concepts:
            labels[c].append(1 if c == 'negative_sentiment' else 0)
        
        # Person entity
        texts.append(np.random.choice(templates['person']))
        for c in all_concepts:
            labels[c].append(1 if c == 'person_entity' else 0)
        
        # Location entity
        texts.append(np.random.choice(templates['location']))
        for c in all_concepts:
            labels[c].append(1 if c == 'location_entity' else 0)
    
    for c in labels:
        labels[c] = np.array(labels[c][:n_samples])
    
    return texts[:n_samples], labels


def extract_activations(model, tokenizer, texts, layer, device, batch_size=16):
    """Extract activations."""
    all_activations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer]
            pooled = hidden.mean(dim=1).cpu().numpy()
            all_activations.append(pooled)
    
    return np.vstack(all_activations)


def train_linear_probe(X, y, seed=42):
    """Train linear probe."""
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    if len(np.unique(y_train)) < 2:
        return None
    
    probe = LogisticRegression(penalty='l1', C=0.01, solver='saga', max_iter=500, random_state=seed)
    probe.fit(X_train, y_train)
    
    val_acc = probe.score(X_val, y_val)
    weights = probe.coef_[0]
    non_zero = np.sum(np.abs(weights) > 1e-6)
    
    return {
        'val_accuracy': float(val_acc),
        'non_zero_weights': int(non_zero),
        'probe_type': 'linear'
    }


def train_mlp_probe(X, y, seed=42):
    """Train MLP probe."""
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    if len(np.unique(y_train)) < 2:
        return None
    
    probe = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1
    )
    probe.fit(X_train, y_train)
    
    val_acc = probe.score(X_val, y_val)
    
    return {
        'val_accuracy': float(val_acc),
        'non_zero_weights': None,  # Not applicable for MLP
        'probe_type': 'mlp'
    }


def run_feature_analysis(checkpoint_step, seed=42):
    """Run feature type analysis for one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Feature Type Analysis - Checkpoint {checkpoint_step}, Seed {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = "EleutherAI/pythia-160m"
    revision = f"step{checkpoint_step}"
    
    print(f"Loading {revision}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model = model.to(device)
    model.eval()
    
    texts, labels_dict = generate_texts(n_samples=1200, seed=seed)
    
    results = {
        'checkpoint': checkpoint_step,
        'seed': seed,
        'linear_probes': {},
        'mlp_probes': {}
    }
    
    for concept in all_concepts:
        print(f"\nConcept: {concept}")
        y = labels_dict[concept]
        
        if len(np.unique(y)) < 2:
            continue
        
        results['linear_probes'][concept] = {}
        results['mlp_probes'][concept] = {}
        
        for layer in layers:
            print(f"  Layer {layer}...", end=' ')
            
            X = extract_activations(model, tokenizer, texts, layer, device)
            
            linear_result = train_linear_probe(X, y, seed=seed)
            mlp_result = train_mlp_probe(X, y, seed=seed)
            
            if linear_result:
                results['linear_probes'][concept][f'layer_{layer}'] = linear_result
            if mlp_result:
                results['mlp_probes'][concept][f'layer_{layer}'] = mlp_result
            
            if linear_result and mlp_result:
                gap = mlp_result['val_accuracy'] - linear_result['val_accuracy']
                print(f"Linear={linear_result['val_accuracy']:.3f}, MLP={mlp_result['val_accuracy']:.3f}, Gap={gap:+.3f}")
            else:
                print("SKIP")
    
    del model
    torch.cuda.empty_cache()
    
    return results


def compute_gap_analysis(all_results):
    """Compute linear vs MLP gap by feature category."""
    gaps_by_category = {cat: [] for cat in ['syntactic', 'semantic']}
    
    for concept in all_concepts:
        # Find category
        category = None
        for cat, concepts in feature_categories.items():
            if concept in concepts:
                category = concepts[concept]
                break
        
        if not category:
            continue
        
        # Compute average gap across checkpoints and layers
        gaps = []
        for result in all_results:
            if concept in result['linear_probes'] and concept in result['mlp_probes']:
                for layer in layers:
                    key = f'layer_{layer}'
                    if key in result['linear_probes'][concept] and key in result['mlp_probes'][concept]:
                        linear_acc = result['linear_probes'][concept][key]['val_accuracy']
                        mlp_acc = result['mlp_probes'][concept][key]['val_accuracy']
                        gaps.append(mlp_acc - linear_acc)
        
        if gaps:
            gaps_by_category[category].append(np.mean(gaps))
    
    summary = {}
    for category, gaps in gaps_by_category.items():
        if gaps:
            summary[category] = {
                'mean_gap': float(np.mean(gaps)),
                'std_gap': float(np.std(gaps)),
                'n_features': len(gaps)
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    all_results = []
    
    for checkpoint_step in checkpoints:
        result = run_feature_analysis(checkpoint_step, seed=args.seed)
        all_results.append(result)
        
        output_file = os.path.join(args.output_dir, f'analysis_ckpt_{checkpoint_step}_seed_{args.seed}.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output_file}")
    
    # Compute gap analysis
    gap_summary = compute_gap_analysis(all_results)
    
    summary = {
        'results_by_checkpoint': all_results,
        'gap_analysis_by_category': gap_summary,
        'interpretation': {
            'syntactic': 'Low gap expected - linearly decodable',
            'semantic': 'Higher gap possible - may need non-linear probes'
        }
    }
    
    summary_file = os.path.join(args.output_dir, f'summary_seed_{args.seed}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Feature type analysis complete.")
    print(f"Gap analysis: {gap_summary}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
