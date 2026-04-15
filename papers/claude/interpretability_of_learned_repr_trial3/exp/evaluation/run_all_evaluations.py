"""Run all evaluations in one script to avoid repeated model loading."""

import sys
import os
import json
import time
import torch
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import defaultdict
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *

import transformer_lens

# === Utility functions ===

def load_model():
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    model.eval()
    return model

def load_sae_weights(layer, seed):
    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    path = os.path.join(sae_base, f"layer_{layer}", f"seed_{seed}", "sae_weights.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cuda")

def get_eval_tokens(model, n_sequences=1000):
    """Generate synthetic eval tokens using random prompts."""
    tokenizer = model.tokenizer
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    # Generate random token sequences (valid tokens only)
    vocab_size = model.cfg.d_vocab
    tokens = torch.randint(1, vocab_size, (n_sequences, CONTEXT_SIZE), device="cuda")
    return tokens

def get_real_eval_tokens(model, n_sequences=1000):
    """Get real eval tokens from dataset, with timeout."""
    try:
        from datasets import load_dataset
        tokenizer = model.tokenizer
        ds = load_dataset(DATASET_PATH, split="train", streaming=True, trust_remote_code=True)
        all_tokens = []
        count = 0
        for item in ds:
            tokens = tokenizer.encode(item["text"], return_tensors="pt")[0]
            if len(tokens) >= CONTEXT_SIZE:
                all_tokens.append(tokens[:CONTEXT_SIZE])
                count += 1
            if count >= n_sequences:
                break
            if count > 0 and count % 100 == 0:
                print(f"  Loaded {count}/{n_sequences} sequences")
        if len(all_tokens) > 0:
            return torch.stack(all_tokens).to("cuda")
    except Exception as e:
        print(f"  Could not load real data: {e}")
    return get_eval_tokens(model, n_sequences)

def compute_sae_activations(hidden_states, sae_weights):
    W_enc = sae_weights["W_enc"]
    b_enc = sae_weights["b_enc"]
    b_dec = sae_weights["b_dec"]
    x_centered = hidden_states - b_dec
    pre_acts = x_centered @ W_enc + b_enc
    topk_values, topk_indices = torch.topk(pre_acts, k=TOPK_K, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_indices, topk_values)
    return acts

def load_consensus_data(layer):
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")
    cs_path = os.path.join(matching_base, f"layer_{layer}", "consensus_scores.npy")
    tl_path = os.path.join(matching_base, f"layer_{layer}", "tier_labels.json")
    if not os.path.exists(cs_path):
        return None, None
    consensus_scores = np.load(cs_path)
    with open(tl_path) as f:
        tier_labels = json.load(f)
    return consensus_scores, tier_labels


# === Causal Importance ===

def eval_causal_importance(model, eval_tokens, layer, sae_weights, consensus_scores, tier_labels):
    """Compute causal importance via feature ablation."""
    print(f"\n  [Causal Importance] Layer {layer}")
    W_dec = sae_weights["W_dec"]
    dict_size = W_dec.shape[0]
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Collect hidden states
    all_hidden = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(eval_tokens), batch_size):
            batch = eval_tokens[i:i+batch_size]
            _, cache = model.run_with_cache(batch, names_filter=[hook_name])
            all_hidden.append(cache[hook_name])

    all_hidden = torch.cat(all_hidden, dim=0)  # (n_seq, seq_len, hidden)
    flat_hidden = all_hidden.reshape(-1, HIDDEN_DIM)  # (n_total, hidden)

    # Compute SAE activations
    print("    Computing SAE activations...")
    acts = compute_sae_activations(flat_hidden, sae_weights)

    # Firing rates
    firing_rates = (acts > 0).float().mean(dim=0).cpu().numpy()

    # Causal importance: relative norm of ablation delta for each feature
    print("    Computing causal importance...")
    causal_importance = np.zeros(dict_size)
    n_active = np.zeros(dict_size)

    for f_idx in range(dict_size):
        active_mask = acts[:, f_idx] > 0
        n_act = active_mask.sum().item()
        n_active[f_idx] = n_act

        if n_act < 10:
            continue

        # Sample active tokens
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        if len(active_indices) > 500:
            perm = torch.randperm(len(active_indices), device="cuda")[:500]
            active_indices = active_indices[perm]

        feat_acts = acts[active_indices, f_idx]
        decoder_vec = W_dec[f_idx]

        # Delta from ablation
        delta = feat_acts.unsqueeze(-1) * decoder_vec.unsqueeze(0)
        delta_norm = delta.norm(dim=-1)
        hidden_norm = flat_hidden[active_indices].norm(dim=-1)
        relative_effect = (delta_norm / (hidden_norm + 1e-8)).mean().item()
        causal_importance[f_idx] = relative_effect

        if (f_idx + 1) % 4096 == 0:
            print(f"    Processed {f_idx+1}/{dict_size} features")

    # Analysis by tier
    active_mask = n_active >= 10
    active_ci = causal_importance[active_mask]
    active_cs = consensus_scores[active_mask]
    active_tiers = [tier_labels[i] for i in range(len(tier_labels)) if active_mask[i]]
    active_fr = firing_rates[active_mask]

    tier_ci = defaultdict(list)
    for ci, tier in zip(active_ci, active_tiers):
        tier_ci[tier].append(ci)

    consensus_ci = np.array(tier_ci.get("consensus", [0]))
    singleton_ci = np.array(tier_ci.get("singleton", [0]))

    if len(consensus_ci) > 5 and len(singleton_ci) > 5:
        u_stat, p_value = stats.mannwhitneyu(consensus_ci, singleton_ci, alternative="greater")
        pooled_std = np.sqrt((consensus_ci.std()**2 + singleton_ci.std()**2) / 2)
        cohens_d = (consensus_ci.mean() - singleton_ci.mean()) / (pooled_std + 1e-10)
        spearman_r, spearman_p = stats.spearmanr(active_cs, active_ci)
        # Partial correlation controlling for firing rate
        from sklearn.linear_model import LinearRegression
        fr_resid_cs = active_cs - LinearRegression().fit(active_fr.reshape(-1,1), active_cs).predict(active_fr.reshape(-1,1))
        fr_resid_ci = active_ci - LinearRegression().fit(active_fr.reshape(-1,1), active_ci).predict(active_fr.reshape(-1,1))
        partial_r, partial_p = stats.spearmanr(fr_resid_cs, fr_resid_ci)
    else:
        u_stat, p_value, cohens_d = 0, 1.0, 0
        spearman_r, spearman_p = 0, 1.0
        partial_r, partial_p = 0, 1.0

    result = {
        "layer": layer,
        "n_active_features": int(active_mask.sum()),
        "tier_counts": {t: len(v) for t, v in tier_ci.items()},
        "tier_mean_causal_importance": {t: float(np.mean(v)) if v else 0 for t, v in tier_ci.items()},
        "tier_median_causal_importance": {t: float(np.median(v)) if v else 0 for t, v in tier_ci.items()},
        "tier_std_causal_importance": {t: float(np.std(v)) if v else 0 for t, v in tier_ci.items()},
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_value),
        "cohens_d": float(cohens_d),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "partial_spearman_r": float(partial_r),
        "partial_spearman_p": float(partial_p),
        "firing_rate_consensus_corr": float(stats.spearmanr(active_cs, active_fr)[0]) if len(active_cs) > 5 else 0,
    }

    print(f"    Consensus CI: {result['tier_mean_causal_importance'].get('consensus', 0):.6f}")
    print(f"    Singleton CI: {result['tier_mean_causal_importance'].get('singleton', 0):.6f}")
    print(f"    Cohen's d: {cohens_d:.3f}, p={p_value:.2e}")
    print(f"    Spearman r: {spearman_r:.3f}, p={spearman_p:.2e}")

    # Save per-feature data
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    layer_dir = os.path.join(eval_dir, f"layer_{layer}")
    os.makedirs(layer_dir, exist_ok=True)
    np.save(os.path.join(layer_dir, "causal_importance.npy"), causal_importance)
    np.save(os.path.join(layer_dir, "firing_rates.npy"), firing_rates)

    # Free GPU memory
    del all_hidden, flat_hidden, acts
    torch.cuda.empty_cache()

    return result, causal_importance, firing_rates


# === Sparse Probing ===

def eval_sparse_probing(model, eval_tokens, layer, sae_weights, tier_labels):
    """Sparse probing evaluation."""
    print(f"\n  [Sparse Probing] Layer {layer}")
    tokenizer = model.tokenizer
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Get SAE activations
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(eval_tokens), 32):
            batch = eval_tokens[i:i+32]
            _, cache = model.run_with_cache(batch, names_filter=[hook_name])
            hidden = cache[hook_name].reshape(-1, HIDDEN_DIM)
            acts = compute_sae_activations(hidden, sae_weights)
            all_acts.append(acts.cpu())

    all_acts = torch.cat(all_acts, dim=0).numpy()
    n_total = all_acts.shape[0]

    # Create simple token-level labels
    tasks = {}
    flat_tokens = eval_tokens.cpu().reshape(-1)

    # Task 1: Capitalization
    labels_cap = np.array([
        1 if tokenizer.decode([t.item()]).strip()[:1].isupper() else 0
        for t in flat_tokens[:n_total]
    ])
    if 0.05 < labels_cap.mean() < 0.95:
        tasks["capitalization"] = labels_cap

    # Task 2: Number detection
    labels_num = np.array([
        1 if any(c.isdigit() for c in tokenizer.decode([t.item()])) else 0
        for t in flat_tokens[:n_total]
    ])
    if 0.05 < labels_num.mean() < 0.95:
        tasks["number_detection"] = labels_num

    # Task 3: Punctuation
    labels_punct = np.array([
        1 if tokenizer.decode([t.item()]).strip()[:1] in string.punctuation else 0
        for t in flat_tokens[:n_total]
    ])
    if 0.05 < labels_punct.mean() < 0.95:
        tasks["punctuation"] = labels_punct

    # Task 4: Word boundary (space-prefixed)
    labels_ws = np.array([
        1 if tokenizer.decode([t.item()])[:1] == ' ' else 0
        for t in flat_tokens[:n_total]
    ])
    if 0.05 < labels_ws.mean() < 0.95:
        tasks["word_boundary"] = labels_ws

    # Tier indices
    tier_indices = defaultdict(list)
    for i, t in enumerate(tier_labels):
        tier_indices[t].append(i)

    k_values = [5, 10, 20, 50]
    results = {}

    for task_name, labels in tasks.items():
        print(f"    Task: {task_name} (positive rate: {labels.mean():.3f})")
        n_train = int(0.8 * n_total)
        X_train, X_test = all_acts[:n_train], all_acts[n_train:]
        y_train, y_test = labels[:n_train], labels[n_train:]

        task_results = {}
        for k in k_values:
            k_results = {}
            for tier_name, feat_idx in tier_indices.items():
                feat_idx = np.array(feat_idx)
                if len(feat_idx) < k:
                    k_results[tier_name] = {"accuracy": 0.5, "n_features": len(feat_idx)}
                    continue

                # Select top-k by variance
                variances = X_train[:, feat_idx].var(axis=0)
                top_k = np.argsort(variances)[-k:]
                selected = feat_idx[top_k]

                clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
                clf.fit(X_train[:, selected], y_train)
                acc = accuracy_score(y_test, clf.predict(X_test[:, selected]))
                k_results[tier_name] = {"accuracy": float(acc), "n_features": len(feat_idx)}

            # Random baseline
            random_accs = []
            for rs in range(5):
                np.random.seed(rs)
                rand_feats = np.random.choice(DICT_SIZE, k, replace=False)
                clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
                clf.fit(X_train[:, rand_feats], y_train)
                random_accs.append(accuracy_score(y_test, clf.predict(X_test[:, rand_feats])))
            k_results["random"] = {"accuracy": float(np.mean(random_accs)), "std": float(np.std(random_accs))}

            task_results[f"k={k}"] = k_results
            print(f"      k={k}: cons={k_results.get('consensus',{}).get('accuracy','N/A'):.3f}, "
                  f"sing={k_results.get('singleton',{}).get('accuracy','N/A'):.3f}, "
                  f"rand={k_results['random']['accuracy']:.3f}")

        results[task_name] = task_results

    del all_acts
    return results


# === Steering ===

def eval_steering(model, eval_tokens, layer, sae_weights, tier_labels, n_features=30):
    """Steering effectiveness evaluation."""
    print(f"\n  [Steering] Layer {layer}")
    W_dec = sae_weights["W_dec"]
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Get firing rates
    all_acts = []
    with torch.no_grad():
        for i in range(0, min(len(eval_tokens), 200), 32):
            batch = eval_tokens[i:i+32]
            _, cache = model.run_with_cache(batch, names_filter=[hook_name])
            hidden = cache[hook_name].reshape(-1, HIDDEN_DIM)
            acts = compute_sae_activations(hidden, sae_weights)
            all_acts.append(acts.cpu())
    all_acts = torch.cat(all_acts, dim=0)
    firing_rates = (all_acts > 0).float().mean(dim=0).numpy()

    # Select features
    consensus_idx = [i for i, t in enumerate(tier_labels) if t == "consensus" and firing_rates[i] > 0.01]
    singleton_idx = [i for i, t in enumerate(tier_labels) if t == "singleton" and firing_rates[i] > 0.01]

    np.random.seed(42)
    if len(consensus_idx) > n_features:
        consensus_idx = list(np.random.choice(consensus_idx, n_features, replace=False))
    if len(singleton_idx) > n_features:
        singleton_idx = list(np.random.choice(singleton_idx, n_features, replace=False))

    def measure_steering(feature_indices):
        effects = []
        for f_idx in feature_indices[:n_features]:
            decoder_vec = W_dec[f_idx].to("cuda")
            mean_act = all_acts[:, f_idx][all_acts[:, f_idx] > 0].mean().item() if (all_acts[:, f_idx] > 0).any() else 1.0

            kl_divs = []
            for seq_i in range(min(20, len(eval_tokens))):
                tokens = eval_tokens[seq_i:seq_i+1]
                with torch.no_grad():
                    orig_logits = model(tokens)

                def hook_fn(activation, hook):
                    return activation + 3.0 * mean_act * decoder_vec

                with torch.no_grad():
                    steered_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])

                orig_probs = torch.softmax(orig_logits[0], dim=-1)
                steered_logprobs = torch.log_softmax(steered_logits[0], dim=-1)
                kl = (orig_probs * (torch.log(orig_probs + 1e-10) - steered_logprobs)).sum(dim=-1)
                kl_divs.extend(kl.cpu().numpy().tolist())

            if kl_divs:
                effects.append(float(np.mean(kl_divs)))
        return effects

    print(f"    Evaluating {len(consensus_idx)} consensus features...")
    consensus_effects = measure_steering(consensus_idx)
    print(f"    Evaluating {len(singleton_idx)} singleton features...")
    singleton_effects = measure_steering(singleton_idx)

    if len(consensus_effects) > 3 and len(singleton_effects) > 3:
        u_stat, p_value = stats.mannwhitneyu(consensus_effects, singleton_effects, alternative="greater")
    else:
        u_stat, p_value = 0, 1.0

    result = {
        "layer": layer,
        "n_consensus": len(consensus_effects),
        "n_singleton": len(singleton_effects),
        "consensus_mean_effect": float(np.mean(consensus_effects)) if consensus_effects else 0,
        "singleton_mean_effect": float(np.mean(singleton_effects)) if singleton_effects else 0,
        "mann_whitney_p": float(p_value),
    }

    print(f"    Consensus mean: {result['consensus_mean_effect']:.6f}")
    print(f"    Singleton mean: {result['singleton_mean_effect']:.6f}")
    print(f"    p-value: {p_value:.4f}")

    del all_acts
    torch.cuda.empty_cache()
    return result


# === Manifold Analysis ===

def eval_manifold(layer, tier_labels):
    """Manifold tiling analysis."""
    from sklearn.cluster import DBSCAN
    print(f"\n  [Manifold Analysis] Layer {layer}")

    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    W_dec = torch.load(os.path.join(sae_base, f"layer_{layer}", "seed_42", "W_dec.pt"), map_location="cpu")
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)
    n_features = W_dec.shape[0]

    tier_indices = defaultdict(list)
    for i, t in enumerate(tier_labels):
        tier_indices[t].append(i)

    # Local density
    print("    Computing local density...")
    neighborhood_sizes = np.zeros(n_features)
    for start in range(0, n_features, 1024):
        end = min(start + 1024, n_features)
        sim = W_dec[start:end] @ W_dec.T
        for i in range(end - start):
            sim[i, start + i] = 0
        neighborhood_sizes[start:end] = (sim > 0.5).sum(dim=-1).numpy()

    tier_density = {}
    for tier in ["consensus", "partial", "singleton"]:
        if tier_indices[tier]:
            d = neighborhood_sizes[tier_indices[tier]]
            tier_density[tier] = {"mean": float(np.mean(d)), "std": float(np.std(d))}

    if tier_indices["consensus"] and tier_indices["singleton"]:
        _, p_density = stats.mannwhitneyu(
            neighborhood_sizes[tier_indices["singleton"]],
            neighborhood_sizes[tier_indices["consensus"]],
            alternative="greater"
        )
    else:
        p_density = 1.0

    # DBSCAN
    print("    Running DBSCAN...")
    n_sample = min(5000, n_features)
    np.random.seed(42)
    sample_idx = np.random.choice(n_features, n_sample, replace=False)
    sample_W = W_dec[sample_idx].numpy()
    sample_tiers = [tier_labels[i] for i in sample_idx]

    cos_sim = sample_W @ sample_W.T
    cos_dist = np.clip(1 - cos_sim, 0, 2)
    clustering = DBSCAN(eps=0.4, min_samples=3, metric="precomputed").fit(cos_dist)
    labels = clustering.labels_

    tier_cluster = defaultdict(int)
    tier_total = defaultdict(int)
    for i, tier in enumerate(sample_tiers):
        tier_total[tier] += 1
        if labels[i] != -1:
            tier_cluster[tier] += 1

    tier_cluster_frac = {t: tier_cluster[t] / max(tier_total[t], 1) for t in ["consensus", "partial", "singleton"]}

    # UMAP
    print("    Computing UMAP...")
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    layer_dir = os.path.join(eval_dir, f"layer_{layer}")
    os.makedirs(layer_dir, exist_ok=True)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=15)
        umap_coords = reducer.fit_transform(sample_W)
        np.save(os.path.join(layer_dir, "umap_coords.npy"), umap_coords)
        consensus_scores, _ = load_consensus_data(layer)
        np.save(os.path.join(layer_dir, "umap_consensus_scores.npy"), consensus_scores[sample_idx])
        np.save(os.path.join(layer_dir, "umap_sample_idx.npy"), sample_idx)
    except Exception as e:
        print(f"    UMAP failed: {e}")

    result = {
        "layer": layer,
        "tier_density": tier_density,
        "density_p_value": float(p_density),
        "tier_cluster_fraction": tier_cluster_frac,
        "n_dbscan_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
    }

    print(f"    Density - Consensus: {tier_density.get('consensus', {}).get('mean', 'N/A'):.1f}, "
          f"Singleton: {tier_density.get('singleton', {}).get('mean', 'N/A'):.1f}")
    print(f"    Cluster frac - Consensus: {tier_cluster_frac.get('consensus', 0):.3f}, "
          f"Singleton: {tier_cluster_frac.get('singleton', 0):.3f}")

    return result


# === Consensus Dictionary ===

def eval_consensus_dictionary(model, eval_tokens, layer, tier_labels):
    """Build and evaluate consensus dictionary."""
    print(f"\n  [Consensus Dictionary] Layer {layer}")

    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Load feature identities
    fi_path = os.path.join(matching_base, f"layer_{layer}", "feature_identities.json")
    if not os.path.exists(fi_path):
        return {}
    with open(fi_path) as f:
        feature_identities = json.load(f)

    # Load decoders
    decoders = {}
    for seed in RANDOM_SEEDS:
        path = os.path.join(sae_base, f"layer_{layer}", f"seed_{seed}", "W_dec.pt")
        if os.path.exists(path):
            decoders[seed] = torch.load(path, map_location="cpu")

    # Build consensus dictionary
    consensus_vectors = []
    for fi in feature_identities:
        if fi["tier"] == "consensus":
            vectors = []
            for seed, idx in fi["members"]:
                if seed in decoders:
                    vec = decoders[seed][idx]
                    vec = vec / vec.norm()
                    vectors.append(vec)
            if vectors:
                centroid = torch.stack(vectors).mean(dim=0)
                centroid = centroid / centroid.norm()
                consensus_vectors.append(centroid)

    consensus_dict = torch.stack(consensus_vectors) if consensus_vectors else torch.zeros(0, HIDDEN_DIM)
    ref_W_dec = decoders[42]

    singleton_idx = [i for i, t in enumerate(tier_labels) if t == "singleton"]
    singleton_dict = ref_W_dec[singleton_idx] if singleton_idx else torch.zeros(0, HIDDEN_DIM)

    n_cons = consensus_dict.shape[0]
    np.random.seed(42)
    random_idx = np.random.choice(DICT_SIZE, min(n_cons, DICT_SIZE), replace=False) if n_cons > 0 else []
    random_dict = ref_W_dec[random_idx] if len(random_idx) > 0 else torch.zeros(0, HIDDEN_DIM)

    def eval_dict(dictionary, name):
        W = dictionary.to("cuda")
        n_f = W.shape[0]
        if n_f == 0:
            return {"name": name, "n_features": 0}
        W_enc = W.T
        cosines = []
        with torch.no_grad():
            for i in range(0, min(len(eval_tokens), 200), 32):
                batch = eval_tokens[i:i+32]
                _, cache = model.run_with_cache(batch, names_filter=[hook_name])
                hidden = cache[hook_name].reshape(-1, HIDDEN_DIM)
                pre_acts = hidden @ W_enc
                k = min(TOPK_K, n_f)
                topk_vals, topk_idx = torch.topk(pre_acts, k=k, dim=-1)
                acts = torch.zeros_like(pre_acts)
                acts.scatter_(-1, topk_idx, torch.relu(topk_vals))
                recon = acts @ W
                cos = torch.nn.functional.cosine_similarity(hidden, recon, dim=-1)
                cosines.extend(cos.cpu().numpy().tolist())
        return {"name": name, "n_features": n_f, "mean_cosine_sim": float(np.mean(cosines))}

    results = {}
    for name, d in [("consensus", consensus_dict), ("full_reference", ref_W_dec),
                    ("singleton", singleton_dict), ("random_subsample", random_dict)]:
        results[name] = eval_dict(d, name)
        print(f"    {name}: {results[name].get('n_features', 0)} features, "
              f"cosine={results[name].get('mean_cosine_sim', 'N/A')}")

    torch.cuda.empty_cache()
    return results


# === Main ===

def main():
    print("=" * 60)
    print("RUNNING ALL EVALUATIONS")
    print("=" * 60)

    eval_dir = os.path.dirname(os.path.abspath(__file__))

    model = load_model()

    # Get evaluation tokens
    print("\nLoading evaluation tokens...")
    eval_tokens = get_real_eval_tokens(model, n_sequences=1000)
    print(f"  Got {len(eval_tokens)} sequences")

    available_layers = []
    for layer in LAYERS:
        sae_w = load_sae_weights(layer, 42)
        cs, tl = load_consensus_data(layer)
        if sae_w is not None and cs is not None:
            available_layers.append(layer)

    print(f"\nAvailable layers: {available_layers}")

    all_causal = {}
    all_probing = {}
    all_steering = {}
    all_manifold = {}
    all_dictionary = {}

    for layer in available_layers:
        sae_weights = load_sae_weights(layer, 42)
        consensus_scores, tier_labels = load_consensus_data(layer)

        # Causal importance
        ci_result, causal_imp, firing_rates = eval_causal_importance(
            model, eval_tokens, layer, sae_weights, consensus_scores, tier_labels
        )
        all_causal[str(layer)] = ci_result

        # Sparse probing (only for layer 6 - most informative)
        if layer == 6:
            probing_result = eval_sparse_probing(model, eval_tokens, layer, sae_weights, tier_labels)
            all_probing = probing_result

            # Steering
            steering_result = eval_steering(model, eval_tokens, layer, sae_weights, tier_labels)
            all_steering = steering_result

            # Consensus dictionary
            dict_result = eval_consensus_dictionary(model, eval_tokens, layer, tier_labels)
            all_dictionary = dict_result

        # Manifold analysis (CPU-bound, can run for all layers)
        manifold_result = eval_manifold(layer, tier_labels)
        all_manifold[str(layer)] = manifold_result

    # Save all results
    with open(os.path.join(eval_dir, "causal_importance_results.json"), "w") as f:
        json.dump(all_causal, f, indent=2)
    with open(os.path.join(eval_dir, "sparse_probing_results.json"), "w") as f:
        json.dump(all_probing, f, indent=2)
    with open(os.path.join(eval_dir, "steering_results.json"), "w") as f:
        json.dump(all_steering, f, indent=2)
    with open(os.path.join(eval_dir, "manifold_analysis_results.json"), "w") as f:
        json.dump(all_manifold, f, indent=2)
    with open(os.path.join(eval_dir, "consensus_dictionary_results.json"), "w") as f:
        json.dump(all_dictionary, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
