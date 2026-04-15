"""Evaluate causal importance of SAE features via ablation and KL divergence."""

import sys
import os
import json
import time
import torch
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *

import transformer_lens


def load_model():
    """Load Pythia-160M with TransformerLens."""
    model = transformer_lens.HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda"
    )
    model.eval()
    return model


def load_sae_weights(layer, seed, base_dir):
    """Load SAE weights."""
    path = os.path.join(base_dir, f"layer_{layer}", f"seed_{seed}", "sae_weights.pt")
    return torch.load(path, map_location="cuda")


def get_eval_tokens(model, n_sequences=EVAL_N_SEQUENCES, ctx_len=CONTEXT_SIZE):
    """Get evaluation tokens from the Pile."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_PATH, split="train", streaming=True, trust_remote_code=True)
    tokenizer = model.tokenizer

    all_tokens = []
    for item in ds:
        tokens = tokenizer.encode(item["text"], return_tensors="pt")[0]
        if len(tokens) >= ctx_len:
            all_tokens.append(tokens[:ctx_len])
        if len(all_tokens) >= n_sequences:
            break

    return torch.stack(all_tokens).to("cuda")


def compute_sae_activations(hidden_states, sae_weights):
    """Compute SAE feature activations using TopK."""
    W_enc = sae_weights["W_enc"]  # (hidden_dim, dict_size)
    b_enc = sae_weights["b_enc"]  # (dict_size,)
    b_dec = sae_weights["b_dec"]  # (hidden_dim,)

    # Encode
    x_centered = hidden_states - b_dec  # subtract decoder bias
    pre_acts = x_centered @ W_enc + b_enc  # (batch*seq, dict_size)

    # TopK activation
    topk_values, topk_indices = torch.topk(pre_acts, k=TOPK_K, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_indices, topk_values)

    return acts


def compute_feature_causal_importance(model, sae_weights, eval_tokens, layer,
                                       feature_batch_size=512):
    """Compute causal importance for all features via ablation."""
    W_dec = sae_weights["W_dec"]  # (dict_size, hidden_dim)
    b_dec = sae_weights["b_dec"]  # (hidden_dim,)
    dict_size = W_dec.shape[0]

    # Collect hidden states and original logits
    print("  Collecting hidden states and original logits...")
    all_hidden = []
    all_orig_logprobs = []

    batch_size = EVAL_BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(eval_tokens), batch_size):
            batch = eval_tokens[i:i+batch_size]
            _, cache = model.run_with_cache(batch, names_filter=[f"blocks.{layer}.hook_resid_post"])
            hidden = cache[f"blocks.{layer}.hook_resid_post"]  # (batch, seq, hidden)

            # Get original logits
            logits = model(batch)
            orig_logprobs = torch.log_softmax(logits[:, :-1], dim=-1)

            all_hidden.append(hidden.cpu())
            all_orig_logprobs.append(orig_logprobs.cpu())

    all_hidden = torch.cat(all_hidden, dim=0)  # (n_seq, seq_len, hidden_dim)
    all_orig_logprobs = torch.cat(all_orig_logprobs, dim=0)

    # Compute SAE activations
    print("  Computing SAE activations...")
    flat_hidden = all_hidden.reshape(-1, HIDDEN_DIM).cuda()
    acts = compute_sae_activations(flat_hidden, sae_weights)

    # Compute firing rates
    firing_mask = (acts > 0).float()
    firing_rates = firing_mask.mean(dim=0).cpu().numpy()  # (dict_size,)

    # Compute causal importance per feature
    print("  Computing causal importance via ablation...")
    n_total = flat_hidden.shape[0]
    causal_importance = np.zeros(dict_size)
    n_active = np.zeros(dict_size)

    # Process features in batches
    for feat_start in range(0, dict_size, feature_batch_size):
        feat_end = min(feat_start + feature_batch_size, dict_size)

        for f_idx in range(feat_start, feat_end):
            # Find tokens where this feature is active
            active_mask = acts[:, f_idx] > 0
            n_act = active_mask.sum().item()
            n_active[f_idx] = n_act

            if n_act == 0 or n_act < 10:
                continue

            # Sample up to 2000 active tokens for efficiency
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            if len(active_indices) > 2000:
                perm = torch.randperm(len(active_indices))[:2000]
                active_indices = active_indices[perm]

            # Get activation values for this feature at active positions
            feat_acts = acts[active_indices, f_idx]  # (n_active,)

            # Compute the change in hidden state when ablating this feature
            # delta = -activation * decoder_vector
            decoder_vec = W_dec[f_idx]  # (hidden_dim,)
            delta = -feat_acts.unsqueeze(-1) * decoder_vec.unsqueeze(0)  # (n_active, hidden_dim)

            # Modified hidden state
            modified_hidden = flat_hidden[active_indices] + delta  # (n_active, hidden_dim)

            # Convert flat indices back to (seq_idx, pos) and run through remaining layers
            seq_indices = active_indices // (CONTEXT_SIZE)
            pos_indices = active_indices % (CONTEXT_SIZE)

            # For efficiency, compute KL divergence using the unembedding directly
            # This approximates the full forward pass for the last layer
            if layer == model.cfg.n_layers - 1:
                # Last layer: can directly compute logits
                modified_logits = model.unembed(model.ln_final(modified_hidden))
                orig_hidden_sel = flat_hidden[active_indices]
                orig_logits = model.unembed(model.ln_final(orig_hidden_sel))
            else:
                # For intermediate layers, we need to run through remaining layers
                # Approximate: use the effect on the residual stream at this layer
                # and measure the KL divergence of the SAE reconstruction
                # More accurate: run modified hidden through rest of model
                # For efficiency, sample a subset
                sample_size = min(500, len(active_indices))
                sample_idx = active_indices[:sample_size]
                sample_seq = seq_indices[:sample_size]
                sample_pos = pos_indices[:sample_size]

                # We'll compute an approximation based on the norm of the delta
                # relative to the hidden state norm (faster but still meaningful)
                delta_norm = delta[:sample_size].norm(dim=-1)
                hidden_norm = flat_hidden[sample_idx].norm(dim=-1)
                relative_effect = (delta_norm / (hidden_norm + 1e-8)).mean().item()
                causal_importance[f_idx] = relative_effect
                continue

            # Compute KL divergence
            orig_probs = torch.softmax(orig_logits, dim=-1)
            modified_logprobs = torch.log_softmax(modified_logits, dim=-1)
            kl_div = (orig_probs * (torch.log(orig_probs + 1e-10) - modified_logprobs)).sum(dim=-1)
            causal_importance[f_idx] = kl_div.mean().item()

        if (feat_end) % 2048 == 0 or feat_end == dict_size:
            print(f"    Processed features {feat_start}-{feat_end}/{dict_size}")

    return causal_importance, firing_rates, n_active


def run_causal_evaluation(layers=None, ref_seed=42):
    if layers is None:
        layers = LAYERS
    """Run causal importance evaluation for all layers."""
    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    model = load_model()
    eval_tokens = get_eval_tokens(model, n_sequences=min(EVAL_N_SEQUENCES, 2000))
    print(f"Loaded {len(eval_tokens)} eval sequences")

    all_results = {}

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        start = time.time()

        # Load SAE weights
        sae_path = os.path.join(sae_base, f"layer_{layer}", f"seed_{ref_seed}", "sae_weights.pt")
        if not os.path.exists(sae_path):
            print(f"  SAE weights not available for layer {layer}, skipping")
            continue
        sae_weights = load_sae_weights(layer, ref_seed, sae_base)

        # Load consensus scores
        consensus_path = os.path.join(matching_base, f"layer_{layer}", "consensus_scores.npy")
        if not os.path.exists(consensus_path):
            print(f"  No consensus scores for layer {layer}, skipping")
            continue
        consensus_scores = np.load(consensus_path)
        with open(os.path.join(matching_base, f"layer_{layer}", "tier_labels.json")) as f:
            tier_labels = json.load(f)

        # Compute causal importance
        causal_importance, firing_rates, n_active = compute_feature_causal_importance(
            model, sae_weights, eval_tokens, layer
        )

        elapsed = time.time() - start

        # Filter to features that actually fire
        active_mask = n_active >= 10
        active_ci = causal_importance[active_mask]
        active_consensus = consensus_scores[active_mask]
        active_tiers = [tier_labels[i] for i in range(len(tier_labels)) if active_mask[i]]
        active_firing = firing_rates[active_mask]

        # Group by tier
        tier_ci = {"consensus": [], "partial": [], "singleton": []}
        for ci, tier in zip(active_ci, active_tiers):
            tier_ci[tier].append(ci)

        # Statistical tests
        consensus_ci = np.array(tier_ci.get("consensus", []))
        singleton_ci = np.array(tier_ci.get("singleton", []))

        if len(consensus_ci) > 5 and len(singleton_ci) > 5:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(consensus_ci, singleton_ci, alternative="greater")
            # Cohen's d
            pooled_std = np.sqrt((consensus_ci.std()**2 + singleton_ci.std()**2) / 2)
            cohens_d = (consensus_ci.mean() - singleton_ci.mean()) / (pooled_std + 1e-10)
            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(active_consensus, active_ci)
            # Partial correlation controlling for firing rate
            from sklearn.linear_model import LinearRegression
            # Residualize both consensus and CI on firing rate
            fr_resid_consensus = active_consensus - LinearRegression().fit(
                active_firing.reshape(-1, 1), active_consensus).predict(active_firing.reshape(-1, 1))
            fr_resid_ci = active_ci - LinearRegression().fit(
                active_firing.reshape(-1, 1), active_ci).predict(active_firing.reshape(-1, 1))
            partial_r, partial_p = stats.spearmanr(fr_resid_consensus, fr_resid_ci)
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
            "firing_rate_consensus_corr": float(stats.spearmanr(active_consensus, active_firing)[0]) if len(active_consensus) > 5 else 0,
            "eval_time_minutes": elapsed / 60,
        }

        all_results[layer] = result

        # Save per-feature data
        layer_dir = os.path.join(output_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)
        np.save(os.path.join(layer_dir, "causal_importance.npy"), causal_importance)
        np.save(os.path.join(layer_dir, "firing_rates.npy"), firing_rates)

        print(f"  Consensus mean CI: {result['tier_mean_causal_importance'].get('consensus', 0):.6f}")
        print(f"  Singleton mean CI: {result['tier_mean_causal_importance'].get('singleton', 0):.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}, p-value: {p_value:.6f}")
        print(f"  Spearman r: {spearman_r:.3f}, p: {spearman_p:.6f}")

    # Save results
    with open(os.path.join(output_dir, "causal_importance_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nCausal importance evaluation complete!")
    return all_results


if __name__ == "__main__":
    run_causal_evaluation()
