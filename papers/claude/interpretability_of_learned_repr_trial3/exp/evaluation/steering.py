"""Feature steering effectiveness: test whether consensus features produce more coherent behavioral changes."""

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
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    model.eval()
    return model


def load_sae_weights(layer, seed, base_dir):
    path = os.path.join(base_dir, f"layer_{layer}", f"seed_{seed}", "sae_weights.pt")
    return torch.load(path, map_location="cuda")


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


def run_steering_evaluation(layer=6, ref_seed=42, n_features_per_tier=50, steering_mult=3.0):
    """Run steering effectiveness evaluation."""
    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    model = load_model()

    # Load consensus data
    consensus_scores = np.load(os.path.join(matching_base, f"layer_{layer}", "consensus_scores.npy"))
    with open(os.path.join(matching_base, f"layer_{layer}", "tier_labels.json")) as f:
        tier_labels = json.load(f)

    sae_weights = load_sae_weights(layer, ref_seed, sae_base)
    W_dec = sae_weights["W_dec"]

    # Get evaluation tokens
    from datasets import load_dataset
    tokenizer = model.tokenizer
    ds = load_dataset(DATASET_PATH, split="train", streaming=True, trust_remote_code=True)
    eval_tokens = []
    for item in ds:
        tokens = tokenizer.encode(item["text"], return_tensors="pt")[0]
        if len(tokens) >= CONTEXT_SIZE:
            eval_tokens.append(tokens[:CONTEXT_SIZE])
        if len(eval_tokens) >= 500:
            break
    eval_tokens = torch.stack(eval_tokens).to("cuda")

    # Compute SAE activations to find firing rates
    print("Computing firing rates...")
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(eval_tokens), EVAL_BATCH_SIZE):
            batch = eval_tokens[i:i+EVAL_BATCH_SIZE]
            _, cache = model.run_with_cache(batch, names_filter=[f"blocks.{layer}.hook_resid_post"])
            hidden = cache[f"blocks.{layer}.hook_resid_post"]
            flat_hidden = hidden.reshape(-1, HIDDEN_DIM)
            acts = compute_sae_activations(flat_hidden, sae_weights)
            all_acts.append(acts.cpu())
    all_acts = torch.cat(all_acts, dim=0)  # (n_tokens, dict_size)

    firing_rates = (all_acts > 0).float().mean(dim=0).numpy()

    # Select features matched by firing rate
    consensus_idx = [i for i, t in enumerate(tier_labels) if t == "consensus" and firing_rates[i] > 0.01]
    singleton_idx = [i for i, t in enumerate(tier_labels) if t == "singleton" and firing_rates[i] > 0.01]

    # Match by firing frequency
    if len(consensus_idx) > n_features_per_tier:
        np.random.seed(42)
        consensus_idx = list(np.random.choice(consensus_idx, n_features_per_tier, replace=False))
    if len(singleton_idx) > n_features_per_tier:
        np.random.seed(42)
        singleton_idx = list(np.random.choice(singleton_idx, n_features_per_tier, replace=False))

    print(f"Selected {len(consensus_idx)} consensus, {len(singleton_idx)} singleton features")

    # Steering evaluation
    def evaluate_steering(feature_indices, label):
        steering_effects = []
        steering_consistencies = []

        for f_idx in feature_indices:
            decoder_vec = W_dec[f_idx].to("cuda")

            # Find tokens where feature is active
            active_mask = all_acts[:, f_idx] > 0
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            if len(active_indices) < 50:
                continue

            # Sample tokens
            sample_idx = active_indices[:200]
            sample_seq = sample_idx // CONTEXT_SIZE
            sample_pos = sample_idx % CONTEXT_SIZE

            # Get mean activation
            mean_act = all_acts[sample_idx, f_idx].mean().item()

            # Compute steering effect: add steering_mult * mean_act * decoder_vec to hidden states
            kl_divs = []
            batch_size = 50
            for bi in range(0, min(len(sample_idx), 200), batch_size):
                be = min(bi + batch_size, len(sample_idx))
                batch_seq_idx = sample_seq[bi:be].unique()

                for seq_i in batch_seq_idx:
                    tokens = eval_tokens[seq_i:seq_i+1]

                    # Original forward pass
                    with torch.no_grad():
                        orig_logits = model(tokens)

                    # Steered forward pass
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    def steering_hook(activation, hook):
                        activation = activation + steering_mult * mean_act * decoder_vec
                        return activation

                    with torch.no_grad():
                        steered_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, steering_hook)])

                    # KL divergence
                    orig_probs = torch.softmax(orig_logits[0], dim=-1)
                    steered_logprobs = torch.log_softmax(steered_logits[0], dim=-1)
                    kl = (orig_probs * (torch.log(orig_probs + 1e-10) - steered_logprobs)).sum(dim=-1)
                    kl_divs.extend(kl.cpu().numpy().tolist())

            if len(kl_divs) > 0:
                steering_effects.append(float(np.mean(kl_divs)))
                steering_consistencies.append(float(1.0 / (np.std(kl_divs) + 1e-6)))

        return steering_effects, steering_consistencies

    print("Evaluating consensus feature steering...")
    consensus_effects, consensus_consistencies = evaluate_steering(consensus_idx, "consensus")
    print("Evaluating singleton feature steering...")
    singleton_effects, singleton_consistencies = evaluate_steering(singleton_idx, "singleton")

    # Statistical comparison
    if len(consensus_effects) > 3 and len(singleton_effects) > 3:
        u_effect, p_effect = stats.mannwhitneyu(consensus_effects, singleton_effects, alternative="greater")
        u_consist, p_consist = stats.mannwhitneyu(consensus_consistencies, singleton_consistencies, alternative="greater")
    else:
        u_effect, p_effect = 0, 1.0
        u_consist, p_consist = 0, 1.0

    results = {
        "layer": layer,
        "n_consensus_features": len(consensus_effects),
        "n_singleton_features": len(singleton_effects),
        "steering_multiplier": steering_mult,
        "consensus_mean_effect": float(np.mean(consensus_effects)) if consensus_effects else 0,
        "consensus_std_effect": float(np.std(consensus_effects)) if consensus_effects else 0,
        "singleton_mean_effect": float(np.mean(singleton_effects)) if singleton_effects else 0,
        "singleton_std_effect": float(np.std(singleton_effects)) if singleton_effects else 0,
        "effect_mann_whitney_p": float(p_effect),
        "consensus_mean_consistency": float(np.mean(consensus_consistencies)) if consensus_consistencies else 0,
        "singleton_mean_consistency": float(np.mean(singleton_consistencies)) if singleton_consistencies else 0,
        "consistency_mann_whitney_p": float(p_consist),
    }

    with open(os.path.join(output_dir, "steering_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSteering results:")
    print(f"  Consensus mean effect: {results['consensus_mean_effect']:.6f}")
    print(f"  Singleton mean effect: {results['singleton_mean_effect']:.6f}")
    print(f"  Effect p-value: {p_effect:.6f}")

    return results


if __name__ == "__main__":
    run_steering_evaluation()
