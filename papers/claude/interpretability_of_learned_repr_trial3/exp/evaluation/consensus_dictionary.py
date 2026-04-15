"""Consensus dictionary construction and evaluation."""

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


def build_consensus_dictionary(layer, ref_seed=42):
    """Build consensus dictionary from centroid decoder vectors."""
    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")

    # Load feature identities
    with open(os.path.join(matching_base, f"layer_{layer}", "feature_identities.json")) as f:
        feature_identities = json.load(f)

    # Load all decoder weight matrices
    decoders = {}
    for seed in RANDOM_SEEDS:
        path = os.path.join(sae_base, f"layer_{layer}", f"seed_{seed}", "W_dec.pt")
        if os.path.exists(path):
            decoders[seed] = torch.load(path, map_location="cpu")

    # Build consensus dictionary: centroid of matched decoder vectors
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

    if consensus_vectors:
        consensus_dict = torch.stack(consensus_vectors)
    else:
        consensus_dict = torch.zeros(0, HIDDEN_DIM)

    return consensus_dict


def evaluate_dictionary(model, dictionary, eval_tokens, layer, name="dict"):
    """Evaluate a dictionary on reconstruction quality."""
    W_dec = dictionary.to("cuda")  # (n_features, hidden_dim)
    n_features = W_dec.shape[0]

    if n_features == 0:
        return {"name": name, "n_features": 0, "error": "empty dictionary"}

    # Pseudo-encoder: project onto dictionary
    W_enc = W_dec.T  # (hidden_dim, n_features) - simple transpose encoder

    all_recon_cosine = []
    all_recon_mse = []

    with torch.no_grad():
        for i in range(0, len(eval_tokens), EVAL_BATCH_SIZE):
            batch = eval_tokens[i:i+EVAL_BATCH_SIZE]
            _, cache = model.run_with_cache(batch, names_filter=[f"blocks.{layer}.hook_resid_post"])
            hidden = cache[f"blocks.{layer}.hook_resid_post"]
            flat_hidden = hidden.reshape(-1, HIDDEN_DIM)

            # Encode with TopK
            pre_acts = flat_hidden @ W_enc
            k = min(TOPK_K, n_features)
            topk_values, topk_indices = torch.topk(pre_acts, k=k, dim=-1)
            acts = torch.zeros_like(pre_acts)
            acts.scatter_(-1, topk_indices, torch.relu(topk_values))

            # Decode
            reconstructed = acts @ W_dec

            # Metrics
            cosine_sim = torch.nn.functional.cosine_similarity(flat_hidden, reconstructed, dim=-1)
            mse = ((flat_hidden - reconstructed) ** 2).mean(dim=-1)

            all_recon_cosine.extend(cosine_sim.cpu().numpy().tolist())
            all_recon_mse.extend(mse.cpu().numpy().tolist())

    return {
        "name": name,
        "n_features": n_features,
        "mean_cosine_sim": float(np.mean(all_recon_cosine)),
        "std_cosine_sim": float(np.std(all_recon_cosine)),
        "mean_mse": float(np.mean(all_recon_mse)),
        "std_mse": float(np.std(all_recon_mse)),
        "cosine_per_feature": float(np.mean(all_recon_cosine) / max(n_features, 1) * 1000),
    }


def run_consensus_dictionary_eval(layer=6, ref_seed=42):
    """Run full consensus dictionary evaluation."""
    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    model = load_model()

    # Get eval tokens
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

    # Build consensus dictionary
    print("Building consensus dictionary...")
    consensus_dict = build_consensus_dictionary(layer, ref_seed)
    print(f"  Consensus dictionary: {consensus_dict.shape[0]} features")

    # Load reference SAE decoder (full)
    ref_W_dec = torch.load(os.path.join(sae_base, f"layer_{layer}", f"seed_{ref_seed}", "W_dec.pt"),
                           map_location="cpu")

    # Load tier labels for reference SAE
    with open(os.path.join(matching_base, f"layer_{layer}", "tier_labels.json")) as f:
        tier_labels = json.load(f)

    # Singleton dictionary
    singleton_idx = [i for i, t in enumerate(tier_labels) if t == "singleton"]
    singleton_dict = ref_W_dec[singleton_idx] if singleton_idx else torch.zeros(0, HIDDEN_DIM)

    # Random subsample (same size as consensus)
    n_consensus = consensus_dict.shape[0]
    np.random.seed(42)
    if n_consensus > 0 and n_consensus < DICT_SIZE:
        random_idx = np.random.choice(DICT_SIZE, n_consensus, replace=False)
        random_dict = ref_W_dec[random_idx]
    else:
        random_dict = ref_W_dec

    # Evaluate all dictionaries
    results = {}
    for name, dictionary in [
        ("consensus", consensus_dict),
        ("full_reference", ref_W_dec),
        ("singleton", singleton_dict),
        ("random_subsample", random_dict),
    ]:
        print(f"Evaluating {name} dictionary ({dictionary.shape[0]} features)...")
        results[name] = evaluate_dictionary(model, dictionary, eval_tokens, layer, name)
        print(f"  Cosine sim: {results[name].get('mean_cosine_sim', 'N/A')}")

    # Save results
    with open(os.path.join(output_dir, "consensus_dictionary_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nConsensus dictionary evaluation complete!")
    return results


if __name__ == "__main__":
    run_consensus_dictionary_eval()
