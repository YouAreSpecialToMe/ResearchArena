"""Sparse probing evaluation with multiple seeds for error bars.

Runs probing with 5 random seeds and reports mean +/- std for each
(task, k, tier) combination.
"""

import sys
import os
import json
import time
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *

import transformer_lens


PROBE_SEEDS = [42, 137, 256, 512, 1024]


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


def create_probing_tasks(model, n_samples=3000):
    """Create probing tasks using simple heuristics (no external datasets needed)."""
    from datasets import load_dataset
    import string

    tokenizer = model.tokenizer
    ds = load_dataset(DATASET_PATH, split="train", streaming=True, trust_remote_code=True)

    all_tokens = []
    all_texts = []
    for item in ds:
        text = item["text"]
        tokens = tokenizer.encode(text, return_tensors="pt")[0]
        if len(tokens) >= CONTEXT_SIZE:
            all_tokens.append(tokens[:CONTEXT_SIZE])
            all_texts.append(text)
        if len(all_tokens) >= n_samples // CONTEXT_SIZE + 10:
            break

    token_ids = torch.stack(all_tokens).to("cuda")

    tasks = {}

    # Task 1: Capitalization detection
    labels_cap = []
    for seq in all_tokens:
        seq_labels = []
        for tid in seq:
            decoded = tokenizer.decode([tid.item()])
            is_cap = 1 if (len(decoded.strip()) > 0 and decoded.strip()[0].isupper()) else 0
            seq_labels.append(is_cap)
        labels_cap.append(seq_labels)
    tasks["capitalization"] = {
        "labels": np.array(labels_cap),
        "description": "Detect capitalized tokens (proxy for NER)",
    }

    # Task 2: Number detection
    labels_num = []
    for seq in all_tokens:
        seq_labels = []
        for tid in seq:
            decoded = tokenizer.decode([tid.item()])
            is_num = 1 if any(c.isdigit() for c in decoded) else 0
            seq_labels.append(is_num)
        labels_num.append(seq_labels)
    tasks["number_detection"] = {
        "labels": np.array(labels_num),
        "description": "Detect tokens containing digits",
    }

    # Task 3: Punctuation detection
    labels_punct = []
    for seq in all_tokens:
        seq_labels = []
        for tid in seq:
            decoded = tokenizer.decode([tid.item()])
            is_punct = 1 if (len(decoded.strip()) > 0 and decoded.strip()[0] in string.punctuation) else 0
            seq_labels.append(is_punct)
        labels_punct.append(seq_labels)
    tasks["punctuation"] = {
        "labels": np.array(labels_punct),
        "description": "Detect punctuation tokens",
    }

    # Task 4: Whitespace-preceded token (proxy for word boundary)
    labels_ws = []
    for seq in all_tokens:
        seq_labels = []
        for tid in seq:
            decoded = tokenizer.decode([tid.item()])
            starts_ws = 1 if (len(decoded) > 0 and decoded[0] == ' ') else 0
            seq_labels.append(starts_ws)
        labels_ws.append(seq_labels)
    tasks["word_boundary"] = {
        "labels": np.array(labels_ws),
        "description": "Detect word-starting tokens",
    }

    return token_ids, tasks


def run_sparse_probing_multiseed(layer=6, ref_seed=42, k_values=[5, 10, 20, 50]):
    """Run sparse probing evaluation with multiple seeds for error bars."""
    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    model = load_model()

    # Load consensus scores and tier labels
    consensus_scores = np.load(os.path.join(matching_base, f"layer_{layer}", "consensus_scores.npy"))
    with open(os.path.join(matching_base, f"layer_{layer}", "tier_labels.json")) as f:
        tier_labels = json.load(f)

    # Load SAE weights
    sae_weights = load_sae_weights(layer, ref_seed, sae_base)

    # Create probing tasks
    print("Creating probing tasks...")
    token_ids, tasks = create_probing_tasks(model, n_samples=3000)

    # Get SAE activations
    print("Computing SAE activations...")
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(token_ids), EVAL_BATCH_SIZE):
            batch = token_ids[i:i+EVAL_BATCH_SIZE]
            _, cache = model.run_with_cache(batch, names_filter=[f"blocks.{layer}.hook_resid_post"])
            hidden = cache[f"blocks.{layer}.hook_resid_post"]
            flat_hidden = hidden.reshape(-1, HIDDEN_DIM)
            acts = compute_sae_activations(flat_hidden, sae_weights)
            all_acts.append(acts.cpu())

    all_acts = torch.cat(all_acts, dim=0).numpy()  # (n_seq * seq_len, dict_size)
    n_total = all_acts.shape[0]

    # Identify features per tier
    tier_indices = {"consensus": [], "partial": [], "singleton": []}
    for i, tier in enumerate(tier_labels):
        tier_indices[tier].append(i)

    print(f"\nTier sizes: consensus={len(tier_indices['consensus'])}, "
          f"partial={len(tier_indices['partial'])}, singleton={len(tier_indices['singleton'])}")
    print(f"Total samples: {n_total}, using seeds: {PROBE_SEEDS}\n")

    results = {}

    for task_name, task_data in tasks.items():
        print(f"\n  Task: {task_name}")
        labels = task_data["labels"].reshape(-1)[:n_total]

        # Check balance
        pos_rate = labels.mean()
        if pos_rate < 0.05 or pos_rate > 0.95:
            print(f"    Skipping {task_name}: too imbalanced ({pos_rate:.3f})")
            continue

        print(f"    Positive rate: {pos_rate:.3f}")

        task_results = {}

        for k in k_values:
            k_results = {}

            for tier_name, feat_indices in tier_indices.items():
                if len(feat_indices) < k:
                    k_results[tier_name] = {
                        "mean": 0.5,
                        "std": 0.0,
                        "n_features": len(feat_indices),
                        "seeds_used": 0,
                        "per_seed": [],
                    }
                    continue

                feat_idx = np.array(feat_indices)
                seed_accs = []

                for probe_seed in PROBE_SEEDS:
                    rng = np.random.RandomState(probe_seed)

                    # Shuffle indices for train/test split
                    perm = rng.permutation(n_total)
                    n_train = int(0.8 * n_total)
                    train_idx = perm[:n_train]
                    test_idx = perm[n_train:]

                    X_train = all_acts[train_idx]
                    X_test = all_acts[test_idx]
                    y_train = labels[train_idx]
                    y_test = labels[test_idx]

                    # Select top-k features by variance on training set
                    X_tier_train = X_train[:, feat_idx]
                    variances = X_tier_train.var(axis=0)
                    top_k_local = np.argsort(variances)[-k:]
                    selected_feats = feat_idx[top_k_local]

                    X_sel_train = X_train[:, selected_feats]
                    X_sel_test = X_test[:, selected_feats]

                    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                             random_state=probe_seed)
                    clf.fit(X_sel_train, y_train)
                    acc = accuracy_score(y_test, clf.predict(X_sel_test))
                    seed_accs.append(float(acc))

                k_results[tier_name] = {
                    "mean": float(np.mean(seed_accs)),
                    "std": float(np.std(seed_accs)),
                    "n_features": len(feat_indices),
                    "seeds_used": len(PROBE_SEEDS),
                    "per_seed": seed_accs,
                }

            # Random baseline with same multi-seed protocol
            seed_accs_random = []
            for probe_seed in PROBE_SEEDS:
                rng = np.random.RandomState(probe_seed)
                perm = rng.permutation(n_total)
                n_train = int(0.8 * n_total)
                train_idx = perm[:n_train]
                test_idx = perm[n_train:]

                X_train = all_acts[train_idx]
                X_test = all_acts[test_idx]
                y_train = labels[train_idx]
                y_test = labels[test_idx]

                # Random features (use a different RNG so feature selection varies)
                feat_rng = np.random.RandomState(probe_seed + 99999)
                rand_feats = feat_rng.choice(DICT_SIZE, k, replace=False)

                X_rand_train = X_train[:, rand_feats]
                X_rand_test = X_test[:, rand_feats]

                clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                         random_state=probe_seed)
                clf.fit(X_rand_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_rand_test))
                seed_accs_random.append(float(acc))

            k_results["random"] = {
                "mean": float(np.mean(seed_accs_random)),
                "std": float(np.std(seed_accs_random)),
                "n_features": DICT_SIZE,
                "seeds_used": len(PROBE_SEEDS),
                "per_seed": seed_accs_random,
            }

            task_results[f"k={k}"] = k_results

            cons_info = k_results.get("consensus", {})
            sing_info = k_results.get("singleton", {})
            rand_info = k_results["random"]
            print(f"    k={k}: "
                  f"consensus={cons_info.get('mean', 'N/A'):.4f}+/-{cons_info.get('std', 0):.4f}, "
                  f"singleton={sing_info.get('mean', 'N/A'):.4f}+/-{sing_info.get('std', 0):.4f}, "
                  f"random={rand_info['mean']:.4f}+/-{rand_info['std']:.4f}")

        results[task_name] = task_results

    # Strip per_seed arrays for the saved output (keep clean format)
    clean_results = {}
    for task_name, task_data in results.items():
        clean_results[task_name] = {}
        for k_key, k_data in task_data.items():
            clean_results[task_name][k_key] = {}
            for tier_name, tier_data in k_data.items():
                clean_results[task_name][k_key][tier_name] = {
                    "mean": tier_data["mean"],
                    "std": tier_data["std"],
                    "n_features": tier_data["n_features"],
                    "seeds_used": tier_data["seeds_used"],
                }

    # Save results
    out_path = os.path.join(output_dir, "sparse_probing_multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nMulti-seed sparse probing evaluation complete!")
    print(f"Results saved to {out_path}")
    return clean_results


if __name__ == "__main__":
    run_sparse_probing_multiseed()
