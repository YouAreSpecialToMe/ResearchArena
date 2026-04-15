"""Sparse probing evaluation: test if consensus features are more useful for downstream tasks."""

import sys
import os
import json
import time
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
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


def create_probing_tasks(model, n_samples=3000):
    """Create probing tasks using simple heuristics (no external datasets needed)."""
    from datasets import load_dataset

    tokenizer = model.tokenizer
    ds = load_dataset(DATASET_PATH, split="train", streaming=True, trust_remote_code=True)

    # Collect tokens and their contexts
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

    # Task 1: Capitalization detection (proxy for NER / proper nouns)
    # Label: whether the token starts with a capital letter
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
    import string
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


def run_sparse_probing(layer=6, ref_seed=42, k_values=[5, 10, 20, 50]):
    """Run sparse probing evaluation."""
    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    model = load_model()

    # Load consensus scores
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

    results = {}

    for task_name, task_data in tasks.items():
        print(f"\n  Task: {task_name}")
        labels = task_data["labels"].reshape(-1)[:n_total]

        # Train/test split
        n_train = int(0.8 * n_total)
        X_train, X_test = all_acts[:n_train], all_acts[n_train:]
        y_train, y_test = labels[:n_train], labels[n_train:]

        # Skip if labels are too imbalanced
        if y_train.mean() < 0.05 or y_train.mean() > 0.95:
            print(f"    Skipping {task_name}: too imbalanced ({y_train.mean():.3f})")
            continue

        task_results = {}

        for k in k_values:
            k_results = {}

            for tier_name, feat_indices in tier_indices.items():
                if len(feat_indices) < k:
                    k_results[tier_name] = {"accuracy": 0.5, "n_features": len(feat_indices)}
                    continue

                # Select top-k features by mutual information
                feat_idx = np.array(feat_indices)
                X_tier_train = X_train[:, feat_idx]

                # Use variance as a fast proxy for MI to rank features
                variances = X_tier_train.var(axis=0)
                top_k_local = np.argsort(variances)[-k:]
                selected_feats = feat_idx[top_k_local]

                X_sel_train = X_train[:, selected_feats]
                X_sel_test = X_test[:, selected_feats]

                # Train probe
                clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
                clf.fit(X_sel_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_sel_test))
                k_results[tier_name] = {"accuracy": float(acc), "n_features": len(feat_indices)}

            # Random baseline
            random_accs = []
            for rs in range(5):
                np.random.seed(rs)
                rand_feats = np.random.choice(DICT_SIZE, k, replace=False)
                X_rand_train = X_train[:, rand_feats]
                X_rand_test = X_test[:, rand_feats]
                clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
                clf.fit(X_rand_train, y_train)
                random_accs.append(accuracy_score(y_test, clf.predict(X_rand_test)))
            k_results["random"] = {
                "accuracy": float(np.mean(random_accs)),
                "std": float(np.std(random_accs)),
            }

            task_results[f"k={k}"] = k_results
            print(f"    k={k}: consensus={k_results.get('consensus', {}).get('accuracy', 'N/A'):.3f}, "
                  f"singleton={k_results.get('singleton', {}).get('accuracy', 'N/A'):.3f}, "
                  f"random={k_results['random']['accuracy']:.3f}")

        results[task_name] = task_results

    # Save results
    with open(os.path.join(output_dir, "sparse_probing_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSparse probing evaluation complete!")
    return results


if __name__ == "__main__":
    run_sparse_probing()
