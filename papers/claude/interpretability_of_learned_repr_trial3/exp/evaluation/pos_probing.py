"""POS tagging probing evaluation: test if consensus features encode semantic-level linguistic information."""

import sys
import os
import json
import time
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *

import transformer_lens
import nltk

# Download required NLTK data
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)


# Simplified POS mapping: map Penn Treebank tags to 4 classes
POS_MAP = {
    # Nouns
    'NN': 0, 'NNS': 0, 'NNP': 0, 'NNPS': 0,
    # Verbs
    'VB': 1, 'VBD': 1, 'VBG': 1, 'VBN': 1, 'VBP': 1, 'VBZ': 1,
    'MD': 1,
    # Adjectives / Adverbs
    'JJ': 2, 'JJR': 2, 'JJS': 2,
    'RB': 2, 'RBR': 2, 'RBS': 2,
    # Other (default)
}
POS_NAMES = {0: 'noun', 1: 'verb', 2: 'adj/adv', 3: 'other'}
DEFAULT_POS = 3


def load_model():
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    model.eval()
    return model


def load_sae_weights(layer, seed):
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "sae_training", "topk")
    path = os.path.join(base, f"layer_{layer}", f"seed_{seed}", "sae_weights.pt")
    return torch.load(path, map_location="cuda", weights_only=True)


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


def get_pos_labels_for_tokens(tokenizer, token_ids_batch):
    """
    For each token in a batch of sequences, assign a simplified POS tag.

    Strategy: decode each token individually, strip whitespace, then use
    NLTK to tag it in a minimal context. For BPE sub-tokens that are partial
    words, we tag the decoded text directly.
    """
    all_labels = []

    for seq_idx, seq in enumerate(token_ids_batch):
        # First, decode the full sequence and get word-level POS tags
        full_text = tokenizer.decode(seq.tolist())

        # Build a mapping: for each token position, figure out its word and POS
        # Approach: decode tokens one by one, accumulate text, match to words
        tokens_decoded = []
        for tid in seq:
            tokens_decoded.append(tokenizer.decode([tid.item()]))

        # Build character-to-token mapping
        # Reconstruct text token by token and track positions
        char_pos = 0
        token_char_ranges = []
        reconstructed = ""
        for i, tok_text in enumerate(tokens_decoded):
            start = len(reconstructed)
            reconstructed += tok_text
            end = len(reconstructed)
            token_char_ranges.append((start, end))

        # Use NLTK to tokenize words and get POS tags from the full text
        try:
            words = nltk.word_tokenize(full_text)
            tagged = nltk.pos_tag(words)
        except Exception:
            # Fallback: tag each decoded token individually
            tagged = []
            for tok_text in tokens_decoded:
                clean = tok_text.strip()
                if clean:
                    t = nltk.pos_tag([clean])
                    tagged.append(t[0])
                else:
                    tagged.append(('', 'OTHER'))

        # Build character ranges for each word in the NLTK tokenization
        word_char_ranges = []
        search_start = 0
        for word, tag in tagged:
            idx = full_text.find(word, search_start)
            if idx == -1:
                # Try case-insensitive or approximate match
                idx = search_start
            word_char_ranges.append((idx, idx + len(word), tag))
            search_start = max(search_start, idx + len(word))

        # For each BPE token, find which word it overlaps with most
        seq_labels = []
        for tok_start, tok_end in token_char_ranges:
            tok_mid = (tok_start + tok_end) / 2.0
            best_tag = 'OTHER'
            best_overlap = 0

            for w_start, w_end, w_tag in word_char_ranges:
                # Compute overlap
                overlap_start = max(tok_start, w_start)
                overlap_end = min(tok_end, w_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_tag = w_tag

            # If no overlap found, try direct tagging of the token
            if best_overlap == 0:
                clean = tokens_decoded[len(seq_labels)].strip() if len(seq_labels) < len(tokens_decoded) else ''
                if clean:
                    try:
                        direct_tag = nltk.pos_tag([clean])[0][1]
                        best_tag = direct_tag
                    except Exception:
                        pass

            pos_class = POS_MAP.get(best_tag, DEFAULT_POS)
            seq_labels.append(pos_class)

        all_labels.append(seq_labels)

    return np.array(all_labels)


def create_pos_probing_data(model, n_sequences=2500):
    """Create POS probing dataset."""
    from datasets import load_dataset

    tokenizer = model.tokenizer
    ds = load_dataset(DATASET_PATH, split="train", streaming=True)

    all_tokens = []
    count = 0
    for item in ds:
        text = item["text"]
        if len(text.strip()) < 50:
            continue
        tokens = tokenizer.encode(text, return_tensors="pt")[0]
        if len(tokens) >= CONTEXT_SIZE:
            all_tokens.append(tokens[:CONTEXT_SIZE])
            count += 1
            if count % 500 == 0:
                print(f"  Collected {count}/{n_sequences} sequences")
        if count >= n_sequences:
            break

    token_ids = torch.stack(all_tokens)

    # Get POS labels
    print(f"  Computing POS tags for {len(token_ids)} sequences...")
    batch_size = 100
    all_labels = []
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i:i+batch_size]
        labels = get_pos_labels_for_tokens(tokenizer, batch)
        all_labels.append(labels)
        if (i // batch_size) % 5 == 0:
            print(f"    POS tagged {min(i+batch_size, len(token_ids))}/{len(token_ids)} sequences")

    all_labels = np.concatenate(all_labels, axis=0)

    # Print class distribution
    flat_labels = all_labels.reshape(-1)
    for cls_id, cls_name in POS_NAMES.items():
        frac = (flat_labels == cls_id).mean()
        print(f"    Class {cls_id} ({cls_name}): {frac:.3f}")

    return token_ids, all_labels


def run_pos_probing(layer=6, ref_seed=42, k_values=[5, 10, 20, 50], n_split_seeds=5):
    """Run POS probing evaluation with multiple seeds."""

    output_dir = os.path.dirname(os.path.abspath(__file__))
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")

    t0 = time.time()

    # Load model
    print("Loading model...")
    model = load_model()

    # Load tier labels
    with open(os.path.join(matching_base, f"layer_{layer}", "tier_labels.json")) as f:
        tier_labels = json.load(f)

    # Load SAE weights
    print("Loading SAE weights...")
    sae_weights = load_sae_weights(layer, ref_seed)

    # Create probing data (reduced to avoid OOM)
    print("Creating POS probing dataset...")
    token_ids, pos_labels = create_pos_probing_data(model, n_sequences=2000)

    # Identify features per tier (do this early to only keep relevant columns)
    tier_indices_dict = {"consensus": [], "partial": [], "singleton": []}
    for i, tier in enumerate(tier_labels):
        if tier in tier_indices_dict:
            tier_indices_dict[tier].append(i)

    # Pre-select candidate feature indices (union of all tiers + random pool)
    # This saves massive memory: instead of storing all 16384 dims, store only ~4000
    all_candidate_indices = set()
    for tier_name, feat_indices in tier_indices_dict.items():
        # Keep top features by selecting all of them (we'll filter by variance later)
        all_candidate_indices.update(feat_indices)
    # Add some random features for the random baseline
    np.random.seed(999)
    random_pool = np.random.choice(DICT_SIZE, 500, replace=False)
    all_candidate_indices.update(random_pool.tolist())
    candidate_indices = sorted(all_candidate_indices)
    candidate_idx_array = np.array(candidate_indices)
    # Map from original index to position in reduced array
    idx_to_reduced = {orig: red for red, orig in enumerate(candidate_indices)}

    print(f"  Keeping {len(candidate_indices)} candidate features (out of {DICT_SIZE})")

    # Get SAE activations - only keep candidate columns to save memory
    print("Computing SAE activations...")
    all_acts_list = []
    batch_size = 16  # smaller batch to avoid GPU OOM
    with torch.no_grad():
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i+batch_size].to("cuda")
            _, cache = model.run_with_cache(batch, names_filter=[f"blocks.{layer}.hook_resid_post"])
            hidden = cache[f"blocks.{layer}.hook_resid_post"]
            flat_hidden = hidden.reshape(-1, HIDDEN_DIM)
            acts = compute_sae_activations(flat_hidden, sae_weights)
            # Only keep candidate columns, convert to float16 to save RAM
            acts_reduced = acts[:, candidate_idx_array].cpu().half()
            all_acts_list.append(acts_reduced)
            del acts, hidden, cache
            torch.cuda.empty_cache()
            if (i // batch_size) % 20 == 0:
                print(f"  Processed {i}/{len(token_ids)} sequences")

    all_acts_reduced = torch.cat(all_acts_list, dim=0).numpy().astype(np.float32)
    del all_acts_list
    flat_labels = pos_labels.reshape(-1)
    n_total = min(all_acts_reduced.shape[0], len(flat_labels))
    all_acts_reduced = all_acts_reduced[:n_total]
    flat_labels = flat_labels[:n_total]

    print(f"  Total tokens for probing: {n_total}")
    print(f"  Reduced activations shape: {all_acts_reduced.shape}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Map tier indices to reduced array positions
    tier_indices_reduced = {}
    for tier_name, orig_indices in tier_indices_dict.items():
        reduced = [idx_to_reduced[i] for i in orig_indices if i in idx_to_reduced]
        tier_indices_reduced[tier_name] = reduced

    # Random baseline: map from random_pool to reduced positions
    random_pool_reduced = [idx_to_reduced[i] for i in random_pool.tolist() if i in idx_to_reduced]

    print(f"\nTier sizes: consensus={len(tier_indices_reduced['consensus'])}, "
          f"partial={len(tier_indices_reduced['partial'])}, "
          f"singleton={len(tier_indices_reduced['singleton'])}")

    results = {}

    for k in k_values:
        print(f"\n--- k={k} ---")
        k_results = {}

        for tier_name in ["consensus", "partial", "singleton", "random"]:
            accs_over_seeds = []

            for split_seed in range(n_split_seeds):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    all_acts_reduced, flat_labels, test_size=0.2, random_state=split_seed, stratify=flat_labels
                )

                if tier_name == "random":
                    np.random.seed(split_seed + 1000)
                    if len(random_pool_reduced) >= k:
                        rand_feats = np.random.choice(random_pool_reduced, k, replace=False)
                    else:
                        rand_feats = np.random.choice(len(candidate_indices), k, replace=False)
                    X_sel_train = X_train[:, rand_feats]
                    X_sel_test = X_test[:, rand_feats]
                else:
                    feat_indices = np.array(tier_indices_reduced[tier_name])
                    if len(feat_indices) < k:
                        accs_over_seeds.append(0.25)  # chance for 4-class
                        continue

                    # Select top-k features by variance on training set
                    X_tier_train = X_train[:, feat_indices]
                    variances = X_tier_train.var(axis=0)
                    top_k_local = np.argsort(variances)[-k:]
                    selected_feats = feat_indices[top_k_local]

                    X_sel_train = X_train[:, selected_feats]
                    X_sel_test = X_test[:, selected_feats]

                # Train multi-class logistic regression probe
                clf = LogisticRegression(
                    max_iter=1000, C=1.0, solver="lbfgs",
                    random_state=split_seed
                )
                clf.fit(X_sel_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_sel_test))
                accs_over_seeds.append(acc)

            mean_acc = float(np.mean(accs_over_seeds))
            std_acc = float(np.std(accs_over_seeds))
            k_results[tier_name] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "accuracies": [float(a) for a in accs_over_seeds],
                "n_features_in_tier": len(tier_indices_dict.get(tier_name, [])) if tier_name != "random" else DICT_SIZE,
            }
            print(f"  {tier_name:12s}: {mean_acc:.4f} ± {std_acc:.4f}")

        # Welch's t-test: consensus vs singleton
        cons_accs = k_results["consensus"]["accuracies"]
        sing_accs = k_results["singleton"]["accuracies"]
        t_stat, p_value = stats.ttest_ind(cons_accs, sing_accs, equal_var=False)
        k_results["welch_ttest_consensus_vs_singleton"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
        }
        print(f"  Welch's t-test (consensus vs singleton): t={t_stat:.4f}, p={p_value:.6f}")

        results[f"k={k}"] = k_results

    # Summary
    elapsed = time.time() - t0
    output = {
        "task": "pos_tagging",
        "description": "4-class POS probing (noun/verb/adj-adv/other) - semantic-level linguistic task",
        "layer": layer,
        "ref_seed": ref_seed,
        "n_sequences": len(token_ids),
        "context_size": CONTEXT_SIZE,
        "n_total_tokens": n_total,
        "n_split_seeds": n_split_seeds,
        "k_values": k_values,
        "pos_classes": POS_NAMES,
        "class_distribution": {
            POS_NAMES[i]: float((flat_labels == i).mean()) for i in range(4)
        },
        "results": results,
        "elapsed_seconds": elapsed,
    }

    out_path = os.path.join(output_dir, "pos_probing_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")

    return output


if __name__ == "__main__":
    run_pos_probing()
