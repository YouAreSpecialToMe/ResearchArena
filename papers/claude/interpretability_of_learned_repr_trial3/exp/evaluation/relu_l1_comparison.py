"""ReLU+L1 SAE comparison: Train 5 SAEs, match features, evaluate causal importance."""

import sys
import os
import json
import time
import uuid
import types

# Monkey-patch wandb.util before sae_lens imports it
import wandb
if not hasattr(wandb, 'util'):
    util_module = types.ModuleType('wandb.util')
    util_module.generate_id = lambda length=8: uuid.uuid4().hex[:length]
    wandb.util = util_module
    sys.modules['wandb.util'] = util_module

import torch
import numpy as np
from scipy import stats
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import MODEL_NAME, HIDDEN_DIM, DATASET_PATH, CONTEXT_SIZE, LR, BATCH_SIZE

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from sae_lens.config import LoggingConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig

# ---- Constants for this experiment ----
LAYER = 6
ALL_SEEDS = [42, 137, 256, 512, 1024]
# Auto-detect which seeds are actually trained
SEEDS = [s for s in ALL_SEEDS if os.path.exists(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "sae_training", "relu_l1", f"layer_6", f"seed_{s}", "W_dec.pt"))]
N_SEEDS = len(SEEDS)
DICT_SIZE = 16384
L1_COEFF = 8e-3
N_TRAINING_TOKENS = 20_000_000
MATCHING_THRESHOLD = 0.7
# For 3 seeds: consensus = 3/3 = 1.0, singleton = 1/3 = 0.33
# For 5 seeds: consensus >= 3/5 = 0.6, singleton <= 2/5 = 0.4
CONSENSUS_HIGH = 0.6
CONSENSUS_LOW = 0.4
EVAL_N_SEQUENCES = 2000
EVAL_BATCH_SIZE = 32
REF_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
SAE_BASE = os.path.join(EXP_DIR, "sae_training", "relu_l1")


# ============================================================
# STAGE 1: Train 5 ReLU+L1 SAEs
# ============================================================
def train_relu_saes():
    """Train 5 ReLU+L1 SAEs at layer 6."""
    print("\n" + "=" * 60)
    print("STAGE 1: Training 5 ReLU+L1 SAEs at layer 6")
    print("=" * 60)

    summaries = []
    for seed in SEEDS:
        output_dir = os.path.join(SAE_BASE, f"layer_{LAYER}", f"seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)

        # Skip if already trained
        if os.path.exists(os.path.join(output_dir, "W_dec.pt")):
            print(f"  Seed {seed}: already trained, skipping")
            summary_path = os.path.join(output_dir, "training_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summaries.append(json.load(f))
            continue

        print(f"\n  Training seed {seed}...")
        hook_point = f"blocks.{LAYER}.hook_resid_post"

        sae_cfg = StandardTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=DICT_SIZE,
            l1_coefficient=L1_COEFF,
            device="cuda",
            dtype="float32",
        )

        cfg = LanguageModelSAERunnerConfig(
            sae=sae_cfg,
            model_name=MODEL_NAME,
            hook_name=hook_point,
            dataset_path=DATASET_PATH,
            streaming=True,
            context_size=CONTEXT_SIZE,
            is_dataset_tokenized=False,
            prepend_bos=True,
            training_tokens=N_TRAINING_TOKENS,
            train_batch_size_tokens=BATCH_SIZE,
            store_batch_size_prompts=32,
            n_batches_in_buffer=32,
            logger=LoggingConfig(log_to_wandb=False),
            device="cuda",
            seed=seed,
            dtype="float32",
            checkpoint_path=output_dir,
            lr=LR,
            lr_warm_up_steps=500,
            output_path=output_dir,
            save_final_checkpoint=True,
            n_checkpoints=0,
            verbose=False,
        )

        start_time = time.time()
        sae = SAETrainingRunner(cfg).run()
        elapsed = time.time() - start_time

        # Save weights
        W_dec = sae.W_dec.detach().cpu()
        torch.save(W_dec, os.path.join(output_dir, "W_dec.pt"))

        state = {
            "W_enc": sae.W_enc.detach().cpu(),
            "W_dec": W_dec,
            "b_enc": sae.b_enc.detach().cpu(),
            "b_dec": sae.b_dec.detach().cpu(),
        }
        torch.save(state, os.path.join(output_dir, "sae_weights.pt"))

        summary = {
            "layer": LAYER,
            "seed": seed,
            "dict_size": DICT_SIZE,
            "architecture": "relu_l1",
            "l1_coefficient": L1_COEFF,
            "training_time_minutes": elapsed / 60,
        }
        with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        summaries.append(summary)
        print(f"  Seed {seed}: {elapsed/60:.1f} min")
        torch.cuda.empty_cache()

    return summaries


# ============================================================
# STAGE 2: Greedy feature matching across 5 seeds
# ============================================================
def greedy_match(sim_matrix, threshold=MATCHING_THRESHOLD):
    """Greedy matching: iteratively pick the highest similarity pair above threshold."""
    n, m = sim_matrix.shape
    matches = []
    used_i = set()
    used_j = set()

    above_threshold = (sim_matrix >= threshold).nonzero(as_tuple=False)
    if len(above_threshold) == 0:
        return matches

    sims = sim_matrix[above_threshold[:, 0], above_threshold[:, 1]]
    sorted_idx = sims.argsort(descending=True)

    for idx in sorted_idx:
        i, j = above_threshold[idx].tolist()
        if i not in used_i and j not in used_j:
            matches.append((i, j, sims[idx].item()))
            used_i.add(i)
            used_j.add(j)

    return matches


def run_feature_matching():
    """Run greedy feature matching across all 5 seeds."""
    print("\n" + "=" * 60)
    print("STAGE 2: Feature matching across 5 seeds")
    print("=" * 60)

    # Load decoder weights
    decoders = {}
    for seed in SEEDS:
        path = os.path.join(SAE_BASE, f"layer_{LAYER}", f"seed_{seed}", "W_dec.pt")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found")
            continue
        W = torch.load(path, map_location="cpu")
        W = W / W.norm(dim=-1, keepdim=True)
        decoders[seed] = W
        print(f"  Loaded decoder for seed {seed}: shape {W.shape}")

    if len(decoders) < 3:
        raise RuntimeError(f"Only {len(decoders)} seeds found, need at least 3")

    seeds = sorted(decoders.keys())
    dict_size = decoders[seeds[0]].shape[0]

    # Union-Find
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for seed in seeds:
        for i in range(dict_size):
            parent[(seed, i)] = (seed, i)

    # Pairwise matching
    total_matches = 0
    for si, sj in combinations(seeds, 2):
        Wi = decoders[si]
        Wj = decoders[sj]
        sim_full = Wi @ Wj.T
        matches = greedy_match(sim_full, MATCHING_THRESHOLD)
        for i, j, s in matches:
            union((si, i), (sj, j))
        total_matches += len(matches)
        print(f"  Pair ({si}, {sj}): {len(matches)} matches")

    # Extract components
    components = defaultdict(set)
    for node in parent:
        root = find(node)
        components[root].add(node)

    # Compute consensus scores and tier assignments
    feature_identities = []
    for root, members in components.items():
        seeds_in = set(s for s, _ in members)
        score = len(seeds_in) / N_SEEDS
        if score >= CONSENSUS_HIGH:
            tier = "consensus"
        elif score <= CONSENSUS_LOW:
            tier = "singleton"
        else:
            tier = "partial"
        feature_identities.append({
            "n_seeds": len(seeds_in),
            "consensus_score": score,
            "tier": tier,
            "members": [(s, int(i)) for s, i in members],
        })

    # Map to reference seed
    consensus_scores = np.zeros(dict_size)
    tier_labels = ["singleton"] * dict_size
    for fi in feature_identities:
        for seed, idx in fi["members"]:
            if seed == REF_SEED:
                consensus_scores[idx] = fi["consensus_score"]
                tier_labels[idx] = fi["tier"]

    # Save matching results
    match_dir = os.path.join(SCRIPT_DIR, "relu_l1_matching")
    os.makedirs(match_dir, exist_ok=True)
    np.save(os.path.join(match_dir, "consensus_scores.npy"), consensus_scores)
    with open(os.path.join(match_dir, "tier_labels.json"), "w") as f:
        json.dump(tier_labels, f)

    tier_counts = defaultdict(int)
    for fi in feature_identities:
        tier_counts[fi["tier"]] += 1

    print(f"\n  Total feature identities: {len(feature_identities)}")
    print(f"  Tier counts: {dict(tier_counts)}")
    print(f"  Total pairwise matches: {total_matches}")

    return consensus_scores, tier_labels, dict(tier_counts), feature_identities


# ============================================================
# STAGE 3: Causal importance evaluation
# ============================================================
def evaluate_causal_importance(consensus_scores, tier_labels):
    """Evaluate causal importance via feature ablation on held-out sequences."""
    print("\n" + "=" * 60)
    print("STAGE 3: Causal importance evaluation")
    print("=" * 60)

    import transformer_lens
    from datasets import load_dataset

    # Load model
    print("  Loading model...")
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    model.eval()

    # Load SAE weights for reference seed
    sae_path = os.path.join(SAE_BASE, f"layer_{LAYER}", f"seed_{REF_SEED}", "sae_weights.pt")
    sae_weights = torch.load(sae_path, map_location="cuda")
    W_enc = sae_weights["W_enc"]
    W_dec = sae_weights["W_dec"]
    b_enc = sae_weights["b_enc"]
    b_dec = sae_weights["b_dec"]

    # Get eval tokens
    print(f"  Loading {EVAL_N_SEQUENCES} eval sequences...")
    ds = load_dataset(DATASET_PATH, split="train", streaming=True, trust_remote_code=True)
    tokenizer = model.tokenizer
    all_tokens = []
    for item in ds:
        tokens = tokenizer.encode(item["text"], return_tensors="pt")[0]
        if len(tokens) >= CONTEXT_SIZE:
            all_tokens.append(tokens[:CONTEXT_SIZE])
        if len(all_tokens) >= EVAL_N_SEQUENCES:
            break
    eval_tokens = torch.stack(all_tokens).to("cuda")
    print(f"  Loaded {len(eval_tokens)} eval sequences")

    # Collect hidden states
    print("  Collecting hidden states...")
    all_hidden = []
    with torch.no_grad():
        for i in range(0, len(eval_tokens), EVAL_BATCH_SIZE):
            batch = eval_tokens[i:i + EVAL_BATCH_SIZE]
            _, cache = model.run_with_cache(
                batch, names_filter=[f"blocks.{LAYER}.hook_resid_post"]
            )
            hidden = cache[f"blocks.{LAYER}.hook_resid_post"]
            all_hidden.append(hidden.cpu())

    all_hidden = torch.cat(all_hidden, dim=0)  # (n_seq, seq_len, hidden_dim)
    flat_hidden = all_hidden.reshape(-1, HIDDEN_DIM)  # keep on CPU
    n_tokens = flat_hidden.shape[0]
    print(f"  Hidden states shape: {flat_hidden.shape}")

    # Compute SAE activations in chunks to avoid OOM
    print("  Computing SAE activations (ReLU) in chunks...")
    dict_size = W_dec.shape[0]
    chunk_size = 4096  # process 4k tokens at a time

    # Accumulators
    firing_rate_sum = torch.zeros(dict_size)
    total_l0 = 0.0
    # For causal importance: we compute vectorized relative effect per chunk
    # relative_effect[f] = mean over active tokens of (|act_f * W_dec[f]| / |hidden|)
    # = mean over active tokens of (act_f * |W_dec[f]| / |hidden|)
    # Since decoder norms are constant, we precompute them
    decoder_norms = W_dec.norm(dim=-1)  # (dict_size,)

    causal_weighted_sum = torch.zeros(dict_size)  # sum of act_f * decoder_norm / hidden_norm
    n_active_total = torch.zeros(dict_size)

    for chunk_start in range(0, n_tokens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_tokens)
        hidden_chunk = flat_hidden[chunk_start:chunk_end].cuda()

        x_centered = hidden_chunk - b_dec
        pre_acts = x_centered @ W_enc + b_enc
        acts_chunk = torch.relu(pre_acts)  # (chunk, dict_size)

        # Firing stats
        active_mask = (acts_chunk > 0)
        firing_rate_sum += active_mask.float().sum(dim=0).cpu()
        total_l0 += active_mask.float().sum(dim=-1).sum().item()
        n_active_total += active_mask.float().sum(dim=0).cpu()

        # Vectorized causal importance:
        # For each token, relative_effect_f = act_f * ||W_dec[f]|| / ||hidden||
        hidden_norms = hidden_chunk.norm(dim=-1, keepdim=True)  # (chunk, 1)
        # acts_chunk * decoder_norms gives the delta norm for each feature
        # Divide by hidden norm to get relative effect
        rel_effects = (acts_chunk * decoder_norms.unsqueeze(0)) / (hidden_norms + 1e-8)  # (chunk, dict_size)
        # Only count active positions
        rel_effects = rel_effects * active_mask.float()
        causal_weighted_sum += rel_effects.sum(dim=0).cpu()

        del hidden_chunk, x_centered, pre_acts, acts_chunk, active_mask, rel_effects, hidden_norms
        torch.cuda.empty_cache()

        if (chunk_end) % (chunk_size * 8) == 0 or chunk_end == n_tokens:
            print(f"    Processed tokens {chunk_start}-{chunk_end}/{n_tokens}")

    # Finalize
    firing_rates = (firing_rate_sum / n_tokens).numpy()
    l0 = total_l0 / n_tokens
    n_active = n_active_total.numpy()

    # Compute mean causal importance per feature
    causal_importance = np.zeros(dict_size)
    valid = n_active >= 10
    causal_importance[valid] = causal_weighted_sum.numpy()[valid] / n_active[valid]

    print(f"  Average L0 (active features per token): {l0:.1f}")

    # Clean up
    del flat_hidden, all_hidden
    torch.cuda.empty_cache()

    return causal_importance, firing_rates, n_active, l0


# ============================================================
# STAGE 4: Statistical analysis and save results
# ============================================================
def analyze_and_save(consensus_scores, tier_labels, tier_counts,
                     causal_importance, firing_rates, n_active, l0):
    """Compute statistics and save results."""
    print("\n" + "=" * 60)
    print("STAGE 4: Statistical analysis")
    print("=" * 60)

    # Filter to features that actually fire
    active_mask = n_active >= 10
    active_ci = causal_importance[active_mask]
    active_consensus = consensus_scores[active_mask]
    active_tiers = [tier_labels[i] for i in range(len(tier_labels)) if active_mask[i]]
    active_firing = firing_rates[active_mask]

    print(f"  Active features (>=10 firings): {active_mask.sum()}")

    # Group by tier
    tier_ci = {"consensus": [], "partial": [], "singleton": []}
    for ci, tier in zip(active_ci, active_tiers):
        tier_ci[tier].append(ci)

    for t in tier_ci:
        arr = tier_ci[t]
        if arr:
            print(f"  {t}: n={len(arr)}, mean_CI={np.mean(arr):.6f}, "
                  f"median_CI={np.median(arr):.6f}")

    # Statistical tests
    consensus_ci = np.array(tier_ci.get("consensus", []))
    singleton_ci = np.array(tier_ci.get("singleton", []))

    if len(consensus_ci) > 5 and len(singleton_ci) > 5:
        u_stat, mw_p = stats.mannwhitneyu(consensus_ci, singleton_ci, alternative="greater")
        pooled_std = np.sqrt((consensus_ci.std() ** 2 + singleton_ci.std() ** 2) / 2)
        cohens_d = (consensus_ci.mean() - singleton_ci.mean()) / (pooled_std + 1e-10)
        spearman_r, spearman_p = stats.spearmanr(active_consensus, active_ci)
    else:
        u_stat, mw_p = 0, 1.0
        cohens_d = 0
        spearman_r, spearman_p = 0, 1.0

    print(f"\n  Spearman r = {spearman_r:.4f}, p = {spearman_p:.2e}")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print(f"  Mann-Whitney p = {mw_p:.2e}")

    # Consensus score distribution
    score_dist = {}
    for s in sorted(set(consensus_scores)):
        count = int((consensus_scores == s).sum())
        score_dist[f"{s:.2f}"] = count

    results = {
        "layer": LAYER,
        "architecture": "relu_l1",
        "l1_coefficient": L1_COEFF,
        "n_seeds": N_SEEDS,
        "dict_size": DICT_SIZE,
        "average_l0": round(l0, 1),
        "n_active_features": int(active_mask.sum()),
        "tier_counts": tier_counts,
        "tier_mean_causal_importance": {
            t: float(np.mean(v)) if v else 0.0 for t, v in tier_ci.items()
        },
        "tier_median_causal_importance": {
            t: float(np.median(v)) if v else 0.0 for t, v in tier_ci.items()
        },
        "tier_std_causal_importance": {
            t: float(np.std(v)) if v else 0.0 for t, v in tier_ci.items()
        },
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "cohens_d": float(cohens_d),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(mw_p),
        "consensus_score_distribution": score_dist,
    }

    output_path = os.path.join(SCRIPT_DIR, "relu_l1_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # Also save per-feature arrays
    match_dir = os.path.join(SCRIPT_DIR, "relu_l1_matching")
    os.makedirs(match_dir, exist_ok=True)
    np.save(os.path.join(match_dir, "causal_importance.npy"), causal_importance)
    np.save(os.path.join(match_dir, "firing_rates.npy"), firing_rates)

    return results


# ============================================================
# Main
# ============================================================
def main():
    total_start = time.time()

    # Stage 1: Train SAEs
    train_summaries = train_relu_saes()

    # Stage 2: Feature matching
    consensus_scores, tier_labels, tier_counts, feature_identities = run_feature_matching()

    # Stage 3: Causal importance
    causal_importance, firing_rates, n_active, l0 = evaluate_causal_importance(
        consensus_scores, tier_labels
    )

    # Stage 4: Analyze and save
    results = analyze_and_save(
        consensus_scores, tier_labels, tier_counts,
        causal_importance, firing_rates, n_active, l0
    )

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {total_elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")

    # Print summary
    print(f"\nArchitecture: ReLU+L1 (l1_coeff={L1_COEFF})")
    print(f"Average L0: {results['average_l0']}")
    print(f"Tier counts: {results['tier_counts']}")
    print(f"Spearman r: {results['spearman_r']:.4f} (p={results['spearman_p']:.2e})")
    print(f"Cohen's d: {results['cohens_d']:.4f}")
    print(f"Mann-Whitney p: {results['mann_whitney_p']:.2e}")
    print(f"Consensus mean CI: {results['tier_mean_causal_importance'].get('consensus', 0):.6f}")
    print(f"Singleton mean CI: {results['tier_mean_causal_importance'].get('singleton', 0):.6f}")


if __name__ == "__main__":
    main()
