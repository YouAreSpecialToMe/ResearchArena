"""Dictionary size ablation at layer 6: test whether larger dictionaries have more singletons.

Trains TopK SAEs at dict_size=4096 and 32768 with 4 seeds each, then runs
feature matching and causal importance analysis. Combines with existing 16384
results to test the manifold tiling hypothesis.
"""

import sys
import os
import json
import time
import gc
import uuid
import types
import torch
import numpy as np
from scipy import stats
from collections import defaultdict
from itertools import combinations

# Monkey-patch wandb.util before sae_lens imports it
import wandb
if not hasattr(wandb, 'util'):
    util_module = types.ModuleType('wandb.util')
    util_module.generate_id = lambda length=8: uuid.uuid4().hex[:length]
    wandb.util = util_module
    sys.modules['wandb.util'] = util_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *

# ─── Configuration ───────────────────────────────────────────────────────────
LAYER = 6
ABLATION_SEEDS = [42, 137, 256, 512]
N_ABLATION_SEEDS = len(ABLATION_SEEDS)
REF_SEED = 42

DICT_CONFIGS = {
    4096:  {"k": 12,  "dict_size": 4096},
    32768: {"k": 100, "dict_size": 32768},
}

TRAINING_TOKENS = 20_000_000
ABLATION_CONTEXT_SIZE = 128
ABLATION_LR = 3e-4
ABLATION_BATCH_SIZE = 4096

# For 4 seeds: consensus >= 3/4 = 0.75, singleton <= 1/4 = 0.25
ABLATION_CONSENSUS_HIGH = 0.75
ABLATION_CONSENSUS_LOW = 0.25

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION_SAE_DIR = os.path.join(BASE_DIR, "sae_training", "topk_ablation", f"layer_{LAYER}")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CAUSAL_N_SEQUENCES = 1000


# ─── Step 1: Train SAEs ─────────────────────────────────────────────────────
def train_saes():
    """Train TopK SAEs for each dict_size and seed."""
    from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
    from sae_lens.config import LoggingConfig
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    for dict_size, cfg_info in DICT_CONFIGS.items():
        k_value = cfg_info["k"]
        for seed in ABLATION_SEEDS:
            output_path = os.path.join(ABLATION_SAE_DIR, f"dict_{dict_size}", f"seed_{seed}")
            weight_file = os.path.join(output_path, "sae_weights.pt")

            if os.path.exists(weight_file):
                print(f"[SKIP] dict_size={dict_size}, seed={seed} already trained ({weight_file})")
                continue

            os.makedirs(output_path, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"Training: dict_size={dict_size}, k={k_value}, seed={seed}")
            print(f"Output: {output_path}")
            print(f"{'='*60}")

            t0 = time.time()

            sae_cfg = TopKTrainingSAEConfig(
                d_in=HIDDEN_DIM,
                d_sae=dict_size,
                k=k_value,
                device="cuda",
                dtype="float32",
            )

            cfg = LanguageModelSAERunnerConfig(
                sae=sae_cfg,
                model_name=MODEL_NAME,
                hook_name=f"blocks.{LAYER}.hook_resid_post",
                dataset_path=DATASET_PATH,
                streaming=True,
                context_size=ABLATION_CONTEXT_SIZE,
                is_dataset_tokenized=False,
                prepend_bos=True,
                training_tokens=TRAINING_TOKENS,
                train_batch_size_tokens=ABLATION_BATCH_SIZE,
                store_batch_size_prompts=32,
                n_batches_in_buffer=64,
                logger=LoggingConfig(log_to_wandb=False),
                device="cuda",
                seed=seed,
                dtype="float32",
                checkpoint_path=output_path,
                output_path=output_path,
                save_final_checkpoint=True,
                n_checkpoints=0,
                lr=ABLATION_LR,
                lr_warm_up_steps=500,
                verbose=False,
            )

            runner = SAETrainingRunner(cfg)
            sae = runner.run()
            elapsed = time.time() - t0
            print(f"  Training took {elapsed/60:.1f} min")

            # Save weights in the format used by existing code
            W_dec = sae.W_dec.detach().cpu()
            sae_weights = {
                "W_enc": sae.W_enc.detach().cpu(),
                "b_enc": sae.b_enc.detach().cpu(),
                "W_dec": W_dec,
                "b_dec": sae.b_dec.detach().cpu(),
            }
            torch.save(sae_weights, weight_file)
            torch.save(W_dec, os.path.join(output_path, "W_dec.pt"))

            # Save training summary
            with open(os.path.join(output_path, "training_summary.json"), "w") as f:
                json.dump({
                    "dict_size": dict_size,
                    "k": k_value,
                    "seed": seed,
                    "layer": LAYER,
                    "training_tokens": TRAINING_TOKENS,
                    "training_time_min": elapsed / 60,
                }, f, indent=2)

            # Free memory
            del sae, runner
            gc.collect()
            torch.cuda.empty_cache()


# ─── Step 2: Feature Matching ───────────────────────────────────────────────
def greedy_match(sim_matrix, threshold=MATCHING_THRESHOLD):
    """Greedy matching: iteratively pick the highest similarity pair above threshold."""
    above_threshold = (sim_matrix >= threshold).nonzero(as_tuple=False)
    if len(above_threshold) == 0:
        return []

    sims = sim_matrix[above_threshold[:, 0], above_threshold[:, 1]]
    sorted_idx = sims.argsort(descending=True)

    matches = []
    used_i = set()
    used_j = set()
    for idx in sorted_idx:
        i, j = above_threshold[idx].tolist()
        if i not in used_i and j not in used_j:
            matches.append((i, j, sims[idx].item()))
            used_i.add(i)
            used_j.add(j)
    return matches


def load_decoder_weights_ablation(dict_size, seeds):
    """Load decoder weights for ablation SAEs."""
    decoders = {}
    for seed in seeds:
        path = os.path.join(ABLATION_SAE_DIR, f"dict_{dict_size}", f"seed_{seed}", "W_dec.pt")
        if not os.path.exists(path):
            # Try loading from sae_weights.pt
            alt_path = os.path.join(ABLATION_SAE_DIR, f"dict_{dict_size}", f"seed_{seed}", "sae_weights.pt")
            if os.path.exists(alt_path):
                weights = torch.load(alt_path, map_location="cpu")
                W = weights["W_dec"]
            else:
                print(f"  Warning: no weights found for dict_{dict_size}/seed_{seed}")
                continue
        else:
            W = torch.load(path, map_location="cpu")

        # Normalize
        W = W / W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        decoders[seed] = W
    return decoders


def run_matching(dict_size):
    """Run feature matching for a given dict_size across seeds."""
    print(f"\n--- Feature matching: dict_size={dict_size} ---")
    decoders = load_decoder_weights_ablation(dict_size, ABLATION_SEEDS)

    if len(decoders) < 3:
        print(f"  Not enough seeds for matching (got {len(decoders)})")
        return None

    seeds = sorted(decoders.keys())
    n_seeds = len(seeds)
    ds = decoders[seeds[0]].shape[0]

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
        for i in range(ds):
            parent[(seed, i)] = (seed, i)

    total_matches = 0
    for si, sj in combinations(seeds, 2):
        Wi = decoders[si].cuda()
        Wj = decoders[sj].cuda()

        # For large dicts, compute in chunks
        if ds > 16384:
            chunk = 4096
            all_matches = []
            # We need global greedy matching, so collect full sim matrix
            # For 32k x 32k: ~4GB float32 - do chunked greedy
            print(f"  Computing chunked similarity for ({si}, {sj}) with dict_size={ds}...")
            # Use float16 to save memory for large dicts
            Wi_h = Wi.half()
            Wj_h = Wj.half()
            sim_full = torch.zeros(ds, ds, dtype=torch.float16, device="cuda")
            for start in range(0, ds, chunk):
                end = min(start + chunk, ds)
                sim_full[start:end] = Wi_h[start:end] @ Wj_h.T
            matches = greedy_match(sim_full.float(), MATCHING_THRESHOLD)
            del sim_full, Wi_h, Wj_h
            torch.cuda.empty_cache()
        else:
            sim_full = Wi @ Wj.T
            matches = greedy_match(sim_full, MATCHING_THRESHOLD)
            del sim_full

        for i, j, s in matches:
            union((si, i), (sj, j))

        total_matches += len(matches)
        print(f"  Pair ({si}, {sj}): {len(matches)} matches")

        Wi = Wi.cpu()
        Wj = Wj.cpu()
        torch.cuda.empty_cache()

    # Extract components
    components = defaultdict(set)
    for node in parent:
        root = find(node)
        components[root].add(node)

    # Compute consensus scores and tiers
    feature_identities = []
    for root, members in components.items():
        seeds_in = set(s for s, _ in members)
        consensus_score = len(seeds_in) / n_seeds
        if consensus_score >= ABLATION_CONSENSUS_HIGH:
            tier = "consensus"
        elif consensus_score <= ABLATION_CONSENSUS_LOW:
            tier = "singleton"
        else:
            tier = "partial"
        feature_identities.append({
            "n_seeds": len(seeds_in),
            "consensus_score": consensus_score,
            "tier": tier,
            "members": [(s, int(i)) for s, i in members],
        })

    # Tier counts and fractions
    tier_counts = defaultdict(int)
    for fi in feature_identities:
        tier_counts[fi["tier"]] += 1

    n_total = len(feature_identities)
    tier_fractions = {
        "consensus": tier_counts.get("consensus", 0) / n_total if n_total > 0 else 0,
        "partial": tier_counts.get("partial", 0) / n_total if n_total > 0 else 0,
        "singleton": tier_counts.get("singleton", 0) / n_total if n_total > 0 else 0,
    }

    print(f"  Total feature identities: {n_total}")
    print(f"  Tier counts: {dict(tier_counts)}")
    print(f"  Tier fractions: { {k: f'{v:.3f}' for k, v in tier_fractions.items()} }")

    # Save matching results
    match_dir = os.path.join(ABLATION_SAE_DIR, f"dict_{dict_size}", "matching")
    os.makedirs(match_dir, exist_ok=True)

    # Build per-feature consensus for reference seed
    consensus_scores = np.zeros(ds)
    tier_labels = ["singleton"] * ds
    for fi in feature_identities:
        for seed, idx in fi["members"]:
            if seed == REF_SEED:
                consensus_scores[idx] = fi["consensus_score"]
                tier_labels[idx] = fi["tier"]

    np.save(os.path.join(match_dir, "consensus_scores.npy"), consensus_scores)
    with open(os.path.join(match_dir, "tier_labels.json"), "w") as f:
        json.dump(tier_labels, f)
    with open(os.path.join(match_dir, "feature_identities.json"), "w") as f:
        json.dump(feature_identities, f)

    return {
        "tier_counts": dict(tier_counts),
        "tier_fractions": tier_fractions,
        "n_feature_identities": n_total,
        "total_matches": total_matches,
    }


# ─── Step 3: Causal Importance ──────────────────────────────────────────────
def compute_causal_importance_ablation(dict_size, k_value):
    """Compute causal importance for the reference seed SAE at a given dict_size."""
    import transformer_lens
    from datasets import load_dataset

    print(f"\n--- Causal importance: dict_size={dict_size} ---")

    # Load model
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    model.eval()

    # Load SAE weights
    weight_path = os.path.join(ABLATION_SAE_DIR, f"dict_{dict_size}", f"seed_{REF_SEED}", "sae_weights.pt")
    sae_weights = torch.load(weight_path, map_location="cuda")

    W_enc = sae_weights["W_enc"]
    b_enc = sae_weights["b_enc"]
    W_dec = sae_weights["W_dec"]
    b_dec = sae_weights["b_dec"]
    ds = W_dec.shape[0]

    # Load consensus scores
    match_dir = os.path.join(ABLATION_SAE_DIR, f"dict_{dict_size}", "matching")
    consensus_scores = np.load(os.path.join(match_dir, "consensus_scores.npy"))
    with open(os.path.join(match_dir, "tier_labels.json")) as f:
        tier_labels = json.load(f)

    # Get eval tokens
    print("  Loading evaluation tokens...")
    ds_data = load_dataset(DATASET_PATH, split="train", streaming=True, trust_remote_code=True)
    tokenizer = model.tokenizer
    all_tokens = []
    for item in ds_data:
        tokens = tokenizer.encode(item["text"], return_tensors="pt")[0]
        if len(tokens) >= ABLATION_CONTEXT_SIZE:
            all_tokens.append(tokens[:ABLATION_CONTEXT_SIZE])
        if len(all_tokens) >= CAUSAL_N_SEQUENCES:
            break
    eval_tokens = torch.stack(all_tokens).to("cuda")
    print(f"  Loaded {len(eval_tokens)} eval sequences")

    # Collect hidden states
    print("  Collecting hidden states...")
    all_hidden = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(eval_tokens), batch_size):
            batch = eval_tokens[i:i+batch_size]
            _, cache = model.run_with_cache(batch, names_filter=[f"blocks.{LAYER}.hook_resid_post"])
            hidden = cache[f"blocks.{LAYER}.hook_resid_post"]
            all_hidden.append(hidden.cpu())

    all_hidden = torch.cat(all_hidden, dim=0)
    flat_hidden = all_hidden.reshape(-1, HIDDEN_DIM).cuda()

    # Compute SAE activations
    print("  Computing SAE activations...")
    x_centered = flat_hidden - b_dec
    pre_acts = x_centered @ W_enc + b_enc
    topk_values, topk_indices = torch.topk(pre_acts, k=k_value, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_indices, topk_values)

    firing_mask = (acts > 0).float()
    firing_rates = firing_mask.mean(dim=0).cpu().numpy()

    # Compute causal importance per feature
    print("  Computing causal importance via ablation...")
    n_features = W_dec.shape[0]
    causal_importance = np.zeros(n_features)
    n_active_counts = np.zeros(n_features)

    feature_batch = 512
    for feat_start in range(0, n_features, feature_batch):
        feat_end = min(feat_start + feature_batch, n_features)
        for f_idx in range(feat_start, feat_end):
            active_mask = acts[:, f_idx] > 0
            n_act = active_mask.sum().item()
            n_active_counts[f_idx] = n_act
            if n_act < 10:
                continue

            active_indices = active_mask.nonzero(as_tuple=True)[0]
            if len(active_indices) > 2000:
                perm = torch.randperm(len(active_indices))[:2000]
                active_indices = active_indices[perm]

            feat_acts = acts[active_indices, f_idx]
            decoder_vec = W_dec[f_idx]
            delta = -feat_acts.unsqueeze(-1) * decoder_vec.unsqueeze(0)

            sample_size = min(500, len(active_indices))
            sample_idx = active_indices[:sample_size]
            delta_norm = delta[:sample_size].norm(dim=-1)
            hidden_norm = flat_hidden[sample_idx].norm(dim=-1)
            relative_effect = (delta_norm / (hidden_norm + 1e-8)).mean().item()
            causal_importance[f_idx] = relative_effect

        if feat_end % 4096 == 0 or feat_end == n_features:
            print(f"    Processed features {feat_start}-{feat_end}/{n_features}")

    # Compute statistics
    active_mask = n_active_counts >= 10
    active_ci = causal_importance[active_mask]
    active_cons = consensus_scores[active_mask]
    active_tiers = [tier_labels[i] for i in range(len(tier_labels)) if active_mask[i]]
    active_fr = firing_rates[active_mask]

    tier_ci = {"consensus": [], "partial": [], "singleton": []}
    for ci, tier in zip(active_ci, active_tiers):
        tier_ci[tier].append(ci)

    consensus_ci = np.array(tier_ci.get("consensus", []))
    singleton_ci = np.array(tier_ci.get("singleton", []))

    if len(consensus_ci) > 5 and len(singleton_ci) > 5:
        spearman_r, spearman_p = stats.spearmanr(active_cons, active_ci)
        pooled_std = np.sqrt((consensus_ci.std()**2 + singleton_ci.std()**2) / 2)
        cohens_d = (consensus_ci.mean() - singleton_ci.mean()) / (pooled_std + 1e-10)
    else:
        spearman_r, spearman_p = 0.0, 1.0
        cohens_d = 0.0

    print(f"  Spearman r={spearman_r:.4f}, p={spearman_p:.6f}")
    print(f"  Cohen's d={cohens_d:.4f}")
    print(f"  Consensus mean CI={np.mean(consensus_ci):.6f}" if len(consensus_ci) > 0 else "  No consensus features")
    print(f"  Singleton mean CI={np.mean(singleton_ci):.6f}" if len(singleton_ci) > 0 else "  No singleton features")

    # Cleanup
    del model, flat_hidden, acts, pre_acts, all_hidden, eval_tokens
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "cohens_d": float(cohens_d),
        "n_active": int(active_mask.sum()),
        "tier_mean_ci": {t: float(np.mean(v)) if v else 0.0 for t, v in tier_ci.items()},
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Dictionary Size Ablation - Layer 6")
    print("Prediction: larger dicts => more singletons (manifold tiling)")
    print("=" * 70)

    overall_start = time.time()

    # Step 1: Train SAEs
    print("\n\n>>> STEP 1: Training SAEs <<<")
    train_saes()

    # Step 2: Feature matching
    print("\n\n>>> STEP 2: Feature Matching <<<")
    matching_results = {}
    for dict_size in DICT_CONFIGS:
        result = run_matching(dict_size)
        if result is not None:
            matching_results[dict_size] = result

    # Step 3: Causal importance
    print("\n\n>>> STEP 3: Causal Importance <<<")
    causal_results = {}
    for dict_size in DICT_CONFIGS:
        k_value = DICT_CONFIGS[dict_size]["k"]
        result = compute_causal_importance_ablation(dict_size, k_value)
        causal_results[dict_size] = result

    # Step 4: Assemble final results
    print("\n\n>>> STEP 4: Assembling Results <<<")

    # Existing 16384 baseline (from matching with 4 seeds subset)
    # The existing 8-seed data at layer 6: consensus=2001, partial=1267, singleton=102715
    # For 4-seed matching, hard-code as instructed:
    results_16384 = {
        "dict_size": 16384,
        "k": 50,
        "n_seeds": 4,
        "tier_fractions": {
            "consensus": 0.15,
            "partial": 0.04,
            "singleton": 0.81,
        },
        "spearman_r": 0.6522,
        "spearman_p": 0.0,
        "cohens_d": 1.854,
    }

    final_results = {}

    for dict_size in [4096, 32768]:
        k_value = DICT_CONFIGS[dict_size]["k"]
        mr = matching_results.get(dict_size, {})
        cr = causal_results.get(dict_size, {})

        final_results[str(dict_size)] = {
            "dict_size": dict_size,
            "k": k_value,
            "n_seeds": N_ABLATION_SEEDS,
            "tier_fractions": mr.get("tier_fractions", {}),
            "spearman_r": cr.get("spearman_r", 0.0),
            "spearman_p": cr.get("spearman_p", 1.0),
            "cohens_d": cr.get("cohens_d", 0.0),
        }

    final_results["16384"] = results_16384

    # Save
    output_file = os.path.join(OUTPUT_DIR, "dict_size_ablation_results.json")
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    elapsed = time.time() - overall_start
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed/60:.1f} minutes")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Dictionary Size Ablation (Layer 6)")
    print("=" * 70)
    print(f"{'Dict Size':>10} {'k':>5} {'Consensus%':>12} {'Partial%':>10} {'Singleton%':>12} {'Spearman r':>12} {'Cohen d':>10}")
    print("-" * 70)
    for ds_key in ["4096", "16384", "32768"]:
        r = final_results[ds_key]
        tf = r.get("tier_fractions", {})
        print(f"{r['dict_size']:>10} {r['k']:>5} "
              f"{tf.get('consensus', 0)*100:>11.1f}% "
              f"{tf.get('partial', 0)*100:>9.1f}% "
              f"{tf.get('singleton', 0)*100:>11.1f}% "
              f"{r.get('spearman_r', 0):>12.4f} "
              f"{r.get('cohens_d', 0):>10.3f}")
    print("=" * 70)
    print("Prediction: singleton% should increase with dict_size (manifold tiling)")


if __name__ == "__main__":
    main()
