"""Cross-seed feature matching using greedy cosine similarity matching."""

import sys
import os
import json
import time
import torch
import numpy as np
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *


def load_decoder_weights(layer, seeds, base_dir):
    """Load and normalize decoder weight matrices for all seeds at a layer."""
    decoders = {}
    for seed in seeds:
        path = os.path.join(base_dir, f"layer_{layer}", f"seed_{seed}", "W_dec.pt")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping seed {seed}")
            continue
        W = torch.load(path, map_location="cpu")
        # Normalize to unit norm
        W = W / W.norm(dim=-1, keepdim=True)
        decoders[seed] = W
    return decoders


def greedy_match(sim_matrix, threshold=MATCHING_THRESHOLD):
    """Greedy matching: iteratively pick the highest similarity pair above threshold."""
    n, m = sim_matrix.shape
    matches = []
    used_i = set()
    used_j = set()

    # Get all similarities above threshold, sorted descending
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


def build_consensus_graph(decoders, layer, threshold=MATCHING_THRESHOLD):
    """Build feature identity graph from pairwise matching."""
    seeds = sorted(decoders.keys())
    n_seeds = len(seeds)
    dict_size = decoders[seeds[0]].shape[0]

    # Union-Find for connected components
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

    # Initialize: each (seed, feature_idx) is its own component
    for seed in seeds:
        for i in range(dict_size):
            node = (seed, i)
            parent[node] = node

    # Pairwise matching between all seed pairs
    pair_count = 0
    total_matches = 0
    for si, sj in combinations(seeds, 2):
        Wi = decoders[si]
        Wj = decoders[sj]

        # Compute cosine similarity in batches to save memory
        batch_size = 2048
        all_matches = []
        for start in range(0, dict_size, batch_size):
            end = min(start + batch_size, dict_size)
            sim = Wi[start:end] @ Wj.T  # (batch, dict_size)
            batch_matches = greedy_match(sim, threshold)
            # Adjust indices
            for i_local, j, s in batch_matches:
                all_matches.append((start + i_local, j, s))

        # Actually, greedy matching needs the full matrix for optimal results
        # Let's use a chunked approach that's still correct
        sim_full = Wi @ Wj.T  # (dict_size, dict_size) - fits in memory for 16k
        matches = greedy_match(sim_full, threshold)

        for i, j, s in matches:
            union((si, i), (sj, j))

        pair_count += 1
        total_matches += len(matches)
        print(f"  Pair ({si}, {sj}): {len(matches)} matches (layer {layer})")

    # Extract connected components
    components = defaultdict(set)
    for node in parent:
        root = find(node)
        components[root].add(node)

    # Compute consensus scores
    feature_identities = []
    for root, members in components.items():
        seeds_in_component = set(s for s, _ in members)
        consensus_score = len(seeds_in_component) / n_seeds
        feature_identities.append({
            "members": [(s, i) for s, i in members],
            "n_seeds": len(seeds_in_component),
            "consensus_score": consensus_score,
            "seeds": sorted(seeds_in_component),
        })

    # Tier assignment
    for fi in feature_identities:
        score = fi["consensus_score"]
        if score >= CONSENSUS_HIGH:
            fi["tier"] = "consensus"
        elif score <= CONSENSUS_LOW:
            fi["tier"] = "singleton"
        else:
            fi["tier"] = "partial"

    return feature_identities


def compute_per_feature_consensus(feature_identities, ref_seed, dict_size):
    """Map consensus scores back to features in the reference SAE."""
    consensus_scores = np.zeros(dict_size)
    tier_labels = ["singleton"] * dict_size

    for fi in feature_identities:
        for seed, idx in fi["members"]:
            if seed == ref_seed:
                consensus_scores[idx] = fi["consensus_score"]
                tier_labels[idx] = fi["tier"]

    return consensus_scores, tier_labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    parser.add_argument("--sae_base", type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            "sae_training", "topk"))
    parser.add_argument("--output_dir", type=str,
                       default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--ref_seed", type=int, default=42)
    args = parser.parse_args()

    all_results = {}

    for layer in args.layers:
        print(f"\n=== Layer {layer} ===")
        start = time.time()

        decoders = load_decoder_weights(layer, args.seeds, args.sae_base)
        if len(decoders) < 3:
            print(f"  Not enough seeds found for layer {layer}, skipping")
            continue

        feature_identities = build_consensus_graph(decoders, layer)
        elapsed = time.time() - start

        # Get per-feature consensus for reference seed
        dict_size = decoders[list(decoders.keys())[0]].shape[0]
        consensus_scores, tier_labels = compute_per_feature_consensus(
            feature_identities, args.ref_seed, dict_size
        )

        # Summary statistics
        tier_counts = defaultdict(int)
        for fi in feature_identities:
            tier_counts[fi["tier"]] += 1

        n_identities = len(feature_identities)
        scores = [fi["consensus_score"] for fi in feature_identities]

        summary = {
            "layer": layer,
            "n_seeds": len(decoders),
            "n_feature_identities": n_identities,
            "tier_counts": dict(tier_counts),
            "consensus_score_stats": {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
            },
            "matching_time_minutes": elapsed / 60,
        }

        print(f"  Feature identities: {n_identities}")
        print(f"  Tier counts: {dict(tier_counts)}")
        print(f"  Mean consensus: {np.mean(scores):.3f}")

        # Save results
        layer_dir = os.path.join(args.output_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

        # Save consensus scores for reference seed
        np.save(os.path.join(layer_dir, "consensus_scores.npy"), consensus_scores)
        with open(os.path.join(layer_dir, "tier_labels.json"), "w") as f:
            json.dump(tier_labels, f)

        # Save summary
        with open(os.path.join(layer_dir, "matching_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Save feature identities (compact form)
        compact_identities = []
        for fi in feature_identities:
            compact_identities.append({
                "n_seeds": fi["n_seeds"],
                "consensus_score": fi["consensus_score"],
                "tier": fi["tier"],
                "members": [(s, int(i)) for s, i in fi["members"]],
            })
        with open(os.path.join(layer_dir, "feature_identities.json"), "w") as f:
            json.dump(compact_identities, f)

        all_results[layer] = summary

    # Save aggregate results
    with open(os.path.join(args.output_dir, "matching_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nFeature matching complete!")


if __name__ == "__main__":
    main()
