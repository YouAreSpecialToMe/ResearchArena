"""Ablation studies: number of seeds, dictionary size, architecture, threshold sensitivity."""

import sys
import os
import json
import time
import torch
import numpy as np
from itertools import combinations
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *

# Import matching functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feature_matching"))
from match_features import load_decoder_weights, build_consensus_graph, compute_per_feature_consensus


def ablation_n_seeds(layer=6, ref_seed=42, sae_base=None, causal_importance=None):
    """Test how many seeds are needed for reliable consensus scoring."""
    if sae_base is None:
        sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "sae_training", "topk")

    all_seeds = RANDOM_SEEDS
    decoders = load_decoder_weights(layer, all_seeds, sae_base)
    available_seeds = sorted(decoders.keys())

    if len(available_seeds) < 4:
        print("Not enough seeds for n_seeds ablation")
        return {}

    results = {}

    for n in range(3, len(available_seeds) + 1):
        n_results = []

        # Try multiple random subsets
        np.random.seed(42)
        if n == len(available_seeds):
            subsets = [available_seeds]
        else:
            # Generate 5 random subsets
            subsets = []
            for _ in range(5):
                subset = list(np.random.choice(available_seeds, n, replace=False))
                subsets.append(sorted(subset))

        for subset in subsets:
            sub_decoders = {s: decoders[s] for s in subset}
            feature_identities = build_consensus_graph(sub_decoders, layer)
            consensus_scores, tier_labels = compute_per_feature_consensus(
                feature_identities, ref_seed, DICT_SIZE
            )

            if causal_importance is not None:
                active_mask = causal_importance > 0
                if active_mask.sum() > 100:
                    spearman_r, spearman_p = stats.spearmanr(
                        consensus_scores[active_mask], causal_importance[active_mask]
                    )
                else:
                    spearman_r, spearman_p = 0, 1.0
            else:
                spearman_r, spearman_p = 0, 1.0

            tier_counts = {"consensus": 0, "partial": 0, "singleton": 0}
            for t in tier_labels:
                tier_counts[t] += 1

            n_results.append({
                "n_seeds": n,
                "seeds": subset,
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "tier_counts": tier_counts,
            })

        results[n] = {
            "mean_spearman_r": float(np.mean([r["spearman_r"] for r in n_results])),
            "std_spearman_r": float(np.std([r["spearman_r"] for r in n_results])),
            "details": n_results,
        }
        print(f"  N={n}: mean Spearman r = {results[n]['mean_spearman_r']:.3f} ± {results[n]['std_spearman_r']:.3f}")

    return results


def ablation_threshold(layer=6, ref_seed=42, causal_importance=None):
    """Test robustness to consensus threshold choice."""
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")

    consensus_scores = np.load(os.path.join(matching_base, f"layer_{layer}", "consensus_scores.npy"))

    thresholds = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    results = {}

    for thresh in thresholds:
        high_mask = consensus_scores >= thresh
        low_mask = consensus_scores <= 0.25  # Keep singleton definition constant

        n_high = int(high_mask.sum())

        if causal_importance is not None:
            high_ci = causal_importance[high_mask & (causal_importance > 0)]
            low_ci = causal_importance[low_mask & (causal_importance > 0)]

            if len(high_ci) > 5 and len(low_ci) > 5:
                pooled_std = np.sqrt((high_ci.std()**2 + low_ci.std()**2) / 2)
                cohens_d = (high_ci.mean() - low_ci.mean()) / (pooled_std + 1e-10)
                u_stat, p_val = stats.mannwhitneyu(high_ci, low_ci, alternative="greater")
            else:
                cohens_d, p_val = 0, 1.0
        else:
            cohens_d, p_val = 0, 1.0

        results[str(thresh)] = {
            "threshold": thresh,
            "n_consensus_features": n_high,
            "cohens_d": float(cohens_d),
            "p_value": float(p_val),
        }
        print(f"  Threshold {thresh:.3f}: n_consensus={n_high}, Cohen's d={cohens_d:.3f}")

    return results


def run_all_ablations():
    """Run all ablation studies."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    layer = 6

    # Try to load causal importance if available
    ci_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "evaluation", f"layer_{layer}", "causal_importance.npy")
    causal_importance = None
    if os.path.exists(ci_path):
        causal_importance = np.load(ci_path)

    results = {}

    print("=== Ablation: Number of Seeds ===")
    results["n_seeds"] = ablation_n_seeds(layer=layer, causal_importance=causal_importance)

    print("\n=== Ablation: Consensus Threshold ===")
    results["threshold"] = ablation_threshold(layer=layer, causal_importance=causal_importance)

    # Save results
    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll ablation studies complete!")
    return results


if __name__ == "__main__":
    run_all_ablations()
