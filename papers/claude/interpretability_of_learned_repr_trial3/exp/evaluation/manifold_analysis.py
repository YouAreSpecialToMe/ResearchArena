"""Manifold tiling analysis: investigate whether singleton features show tiling signatures."""

import sys
import os
import json
import time
import torch
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *


def run_manifold_analysis(layers=LAYERS, ref_seed=42):
    """Analyze manifold tiling signatures of singleton vs consensus features."""
    sae_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sae_training", "topk")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    all_results = {}

    for layer in layers:
        print(f"\n=== Layer {layer}: Manifold Analysis ===")

        # Load decoder weights
        W_dec_path = os.path.join(sae_base, f"layer_{layer}", f"seed_{ref_seed}", "W_dec.pt")
        if not os.path.exists(W_dec_path):
            print(f"  No decoder weights for layer {layer}")
            continue
        W_dec = torch.load(W_dec_path, map_location="cpu")
        W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)

        # Load consensus data
        consensus_path = os.path.join(matching_base, f"layer_{layer}", "consensus_scores.npy")
        if not os.path.exists(consensus_path):
            print(f"  No consensus scores for layer {layer}")
            continue
        consensus_scores = np.load(consensus_path)
        with open(os.path.join(matching_base, f"layer_{layer}", "tier_labels.json")) as f:
            tier_labels = json.load(f)

        tier_indices = {"consensus": [], "partial": [], "singleton": []}
        for i, t in enumerate(tier_labels):
            tier_indices[t].append(i)

        # 1. Local density: count neighbors within cosine similarity > 0.5
        print("  Computing local density...")
        n_features = W_dec.shape[0]
        neighborhood_sizes = np.zeros(n_features)

        # Process in chunks to avoid memory issues
        chunk_size = 1024
        for start in range(0, n_features, chunk_size):
            end = min(start + chunk_size, n_features)
            sim = W_dec[start:end] @ W_dec.T  # (chunk, n_features)
            # Don't count self-similarity
            for i in range(end - start):
                sim[i, start + i] = 0
            neighborhood_sizes[start:end] = (sim > 0.5).sum(dim=-1).numpy()

        # Compare by tier
        tier_density = {}
        for tier, indices in tier_indices.items():
            if indices:
                densities = neighborhood_sizes[indices]
                tier_density[tier] = {
                    "mean": float(np.mean(densities)),
                    "median": float(np.median(densities)),
                    "std": float(np.std(densities)),
                }

        # Statistical test
        if len(tier_indices["consensus"]) > 10 and len(tier_indices["singleton"]) > 10:
            cons_density = neighborhood_sizes[tier_indices["consensus"]]
            sing_density = neighborhood_sizes[tier_indices["singleton"]]
            u_density, p_density = stats.mannwhitneyu(sing_density, cons_density, alternative="greater")
        else:
            u_density, p_density = 0, 1.0

        # 2. Activation correlation (if activations available)
        acts_path = os.path.join(output_dir, f"layer_{layer}", "firing_rates.npy")
        activation_corr_results = {}

        # 3. DBSCAN clustering
        print("  Running DBSCAN clustering...")
        # Use cosine distance = 1 - cosine_similarity
        # For efficiency, sample features
        n_sample = min(5000, n_features)
        np.random.seed(42)
        sample_idx = np.random.choice(n_features, n_sample, replace=False)
        sample_W = W_dec[sample_idx].numpy()
        sample_tiers = [tier_labels[i] for i in sample_idx]

        # Compute distance matrix
        cos_sim = sample_W @ sample_W.T
        cos_dist = 1 - cos_sim

        clustering = DBSCAN(eps=0.4, min_samples=3, metric="precomputed").fit(cos_dist)
        labels = clustering.labels_

        # Analyze cluster composition
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        clustered_mask = labels != -1

        tier_in_cluster = {"consensus": 0, "partial": 0, "singleton": 0}
        tier_total = {"consensus": 0, "partial": 0, "singleton": 0}
        for i, tier in enumerate(sample_tiers):
            tier_total[tier] += 1
            if clustered_mask[i]:
                tier_in_cluster[tier] += 1

        tier_cluster_frac = {}
        for tier in ["consensus", "partial", "singleton"]:
            if tier_total[tier] > 0:
                tier_cluster_frac[tier] = tier_in_cluster[tier] / tier_total[tier]
            else:
                tier_cluster_frac[tier] = 0

        # 4. UMAP visualization data (save coordinates for later plotting)
        print("  Computing UMAP projection...")
        try:
            import umap
            reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=15)
            umap_coords = reducer.fit_transform(sample_W)

            np.save(os.path.join(output_dir, f"layer_{layer}", "umap_coords.npy"), umap_coords)
            np.save(os.path.join(output_dir, f"layer_{layer}", "umap_consensus_scores.npy"),
                    consensus_scores[sample_idx])
            np.save(os.path.join(output_dir, f"layer_{layer}", "umap_sample_idx.npy"), sample_idx)
        except ImportError:
            print("  UMAP not available, skipping visualization")
            umap_coords = None

        result = {
            "layer": layer,
            "n_features": n_features,
            "tier_density": tier_density,
            "density_mannwhitney_p": float(p_density),
            "density_hypothesis": "singletons have denser neighborhoods",
            "n_dbscan_clusters": n_clusters,
            "tier_cluster_fraction": tier_cluster_frac,
            "cluster_hypothesis": "singletons are more likely to be in dense clusters",
            "tier_counts": {t: len(v) for t, v in tier_indices.items()},
        }

        all_results[layer] = result
        print(f"  Density - Consensus: {tier_density.get('consensus', {}).get('mean', 'N/A'):.1f}, "
              f"Singleton: {tier_density.get('singleton', {}).get('mean', 'N/A'):.1f}")
        print(f"  Cluster fraction - Consensus: {tier_cluster_frac.get('consensus', 0):.3f}, "
              f"Singleton: {tier_cluster_frac.get('singleton', 0):.3f}")

    # Save results
    with open(os.path.join(output_dir, "manifold_analysis_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nManifold analysis complete!")
    return all_results


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer_2"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer_6"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer_10"), exist_ok=True)
    run_manifold_analysis()
