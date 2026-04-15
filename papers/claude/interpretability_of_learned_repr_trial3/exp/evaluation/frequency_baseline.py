"""Frequency baseline: Compare top-N features by firing frequency vs consensus features.

Tests whether consensus score provides signal beyond simple frequency selection.
For the TopK SAEs at layer 6, selects top-N features by firing frequency
(where N = number of consensus features) and compares their causal importance
against the consensus features.
"""

import sys
import os
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import (
    LAYERS, HIDDEN_DIM, CONSENSUS_HIGH, CONSENSUS_LOW, DICT_SIZE
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
MATCHING_DIR = os.path.join(EXP_DIR, "feature_matching")
EVAL_DIR = SCRIPT_DIR


def run_frequency_baseline(layer=6):
    """Run frequency baseline comparison for a given layer."""
    print(f"\n{'=' * 60}")
    print(f"Frequency Baseline Experiment - Layer {layer}")
    print(f"{'=' * 60}")

    # Load data
    consensus_scores = np.load(
        os.path.join(MATCHING_DIR, f"layer_{layer}", "consensus_scores.npy")
    )
    causal_importance = np.load(
        os.path.join(EVAL_DIR, f"layer_{layer}", "causal_importance.npy")
    )
    firing_rates = np.load(
        os.path.join(EVAL_DIR, f"layer_{layer}", "firing_rates.npy")
    )
    with open(os.path.join(MATCHING_DIR, f"layer_{layer}", "tier_labels.json")) as f:
        tier_labels = json.load(f)

    print(f"  Loaded {len(consensus_scores)} features")
    print(f"  Firing rate range: [{firing_rates.min():.6f}, {firing_rates.max():.6f}]")

    # Filter to features with meaningful causal importance (active features)
    active_mask = causal_importance > 0
    n_active = active_mask.sum()
    print(f"  Active features (CI > 0): {n_active}")

    # Identify consensus features
    consensus_mask = np.array([t == "consensus" for t in tier_labels])
    n_consensus = consensus_mask.sum()
    print(f"  Consensus features: {n_consensus}")

    # Identify singleton features
    singleton_mask = np.array([t == "singleton" for t in tier_labels])
    n_singleton = singleton_mask.sum()
    print(f"  Singleton features: {n_singleton}")

    # --- Frequency baseline: top-N by firing rate ---
    N = n_consensus  # match the number of consensus features
    sorted_by_freq = np.argsort(-firing_rates)  # descending
    top_freq_indices = sorted_by_freq[:N]
    top_freq_mask = np.zeros(len(firing_rates), dtype=bool)
    top_freq_mask[top_freq_indices] = True

    # Compute overlap between frequency-selected and consensus features
    overlap = (top_freq_mask & consensus_mask).sum()
    overlap_pct = overlap / N * 100
    print(f"\n  Top-{N} frequency features overlap with consensus: "
          f"{overlap}/{N} ({overlap_pct:.1f}%)")

    # --- Causal importance comparison ---
    consensus_ci = causal_importance[consensus_mask]
    singleton_ci = causal_importance[singleton_mask]
    freq_ci = causal_importance[top_freq_mask]

    # Also compute for frequency-only (top freq but NOT consensus)
    freq_only_mask = top_freq_mask & ~consensus_mask
    freq_only_ci = causal_importance[freq_only_mask]

    # And consensus-only (consensus but NOT in top freq)
    consensus_only_mask = consensus_mask & ~top_freq_mask
    consensus_only_ci = causal_importance[consensus_only_mask]

    print(f"\n  --- Causal Importance Statistics ---")
    for name, arr in [
        ("Consensus (all)", consensus_ci),
        ("Top-N frequency (all)", freq_ci),
        ("Frequency-only (not consensus)", freq_only_ci),
        ("Consensus-only (not freq)", consensus_only_ci),
        ("Singleton", singleton_ci),
    ]:
        if len(arr) > 0:
            print(f"  {name}: n={len(arr)}, "
                  f"mean={np.mean(arr):.6f}, median={np.median(arr):.6f}, "
                  f"std={np.std(arr):.6f}")
        else:
            print(f"  {name}: n=0")

    # --- Statistical tests ---
    print(f"\n  --- Statistical Tests ---")
    results_tests = {}

    # Test 1: Consensus vs Frequency (all)
    if len(consensus_ci) > 5 and len(freq_ci) > 5:
        u, p = stats.mannwhitneyu(consensus_ci, freq_ci, alternative="greater")
        pooled_std = np.sqrt((consensus_ci.std()**2 + freq_ci.std()**2) / 2)
        d = (consensus_ci.mean() - freq_ci.mean()) / (pooled_std + 1e-10)
        print(f"  Consensus vs Top-N Freq: U={u:.0f}, p={p:.2e}, Cohen's d={d:.4f}")
        results_tests["consensus_vs_freq"] = {
            "mann_whitney_u": float(u), "mann_whitney_p": float(p),
            "cohens_d": float(d)
        }

    # Test 2: Consensus vs Frequency-only (disjoint sets)
    if len(consensus_ci) > 5 and len(freq_only_ci) > 5:
        u, p = stats.mannwhitneyu(consensus_ci, freq_only_ci, alternative="greater")
        pooled_std = np.sqrt((consensus_ci.std()**2 + freq_only_ci.std()**2) / 2)
        d = (consensus_ci.mean() - freq_only_ci.mean()) / (pooled_std + 1e-10)
        print(f"  Consensus vs Freq-only: U={u:.0f}, p={p:.2e}, Cohen's d={d:.4f}")
        results_tests["consensus_vs_freq_only"] = {
            "mann_whitney_u": float(u), "mann_whitney_p": float(p),
            "cohens_d": float(d)
        }

    # Test 3: Consensus-only vs Frequency-only (strictly non-overlapping)
    if len(consensus_only_ci) > 5 and len(freq_only_ci) > 5:
        u, p = stats.mannwhitneyu(consensus_only_ci, freq_only_ci, alternative="greater")
        pooled_std = np.sqrt((consensus_only_ci.std()**2 + freq_only_ci.std()**2) / 2)
        d = (consensus_only_ci.mean() - freq_only_ci.mean()) / (pooled_std + 1e-10)
        print(f"  Consensus-only vs Freq-only: U={u:.0f}, p={p:.2e}, Cohen's d={d:.4f}")
        results_tests["consensus_only_vs_freq_only"] = {
            "mann_whitney_u": float(u), "mann_whitney_p": float(p),
            "cohens_d": float(d)
        }

    # Test 4: Top-N freq vs Singleton
    if len(freq_ci) > 5 and len(singleton_ci) > 5:
        u, p = stats.mannwhitneyu(freq_ci, singleton_ci, alternative="greater")
        pooled_std = np.sqrt((freq_ci.std()**2 + singleton_ci.std()**2) / 2)
        d = (freq_ci.mean() - singleton_ci.mean()) / (pooled_std + 1e-10)
        print(f"  Top-N Freq vs Singleton: U={u:.0f}, p={p:.2e}, Cohen's d={d:.4f}")
        results_tests["freq_vs_singleton"] = {
            "mann_whitney_u": float(u), "mann_whitney_p": float(p),
            "cohens_d": float(d)
        }

    # Test 5: Consensus vs Singleton (reference)
    if len(consensus_ci) > 5 and len(singleton_ci) > 5:
        u, p = stats.mannwhitneyu(consensus_ci, singleton_ci, alternative="greater")
        pooled_std = np.sqrt((consensus_ci.std()**2 + singleton_ci.std()**2) / 2)
        d = (consensus_ci.mean() - singleton_ci.mean()) / (pooled_std + 1e-10)
        print(f"  Consensus vs Singleton: U={u:.0f}, p={p:.2e}, Cohen's d={d:.4f}")
        results_tests["consensus_vs_singleton"] = {
            "mann_whitney_u": float(u), "mann_whitney_p": float(p),
            "cohens_d": float(d)
        }

    # --- Correlation analysis ---
    print(f"\n  --- Correlation Analysis ---")
    # Spearman: consensus score vs CI
    active_idx = causal_importance > 0
    r_consensus, p_consensus = stats.spearmanr(
        consensus_scores[active_idx], causal_importance[active_idx]
    )
    print(f"  Consensus score vs CI (Spearman): r={r_consensus:.4f}, p={p_consensus:.2e}")

    # Spearman: firing rate vs CI
    r_freq, p_freq = stats.spearmanr(
        firing_rates[active_idx], causal_importance[active_idx]
    )
    print(f"  Firing rate vs CI (Spearman): r={r_freq:.4f}, p={p_freq:.2e}")

    # Spearman: consensus score vs firing rate
    r_cs_fr, p_cs_fr = stats.spearmanr(
        consensus_scores[active_idx], firing_rates[active_idx]
    )
    print(f"  Consensus score vs Firing rate (Spearman): r={r_cs_fr:.4f}, p={p_cs_fr:.2e}")

    # Partial correlation: consensus vs CI controlling for firing rate
    from sklearn.linear_model import LinearRegression
    fr_active = firing_rates[active_idx].reshape(-1, 1)
    cs_active = consensus_scores[active_idx]
    ci_active = causal_importance[active_idx]

    resid_cs = cs_active - LinearRegression().fit(fr_active, cs_active).predict(fr_active)
    resid_ci = ci_active - LinearRegression().fit(fr_active, ci_active).predict(fr_active)
    r_partial, p_partial = stats.spearmanr(resid_cs, resid_ci)
    print(f"  Partial corr (consensus vs CI | firing rate): r={r_partial:.4f}, p={p_partial:.2e}")

    # --- Binned analysis: frequency deciles ---
    print(f"\n  --- Frequency Decile Analysis ---")
    active_fr = firing_rates[active_idx]
    active_cs = consensus_scores[active_idx]
    active_ci_arr = causal_importance[active_idx]

    n_bins = 10
    bin_edges = np.percentile(active_fr, np.linspace(0, 100, n_bins + 1))
    decile_results = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            in_bin = (active_fr >= lo) & (active_fr < hi)
        else:
            in_bin = (active_fr >= lo) & (active_fr <= hi)
        if in_bin.sum() < 10:
            continue
        bin_cs = active_cs[in_bin]
        bin_ci = active_ci_arr[in_bin]
        r_bin, p_bin = stats.spearmanr(bin_cs, bin_ci)
        decile_results.append({
            "decile": i + 1,
            "freq_range": [float(lo), float(hi)],
            "n_features": int(in_bin.sum()),
            "spearman_r": float(r_bin),
            "spearman_p": float(p_bin),
            "mean_ci": float(bin_ci.mean()),
        })
        print(f"    Decile {i+1} (freq {lo:.4f}-{hi:.4f}): n={in_bin.sum()}, "
              f"r={r_bin:.4f}, p={p_bin:.2e}")

    # --- Random baseline: N random features ---
    print(f"\n  --- Random Baseline (1000 trials) ---")
    n_trials = 1000
    random_means = []
    rng = np.random.RandomState(42)
    all_active_indices = np.where(active_idx)[0]
    for _ in range(n_trials):
        rand_idx = rng.choice(all_active_indices, size=N, replace=False)
        random_means.append(causal_importance[rand_idx].mean())
    random_means = np.array(random_means)

    # Where does consensus mean fall in the random distribution?
    consensus_mean = consensus_ci.mean()
    freq_mean = freq_ci.mean()
    pct_consensus = (random_means < consensus_mean).mean() * 100
    pct_freq = (random_means < freq_mean).mean() * 100

    print(f"  Random baseline: mean={random_means.mean():.6f}, "
          f"std={random_means.std():.6f}")
    print(f"  Consensus mean CI ({consensus_mean:.6f}) exceeds "
          f"{pct_consensus:.1f}% of random samples")
    print(f"  Top-freq mean CI ({freq_mean:.6f}) exceeds "
          f"{pct_freq:.1f}% of random samples")

    # --- Compile and save results ---
    results = {
        "layer": layer,
        "n_consensus_features": int(n_consensus),
        "n_singleton_features": int(n_singleton),
        "n_top_freq_features": int(N),
        "overlap_consensus_topfreq": int(overlap),
        "overlap_pct": float(overlap_pct),
        "causal_importance_stats": {
            "consensus": {
                "n": int(len(consensus_ci)),
                "mean": float(np.mean(consensus_ci)),
                "median": float(np.median(consensus_ci)),
                "std": float(np.std(consensus_ci)),
            },
            "top_freq": {
                "n": int(len(freq_ci)),
                "mean": float(np.mean(freq_ci)),
                "median": float(np.median(freq_ci)),
                "std": float(np.std(freq_ci)),
            },
            "freq_only": {
                "n": int(len(freq_only_ci)),
                "mean": float(np.mean(freq_only_ci)) if len(freq_only_ci) > 0 else 0,
                "median": float(np.median(freq_only_ci)) if len(freq_only_ci) > 0 else 0,
                "std": float(np.std(freq_only_ci)) if len(freq_only_ci) > 0 else 0,
            },
            "consensus_only": {
                "n": int(len(consensus_only_ci)),
                "mean": float(np.mean(consensus_only_ci)) if len(consensus_only_ci) > 0 else 0,
                "median": float(np.median(consensus_only_ci)) if len(consensus_only_ci) > 0 else 0,
                "std": float(np.std(consensus_only_ci)) if len(consensus_only_ci) > 0 else 0,
            },
            "singleton": {
                "n": int(len(singleton_ci)),
                "mean": float(np.mean(singleton_ci)),
                "median": float(np.median(singleton_ci)),
                "std": float(np.std(singleton_ci)),
            },
        },
        "statistical_tests": results_tests,
        "correlations": {
            "consensus_vs_ci_spearman": {"r": float(r_consensus), "p": float(p_consensus)},
            "firing_rate_vs_ci_spearman": {"r": float(r_freq), "p": float(p_freq)},
            "consensus_vs_firing_rate_spearman": {"r": float(r_cs_fr), "p": float(p_cs_fr)},
            "partial_consensus_vs_ci_controlling_freq": {"r": float(r_partial), "p": float(p_partial)},
        },
        "frequency_decile_analysis": decile_results,
        "random_baseline": {
            "n_trials": n_trials,
            "random_mean_ci": float(random_means.mean()),
            "random_std_ci": float(random_means.std()),
            "consensus_percentile": float(pct_consensus),
            "freq_percentile": float(pct_freq),
        },
    }

    output_path = os.path.join(SCRIPT_DIR, "frequency_baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return results


def run_all_layers():
    """Run frequency baseline for all layers with available data."""
    all_results = {}
    for layer in LAYERS:
        ci_path = os.path.join(EVAL_DIR, f"layer_{layer}", "causal_importance.npy")
        cs_path = os.path.join(MATCHING_DIR, f"layer_{layer}", "consensus_scores.npy")
        fr_path = os.path.join(EVAL_DIR, f"layer_{layer}", "firing_rates.npy")
        if os.path.exists(ci_path) and os.path.exists(cs_path) and os.path.exists(fr_path):
            all_results[layer] = run_frequency_baseline(layer)
        else:
            print(f"  Skipping layer {layer}: missing data")

    # Save combined results
    output_path = os.path.join(SCRIPT_DIR, "frequency_baseline_all_layers.json")
    serializable = {str(k): v for k, v in all_results.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAll-layer results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_all_layers()
