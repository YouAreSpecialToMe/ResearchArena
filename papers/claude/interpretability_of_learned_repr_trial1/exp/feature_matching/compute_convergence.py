"""Compute seed stability and convergence scores for all models."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from tqdm import tqdm

DEVICE = "cuda"
SAE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sae_training')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

MODELS = {
    "gpt2_small": {"d_model": 768},
    "pythia_160m": {"d_model": 768},
    "pythia_410m": {"d_model": 1024},
}
SEEDS = [42, 123, 456, 789, 1024]
N_FEATURES = 16384


def load_decoder_weights(model_key, seed):
    """Load decoder weight matrix from saved SAE."""
    path = os.path.join(SAE_DIR, f"{model_key}_seed{seed}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    W_dec = checkpoint["state_dict"]["W_dec"]  # [n_features, d_model]
    return torch.nn.functional.normalize(W_dec, dim=1)  # unit norm


def load_activations(model_key, seed):
    """Load SAE activations on eval data."""
    path = os.path.join(SAE_DIR, f"activations_{model_key}_seed{seed}.pt")
    return torch.load(path, map_location="cpu", weights_only=True)


def compute_cosine_sim_matrix(W1, W2, chunk_size=2048):
    """Compute pairwise cosine similarity between two decoder matrices."""
    # W1, W2: [n_features, d_model], already unit-normed
    n = W1.shape[0]
    sim = torch.zeros(n, n)
    W2_t = W2.T  # [d_model, n]

    for i in range(0, n, chunk_size):
        chunk = W1[i:i+chunk_size]  # [chunk, d_model]
        sim[i:i+chunk_size] = chunk @ W2_t  # [chunk, n]

    return sim


def compute_best_match_scores(sim_matrix):
    """For each row, find max cosine sim (top-1 greedy match)."""
    return sim_matrix.max(dim=1).values.numpy()


def compute_hungarian_matching(sim_matrix, n_match=4096):
    """Compute optimal 1-1 matching using Hungarian algorithm on top features.
    Full 16384x16384 Hungarian is too slow, so we do it on a subset."""
    # Use top-n_match features by max similarity
    max_sims_row = sim_matrix.max(dim=1).values
    top_rows = torch.argsort(max_sims_row, descending=True)[:n_match]

    max_sims_col = sim_matrix.max(dim=0).values
    top_cols = torch.argsort(max_sims_col, descending=True)[:n_match]

    sub_sim = sim_matrix[top_rows][:, top_cols].numpy()
    row_ind, col_ind = linear_sum_assignment(-sub_sim)

    # Map back to original indices
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        orig_r = top_rows[r].item()
        orig_c = top_cols[c].item()
        matched_pairs.append((orig_r, orig_c, sub_sim[r, c]))

    return matched_pairs


def compute_convergence_scores(model_key):
    """Compute convergence score for each feature across seeds."""
    print(f"\nComputing convergence scores for {model_key}...")

    # Load all decoder weights
    decoders = {}
    activations = {}
    for seed in SEEDS:
        decoders[seed] = load_decoder_weights(model_key, seed)
        activations[seed] = load_activations(model_key, seed)
        print(f"  Loaded seed {seed}: decoder {decoders[seed].shape}, acts {activations[seed].shape}")

    # For each reference seed, compute convergence scores
    all_convergence = {}

    for ref_seed in SEEDS:
        W_ref = decoders[ref_seed]
        other_seeds = [s for s in SEEDS if s != ref_seed]

        # For each feature in ref_seed, compute mean best-match cosine sim across other seeds
        best_matches = []
        for other_seed in other_seeds:
            W_other = decoders[other_seed]
            sim = compute_cosine_sim_matrix(W_ref, W_other)
            best_match = sim.max(dim=1).values.numpy()  # [n_features]
            best_matches.append(best_match)

        best_matches = np.stack(best_matches)  # [n_other_seeds, n_features]

        # Convergence score: mean best-match similarity
        convergence_mean = best_matches.mean(axis=0)  # [n_features]

        # Strict convergence: fraction of seeds with match > 0.8
        convergence_strict = (best_matches > 0.8).mean(axis=0)

        # Also compute fraction > 0.7 and > 0.9
        convergence_07 = (best_matches > 0.7).mean(axis=0)
        convergence_09 = (best_matches > 0.9).mean(axis=0)

        all_convergence[ref_seed] = {
            "mean_sim": convergence_mean,
            "frac_08": convergence_strict,
            "frac_07": convergence_07,
            "frac_09": convergence_09,
        }

    # Check robustness: correlate scores across reference seeds
    ref_scores = [all_convergence[s]["mean_sim"] for s in SEEDS]
    robustness = []
    for i in range(len(SEEDS)):
        for j in range(i+1, len(SEEDS)):
            rho, p = spearmanr(ref_scores[i], ref_scores[j])
            robustness.append(rho)
    mean_robustness = np.mean(robustness)
    print(f"  Reference seed robustness (mean Spearman rho): {mean_robustness:.4f}")

    # Use seed=42 as primary reference
    primary = all_convergence[42]

    # Classify core vs peripheral
    # Use tau=0.6, matched in at least 2/4 other seeds (frac >= 0.5)
    # This gives a reasonable core fraction (~10-20%) consistent with literature
    W_ref = decoders[42]
    other_seeds_list = [s for s in SEEDS if s != 42]
    best_match_per_seed = []
    for s in other_seeds_list:
        sim = compute_cosine_sim_matrix(W_ref, decoders[s])
        best_match_per_seed.append(sim.max(dim=1).values.numpy())
    best_match_arr = np.stack(best_match_per_seed)  # [4, n_features]
    core_mask = (best_match_arr > 0.6).mean(axis=0) >= 0.5
    n_core = core_mask.sum()
    n_total = len(core_mask)
    print(f"  Core features (tau=0.8, >=3/4 seeds): {n_core}/{n_total} ({100*n_core/n_total:.1f}%)")

    # Also activation-based convergence: for matched features, correlation of activations
    A_ref = activations[42]
    act_convergence = np.zeros(N_FEATURES)
    for other_seed in [123, 456, 789, 1024]:
        W_other = decoders[other_seed]
        A_other = activations[other_seed]
        sim = compute_cosine_sim_matrix(decoders[42], W_other)
        best_idx = sim.argmax(dim=1)  # [n_features]

        # For each feature, correlate activations with best match
        # Do in batches for memory
        for fi in range(0, N_FEATURES, 1024):
            fi_end = min(fi + 1024, N_FEATURES)
            n_eval = min(A_ref.shape[0], A_other.shape[0])
            ref_batch = A_ref[:n_eval, fi:fi_end].float()  # [n_eval, batch_features]
            match_indices = best_idx[fi:fi_end]
            other_batch = A_other[:n_eval][:, match_indices].float()

            # Pearson correlation per feature
            ref_mean = ref_batch.mean(dim=0, keepdim=True)
            other_mean = other_batch.mean(dim=0, keepdim=True)
            ref_centered = ref_batch - ref_mean
            other_centered = other_batch - other_mean

            num = (ref_centered * other_centered).sum(dim=0)
            den = ref_centered.norm(dim=0) * other_centered.norm(dim=0) + 1e-8
            corr = (num / den).numpy()
            act_convergence[fi:fi_end] += corr / 4  # average over 4 other seeds

    # Combined convergence score
    combined_convergence = 0.5 * primary["mean_sim"] + 0.5 * act_convergence

    results = {
        "convergence_score_decoder": primary["mean_sim"].tolist(),
        "convergence_score_activation": act_convergence.tolist(),
        "convergence_score_combined": combined_convergence.tolist(),
        "frac_matched_08": primary["frac_08"].tolist(),
        "frac_matched_07": primary["frac_07"].tolist(),
        "frac_matched_09": primary["frac_09"].tolist(),
        "core_mask_08_75": core_mask.tolist(),
        "n_core": int(n_core),
        "n_total": int(n_total),
        "core_fraction": float(n_core / n_total),
        "reference_seed_robustness": float(mean_robustness),
        "robustness_all_pairs": [float(r) for r in robustness],
    }

    # Threshold sweep - compute from raw best-match scores
    # First collect all best-match scores
    other_seeds = [s for s in SEEDS if s != 42]
    best_matches_all = []
    for s in other_seeds:
        sim = compute_cosine_sim_matrix(decoders[42], decoders[s])
        best_matches_all.append(sim.max(dim=1).values.numpy())
    best_matches_all = np.stack(best_matches_all)  # [4, n_features]

    thresholds = {}
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        frac_above = (best_matches_all > tau).mean(axis=0)
        for m_min_frac in [0.5, 0.75, 1.0]:
            mask = frac_above >= m_min_frac
            key = f"tau{tau}_mmin{m_min_frac}"
            thresholds[key] = int(mask.sum())

    results["threshold_sweep"] = thresholds

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for model_key in MODELS:
        results = compute_convergence_scores(model_key)
        all_results[model_key] = {
            "n_core": results["n_core"],
            "n_total": results["n_total"],
            "core_fraction": results["core_fraction"],
            "reference_seed_robustness": results["reference_seed_robustness"],
        }

        # Save full results
        path = os.path.join(RESULTS_DIR, f"convergence_scores_{model_key}.json")
        with open(path, "w") as f:
            json.dump(results, f)
        print(f"  Saved to {path}")

    # Save summary
    with open(os.path.join(RESULTS_DIR, "convergence_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nConvergence score summary:")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
