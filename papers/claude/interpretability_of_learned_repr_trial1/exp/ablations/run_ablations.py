"""Ablation studies on convergence score methodology."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

SAE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sae_training')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

MODELS = {
    "gpt2_small": {"d_model": 768},
    "pythia_160m": {"d_model": 768},
    "pythia_410m": {"d_model": 1024},
}
SEEDS = [42, 123, 456, 789, 1024]
N_FEATURES = 16384


def load_decoder(model_key, seed):
    path = os.path.join(SAE_DIR, f"{model_key}_seed{seed}.pt")
    cp = torch.load(path, map_location="cpu", weights_only=True)
    W = cp["state_dict"]["W_dec"]
    return torch.nn.functional.normalize(W, dim=1)


def load_activations(model_key, seed):
    path = os.path.join(SAE_DIR, f"activations_{model_key}_seed{seed}.pt")
    return torch.load(path, map_location="cpu", weights_only=True)


def cosine_sim_matrix(W1, W2):
    return (W1 @ W2.T).numpy()


def convergence_from_seeds(model_key, seed_subset, method="top1", metric="decoder"):
    """Compute convergence scores using a subset of seeds and a specific method."""
    ref_seed = seed_subset[0]
    other_seeds = seed_subset[1:]

    if metric in ("decoder", "combined"):
        W_ref = load_decoder(model_key, ref_seed)

    if metric in ("activation", "combined"):
        A_ref = load_activations(model_key, ref_seed)

    scores = np.zeros(N_FEATURES)

    for other_seed in other_seeds:
        if metric in ("decoder", "combined"):
            W_other = load_decoder(model_key, other_seed)
            sim = cosine_sim_matrix(W_ref, W_other)

            if method == "top1":
                best = sim.max(axis=1)
            elif method == "greedy":
                best = np.zeros(N_FEATURES)
                used = set()
                order = np.argsort(-sim.max(axis=1))
                for i in order:
                    available = [j for j in range(N_FEATURES) if j not in used]
                    if not available:
                        break
                    j = available[np.argmax(sim[i, available])]
                    best[i] = sim[i, j]
                    used.add(j)
            elif method == "hungarian":
                # Use subset for speed
                n_sub = min(4096, N_FEATURES)
                top_idx = np.argsort(-sim.max(axis=1))[:n_sub]
                sub_sim = sim[np.ix_(top_idx, top_idx)]
                ri, ci = linear_sum_assignment(-sub_sim)
                best = sim.max(axis=1)  # fallback for non-matched
                for r, c in zip(ri, ci):
                    best[top_idx[r]] = sub_sim[r, c]
            else:
                best = sim.max(axis=1)

            if metric == "decoder":
                scores += best / len(other_seeds)

        if metric in ("activation", "combined"):
            A_other = load_activations(model_key, other_seed)
            W_other_dec = load_decoder(model_key, other_seed)
            sim_dec = cosine_sim_matrix(load_decoder(model_key, ref_seed), W_other_dec)
            best_idx = sim_dec.argmax(axis=1)

            n_eval = min(A_ref.shape[0], A_other.shape[0])
            act_corr = np.zeros(N_FEATURES)
            for fi in range(0, N_FEATURES, 1024):
                fi_end = min(fi + 1024, N_FEATURES)
                ref_batch = A_ref[:n_eval, fi:fi_end].float()
                match_indices = best_idx[fi:fi_end]
                other_batch = A_other[:n_eval][:, match_indices].float()

                ref_c = ref_batch - ref_batch.mean(dim=0, keepdim=True)
                other_c = other_batch - other_batch.mean(dim=0, keepdim=True)
                num = (ref_c * other_c).sum(dim=0)
                den = ref_c.norm(dim=0) * other_c.norm(dim=0) + 1e-8
                act_corr[fi:fi_end] = (num / den).numpy()

            if metric == "activation":
                scores += act_corr / len(other_seeds)
            elif metric == "combined":
                scores += (best + act_corr) / (2 * len(other_seeds))

    return scores


def run_ablation_num_seeds(model_key, kl_divergences):
    """Ablation 1: Number of seeds."""
    results = {}
    for n_seeds in [2, 3, 4, 5]:
        seed_subset = SEEDS[:n_seeds]
        conv = convergence_from_seeds(model_key, seed_subset)
        rho, p = spearmanr(conv, kl_divergences)
        top20 = kl_divergences >= np.percentile(kl_divergences, 80)
        auc = roc_auc_score(top20, conv) if len(np.unique(top20)) > 1 else 0.5
        results[str(n_seeds)] = {"spearman_rho": float(rho), "p": float(p), "auc": float(auc)}
    return results


def run_ablation_matching(model_key, kl_divergences):
    """Ablation 2: Matching method."""
    results = {}
    for method in ["top1", "greedy", "hungarian"]:
        conv = convergence_from_seeds(model_key, SEEDS, method=method)
        rho, p = spearmanr(conv, kl_divergences)
        top20 = kl_divergences >= np.percentile(kl_divergences, 80)
        auc = roc_auc_score(top20, conv) if len(np.unique(top20)) > 1 else 0.5
        results[method] = {"spearman_rho": float(rho), "p": float(p), "auc": float(auc)}
    return results


def run_ablation_metric(model_key, kl_divergences):
    """Ablation 3: Similarity metric."""
    results = {}
    for metric in ["decoder", "activation", "combined"]:
        conv = convergence_from_seeds(model_key, SEEDS, metric=metric)
        rho, p = spearmanr(conv, kl_divergences)
        top20 = kl_divergences >= np.percentile(kl_divergences, 80)
        auc = roc_auc_score(top20, conv) if len(np.unique(top20)) > 1 else 0.5
        results[metric] = {"spearman_rho": float(rho), "p": float(p), "auc": float(auc)}
    return results


def run_ablation_threshold(model_key, kl_divergences):
    """Ablation 4: Threshold sensitivity."""
    conv_path = os.path.join(RESULTS_DIR, f"convergence_scores_{model_key}.json")
    with open(conv_path) as f:
        conv_data = json.load(f)

    conv_scores = np.array(conv_data["convergence_score_decoder"])

    results = {}
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        # Compute core mask at this threshold
        W_ref = load_decoder(model_key, 42)
        frac_above = np.zeros(N_FEATURES)
        for other_seed in [123, 456, 789, 1024]:
            W_other = load_decoder(model_key, other_seed)
            sim = cosine_sim_matrix(W_ref, W_other)
            frac_above += (sim.max(axis=1) > tau) / 4

        core_mask = frac_above >= 0.75
        n_core = core_mask.sum()

        if n_core > 0 and n_core < N_FEATURES:
            core_kl = kl_divergences[core_mask]
            periph_kl = kl_divergences[~core_mask]
            n1, n2 = len(core_kl), len(periph_kl)
            v1, v2 = np.var(core_kl, ddof=1), np.var(periph_kl, ddof=1)
            pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
            d = float((core_kl.mean() - periph_kl.mean()) / (pooled + 1e-10))
        else:
            d = 0.0

        results[str(tau)] = {
            "n_core": int(n_core),
            "core_fraction": float(n_core / N_FEATURES),
            "cohens_d": float(d),
        }

    return results


def run_combined_baseline(model_key, kl_divergences):
    """Strong baseline: logistic regression on multiple feature statistics."""
    sae_acts = load_activations(model_key, 42)
    cp = torch.load(os.path.join(SAE_DIR, f"{model_key}_seed42.pt"),
                    map_location="cpu", weights_only=True)
    W_dec = cp["state_dict"]["W_dec"]
    W_enc = cp["state_dict"]["W_enc"]

    act_freq = (sae_acts > 0).float().mean(dim=0).numpy()
    dec_norm = W_dec.norm(dim=1).numpy()
    act_mag = sae_acts.sum(dim=0).numpy() / (sae_acts > 0).float().sum(dim=0).numpy().clip(min=1)
    enc_dec = torch.nn.functional.cosine_similarity(W_enc, W_dec, dim=1).numpy()

    # Focus on active features only
    active = kl_divergences > 1e-8
    if active.sum() < 100:
        return {"mean_auc": 0.5, "std_auc": 0.0, "fold_aucs": [0.5]*5}

    X = np.column_stack([act_freq[active], dec_norm[active], act_mag[active], enc_dec[active]])
    active_kl = kl_divergences[active]
    top20 = (active_kl >= np.percentile(active_kl, 80)).astype(int)

    if len(np.unique(top20)) < 2:
        return {"mean_auc": 0.5, "std_auc": 0.0, "fold_aucs": [0.5]*5}

    clf = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(clf, X, top20, cv=5, scoring='roc_auc')

    return {
        "mean_auc": float(cv_scores.mean()),
        "std_auc": float(cv_scores.std()),
        "fold_aucs": [float(s) for s in cv_scores],
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for model_key in MODELS:
        print(f"\n{'='*60}")
        print(f"Ablations: {model_key}")
        print(f"{'='*60}")

        # Load KL divergences
        causal_path = os.path.join(RESULTS_DIR, f"causal_importance_{model_key}.json")
        if not os.path.exists(causal_path):
            print(f"  Skipping {model_key} - no causal importance data")
            continue
        with open(causal_path) as f:
            causal_data = json.load(f)
        kl_div = np.array(causal_data["kl_divergences"])

        print("  Ablation 1: Number of seeds...")
        abl_seeds = run_ablation_num_seeds(model_key, kl_div)
        print(f"    {json.dumps(abl_seeds, indent=2)}")

        print("  Ablation 2: Matching method...")
        abl_match = run_ablation_matching(model_key, kl_div)
        print(f"    {json.dumps(abl_match, indent=2)}")

        print("  Ablation 3: Similarity metric...")
        abl_metric = run_ablation_metric(model_key, kl_div)
        print(f"    {json.dumps(abl_metric, indent=2)}")

        print("  Ablation 4: Threshold sensitivity...")
        abl_thresh = run_ablation_threshold(model_key, kl_div)
        print(f"    {json.dumps(abl_thresh, indent=2)}")

        print("  Combined baseline (logistic regression)...")
        combined_bl = run_combined_baseline(model_key, kl_div)
        print(f"    {json.dumps(combined_bl, indent=2)}")

        all_results[model_key] = {
            "num_seeds": abl_seeds,
            "matching_method": abl_match,
            "similarity_metric": abl_metric,
            "threshold_sensitivity": abl_thresh,
            "combined_baseline": combined_bl,
        }

    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nAblation results saved.")


if __name__ == "__main__":
    main()
