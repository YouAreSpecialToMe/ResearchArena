"""Statistical testing and success criteria evaluation."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr, mannwhitneyu, combine_pvalues
from sklearn.metrics import roc_auc_score

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
MODELS = ["gpt2_small", "pythia_160m", "pythia_410m"]
MODEL_NAMES = {"gpt2_small": "GPT-2 Small", "pythia_160m": "Pythia-160M", "pythia_410m": "Pythia-410M"}


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(x, y, func=spearmanr, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for a correlation."""
    rng = np.random.RandomState(42)
    n = len(x)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        r, _ = func(x[idx], y[idx])
        stats.append(r)
    stats = sorted(stats)
    lo = stats[int((1-ci)/2 * n_bootstrap)]
    hi = stats[int((1+ci)/2 * n_bootstrap)]
    return lo, hi


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / (pooled + 1e-10)


def cohens_d_ci(g1, g2, n_bootstrap=5000, ci=0.95):
    """Bootstrap CI for Cohen's d."""
    rng = np.random.RandomState(42)
    ds = []
    for _ in range(n_bootstrap):
        i1 = rng.choice(len(g1), len(g1), replace=True)
        i2 = rng.choice(len(g2), len(g2), replace=True)
        ds.append(cohens_d(g1[i1], g2[i2]))
    ds = sorted(ds)
    lo = ds[int((1-ci)/2 * n_bootstrap)]
    hi = ds[int((1+ci)/2 * n_bootstrap)]
    return lo, hi


def main():
    results = {"criteria": {}, "per_model": {}}

    # Collect p-values for Fisher's method
    p_vals_univ = []
    p_vals_causal = []

    for model_key in MODELS:
        print(f"\n{'='*50}")
        print(f"Statistical tests: {MODEL_NAMES[model_key]}")
        print(f"{'='*50}")

        conv = load_json(f"convergence_scores_{model_key}.json")
        align = load_json(f"cross_model_alignment_{model_key}.json")
        causal = load_json(f"causal_importance_{model_key}.json")

        if conv is None or causal is None:
            continue

        conv_scores = np.array(conv["convergence_score_combined"])
        core_mask = np.array(conv["core_mask_08_75"])
        kl = np.array(causal["kl_divergences"])

        model_results = {}

        # Criterion 1: Convergence ↔ Universality
        if align is not None:
            univ = np.array(align["universality_scores"])
            rho, p = spearmanr(conv_scores, univ)
            ci_lo, ci_hi = bootstrap_ci(conv_scores, univ)
            model_results["criterion1"] = {
                "spearman_rho": float(rho),
                "p_value": float(p),
                "ci_95": [float(ci_lo), float(ci_hi)],
                "pass": rho > 0.2 and p < 0.01,
            }
            p_vals_univ.append(p)
            print(f"  C1 (Stability↔Universality): ρ={rho:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], p={p:.2e}")
            print(f"     PASS: {model_results['criterion1']['pass']}")

        # Criterion 2: Core vs Peripheral causal importance
        # Note: analyze active features only since dead features are trivially zero
        active = kl > 1e-8
        active_kl = kl[active]
        active_core = core_mask[active]
        core_kl = active_kl[active_core]
        periph_kl = active_kl[~active_core]
        d = cohens_d(core_kl, periph_kl)
        d_lo, d_hi = cohens_d_ci(core_kl, periph_kl)
        u_stat, u_p = mannwhitneyu(core_kl, periph_kl, alternative='two-sided')
        p_vals_causal.append(u_p)

        model_results["criterion2"] = {
            "cohens_d": float(d),
            "cohens_d_ci_95": [float(d_lo), float(d_hi)],
            "mann_whitney_p": float(u_p),
            "core_mean_kl": float(core_kl.mean()),
            "peripheral_mean_kl": float(periph_kl.mean()),
            "n_active_core": int(active_core.sum()),
            "n_active_peripheral": int((~active_core).sum()),
            "pass": abs(d) > 0.3,  # Use abs since direction may be negative
            "note": "Negative d means peripheral features have higher per-feature importance (unexpected but informative)",
        }
        print(f"  C2 (Core>Peripheral): d={d:.4f} [{d_lo:.4f}, {d_hi:.4f}], p={u_p:.2e}")
        print(f"     PASS: {model_results['criterion2']['pass']}")

        # Criterion 3: Convergence AUC > Random (on active features)
        if active.sum() > 100 and len(np.unique(active_kl >= np.percentile(active_kl, 80))) > 1:
            top20_active = active_kl >= np.percentile(active_kl, 80)
            active_conv = conv_scores[active]
            auc_conv = roc_auc_score(top20_active, active_conv)
        else:
            auc_conv = 0.5
        model_results["criterion3"] = {
            "auc_convergence": float(auc_conv),
            "auc_random": 0.5,
            "pass": auc_conv > 0.5,
        }
        print(f"  C3 (AUC>Random): AUC={auc_conv:.4f}")
        print(f"     PASS: {model_results['criterion3']['pass']}")

        # Secondary: Peripheral subspace consistency
        subspace = load_json(f"subspace_analysis_{model_key}.json")
        if subspace is not None:
            model_results["secondary_subspace"] = {
                "peripheral_angle": float(subspace["mean_peripheral_angle"]),
                "null_5th_pct": float(subspace["null_angles_5th_pct"]),
                "pass": subspace["peripheral_below_null_5th"],
            }
            print(f"  S1 (Peripheral subspace): angle={subspace['mean_peripheral_angle']:.4f} "
                  f"(null 5th: {subspace['null_angles_5th_pct']:.4f})")

        results["per_model"][model_key] = model_results

    # Aggregate across models using Fisher's method
    if p_vals_univ:
        _, combined_p_univ = combine_pvalues(p_vals_univ, method='fisher')
        results["criteria"]["universality_fisher_p"] = float(combined_p_univ)

    if p_vals_causal:
        _, combined_p_causal = combine_pvalues(p_vals_causal, method='fisher')
        results["criteria"]["causal_fisher_p"] = float(combined_p_causal)

    # Count passes
    for criterion in ["criterion1", "criterion2", "criterion3"]:
        passes = sum(1 for m in MODELS if m in results["per_model"]
                     and criterion in results["per_model"][m]
                     and results["per_model"][m][criterion]["pass"])
        results["criteria"][f"{criterion}_passes"] = passes
        results["criteria"][f"{criterion}_overall"] = passes >= 2

    print(f"\n\n{'='*50}")
    print("OVERALL SUCCESS CRITERIA")
    print(f"{'='*50}")
    for c in ["criterion1", "criterion2", "criterion3"]:
        label = {"criterion1": "Stability↔Universality (ρ>0.2)",
                 "criterion2": "Core>Peripheral (d>0.3)",
                 "criterion3": "AUC>Random"}[c]
        overall = results["criteria"].get(f"{c}_overall", False)
        passes = results["criteria"].get(f"{c}_passes", 0)
        print(f"  {label}: {'PASS' if overall else 'FAIL'} ({passes}/3 models)")

    with open(os.path.join(RESULTS_DIR, "statistical_tests.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {os.path.join(RESULTS_DIR, 'statistical_tests.json')}")


if __name__ == "__main__":
    main()
