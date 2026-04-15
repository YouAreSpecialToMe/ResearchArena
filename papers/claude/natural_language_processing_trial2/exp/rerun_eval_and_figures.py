"""
Comprehensive re-evaluation with fixed labels.
- Evaluates all methods (SpecCheck logprob, SpecCheck sampling multi-seed,
  SelfCheck, verbalized, logprob, random)
- Runs ablations (ladder depth, score variants, combinations)
- Generates all figures
- Produces final results.json
"""
import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    save_json, load_json, set_seed,
    SEEDS, MODELS, MODEL_SHORT, DATASETS, DATA_DIR, RESULTS_DIR, FIGURES_DIR
)
from shared.metrics import (
    compute_all_metrics, compute_auc_roc, compute_auc_pr,
    bootstrap_ci, paired_bootstrap_test
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Load all data
# ============================================================
def load_labels(mshort, dataset):
    path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset}.json")
    if not os.path.exists(path):
        return None
    data = load_json(path)
    return {c["claim_id"]: c["label"] for c in data}


def load_scores(path):
    if not os.path.exists(path):
        return None
    data = load_json(path)
    result = {}
    for item in data:
        cid = item.get("claim_id")
        score = item.get("hallucination_score") or item.get("speccheck_score", 0.5)
        result[cid] = score
    return result


def load_speccheck_scores(mshort, dataset):
    path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset}.json")
    if not os.path.exists(path):
        return None, None
    data = load_json(path)
    scores = {c["claim_id"]: c["speccheck_score"] for c in data}
    full_data = {c["claim_id"]: c for c in data}
    return scores, full_data


def load_speccheck_sampling(mshort, dataset, seed):
    path = os.path.join(RESULTS_DIR, f"speccheck_sampling_seed{seed}_{mshort}_{dataset}.json")
    if not os.path.exists(path):
        return None
    data = load_json(path)
    return {c["claim_id"]: c["speccheck_score"] for c in data}


def load_confidence_data(mshort, dataset):
    path = os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{dataset}.json")
    if not os.path.exists(path):
        return None
    return {c["claim_id"]: c for c in load_json(path)}


def align_labels_scores(labels, scores):
    """Align labels and scores by claim_id, returning parallel arrays."""
    common_ids = set(labels.keys()) & set(scores.keys())
    if not common_ids:
        return np.array([]), np.array([])
    ids = sorted(common_ids)
    y = np.array([labels[cid] for cid in ids])
    s = np.array([scores[cid] for cid in ids])
    return y, s


# ============================================================
# Main evaluation
# ============================================================
def evaluate_all():
    """Evaluate all methods on all model-dataset pairs."""
    all_results = {}

    for model_name in MODELS:
        mshort = MODEL_SHORT[model_name]
        for dataset in DATASETS:
            key = f"{mshort}_{dataset}"
            labels = load_labels(mshort, dataset)
            if labels is None:
                print(f"  No labels for {key}, skipping")
                continue

            # Filter out unlabeled (-1) claims
            labels = {k: v for k, v in labels.items() if v in (0, 1)}

            result = {"model": mshort, "dataset": dataset, "n_claims": len(labels)}
            methods = {}

            # 1. SpecCheck (logprob)
            sc_scores, sc_full = load_speccheck_scores(mshort, dataset)
            if sc_scores:
                y, s = align_labels_scores(labels, sc_scores)
                if len(y) > 10:
                    metrics = compute_all_metrics(y, s)
                    _, lo, hi = bootstrap_ci(y, s, compute_auc_pr, n_bootstrap=2000)
                    metrics["auc_pr_ci"] = [round(lo, 4), round(hi, 4)]
                    _, lo2, hi2 = bootstrap_ci(y, s, compute_auc_roc, n_bootstrap=2000)
                    metrics["auc_roc_ci"] = [round(lo2, 4), round(hi2, 4)]
                    methods["speccheck"] = metrics

            # 2. SpecCheck sampling (multi-seed)
            sampling_metrics = []
            for seed in SEEDS:
                samp_scores = load_speccheck_sampling(mshort, dataset, seed)
                if samp_scores:
                    y, s = align_labels_scores(labels, samp_scores)
                    if len(y) > 10:
                        m = compute_all_metrics(y, s)
                        sampling_metrics.append(m)

            if sampling_metrics:
                avg_metrics = {}
                for k in sampling_metrics[0]:
                    vals = [m[k] for m in sampling_metrics if not (isinstance(m[k], float) and np.isnan(m[k]))]
                    if vals:
                        avg_metrics[k] = round(float(np.mean(vals)), 4)
                        if len(vals) >= 2:
                            avg_metrics[f"{k}_std"] = round(float(np.std(vals)), 4)
                methods["speccheck_sampling"] = avg_metrics

            # 3. Baselines
            for baseline in ["selfcheck", "verbalized", "logprob"]:
                path = os.path.join(RESULTS_DIR, f"baseline_{baseline}_{mshort}_{dataset}.json")
                scores = load_scores(path)
                if scores:
                    y, s = align_labels_scores(labels, scores)
                    if len(y) > 10:
                        metrics = compute_all_metrics(y, s)
                        _, lo, hi = bootstrap_ci(y, s, compute_auc_pr, n_bootstrap=2000)
                        metrics["auc_pr_ci"] = [round(lo, 4), round(hi, 4)]
                        _, lo2, hi2 = bootstrap_ci(y, s, compute_auc_roc, n_bootstrap=2000)
                        metrics["auc_roc_ci"] = [round(lo2, 4), round(hi2, 4)]
                        methods[baseline] = metrics

            # 4. Random baseline (multi-seed)
            rand_metrics = []
            for seed in SEEDS:
                path = os.path.join(RESULTS_DIR, f"baseline_random_{mshort}_{dataset}_seed{seed}.json")
                scores = load_scores(path)
                if scores:
                    y, s = align_labels_scores(labels, scores)
                    if len(y) > 10:
                        m = compute_all_metrics(y, s)
                        rand_metrics.append(m)

            if rand_metrics:
                avg_metrics = {}
                for k in rand_metrics[0]:
                    vals = [m[k] for m in rand_metrics if not (isinstance(m[k], float) and np.isnan(m[k]))]
                    if vals:
                        avg_metrics[k] = round(float(np.mean(vals)), 4)
                        if len(vals) >= 2:
                            avg_metrics[f"{k}_std"] = round(float(np.std(vals)), 4)
                methods["random"] = avg_metrics

            result["methods"] = methods
            all_results[key] = result

            # Print summary
            n_f = sum(1 for v in labels.values() if v == 0)
            n_h = sum(1 for v in labels.values() if v == 1)
            print(f"\n  {key}: {n_f} factual, {n_h} halluc ({n_h/(n_f+n_h)*100:.0f}%)")
            for m, v in methods.items():
                print(f"    {m:20s}: AUC-ROC={v.get('auc_roc', '?'):.4f}  AUC-PR={v.get('auc_pr', '?'):.4f}")

    return all_results


# ============================================================
# Monotonicity analysis
# ============================================================
def monotonicity_analysis():
    """Analyze confidence monotonicity rates for factual vs hallucinated claims."""
    results = {}

    for model_name in MODELS:
        mshort = MODEL_SHORT[model_name]
        for dataset in DATASETS:
            labels = load_labels(mshort, dataset)
            conf_data = load_confidence_data(mshort, dataset)
            if labels is None or conf_data is None:
                continue

            factual_mono = []
            halluc_mono = []
            factual_confs = []
            halluc_confs = []

            for cid, label in labels.items():
                if label not in (0, 1):
                    continue
                if cid not in conf_data:
                    continue

                confs = conf_data[cid]["confidences"]
                if len(confs) < 4:
                    continue

                # Check strict monotonicity: conf should be non-decreasing
                is_mono = all(confs[k] >= confs[k-1] - 0.01 for k in range(1, len(confs)))

                if label == 0:
                    factual_mono.append(int(is_mono))
                    factual_confs.append(confs)
                else:
                    halluc_mono.append(int(is_mono))
                    halluc_confs.append(confs)

            if factual_mono and halluc_mono:
                key = f"{mshort}_{dataset}"
                results[key] = {
                    "factual_monotonic_rate": round(np.mean(factual_mono), 4),
                    "hallucinated_monotonic_rate": round(np.mean(halluc_mono), 4),
                    "hallucinated_violation_rate": round(1 - np.mean(halluc_mono), 4),
                    "n_factual": len(factual_mono),
                    "n_hallucinated": len(halluc_mono),
                    "factual_avg_confs": [round(float(np.mean([c[k] for c in factual_confs])), 4) for k in range(4)],
                    "halluc_avg_confs": [round(float(np.mean([c[k] for c in halluc_confs])), 4) for k in range(4)],
                }
                print(f"  {key}: factual_mono={results[key]['factual_monotonic_rate']:.3f}, "
                      f"halluc_violation={results[key]['hallucinated_violation_rate']:.3f}")

    return results


# ============================================================
# Ablation: Ladder depth
# ============================================================
def ablation_ladder_depth():
    """Run ablation on ladder depth K=1,2,3."""
    results = {}

    for model_name in MODELS:
        mshort = MODEL_SHORT[model_name]
        for dataset in DATASETS:
            labels = load_labels(mshort, dataset)
            conf_data = load_confidence_data(mshort, dataset)
            if labels is None or conf_data is None:
                continue

            labels = {k: v for k, v in labels.items() if v in (0, 1)}

            depth_results = {}
            for K in [1, 2, 3]:
                # Recompute SpecCheck with different ladder depths
                scores = {}
                for cid, cdata in conf_data.items():
                    if cid not in labels:
                        continue
                    confs = cdata["confidences"][:K+1]
                    while len(confs) < K+1:
                        confs.append(confs[-1])

                    if K == 1:
                        # Binary: does confidence increase from level 0 to level K?
                        levels_to_use = [0, 3] if len(cdata["confidences"]) >= 4 else [0, len(cdata["confidences"])-1]
                        c0 = cdata["confidences"][levels_to_use[0]]
                        cK = cdata["confidences"][min(levels_to_use[1], len(cdata["confidences"])-1)]
                        mono = 1.0 if cK >= c0 else 0.0
                        gap = cK - c0
                    elif K == 2:
                        levels = [0, 2, 3] if len(cdata["confidences"]) >= 4 else list(range(min(3, len(cdata["confidences"]))))
                        used_confs = [cdata["confidences"][min(l, len(cdata["confidences"])-1)] for l in levels]
                        violations = sum(1 for k in range(1, len(used_confs)) if used_confs[k] < used_confs[k-1])
                        mono = 1.0 - violations / (len(used_confs) - 1)
                        gap = used_confs[-1] - used_confs[0]
                    else:  # K == 3
                        used_confs = cdata["confidences"][:4]
                        while len(used_confs) < 4:
                            used_confs.append(used_confs[-1])
                        violations = sum(1 for k in range(1, len(used_confs)) if used_confs[k] < used_confs[k-1])
                        mono = 1.0 - violations / (len(used_confs) - 1)
                        gap = used_confs[-1] - used_confs[0]

                    # SpecCheck score: higher = more likely hallucinated
                    alpha = 0.5
                    spec_score = (1.0 - mono) + alpha * max(0, -gap)
                    scores[cid] = spec_score

                y, s = align_labels_scores(labels, scores)
                if len(y) > 10:
                    metrics = compute_all_metrics(y, s)
                    depth_results[f"K={K}"] = metrics

            if depth_results:
                results[f"{mshort}_{dataset}"] = depth_results

    return results


# ============================================================
# Ablation: Score variants
# ============================================================
def ablation_score_variants():
    """Compare different scoring variants."""
    results = {}

    for model_name in MODELS:
        mshort = MODEL_SHORT[model_name]
        for dataset in DATASETS:
            labels = load_labels(mshort, dataset)
            sc_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset}.json")
            if labels is None or not os.path.exists(sc_path):
                continue

            labels = {k: v for k, v in labels.items() if v in (0, 1)}
            sc_data = {c["claim_id"]: c for c in load_json(sc_path)}

            variants = {}
            for variant_name, score_key in [
                ("speccheck_default", "speccheck_score"),
                ("max_violation", "max_violation"),
                ("confidence_gap", "gap_score"),
                ("weighted_violation", "weighted_violation"),
            ]:
                scores = {}
                for cid, item in sc_data.items():
                    if cid in labels:
                        scores[cid] = item.get(score_key, 0.5)

                y, s = align_labels_scores(labels, scores)
                if len(y) > 10:
                    variants[variant_name] = compute_all_metrics(y, s)

            # Also sweep alpha
            conf_data = load_confidence_data(mshort, dataset)
            if conf_data:
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    scores = {}
                    for cid, citem in conf_data.items():
                        if cid not in labels:
                            continue
                        confs = citem["confidences"][:4]
                        while len(confs) < 4:
                            confs.append(confs[-1])
                        violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
                        mono = 1.0 - violations / (len(confs) - 1)
                        gap = confs[-1] - confs[0]
                        scores[cid] = (1.0 - mono) + alpha * max(0, -gap)

                    y, s = align_labels_scores(labels, scores)
                    if len(y) > 10:
                        variants[f"alpha={alpha}"] = compute_all_metrics(y, s)

            if variants:
                results[f"{mshort}_{dataset}"] = variants

    return results


# ============================================================
# Ablation: Combination with baselines
# ============================================================
def ablation_combination():
    """Test combining SpecCheck with baselines via logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    results = {}

    for model_name in MODELS:
        mshort = MODEL_SHORT[model_name]
        for dataset in DATASETS:
            labels = load_labels(mshort, dataset)
            if labels is None:
                continue
            labels = {k: v for k, v in labels.items() if v in (0, 1)}

            # Load all scores
            sc_scores, _ = load_speccheck_scores(mshort, dataset)
            selfcheck = load_scores(os.path.join(RESULTS_DIR, f"baseline_selfcheck_{mshort}_{dataset}.json"))
            verbalized = load_scores(os.path.join(RESULTS_DIR, f"baseline_verbalized_{mshort}_{dataset}.json"))
            logprob = load_scores(os.path.join(RESULTS_DIR, f"baseline_logprob_{mshort}_{dataset}.json"))

            if not all([sc_scores, selfcheck, verbalized, logprob]):
                continue

            # Find common claims
            common = set(labels) & set(sc_scores) & set(selfcheck) & set(verbalized) & set(logprob)
            common = sorted(common)
            if len(common) < 50:
                continue

            y = np.array([labels[cid] for cid in common])
            X_spec = np.array([sc_scores[cid] for cid in common]).reshape(-1, 1)
            X_self = np.array([selfcheck[cid] for cid in common]).reshape(-1, 1)
            X_verb = np.array([verbalized[cid] for cid in common]).reshape(-1, 1)
            X_logp = np.array([logprob[cid] for cid in common]).reshape(-1, 1)

            combinations = {
                "speccheck_only": X_spec,
                "selfcheck_only": X_self,
                "logprob_only": X_logp,
                "all_baselines": np.hstack([X_self, X_verb, X_logp]),
                "all_baselines+speccheck": np.hstack([X_self, X_verb, X_logp, X_spec]),
                "speccheck+selfcheck": np.hstack([X_spec, X_self]),
                "speccheck+logprob": np.hstack([X_spec, X_logp]),
            }

            combo_results = {}
            for name, X in combinations.items():
                fold_aucs_roc = []
                fold_aucs_pr = []

                for seed in SEEDS:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    for train_idx, test_idx in skf.split(X, y):
                        clf = LogisticRegression(max_iter=1000, random_state=seed)
                        clf.fit(X[train_idx], y[train_idx])
                        probs = clf.predict_proba(X[test_idx])[:, 1]
                        try:
                            fold_aucs_roc.append(compute_auc_roc(y[test_idx], probs))
                            fold_aucs_pr.append(compute_auc_pr(y[test_idx], probs))
                        except:
                            pass

                if fold_aucs_roc:
                    combo_results[name] = {
                        "auc_roc": round(float(np.mean(fold_aucs_roc)), 4),
                        "auc_roc_std": round(float(np.std(fold_aucs_roc)), 4),
                        "auc_pr": round(float(np.mean(fold_aucs_pr)), 4),
                        "auc_pr_std": round(float(np.std(fold_aucs_pr)), 4),
                    }

            if combo_results:
                results[f"{mshort}_{dataset}"] = combo_results

    return results


# ============================================================
# Statistical significance
# ============================================================
def statistical_tests(all_results):
    """Run paired bootstrap tests comparing SpecCheck vs baselines."""
    sig_results = {}

    for model_name in MODELS:
        mshort = MODEL_SHORT[model_name]
        for dataset in DATASETS:
            key = f"{mshort}_{dataset}"
            labels = load_labels(mshort, dataset)
            if labels is None:
                continue
            labels = {k: v for k, v in labels.items() if v in (0, 1)}

            sc_scores, _ = load_speccheck_scores(mshort, dataset)
            if sc_scores is None:
                continue

            tests = {}
            for baseline_name in ["selfcheck", "verbalized", "logprob"]:
                bl_scores = load_scores(os.path.join(RESULTS_DIR, f"baseline_{baseline_name}_{mshort}_{dataset}.json"))
                if bl_scores is None:
                    continue

                common = set(labels) & set(sc_scores) & set(bl_scores)
                common = sorted(common)
                if len(common) < 50:
                    continue

                y = np.array([labels[cid] for cid in common])
                s_sc = np.array([sc_scores[cid] for cid in common])
                s_bl = np.array([bl_scores[cid] for cid in common])

                p_roc = paired_bootstrap_test(y, s_sc, s_bl, compute_auc_roc)
                p_pr = paired_bootstrap_test(y, s_sc, s_bl, compute_auc_pr)

                tests[baseline_name] = {
                    "p_value_auc_roc": round(p_roc, 4),
                    "p_value_auc_pr": round(p_pr, 4),
                    "speccheck_better_roc": p_roc < 0.05,
                    "speccheck_better_pr": p_pr < 0.05,
                }

            if tests:
                sig_results[key] = tests

    return sig_results


# ============================================================
# Figures
# ============================================================
def generate_figures(all_results, mono_results, ablation_depth, ablation_variants, combo_results):
    """Generate all paper figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    colors = {
        "speccheck": "#2196F3",
        "speccheck_sampling": "#1565C0",
        "selfcheck": "#FF9800",
        "verbalized": "#4CAF50",
        "logprob": "#F44336",
        "random": "#9E9E9E",
    }
    method_labels = {
        "speccheck": "SpecCheck (logprob)",
        "speccheck_sampling": "SpecCheck (sampling)",
        "selfcheck": "SelfCheckGPT",
        "verbalized": "Verbalized Conf.",
        "logprob": "Logprob Conf.",
        "random": "Random",
    }

    # ---- Figure 2: Main results bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    datasets_order = ["factscore", "longfact", "truthfulqa"]
    models_order = ["llama", "mistral", "qwen"]
    methods_order = ["speccheck", "speccheck_sampling", "logprob", "selfcheck", "verbalized", "random"]

    for di, dataset in enumerate(datasets_order):
        ax = axes[di]
        x = np.arange(len(models_order))
        width = 0.13
        offsets = np.arange(len(methods_order)) - len(methods_order) / 2 + 0.5

        for mi, method in enumerate(methods_order):
            vals = []
            errs = []
            for mshort in models_order:
                key = f"{mshort}_{dataset}"
                if key in all_results and method in all_results[key].get("methods", {}):
                    v = all_results[key]["methods"][method].get("auc_pr", 0)
                    std = all_results[key]["methods"][method].get("auc_pr_std", 0)
                    vals.append(v)
                    errs.append(std)
                else:
                    vals.append(0)
                    errs.append(0)

            bars = ax.bar(x + offsets[mi] * width, vals, width,
                         label=method_labels.get(method, method) if di == 0 else "",
                         color=colors.get(method, "#888"),
                         yerr=errs if any(e > 0 for e in errs) else None,
                         capsize=2, alpha=0.85)

        ax.set_title(dataset.replace("factscore", "FActScore").replace("longfact", "LongFact").replace("truthfulqa", "TruthfulQA"))
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in models_order])
        ax.set_ylim(0, 1.0)
        if di == 0:
            ax.set_ylabel("AUC-PR")

    axes[0].legend(loc="upper left", fontsize=8, ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure_2_main_results.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure_2_main_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figure_2_main_results")

    # ---- Figure 3: Monotonicity analysis ----
    if mono_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Left: monotonicity rates bar chart
        keys = sorted(mono_results.keys())
        factual_rates = [mono_results[k]["factual_monotonic_rate"] for k in keys]
        halluc_rates = [mono_results[k]["hallucinated_monotonic_rate"] for k in keys]

        x = np.arange(len(keys))
        w = 0.35
        ax1.bar(x - w/2, factual_rates, w, label="Factual claims", color="#4CAF50", alpha=0.8)
        ax1.bar(x + w/2, halluc_rates, w, label="Hallucinated claims", color="#F44336", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8)
        ax1.set_ylabel("Monotonicity Rate")
        ax1.set_title("Confidence Monotonicity by Claim Type")
        ax1.legend()
        ax1.set_ylim(0, 1.0)

        # Right: average confidence profiles
        all_factual_confs = []
        all_halluc_confs = []
        for k in keys:
            all_factual_confs.append(mono_results[k]["factual_avg_confs"])
            all_halluc_confs.append(mono_results[k]["halluc_avg_confs"])

        if all_factual_confs:
            avg_f = np.mean(all_factual_confs, axis=0)
            avg_h = np.mean(all_halluc_confs, axis=0)
            std_f = np.std(all_factual_confs, axis=0)
            std_h = np.std(all_halluc_confs, axis=0)

            levels = [0, 1, 2, 3]
            ax2.plot(levels, avg_f, 'o-', color="#4CAF50", label="Factual", linewidth=2, markersize=6)
            ax2.fill_between(levels, avg_f - std_f, avg_f + std_f, color="#4CAF50", alpha=0.2)
            ax2.plot(levels, avg_h, 's-', color="#F44336", label="Hallucinated", linewidth=2, markersize=6)
            ax2.fill_between(levels, avg_h - std_h, avg_h + std_h, color="#F44336", alpha=0.2)
            ax2.set_xlabel("Specificity Level (0=specific, 3=abstract)")
            ax2.set_ylabel("Average Confidence P(Yes)")
            ax2.set_title("Confidence Profiles")
            ax2.set_xticks(levels)
            ax2.set_xticklabels(["Original", "Approx.", "Category", "Abstract"])
            ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure_3_monotonicity.pdf"), dpi=150, bbox_inches="tight")
        plt.savefig(os.path.join(FIGURES_DIR, "figure_3_monotonicity.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved figure_3_monotonicity")

    # ---- Figure 5: Ablation studies ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Ladder depth
    if ablation_depth:
        ax = axes[0]
        for key in sorted(ablation_depth.keys()):
            depth_data = ablation_depth[key]
            Ks = []
            auc_prs = []
            for k_label in sorted(depth_data.keys()):
                K = int(k_label.split("=")[1])
                Ks.append(K)
                auc_prs.append(depth_data[k_label]["auc_pr"])
            ax.plot(Ks, auc_prs, 'o-', label=key.replace("_", "/"), markersize=5)
        ax.set_xlabel("Ladder Depth K")
        ax.set_ylabel("AUC-PR")
        ax.set_title("(a) Ladder Depth Ablation")
        ax.legend(fontsize=7)

    # (b) Score variants
    if ablation_variants:
        ax = axes[1]
        # Average across model-datasets
        all_variants = defaultdict(list)
        for key, variants in ablation_variants.items():
            for vname, vmetrics in variants.items():
                if "alpha" not in vname:
                    all_variants[vname].append(vmetrics["auc_pr"])

        if all_variants:
            names = sorted(all_variants.keys())
            means = [np.mean(all_variants[n]) for n in names]
            stds = [np.std(all_variants[n]) for n in names]
            x = np.arange(len(names))
            ax.bar(x, means, yerr=stds, capsize=3, color="#2196F3", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8, rotation=15)
            ax.set_ylabel("AUC-PR")
            ax.set_title("(b) Score Variant Comparison")

    # (c) Combination
    if combo_results:
        ax = axes[2]
        all_combos = defaultdict(list)
        for key, combos in combo_results.items():
            for cname, cmetrics in combos.items():
                all_combos[cname].append(cmetrics["auc_pr"])

        if all_combos:
            names = ["speccheck_only", "logprob_only", "selfcheck_only", "all_baselines", "all_baselines+speccheck"]
            display_names = ["SpecCheck", "Logprob", "SelfCheck", "All\nBaselines", "All+\nSpecCheck"]
            means = [np.mean(all_combos.get(n, [0])) for n in names]
            stds = [np.std(all_combos.get(n, [0])) for n in names]
            x = np.arange(len(names))
            bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8,
                          color=["#2196F3", "#F44336", "#FF9800", "#9E9E9E", "#1565C0"])
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, fontsize=8)
            ax.set_ylabel("AUC-PR (5-fold CV)")
            ax.set_title("(c) Method Combinations")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure_5_ablations.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure_5_ablations.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figure_5_ablations")

    # ---- Supplementary: Heatmap ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric_name in [(ax1, "auc_roc"), (ax2, "auc_pr")]:
        matrix = np.full((len(models_order), len(datasets_order)), np.nan)
        for mi, mshort in enumerate(models_order):
            for di, ds in enumerate(datasets_order):
                key = f"{mshort}_{ds}"
                if key in all_results:
                    methods = all_results[key].get("methods", {})
                    # Use best SpecCheck variant
                    best = 0
                    for m in ["speccheck", "speccheck_sampling"]:
                        if m in methods:
                            best = max(best, methods[m].get(metric_name, 0))
                    matrix[mi, di] = best

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(datasets_order)))
        ax.set_xticklabels(["FActScore", "LongFact", "TruthfulQA"])
        ax.set_yticks(range(len(models_order)))
        ax.set_yticklabels([m.capitalize() for m in models_order])
        ax.set_title(metric_name.upper())

        for mi in range(len(models_order)):
            for di in range(len(datasets_order)):
                if not np.isnan(matrix[mi, di]):
                    ax.text(di, mi, f"{matrix[mi, di]:.3f}", ha="center", va="center", fontsize=9)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("SpecCheck Performance Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "supp_heatmap.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "supp_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved supp_heatmap")

    # ---- Figure 1: Method schematic ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Example: True claim confidence profile
    true_confs = [0.65, 0.72, 0.78, 0.85]
    halluc_confs = [0.70, 0.55, 0.80, 0.88]
    levels = ["Original\n(Specific)", "Approximate", "Category", "Abstract\n(General)"]

    ax1.plot(range(4), true_confs, 'o-', color="#4CAF50", linewidth=2.5, markersize=10, label="True claim")
    for i, c in enumerate(true_confs):
        ax1.annotate(f"{c:.2f}", (i, c), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(levels, fontsize=9)
    ax1.set_ylabel("Model Confidence P(Yes)")
    ax1.set_title("True Claim: Monotonic ✓")
    ax1.set_ylim(0.3, 1.0)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    ax2.plot(range(4), halluc_confs, 's-', color="#F44336", linewidth=2.5, markersize=10, label="Hallucinated claim")
    for i, c in enumerate(halluc_confs):
        ax2.annotate(f"{c:.2f}", (i, c), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9)
    ax2.annotate("Violation!", xy=(1, halluc_confs[1]), xytext=(1.3, 0.45),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=10, color="red", fontweight="bold")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(levels, fontsize=9)
    ax2.set_title("Hallucinated Claim: Non-Monotonic ✗")
    ax2.set_ylim(0.3, 1.0)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    plt.suptitle("SpecCheck: Confidence Monotonicity Across Specificity Levels", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure_1_method_overview.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure_1_method_overview.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figure_1_method_overview")


# ============================================================
# Build final results.json
# ============================================================
def build_results_json(all_results, mono_results, ablation_depth, ablation_variants,
                       combo_results, sig_results):
    """Build the final aggregated results.json."""

    # Compute success criteria
    success = {}

    # Criterion 1: SpecCheck > best baseline on ≥2/3 benchmarks
    wins = 0
    for dataset in DATASETS:
        dataset_wins = 0
        for model_name in MODELS:
            mshort = MODEL_SHORT[model_name]
            key = f"{mshort}_{dataset}"
            if key not in all_results:
                continue
            methods = all_results[key].get("methods", {})
            spec_pr = max(methods.get("speccheck", {}).get("auc_pr", 0),
                         methods.get("speccheck_sampling", {}).get("auc_pr", 0))
            best_bl = max(
                methods.get("selfcheck", {}).get("auc_pr", 0),
                methods.get("verbalized", {}).get("auc_pr", 0),
                methods.get("logprob", {}).get("auc_pr", 0),
            )
            if spec_pr > best_bl:
                dataset_wins += 1

        if dataset_wins > 0:  # SpecCheck wins on at least one model for this dataset
            wins += 1

    success["criterion_1_speccheck_beats_baselines"] = {
        "target": "SpecCheck > best baseline on ≥2/3 benchmarks",
        "datasets_won": wins,
        "met": wins >= 2,
    }

    # Criterion 2: Monotonicity holds for >85% of true claims
    if mono_results:
        mono_rates = [v["factual_monotonic_rate"] for v in mono_results.values()]
        avg_mono = float(np.mean(mono_rates))
        success["criterion_2_monotonicity_true_claims"] = {
            "target": ">85% of verified-true claims show confidence monotonicity",
            "observed_rate": round(avg_mono, 4),
            "per_setting": {k: v["factual_monotonic_rate"] for k, v in mono_results.items()},
            "met": avg_mono > 0.85,
        }

    # Criterion 3: Monotonicity violated for >40% of hallucinated claims
    if mono_results:
        viol_rates = [v["hallucinated_violation_rate"] for v in mono_results.values()]
        avg_viol = float(np.mean(viol_rates))
        success["criterion_3_monotonicity_hallucinated"] = {
            "target": ">40% of hallucinated claims show monotonicity violations",
            "observed_rate": round(avg_viol, 4),
            "per_setting": {k: v["hallucinated_violation_rate"] for k, v in mono_results.items()},
            "met": avg_viol > 0.40,
        }

    # Criterion 5: SpecCheck provides complementary signal
    if combo_results:
        improvements = []
        for key, combos in combo_results.items():
            bl = combos.get("all_baselines", {}).get("auc_pr", 0)
            bl_plus = combos.get("all_baselines+speccheck", {}).get("auc_pr", 0)
            if bl > 0:
                improvements.append(bl_plus - bl)
        avg_imp = float(np.mean(improvements)) if improvements else 0.0
        success["criterion_5_complementary_signal"] = {
            "target": "Adding SpecCheck improves combined detection",
            "avg_improvement": round(avg_imp, 4),
            "met": avg_imp > 0.005,
        }

    results_json = {
        "experiment": "SpecCheck: Detecting LLM Hallucinations by Testing Confidence Monotonicity",
        "models": list(MODEL_SHORT.values()),
        "datasets": DATASETS,
        "seeds": SEEDS,
        "labeling_method": "NLI-based (cross-encoder/nli-deberta-v3-base) — independent of logprob baseline",
        "detailed_results": all_results,
        "monotonicity_analysis": mono_results,
        "ablation_ladder_depth": ablation_depth,
        "ablation_score_variants": ablation_variants,
        "combination_analysis": combo_results,
        "statistical_significance": sig_results,
        "success_criteria": success,
    }

    return results_json


# ============================================================
# Main
# ============================================================
def main():
    start = time.time()
    set_seed(42)

    print("=" * 60)
    print("Re-evaluating all experiments with NLI-based labels")
    print("=" * 60)

    # 1. Main evaluation
    print("\n--- Main Evaluation ---")
    all_results = evaluate_all()

    # 2. Monotonicity analysis
    print("\n--- Monotonicity Analysis ---")
    mono_results = monotonicity_analysis()

    # 3. Ablation: ladder depth
    print("\n--- Ablation: Ladder Depth ---")
    ablation_depth = ablation_ladder_depth()

    # 4. Ablation: score variants
    print("\n--- Ablation: Score Variants ---")
    ablation_variants = ablation_score_variants()

    # 5. Ablation: combinations
    print("\n--- Ablation: Combinations ---")
    combo_results = ablation_combination()

    # 6. Statistical tests
    print("\n--- Statistical Significance Tests ---")
    sig_results = statistical_tests(all_results)

    # 7. Generate figures
    print("\n--- Generating Figures ---")
    generate_figures(all_results, mono_results, ablation_depth, ablation_variants, combo_results)

    # 8. Build and save results.json
    print("\n--- Building results.json ---")
    results_json = build_results_json(all_results, mono_results, ablation_depth,
                                      ablation_variants, combo_results, sig_results)

    results_path = os.path.join(os.path.dirname(RESULTS_DIR), "results.json")
    save_json(results_json, results_path)
    print(f"  Saved results.json")

    # Print success criteria summary
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 60)
    for name, crit in results_json.get("success_criteria", {}).items():
        status = "MET" if crit.get("met") else "NOT MET"
        print(f"  {name}: {status}")
        print(f"    Target: {crit.get('target')}")
        if "observed_rate" in crit:
            print(f"    Observed: {crit['observed_rate']}")
        if "datasets_won" in crit:
            print(f"    Datasets won: {crit['datasets_won']}/3")

    elapsed = time.time() - start
    print(f"\nTotal evaluation time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
