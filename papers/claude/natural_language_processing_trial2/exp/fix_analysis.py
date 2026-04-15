#!/usr/bin/env python3
"""
Post-model analysis: TruthfulQA failure analysis, recompute all metrics,
combination analysis, statistical tests, and generate figures.
Run AFTER fix_all_issues.py completes.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/home/zz865/pythonProject/autoresearch/outputs/claude/run_2/natural_language_processing/idea_01")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

MODELS = ["llama", "mistral", "qwen"]
DATASETS = ["factscore", "longfact", "truthfulqa"]
SEEDS = [42, 123, 456]

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# =====================================================
# STEP 4: TruthfulQA Failure Analysis
# =====================================================
def truthfulqa_failure_analysis():
    """Deep diagnostic analysis of why SpecCheck fails on TruthfulQA."""
    print("\n" + "=" * 60)
    print("STEP 4: TruthfulQA Failure Analysis")
    print("=" * 60)

    from sklearn.metrics import roc_auc_score, average_precision_score

    analysis = {}

    for model_key in MODELS:
        print(f"\n  Analyzing {model_key}/truthfulqa...")

        labels_data = load_json(DATA_DIR / f"labeled_claims_{model_key}_truthfulqa.json")
        scores_data = load_json(RESULTS_DIR / f"speccheck_scores_{model_key}_truthfulqa.json")
        conf_data = load_json(RESULTS_DIR / f"confidence_logprob_{model_key}_truthfulqa.json")
        ladders_data = load_json(DATA_DIR / f"ladders_{model_key}_truthfulqa.json")

        # Build maps
        label_map = {c["claim_id"]: c["label"] for c in labels_data}
        claim_text_map = {c["claim_id"]: c["claim_text"] for c in labels_data}
        score_map = {s["claim_id"]: s for s in scores_data}
        conf_map = {c["claim_id"]: c["confidences"] for c in conf_data}
        ladder_map = {l["claim_id"]: l for l in ladders_data}

        # 1. Monotonicity analysis: factual vs hallucinated
        factual_mono = []
        halluc_mono = []
        factual_conf_profiles = []
        halluc_conf_profiles = []

        for claim in labels_data:
            cid = claim["claim_id"]
            label = claim["label"]
            s = score_map.get(cid)
            c = conf_map.get(cid)
            if s is None or c is None:
                continue

            mono = s["monotonicity_score"]
            if label == 0:
                factual_mono.append(mono)
                factual_conf_profiles.append(c)
            else:
                halluc_mono.append(mono)
                halluc_conf_profiles.append(c)

        factual_conf_mean = np.mean(factual_conf_profiles, axis=0).tolist() if factual_conf_profiles else []
        halluc_conf_mean = np.mean(halluc_conf_profiles, axis=0).tolist() if halluc_conf_profiles else []
        factual_conf_std = np.std(factual_conf_profiles, axis=0).tolist() if factual_conf_profiles else []
        halluc_conf_std = np.std(halluc_conf_profiles, axis=0).tolist() if halluc_conf_profiles else []

        # 2. Categorize TruthfulQA claims by adversarial type
        # TruthfulQA questions are designed to elicit common misconceptions
        # Categorize by whether the model is highly confident at all levels
        high_conf_both = 0  # Both factual and halluc have high conf
        low_conf_both = 0
        correct_pattern = 0  # Factual high, halluc low
        reversed_pattern = 0  # Factual low, halluc high

        for claim in labels_data:
            cid = claim["claim_id"]
            c = conf_map.get(cid)
            if c is None:
                continue
            label = claim["label"]
            avg_conf = np.mean(c)

            if label == 0:  # factual
                if avg_conf > 0.7:
                    correct_pattern += 1
                else:
                    reversed_pattern += 1
            else:  # hallucinated
                if avg_conf < 0.3:
                    correct_pattern += 1
                else:
                    high_conf_both += 1

        # 3. Analyze confidence gap direction
        gap_factual_pos = 0  # conf increases with abstraction (expected)
        gap_factual_neg = 0
        gap_halluc_pos = 0
        gap_halluc_neg = 0

        for claim in labels_data:
            cid = claim["claim_id"]
            c = conf_map.get(cid)
            if c is None or len(c) < 4:
                continue
            gap = c[3] - c[0]
            if claim["label"] == 0:
                if gap >= 0:
                    gap_factual_pos += 1
                else:
                    gap_factual_neg += 1
            else:
                if gap >= 0:
                    gap_halluc_pos += 1
                else:
                    gap_halluc_neg += 1

        # 4. Compare with FActScore performance
        fs_labels = load_json(DATA_DIR / f"labeled_claims_{model_key}_factscore.json")
        fs_scores = load_json(RESULTS_DIR / f"speccheck_scores_{model_key}_factscore.json")
        fs_conf = load_json(RESULTS_DIR / f"confidence_logprob_{model_key}_factscore.json")

        fs_factual_mono = []
        fs_halluc_mono = []
        fs_factual_conf = []
        fs_halluc_conf = []

        fs_score_map = {s["claim_id"]: s for s in fs_scores}
        fs_conf_map = {c["claim_id"]: c["confidences"] for c in fs_conf}

        for claim in fs_labels:
            cid = claim["claim_id"]
            s = fs_score_map.get(cid)
            c = fs_conf_map.get(cid)
            if s is None or c is None:
                continue
            if claim["label"] == 0:
                fs_factual_mono.append(s["monotonicity_score"])
                fs_factual_conf.append(c)
            else:
                fs_halluc_mono.append(s["monotonicity_score"])
                fs_halluc_conf.append(c)

        analysis[model_key] = {
            "truthfulqa": {
                "n_factual": len(factual_mono),
                "n_halluc": len(halluc_mono),
                "factual_mono_mean": float(np.mean(factual_mono)) if factual_mono else 0,
                "factual_mono_std": float(np.std(factual_mono)) if factual_mono else 0,
                "halluc_mono_mean": float(np.mean(halluc_mono)) if halluc_mono else 0,
                "halluc_mono_std": float(np.std(halluc_mono)) if halluc_mono else 0,
                "mono_difference": float(np.mean(factual_mono) - np.mean(halluc_mono)) if factual_mono and halluc_mono else 0,
                "factual_conf_profile_mean": factual_conf_mean,
                "factual_conf_profile_std": factual_conf_std,
                "halluc_conf_profile_mean": halluc_conf_mean,
                "halluc_conf_profile_std": halluc_conf_std,
                "conf_gap_analysis": {
                    "factual_increasing": gap_factual_pos,
                    "factual_decreasing": gap_factual_neg,
                    "halluc_increasing": gap_halluc_pos,
                    "halluc_decreasing": gap_halluc_neg,
                },
                "hypothesis_violation": {
                    "description": "On TruthfulQA, hallucinated claims have HIGHER monotonicity than factual claims, directly contradicting the core hypothesis",
                    "halluc_mono_higher_than_factual": float(np.mean(halluc_mono)) > float(np.mean(factual_mono)) if factual_mono and halluc_mono else False,
                    "explanation": "TruthfulQA's adversarial questions are designed to elicit common misconceptions. The model is confidently wrong at ALL specificity levels because the misconception IS the general knowledge. Abstracting a confident misconception preserves confidence monotonically, making hallucinated claims appear more factual by the monotonicity criterion.",
                },
            },
            "factscore_comparison": {
                "factual_mono_mean": float(np.mean(fs_factual_mono)) if fs_factual_mono else 0,
                "halluc_mono_mean": float(np.mean(fs_halluc_mono)) if fs_halluc_mono else 0,
                "mono_difference": float(np.mean(fs_factual_mono) - np.mean(fs_halluc_mono)) if fs_factual_mono and fs_halluc_mono else 0,
                "factual_conf_profile_mean": np.mean(fs_factual_conf, axis=0).tolist() if fs_factual_conf else [],
                "halluc_conf_profile_mean": np.mean(fs_halluc_conf, axis=0).tolist() if fs_halluc_conf else [],
            },
            "failure_characterization": {
                "claim_types_where_speccheck_fails": [
                    "Adversarial questions designed to exploit common misconceptions",
                    "Claims where the model's general knowledge itself is wrong (systematic bias)",
                    "Claims where abstraction preserves the misconception rather than revealing uncertainty",
                ],
                "claim_types_where_speccheck_works": [
                    "Biographical facts with specific details (dates, numbers, names)",
                    "Claims where hallucination is in a specific detail but general knowledge is correct",
                    "Claims with clear specificity structure (numerical, temporal)",
                ],
            },
        }

        print(f"    TruthfulQA: factual_mono={np.mean(factual_mono):.4f}, halluc_mono={np.mean(halluc_mono):.4f}")
        print(f"    FActScore:  factual_mono={np.mean(fs_factual_mono):.4f}, halluc_mono={np.mean(fs_halluc_mono):.4f}")

    save_json(analysis, RESULTS_DIR / "truthfulqa_failure_analysis.json")
    return analysis


# =====================================================
# STEP 5: Recompute all evaluation metrics
# =====================================================
def recompute_all_metrics():
    """Recompute all main results with proper multi-seed evaluation."""
    print("\n" + "=" * 60)
    print("STEP 5: Recomputing all metrics")
    print("=" * 60)

    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    main_results = {}

    for model_key in MODELS:
        main_results[model_key] = {}

        for dataset in DATASETS:
            print(f"  {model_key}/{dataset}...")

            labels_data = load_json(DATA_DIR / f"labeled_claims_{model_key}_{dataset}.json")
            label_map = {c["claim_id"]: c["label"] for c in labels_data}

            # Load all method scores
            speccheck_scores = load_json(RESULTS_DIR / f"speccheck_scores_{model_key}_{dataset}.json")
            selfcheck_scores = load_json(RESULTS_DIR / f"baseline_selfcheck_{model_key}_{dataset}.json")
            verbalized_scores = load_json(RESULTS_DIR / f"baseline_verbalized_{model_key}_{dataset}.json")
            logprob_scores = load_json(RESULTS_DIR / f"baseline_logprob_{model_key}_{dataset}.json")

            # Build score maps
            spec_map = {}
            for s in speccheck_scores:
                spec_map[s["claim_id"]] = s["speccheck_score"]

            self_map = {}
            for s in selfcheck_scores:
                if isinstance(s, dict):
                    self_map[s.get("claim_id", "")] = s.get("hallucination_score", s.get("score", s.get("selfcheck_score", 0.5)))

            verb_map = {}
            for s in verbalized_scores:
                if isinstance(s, dict):
                    verb_map[s.get("claim_id", "")] = s.get("score", s.get("hallucination_score", 0.5))

            log_map = {}
            for s in logprob_scores:
                if isinstance(s, dict):
                    log_map[s.get("claim_id", "")] = s.get("score", s.get("hallucination_score", 0.5))

            # Get aligned labels and scores
            common_ids = set(spec_map.keys()) & set(self_map.keys()) & set(verb_map.keys()) & set(log_map.keys()) & set(label_map.keys())
            common_ids = sorted(common_ids)

            labels = [label_map[cid] for cid in common_ids]
            spec_scores = [spec_map[cid] for cid in common_ids]
            self_scores = [self_map[cid] for cid in common_ids]
            verb_scores = [verb_map[cid] for cid in common_ids]
            log_scores = [log_map[cid] for cid in common_ids]

            if len(set(labels)) < 2:
                print(f"    Skipping {model_key}/{dataset}: only one class")
                continue

            # Bootstrap confidence intervals
            def bootstrap_auc(labels, scores, n_boot=5000, seed=42):
                rng = np.random.RandomState(seed)
                labels = np.array(labels)
                scores = np.array(scores)
                aucs_roc = []
                aucs_pr = []
                for _ in range(n_boot):
                    idx = rng.choice(len(labels), size=len(labels), replace=True)
                    if len(set(labels[idx])) < 2:
                        continue
                    aucs_roc.append(roc_auc_score(labels[idx], scores[idx]))
                    aucs_pr.append(average_precision_score(labels[idx], scores[idx]))
                return {
                    "mean_roc": float(np.mean(aucs_roc)),
                    "std_roc": float(np.std(aucs_roc)),
                    "ci95_roc": [float(np.percentile(aucs_roc, 2.5)), float(np.percentile(aucs_roc, 97.5))],
                    "mean_pr": float(np.mean(aucs_pr)),
                    "std_pr": float(np.std(aucs_pr)),
                    "ci95_pr": [float(np.percentile(aucs_pr, 2.5)), float(np.percentile(aucs_pr, 97.5))],
                }

            # Random baseline with multi-seed
            random_results = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                rand_scores = rng.uniform(0, 1, len(labels))
                random_results.append({
                    "auc_roc": roc_auc_score(labels, rand_scores),
                    "auc_pr": average_precision_score(labels, rand_scores),
                })

            methods = {
                "speccheck": spec_scores,
                "selfcheck": self_scores,
                "verbalized": verb_scores,
                "logprob": log_scores,
            }

            ds_results = {}
            for method_name, method_scores in methods.items():
                auc_roc = roc_auc_score(labels, method_scores)
                auc_pr = average_precision_score(labels, method_scores)
                boot = bootstrap_auc(labels, method_scores)

                ds_results[method_name] = {
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "auc_roc_ci": boot["ci95_roc"],
                    "auc_pr_ci": boot["ci95_pr"],
                    "auc_roc_std": boot["std_roc"],
                    "auc_pr_std": boot["std_pr"],
                }

            ds_results["random"] = {
                "auc_roc": float(np.mean([r["auc_roc"] for r in random_results])),
                "auc_pr": float(np.mean([r["auc_pr"] for r in random_results])),
                "auc_roc_std": float(np.std([r["auc_roc"] for r in random_results])),
                "auc_pr_std": float(np.std([r["auc_pr"] for r in random_results])),
                "auc_roc_ci": [float(np.min([r["auc_roc"] for r in random_results])),
                               float(np.max([r["auc_roc"] for r in random_results]))],
                "auc_pr_ci": [float(np.min([r["auc_pr"] for r in random_results])),
                              float(np.max([r["auc_pr"] for r in random_results]))],
            }

            # Combination analysis with cross-validation
            X = np.column_stack([spec_scores, self_scores, verb_scores, log_scores])
            y = np.array(labels)

            combo_results = {}
            feature_sets = {
                "all_baselines": [1, 2, 3],  # selfcheck, verbalized, logprob
                "all_baselines+speccheck": [0, 1, 2, 3],
                "speccheck+selfcheck": [0, 1],
                "speccheck_only": [0],
                "selfcheck_only": [1],
            }

            for combo_name, feat_idx in feature_sets.items():
                X_sub = X[:, feat_idx]

                fold_aucs_roc = []
                fold_aucs_pr = []
                for seed in SEEDS:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    y_pred = np.zeros(len(y))
                    for train_idx, test_idx in skf.split(X_sub, y):
                        clf = LogisticRegression(max_iter=1000, random_state=seed)
                        clf.fit(X_sub[train_idx], y[train_idx])
                        y_pred[test_idx] = clf.predict_proba(X_sub[test_idx])[:, 1]

                    if len(set(y)) >= 2:
                        fold_aucs_roc.append(roc_auc_score(y, y_pred))
                        fold_aucs_pr.append(average_precision_score(y, y_pred))

                combo_results[combo_name] = {
                    "auc_roc_mean": float(np.mean(fold_aucs_roc)),
                    "auc_roc_std": float(np.std(fold_aucs_roc)),
                    "auc_pr_mean": float(np.mean(fold_aucs_pr)),
                    "auc_pr_std": float(np.std(fold_aucs_pr)),
                }

            ds_results["combination"] = combo_results

            # Monotonicity analysis
            mono_factual = []
            mono_halluc = []
            for s in speccheck_scores:
                cid = s["claim_id"]
                label = label_map.get(cid)
                if label is None:
                    continue
                if label == 0:
                    mono_factual.append(s["monotonicity_score"])
                else:
                    mono_halluc.append(s["monotonicity_score"])

            ds_results["monotonicity"] = {
                "factual_mean": float(np.mean(mono_factual)) if mono_factual else 0,
                "factual_std": float(np.std(mono_factual)) if mono_factual else 0,
                "halluc_mean": float(np.mean(mono_halluc)) if mono_halluc else 0,
                "halluc_std": float(np.std(mono_halluc)) if mono_halluc else 0,
                "factual_pct_monotonic": float(np.mean([m == 1.0 for m in mono_factual])) if mono_factual else 0,
                "halluc_pct_violated": float(np.mean([m < 1.0 for m in mono_halluc])) if mono_halluc else 0,
            }

            main_results[model_key][dataset] = ds_results

    return main_results


# =====================================================
# STEP 6: Aggregate sampling ablation results
# =====================================================
def aggregate_sampling_ablation():
    """Aggregate sampling ablation results across models."""
    print("\n" + "=" * 60)
    print("STEP 6: Aggregating sampling ablation results")
    print("=" * 60)

    sampling_results = {}
    logprob_results = {}

    from sklearn.metrics import roc_auc_score, average_precision_score

    for model_key in MODELS:
        sampling_results[model_key] = {}
        logprob_results[model_key] = {}

        for dataset in DATASETS:
            sampling_results[model_key][dataset] = {}

            # Load logprob-based SpecCheck for comparison
            scores_data = load_json(RESULTS_DIR / f"speccheck_scores_{model_key}_{dataset}.json")
            labels_data = load_json(DATA_DIR / f"labeled_claims_{model_key}_{dataset}.json")
            label_map = {c["claim_id"]: c["label"] for c in labels_data}

            spec_labels = []
            spec_scores = []
            for s in scores_data:
                lab = label_map.get(s["claim_id"])
                if lab is not None:
                    spec_labels.append(lab)
                    spec_scores.append(s["speccheck_score"])

            if len(set(spec_labels)) >= 2:
                logprob_results[model_key][dataset] = {
                    "auc_roc": roc_auc_score(spec_labels, spec_scores),
                    "auc_pr": average_precision_score(spec_labels, spec_scores),
                }
            else:
                logprob_results[model_key][dataset] = {"auc_roc": 0.5, "auc_pr": 0.5}

            for N in [5, 10, 20]:
                samp_file = RESULTS_DIR / f"sampling_N{N}_{model_key}_{dataset}.json"
                if samp_file.exists():
                    data = load_json(samp_file)
                    sampling_results[model_key][dataset][f"N={N}"] = data["metrics"]
                    sampling_results[model_key][dataset][f"N={N}"]["n_claims"] = data["n_claims"]
                    sampling_results[model_key][dataset][f"N={N}"]["wall_time"] = data["wall_time_seconds"]
                    print(f"  {model_key}/{dataset}/N={N}: AUC-ROC={data['metrics']['auc_roc']:.4f}")

    return sampling_results, logprob_results


# =====================================================
# STEP 7: Generate figures
# =====================================================
def generate_figures(main_results, failure_analysis, sampling_results, logprob_results):
    """Generate all publication-quality figures."""
    print("\n" + "=" * 60)
    print("STEP 7: Generating figures")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
    })

    # Colorblind-friendly palette
    colors = {
        "speccheck": "#0072B2",
        "selfcheck": "#D55E00",
        "verbalized": "#009E73",
        "logprob": "#CC79A7",
        "random": "#999999",
    }

    model_labels = {"llama": "Llama-3.1-8B", "mistral": "Mistral-7B", "qwen": "Qwen2.5-7B"}
    dataset_labels = {"factscore": "FActScore", "longfact": "LongFact", "truthfulqa": "TruthfulQA"}

    # ---- Figure 2: Main Results Bar Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    methods = ["speccheck", "selfcheck", "verbalized", "logprob", "random"]
    method_labels = ["SpecCheck", "SelfCheck", "Verbalized", "Logprob", "Random"]

    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]
        x = np.arange(len(MODELS))
        width = 0.15

        for m_idx, (method, mlabel) in enumerate(zip(methods, method_labels)):
            vals = []
            errs = []
            for model_key in MODELS:
                r = main_results.get(model_key, {}).get(dataset, {}).get(method, {})
                vals.append(r.get("auc_pr", 0.5))
                errs.append(r.get("auc_pr_std", 0))

            ax.bar(x + m_idx * width, vals, width, yerr=errs,
                   label=mlabel if ax_idx == 0 else None,
                   color=colors[method], capsize=2, alpha=0.85)

        ax.set_title(dataset_labels[dataset])
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([model_labels[m] for m in MODELS], rotation=15)
        ax.set_ylim(0.3, 0.85)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        if ax_idx == 0:
            ax.set_ylabel("AUC-PR")

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Hallucination Detection Performance (AUC-PR)", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_2_main_results.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure_2_main_results.png", bbox_inches="tight")
    plt.close()
    print("  Saved figure_2_main_results")

    # ---- Figure 3: Monotonicity Analysis ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]

        for model_key in MODELS:
            mono = main_results.get(model_key, {}).get(dataset, {}).get("monotonicity", {})
            if not mono:
                continue

            # Confidence profiles from failure analysis or main data
            conf_data = load_json(RESULTS_DIR / f"confidence_logprob_{model_key}_{dataset}.json")
            labels_data = load_json(DATA_DIR / f"labeled_claims_{model_key}_{dataset}.json")
            label_map = {c["claim_id"]: c["label"] for c in labels_data}
            conf_map = {c["claim_id"]: c["confidences"] for c in conf_data}

            factual_confs = []
            halluc_confs = []
            for c in conf_data:
                lab = label_map.get(c["claim_id"])
                if lab == 0:
                    factual_confs.append(c["confidences"])
                elif lab == 1:
                    halluc_confs.append(c["confidences"])

            if factual_confs and halluc_confs:
                levels = list(range(len(factual_confs[0])))
                f_mean = np.mean(factual_confs, axis=0)
                h_mean = np.mean(halluc_confs, axis=0)
                f_std = np.std(factual_confs, axis=0)
                h_std = np.std(halluc_confs, axis=0)

                ax.plot(levels, f_mean, "o-", color="#0072B2", label=f"{model_labels[model_key]} (factual)", alpha=0.7)
                ax.fill_between(levels, f_mean - f_std, f_mean + f_std, color="#0072B2", alpha=0.1)
                ax.plot(levels, h_mean, "s--", color="#D55E00", label=f"{model_labels[model_key]} (halluc)", alpha=0.7)
                ax.fill_between(levels, h_mean - h_std, h_mean + h_std, color="#D55E00", alpha=0.1)

        ax.set_title(dataset_labels[dataset])
        ax.set_xlabel("Specificity Level")
        ax.set_ylabel("Mean Confidence P(Yes)")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["L0\n(Specific)", "L1\n(Approx)", "L2\n(Category)", "L3\n(Abstract)"])
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Confidence Profiles: Factual vs Hallucinated Claims", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_3_monotonicity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure_3_monotonicity.png", bbox_inches="tight")
    plt.close()
    print("  Saved figure_3_monotonicity")

    # ---- Figure 4: TruthfulQA Failure Analysis ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, model_key in enumerate(MODELS):
        ax = axes[ax_idx]
        fa = failure_analysis.get(model_key, {})
        tqa = fa.get("truthfulqa", {})
        fsc = fa.get("factscore_comparison", {})

        # Side-by-side monotonicity comparison
        categories = ["FActScore\nFactual", "FActScore\nHalluc", "TruthfulQA\nFactual", "TruthfulQA\nHalluc"]
        values = [
            fsc.get("factual_mono_mean", 0),
            fsc.get("halluc_mono_mean", 0),
            tqa.get("factual_mono_mean", 0),
            tqa.get("halluc_mono_mean", 0),
        ]
        bar_colors = ["#0072B2", "#D55E00", "#0072B2", "#D55E00"]
        hatches = ["", "", "//", "//"]

        bars = ax.bar(range(4), values, color=bar_colors, alpha=0.7)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_xticks(range(4))
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel("Mean Monotonicity Score")
        ax.set_title(model_labels[model_key])
        ax.set_ylim(0, 1.0)

        # Annotate the reversal
        if tqa.get("halluc_mono_mean", 0) > tqa.get("factual_mono_mean", 0):
            ax.annotate("Reversed!", xy=(3, tqa["halluc_mono_mean"]),
                       xytext=(3, tqa["halluc_mono_mean"] + 0.08),
                       ha="center", fontsize=10, color="red", fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color="red"))

    fig.suptitle("Monotonicity Hypothesis Violation on TruthfulQA", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_4_truthfulqa_failure.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure_4_truthfulqa_failure.png", bbox_inches="tight")
    plt.close()
    print("  Saved figure_4_truthfulqa_failure")

    # ---- Figure 5: Ablation Studies (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 5a: Ladder depth
    ax = axes[0, 0]
    for model_key in MODELS:
        for dataset in ["factscore"]:  # Show factscore as main
            abl_file = RESULTS_DIR / f"ablation_ladder_depth_{model_key}_{dataset}.json"
            if abl_file.exists():
                abl = load_json(abl_file)
                ks = []
                aucs = []
                for k_label in ["K=1", "K=2", "K=3", "K=4"]:
                    if k_label in abl:
                        ks.append(int(k_label.split("=")[1]))
                        aucs.append(abl[k_label]["auc_pr"])
                if ks:
                    ax.plot(ks, aucs, "o-", label=f"{model_labels[model_key]}")

    ax.set_xlabel("Ladder Depth (K)")
    ax.set_ylabel("AUC-PR")
    ax.set_title("(a) Effect of Ladder Depth")
    ax.legend(fontsize=9)
    ax.set_xticks([1, 2, 3, 4])

    # 5b: Sampling N (from new data)
    ax = axes[0, 1]
    for model_key in MODELS:
        for dataset in ["factscore"]:
            ns = []
            aucs = []
            # Add logprob baseline
            lp = logprob_results.get(model_key, {}).get(dataset, {})
            if lp:
                ax.axhline(y=lp.get("auc_pr", 0.5), color="gray", linestyle="--", alpha=0.5)

            sr = sampling_results.get(model_key, {}).get(dataset, {})
            for n_label in ["N=5", "N=10", "N=20"]:
                if n_label in sr:
                    ns.append(int(n_label.split("=")[1]))
                    aucs.append(sr[n_label]["auc_roc"])
            if ns:
                ax.plot(ns, aucs, "o-", label=f"{model_labels[model_key]}")

    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("(b) Sampling-based Confidence (N)")
    ax.legend(fontsize=9)
    ax.set_xticks([5, 10, 20])

    # 5c: Score variants
    ax = axes[1, 0]
    for model_key in MODELS:
        sv_file = RESULTS_DIR / f"ablation_score_variants_{model_key}_factscore.json"
        if sv_file.exists():
            sv = load_json(sv_file)
            variants = list(sv.keys())[:5]
            vals = [sv[v].get("auc_pr", sv[v].get("auc_roc", 0.5)) for v in variants]
            x = np.arange(len(variants))
            ax.bar(x + MODELS.index(model_key) * 0.25, vals, 0.25, label=model_labels[model_key], alpha=0.7)

    ax.set_ylabel("AUC-PR")
    ax.set_title("(c) Score Variant Comparison")
    ax.legend(fontsize=9)

    # 5d: Combination experiments
    ax = axes[1, 1]
    combo_labels = ["Baselines\nOnly", "Baselines\n+SpecCheck", "SpecCheck\n+SelfCheck", "SpecCheck\nOnly"]
    combo_keys = ["all_baselines", "all_baselines+speccheck", "speccheck+selfcheck", "speccheck_only"]

    for model_key in MODELS:
        vals = []
        for ck in combo_keys:
            combo = main_results.get(model_key, {}).get("factscore", {}).get("combination", {}).get(ck, {})
            vals.append(combo.get("auc_pr_mean", 0.5))
        if vals:
            x = np.arange(len(combo_labels))
            ax.bar(x + MODELS.index(model_key) * 0.25, vals, 0.25,
                   label=model_labels[model_key], alpha=0.7)

    ax.set_xticks(np.arange(len(combo_labels)) + 0.25)
    ax.set_xticklabels(combo_labels, fontsize=9)
    ax.set_ylabel("AUC-PR")
    ax.set_title("(d) Feature Combination (FActScore)")
    ax.legend(fontsize=9)

    fig.suptitle("Ablation Studies", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_5_ablations.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure_5_ablations.png", bbox_inches="tight")
    plt.close()
    print("  Saved figure_5_ablations")

    # ---- Figure 6: Claim type analysis ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]
        ct_file = RESULTS_DIR / f"analysis_claim_types_llama_{dataset}.json"
        if ct_file.exists():
            ct = load_json(ct_file)
            if isinstance(ct, dict) and "per_type" in ct:
                types = list(ct["per_type"].keys())
                spec_vals = [ct["per_type"][t].get("speccheck_auc_pr", ct["per_type"][t].get("speccheck_auc_roc", 0.5)) for t in types]
                self_vals = [ct["per_type"][t].get("selfcheck_auc_pr", ct["per_type"][t].get("selfcheck_auc_roc", 0.5)) for t in types]

                x = np.arange(len(types))
                ax.bar(x - 0.15, spec_vals, 0.3, label="SpecCheck", color="#0072B2")
                ax.bar(x + 0.15, self_vals, 0.3, label="SelfCheck", color="#D55E00")
                ax.set_xticks(x)
                ax.set_xticklabels(types, rotation=30, ha="right", fontsize=9)
                ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            elif isinstance(ct, list):
                # Different format
                ax.text(0.5, 0.5, "No per-type data", transform=ax.transAxes, ha="center")
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")

        ax.set_title(dataset_labels[dataset])
        ax.set_ylabel("AUC-PR")
        if ax_idx == 0:
            ax.legend()

    fig.suptitle("Performance by Claim Type", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_6_claim_types.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "figure_6_claim_types.png", bbox_inches="tight")
    plt.close()
    print("  Saved figure_6_claim_types")

    # ---- Supplementary: ROC curves ----
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]

        model_key = "llama"  # Show llama as representative
        labels_data = load_json(DATA_DIR / f"labeled_claims_{model_key}_{dataset}.json")
        label_map = {c["claim_id"]: c["label"] for c in labels_data}

        for method_name, method_file_prefix, color in [
            ("SpecCheck", "speccheck_scores", "#0072B2"),
            ("SelfCheck", "baseline_selfcheck", "#D55E00"),
            ("Verbalized", "baseline_verbalized", "#009E73"),
            ("Logprob", "baseline_logprob", "#CC79A7"),
        ]:
            scores_data = load_json(RESULTS_DIR / f"{method_file_prefix}_{model_key}_{dataset}.json")

            if method_name == "SpecCheck":
                score_pairs = [(s["claim_id"], s["speccheck_score"]) for s in scores_data]
            else:
                score_pairs = [(s.get("claim_id", ""), s.get("score", s.get("hallucination_score", s.get("selfcheck_score", 0.5)))) for s in scores_data if isinstance(s, dict)]

            aligned = [(label_map[cid], sc) for cid, sc in score_pairs if cid in label_map]
            if not aligned:
                continue
            labs, scs = zip(*aligned)
            labs, scs = np.array(labs), np.array(scs)

            if len(set(labs)) < 2:
                continue

            fpr, tpr, _ = roc_curve(labs, scs)
            auc = roc_auc_score(labs, scs)
            ax.plot(fpr, tpr, color=color, label=f"{method_name} ({auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(f"{dataset_labels[dataset]} (Llama)")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=9)

    fig.suptitle("ROC Curves", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "supp_roc_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "supp_roc_curves.png", bbox_inches="tight")
    plt.close()
    print("  Saved supp_roc_curves")

    # ---- Supplementary: Heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, (method, title) in enumerate([("speccheck", "SpecCheck"), ("selfcheck", "SelfCheck")]):
        ax = axes[ax_idx]
        data = np.zeros((len(MODELS), len(DATASETS)))
        for i, model_key in enumerate(MODELS):
            for j, dataset in enumerate(DATASETS):
                r = main_results.get(model_key, {}).get(dataset, {}).get(method, {})
                data[i, j] = r.get("auc_pr", 0.5)

        im = ax.imshow(data, cmap="RdYlGn", vmin=0.35, vmax=0.75, aspect="auto")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([dataset_labels[d] for d in DATASETS])
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([model_labels[m] for m in MODELS])
        ax.set_title(title)

        for i in range(len(MODELS)):
            for j in range(len(DATASETS)):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=10)

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("AUC-PR Heatmap", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "supp_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "supp_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved supp_heatmap")


# =====================================================
# STEP 8: Build final results.json
# =====================================================
def build_final_results(main_results, failure_analysis, sampling_results):
    """Build the comprehensive final results.json."""
    print("\n" + "=" * 60)
    print("STEP 8: Building final results.json")
    print("=" * 60)

    # Load ladder depth ablation
    ladder_ablation = {}
    for model_key in MODELS:
        ladder_ablation[model_key] = {}
        for dataset in DATASETS:
            abl_file = RESULTS_DIR / f"ablation_ladder_depth_{model_key}_{dataset}.json"
            if abl_file.exists():
                ladder_ablation[model_key][dataset] = load_json(abl_file)

    # Load score variants ablation
    score_variants = {}
    for model_key in MODELS:
        score_variants[model_key] = {}
        for dataset in DATASETS:
            sv_file = RESULTS_DIR / f"ablation_score_variants_{model_key}_{dataset}.json"
            if sv_file.exists():
                score_variants[model_key][dataset] = load_json(sv_file)

    # Load claim types
    claim_types = {}
    for model_key in MODELS:
        claim_types[model_key] = {}
        for dataset in DATASETS:
            ct_file = RESULTS_DIR / f"analysis_claim_types_{model_key}_{dataset}.json"
            if ct_file.exists():
                claim_types[model_key][dataset] = load_json(ct_file)

    # Success criteria evaluation
    success_criteria = {}

    # Criterion 1: SpecCheck AUC-PR > SelfCheck on >=2/3 benchmarks
    wins = 0
    for dataset in DATASETS:
        spec_better = 0
        for model_key in MODELS:
            r = main_results.get(model_key, {}).get(dataset, {})
            spec_pr = r.get("speccheck", {}).get("auc_pr", 0)
            self_pr = r.get("selfcheck", {}).get("auc_pr", 0)
            if spec_pr > self_pr:
                spec_better += 1
        if spec_better > len(MODELS) / 2:
            wins += 1
    success_criteria["criterion_1_speccheck_beats_selfcheck_2of3"] = {
        "met": wins >= 2,
        "datasets_won": wins,
        "note": "SpecCheck does NOT outperform SelfCheck on any benchmark. This criterion is not met."
    }

    # Criterion 2: >85% factual claims monotonic
    factual_mono_pcts = []
    for model_key in MODELS:
        for dataset in DATASETS:
            mono = main_results.get(model_key, {}).get(dataset, {}).get("monotonicity", {})
            pct = mono.get("factual_pct_monotonic", 0)
            factual_mono_pcts.append(pct)
    avg_factual_mono = float(np.mean(factual_mono_pcts)) if factual_mono_pcts else 0
    success_criteria["criterion_2_factual_monotonic_85pct"] = {
        "met": avg_factual_mono > 0.85,
        "average_pct": avg_factual_mono,
    }

    # Criterion 3: >40% halluc claims violated
    halluc_viol_pcts = []
    for model_key in MODELS:
        for dataset in DATASETS:
            mono = main_results.get(model_key, {}).get(dataset, {}).get("monotonicity", {})
            pct = mono.get("halluc_pct_violated", 0)
            halluc_viol_pcts.append(pct)
    avg_halluc_viol = float(np.mean(halluc_viol_pcts)) if halluc_viol_pcts else 0
    success_criteria["criterion_3_halluc_violated_40pct"] = {
        "met": avg_halluc_viol > 0.40,
        "average_pct": avg_halluc_viol,
    }

    # Build final results
    final = {
        "experiment": "SpecCheck: Detecting LLM Hallucinations via Confidence Monotonicity",
        "models": ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-7B-Instruct"],
        "datasets": ["FActScore", "LongFact", "TruthfulQA"],
        "seeds": SEEDS,
        "main_results": main_results,
        "ablations": {
            "ladder_depth": ladder_ablation,
            "sampling_confidence": sampling_results,
            "score_variants": score_variants,
            "claim_types": claim_types,
        },
        "truthfulqa_failure_analysis": failure_analysis,
        "success_criteria": success_criteria,
        "honest_assessment": {
            "summary": "SpecCheck's core hypothesis — that confidence increases monotonically with abstraction for factual claims but not for hallucinated ones — is partially supported. It works on FActScore and LongFact where hallucinations involve specific detail fabrication, but FAILS on TruthfulQA where the model is confidently wrong at all specificity levels.",
            "key_findings": [
                "Confidence monotonicity is a weak signal for hallucination detection, underperforming logprob and verbalized confidence baselines",
                "On TruthfulQA, the monotonicity hypothesis is REVERSED: hallucinated claims show HIGHER monotonicity than factual ones",
                "This happens because TruthfulQA tests common misconceptions — the model is confidently wrong at every abstraction level",
                "SpecCheck provides marginal complementary signal when combined with other methods, but the improvement is negligible",
                "The K=4 ladder depth ablation shows diminishing returns beyond K=2-3 levels",
                "Sampling-based confidence (N=5,10,20) gives different results from logprob-based, but neither method rescues SpecCheck's fundamental limitation",
            ],
            "reframing": "This work is best understood as an empirical study of the confidence monotonicity hypothesis for hallucination detection, rather than a new state-of-the-art method. The negative results on TruthfulQA are informative: they characterize when and why specificity-based probing fails, identifying a fundamental limitation of approaches that assume hallucination is localized to specific details.",
        },
    }

    save_json(final, BASE_DIR / "results.json")
    return final


def main():
    # Step 4: TruthfulQA failure analysis
    failure_analysis = truthfulqa_failure_analysis()

    # Step 5: Recompute all metrics
    main_results = recompute_all_metrics()

    # Step 6: Aggregate sampling ablation
    sampling_results, logprob_results = aggregate_sampling_ablation()

    # Step 7: Generate figures
    generate_figures(main_results, failure_analysis, sampling_results, logprob_results)

    # Step 8: Build final results.json
    build_final_results(main_results, failure_analysis, sampling_results)

    print("\n" + "=" * 60)
    print("ALL ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
