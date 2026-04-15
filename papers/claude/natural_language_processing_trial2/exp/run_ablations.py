"""
Ablation studies for SpecCheck:
1. Ladder depth (K=1,2,3,4)
2. Confidence estimation method (sampling vs logprob)
3. Claim type analysis
4. SpecCheck + baselines combination
5. Score variant comparison
"""
import os
import sys
import json
import re
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json, get_model_short,
    SEEDS, MODELS, DATASETS, DATA_DIR, RESULTS_DIR
)
from shared.metrics import compute_auc_roc, compute_auc_pr, compute_all_metrics


def ablation_ladder_depth(model_name, dataset_name):
    """Ablation 1: Vary ladder depth K=1,2,3."""
    mshort = get_model_short(model_name)
    conf_path = os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{dataset_name}.json")
    label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")

    if not os.path.exists(conf_path) or not os.path.exists(label_path):
        return None

    conf_data = load_json(conf_path)
    labeled = load_json(label_path)
    label_map = {c["claim_id"]: c["label"] for c in labeled}

    results = {}
    for K in [1, 2, 3]:
        scores = []
        labels = []
        for item in conf_data:
            cid = item["claim_id"]
            if cid not in label_map:
                continue
            confs = item["confidences"]
            if len(confs) < K + 1:
                continue

            if K == 1:
                # Only use levels 0 and 3
                selected = [confs[0], confs[-1]]
            elif K == 2:
                # Levels 0, 2, 3
                selected = [confs[0], confs[min(2, len(confs)-1)], confs[-1]]
            else:
                # All 4 levels (K=3 transitions)
                selected = confs[:4]

            n_trans = len(selected) - 1
            violations = sum(1 for k in range(1, len(selected)) if selected[k] < selected[k-1])
            mono_score = 1.0 - violations / n_trans
            gap = selected[-1] - selected[0]
            spec_score = (1.0 - mono_score) + 0.5 * max(0, -gap)

            scores.append(spec_score)
            labels.append(label_map[cid])

        if len(scores) > 10:
            auc_roc = compute_auc_roc(labels, scores)
            auc_pr = compute_auc_pr(labels, scores)
            results[f"K={K}"] = {
                "auc_roc": round(auc_roc, 4),
                "auc_pr": round(auc_pr, 4),
                "n_claims": len(scores),
            }

    out_path = os.path.join(RESULTS_DIR, f"ablation_ladder_depth_{mshort}_{dataset_name}.json")
    save_json(results, out_path)
    print(f"  Ladder depth ablation ({mshort}/{dataset_name}): {results}")
    return results


def ablation_score_variants(model_name, dataset_name):
    """Ablation 5: Compare different scoring methods and alpha values."""
    mshort = get_model_short(model_name)
    scores_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json")
    label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")

    if not os.path.exists(scores_path) or not os.path.exists(label_path):
        return None

    scores_data = load_json(scores_path)
    labeled = load_json(label_path)
    label_map = {c["claim_id"]: c["label"] for c in labeled}

    # Collect score variants
    variant_scores = {"speccheck": [], "monotonicity": [], "max_violation": [],
                      "gap_score": [], "weighted_violation": []}
    labels = []

    for item in scores_data:
        cid = item["claim_id"]
        if cid not in label_map:
            continue
        labels.append(label_map[cid])
        variant_scores["speccheck"].append(item["speccheck_score"])
        variant_scores["monotonicity"].append(1.0 - item["monotonicity_score"])
        variant_scores["max_violation"].append(item["max_violation"])
        variant_scores["gap_score"].append(item["gap_score"])
        variant_scores["weighted_violation"].append(item["weighted_violation"])

    results = {}
    for name, scores in variant_scores.items():
        if len(scores) > 10:
            results[name] = {
                "auc_roc": round(compute_auc_roc(labels, scores), 4),
                "auc_pr": round(compute_auc_pr(labels, scores), 4),
            }

    # Alpha sweep
    alpha_results = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        alpha_scores = []
        for item in scores_data:
            cid = item["claim_id"]
            if cid not in label_map:
                continue
            confs = item["confidences"]
            n_trans = len(confs) - 1
            violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
            mono = 1.0 - violations / n_trans
            gap = confs[-1] - confs[0]
            score = (1.0 - mono) + alpha * max(0, -gap)
            alpha_scores.append(score)

        if len(alpha_scores) > 10:
            alpha_results[f"alpha={alpha}"] = {
                "auc_roc": round(compute_auc_roc(labels, alpha_scores), 4),
                "auc_pr": round(compute_auc_pr(labels, alpha_scores), 4),
            }
    results["alpha_sweep"] = alpha_results

    out_path = os.path.join(RESULTS_DIR, f"ablation_score_variants_{mshort}_{dataset_name}.json")
    save_json(results, out_path)
    print(f"  Score variants ({mshort}/{dataset_name}): {json.dumps(results, indent=2)}")
    return results


def ablation_claim_types(model_name, dataset_name):
    """Ablation 3: Per-claim-type analysis."""
    mshort = get_model_short(model_name)
    label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")
    scores_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json")
    selfcheck_path = os.path.join(RESULTS_DIR, f"baseline_selfcheck_{mshort}_{dataset_name}.json")

    if not all(os.path.exists(p) for p in [label_path, scores_path]):
        return None

    labeled = load_json(label_path)
    scores_data = load_json(scores_path)
    selfcheck_data = load_json(selfcheck_path) if os.path.exists(selfcheck_path) else []

    label_map = {c["claim_id"]: c["label"] for c in labeled}
    spec_map = {s["claim_id"]: s["speccheck_score"] for s in scores_data}
    self_map = {s["claim_id"]: s["hallucination_score"] for s in selfcheck_data}

    # Categorize claims
    categories = {}
    for claim in labeled:
        text = claim["claim_text"]
        if re.search(r'\b\d{4}\b|\b\d+\s*(years?|months?|days?)', text):
            cat = "temporal"
        elif re.search(r'\b\d+(\.\d+)?(%|\s*(meters?|km|miles?|kg|pounds?|dollars?|million|billion))', text):
            cat = "numerical"
        elif re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text):
            cat = "entity"
        elif re.search(r'(is a|was a|are |were |belongs? to|part of)', text.lower()):
            cat = "relational"
        else:
            cat = "general"

        if cat not in categories:
            categories[cat] = {"labels": [], "speccheck": [], "selfcheck": []}
        cid = claim["claim_id"]
        if cid in spec_map and cid in label_map:
            categories[cat]["labels"].append(label_map[cid])
            categories[cat]["speccheck"].append(spec_map[cid])
            categories[cat]["selfcheck"].append(self_map.get(cid, 0.5))

    results = {}
    for cat, data in categories.items():
        if len(data["labels"]) > 10 and len(set(data["labels"])) >= 2:
            results[cat] = {
                "n_claims": len(data["labels"]),
                "halluc_rate": round(np.mean(data["labels"]), 3),
                "speccheck_auc_pr": round(compute_auc_pr(data["labels"], data["speccheck"]), 4),
                "selfcheck_auc_pr": round(compute_auc_pr(data["labels"], data["selfcheck"]), 4),
                "speccheck_auc_roc": round(compute_auc_roc(data["labels"], data["speccheck"]), 4),
                "selfcheck_auc_roc": round(compute_auc_roc(data["labels"], data["selfcheck"]), 4),
            }

    out_path = os.path.join(RESULTS_DIR, f"analysis_claim_types_{mshort}_{dataset_name}.json")
    save_json(results, out_path)
    print(f"  Claim types ({mshort}/{dataset_name}): {list(results.keys())}")
    return results


def ablation_combination(model_name, dataset_name):
    """Ablation 4: Combine SpecCheck with baselines via logistic regression."""
    mshort = get_model_short(model_name)
    label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")

    if not os.path.exists(label_path):
        return None

    labeled = load_json(label_path)
    label_map = {c["claim_id"]: c["label"] for c in labeled}

    # Load all score files
    methods = {
        "speccheck": os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json"),
        "selfcheck": os.path.join(RESULTS_DIR, f"baseline_selfcheck_{mshort}_{dataset_name}.json"),
        "verbalized": os.path.join(RESULTS_DIR, f"baseline_verbalized_{mshort}_{dataset_name}.json"),
        "logprob": os.path.join(RESULTS_DIR, f"baseline_logprob_{mshort}_{dataset_name}.json"),
    }

    score_maps = {}
    for name, path in methods.items():
        if os.path.exists(path):
            data = load_json(path)
            key = "speccheck_score" if name == "speccheck" else "hallucination_score"
            score_maps[name] = {d["claim_id"]: d.get(key, d.get("hallucination_score", 0.5)) for d in data}

    # Align claims across all methods
    common_ids = set(label_map.keys())
    for sm in score_maps.values():
        common_ids &= set(sm.keys())
    common_ids = sorted(common_ids)

    if len(common_ids) < 20:
        print(f"  Not enough common claims for combination: {len(common_ids)}")
        return None

    labels = np.array([label_map[cid] for cid in common_ids])
    feature_names = sorted(score_maps.keys())
    X = np.column_stack([
        [score_maps[name][cid] for cid in common_ids]
        for name in feature_names
    ])
    # Replace NaN with 0.5
    X = np.nan_to_num(X, nan=0.5)

    results = {}
    # Individual methods
    for i, name in enumerate(feature_names):
        scores = X[:, i]
        results[name] = {
            "auc_roc": round(compute_auc_roc(labels, scores), 4),
            "auc_pr": round(compute_auc_pr(labels, scores), 4),
        }

    # Combinations via logistic regression with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    combos = {
        "all_baselines": [n for n in feature_names if n != "speccheck"],
        "all_baselines+speccheck": feature_names,
        "speccheck+selfcheck": [n for n in ["speccheck", "selfcheck"] if n in feature_names],
    }

    for combo_name, method_list in combos.items():
        if not all(m in feature_names for m in method_list):
            continue
        idx = [feature_names.index(m) for m in method_list]
        X_sub = X[:, idx]

        auc_rocs = []
        auc_prs = []
        for seed in SEEDS:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_idx, test_idx in cv.split(X_sub, labels):
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_sub[train_idx], labels[train_idx])
                preds = clf.predict_proba(X_sub[test_idx])[:, 1]
                if len(set(labels[test_idx])) >= 2:
                    auc_rocs.append(compute_auc_roc(labels[test_idx], preds))
                    auc_prs.append(compute_auc_pr(labels[test_idx], preds))

        if auc_rocs:
            results[combo_name] = {
                "auc_roc_mean": round(np.mean(auc_rocs), 4),
                "auc_roc_std": round(np.std(auc_rocs), 4),
                "auc_pr_mean": round(np.mean(auc_prs), 4),
                "auc_pr_std": round(np.std(auc_prs), 4),
                "methods": method_list,
            }

    # Feature importance from full model
    if len(feature_names) >= 2 and len(set(labels)) >= 2:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, labels)
        results["feature_importance"] = {
            name: round(float(coef), 4)
            for name, coef in zip(feature_names, clf.coef_[0])
        }

    out_path = os.path.join(RESULTS_DIR, f"analysis_combination_{mshort}_{dataset_name}.json")
    save_json(results, out_path)
    print(f"  Combination ({mshort}/{dataset_name}): {list(results.keys())}")
    return results


def run_all_ablations():
    """Run all ablation studies."""
    print("\n" + "="*60)
    print("RUNNING ABLATION STUDIES")
    print("="*60)

    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for dataset_name in DATASETS:
            print(f"\n--- {mshort}/{dataset_name} ---")
            ablation_ladder_depth(model_name, dataset_name)
            ablation_score_variants(model_name, dataset_name)
            ablation_claim_types(model_name, dataset_name)
            ablation_combination(model_name, dataset_name)


if __name__ == "__main__":
    run_all_ablations()
    print("\nAll ablations complete!")
