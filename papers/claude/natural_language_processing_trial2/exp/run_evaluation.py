"""
Statistical evaluation and results aggregation.
Computes all final metrics, statistical significance tests,
and produces the aggregated results.json.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json, get_model_short,
    SEEDS, MODELS, DATASETS, DATA_DIR, RESULTS_DIR, BASE_DIR
)
from shared.metrics import (
    compute_all_metrics, compute_auc_roc, compute_auc_pr,
    bootstrap_ci, paired_bootstrap_test
)


def collect_all_scores(model_name, dataset_name):
    """Collect all method scores aligned by claim_id."""
    mshort = get_model_short(model_name)
    label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")
    if not os.path.exists(label_path):
        return None

    labeled = load_json(label_path)
    label_map = {c["claim_id"]: c["label"] for c in labeled}

    methods = {}
    # SpecCheck
    p = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json")
    if os.path.exists(p):
        data = load_json(p)
        methods["speccheck"] = {d["claim_id"]: d["speccheck_score"] for d in data}

    # SelfCheck
    p = os.path.join(RESULTS_DIR, f"baseline_selfcheck_{mshort}_{dataset_name}.json")
    if os.path.exists(p):
        data = load_json(p)
        methods["selfcheck"] = {d["claim_id"]: d["hallucination_score"] for d in data}

    # Verbalized
    p = os.path.join(RESULTS_DIR, f"baseline_verbalized_{mshort}_{dataset_name}.json")
    if os.path.exists(p):
        data = load_json(p)
        methods["verbalized"] = {d["claim_id"]: d["hallucination_score"] for d in data}

    # Logprob
    p = os.path.join(RESULTS_DIR, f"baseline_logprob_{mshort}_{dataset_name}.json")
    if os.path.exists(p):
        data = load_json(p)
        methods["logprob"] = {d["claim_id"]: d["hallucination_score"] for d in data}

    # Random (average of seeds)
    random_scores = {}
    for seed in SEEDS:
        p = os.path.join(RESULTS_DIR, f"baseline_random_{mshort}_{dataset_name}_seed{seed}.json")
        if os.path.exists(p):
            data = load_json(p)
            for d in data:
                if d["claim_id"] not in random_scores:
                    random_scores[d["claim_id"]] = []
                random_scores[d["claim_id"]].append(d["hallucination_score"])
    if random_scores:
        methods["random"] = {cid: np.mean(scores) for cid, scores in random_scores.items()}

    # Find common claim IDs
    all_ids = set(label_map.keys())
    for m_scores in methods.values():
        all_ids &= set(m_scores.keys())
    all_ids = sorted(all_ids)

    if len(all_ids) < 10:
        return None

    labels = [label_map[cid] for cid in all_ids]
    scores = {name: [m[cid] for cid in all_ids] for name, m in methods.items()}

    return {"claim_ids": all_ids, "labels": labels, "scores": scores}


def evaluate_model_dataset(model_name, dataset_name):
    """Full evaluation for one model-dataset pair."""
    mshort = get_model_short(model_name)
    data = collect_all_scores(model_name, dataset_name)
    if data is None:
        return None

    labels = data["labels"]
    results = {"model": mshort, "dataset": dataset_name, "n_claims": len(labels)}

    # Compute metrics for each method
    method_results = {}
    for name, scores in data["scores"].items():
        metrics = compute_all_metrics(labels, scores)
        # Bootstrap CI
        auc_pr_mean, auc_pr_lo, auc_pr_hi = bootstrap_ci(labels, scores, compute_auc_pr)
        auc_roc_mean, auc_roc_lo, auc_roc_hi = bootstrap_ci(labels, scores, compute_auc_roc)
        metrics["auc_pr_ci"] = [round(auc_pr_lo, 4), round(auc_pr_hi, 4)]
        metrics["auc_roc_ci"] = [round(auc_roc_lo, 4), round(auc_roc_hi, 4)]
        method_results[name] = metrics

    results["methods"] = method_results

    # Significance tests: SpecCheck vs each baseline
    if "speccheck" in data["scores"]:
        sig_tests = {}
        for baseline in ["selfcheck", "verbalized", "logprob", "random"]:
            if baseline in data["scores"]:
                p_val = paired_bootstrap_test(
                    labels, data["scores"]["speccheck"],
                    data["scores"][baseline], compute_auc_pr
                )
                sig_tests[f"speccheck_vs_{baseline}"] = {
                    "p_value": round(p_val, 4),
                    "significant_at_005": p_val < 0.05,
                }
        results["significance_tests"] = sig_tests

    # Success criteria evaluation
    if "speccheck" in data["scores"]:
        spec_scores_data = load_json(
            os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json")
        )
        label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")
        labeled = load_json(label_path)
        label_map = {c["claim_id"]: c["label"] for c in labeled}

        # Monotonicity rates
        factual_mono = []
        halluc_mono = []
        for item in spec_scores_data:
            cid = item["claim_id"]
            if cid not in label_map:
                continue
            mono = item["monotonicity_score"]
            if label_map[cid] == 0:
                factual_mono.append(mono)
            else:
                halluc_mono.append(mono)

        results["success_criteria"] = {
            "factual_monotonicity_rate": round(np.mean([m == 1.0 for m in factual_mono]), 4) if factual_mono else None,
            "factual_mono_mean": round(np.mean(factual_mono), 4) if factual_mono else None,
            "halluc_violation_rate": round(np.mean([m < 1.0 for m in halluc_mono]), 4) if halluc_mono else None,
            "halluc_mono_mean": round(np.mean(halluc_mono), 4) if halluc_mono else None,
            "n_factual": len(factual_mono),
            "n_hallucinated": len(halluc_mono),
        }

    return results


def run_full_evaluation():
    """Run evaluation across all models and datasets."""
    print("\n" + "="*60)
    print("RUNNING FULL EVALUATION")
    print("="*60)

    all_results = {}
    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for dataset_name in DATASETS:
            key = f"{mshort}_{dataset_name}"
            print(f"\nEvaluating {key}...")
            result = evaluate_model_dataset(model_name, dataset_name)
            if result:
                all_results[key] = result
                # Print summary
                if "methods" in result:
                    for m, metrics in result["methods"].items():
                        print(f"  {m:15s}: AUC-ROC={metrics['auc_roc']:.4f}  AUC-PR={metrics['auc_pr']:.4f}")

    # Save detailed evaluation
    save_json(all_results, os.path.join(RESULTS_DIR, "evaluation_statistics.json"))

    # Aggregate into results.json at workspace root
    aggregate = {
        "experiment": "SpecCheck: Detecting LLM Hallucinations by Testing Confidence Monotonicity",
        "models": [get_model_short(m) for m in MODELS],
        "datasets": DATASETS,
        "seeds": SEEDS,
        "detailed_results": {},
        "summary": {},
    }

    # Build summary tables
    for key, result in all_results.items():
        if "methods" not in result:
            continue
        aggregate["detailed_results"][key] = result

        for method, metrics in result["methods"].items():
            if method not in aggregate["summary"]:
                aggregate["summary"][method] = {"auc_roc": [], "auc_pr": []}
            aggregate["summary"][method]["auc_roc"].append(metrics["auc_roc"])
            aggregate["summary"][method]["auc_pr"].append(metrics["auc_pr"])

    # Compute means
    for method in aggregate["summary"]:
        for metric in ["auc_roc", "auc_pr"]:
            vals = aggregate["summary"][method][metric]
            aggregate["summary"][method][f"{metric}_mean"] = round(np.mean(vals), 4)
            aggregate["summary"][method][f"{metric}_std"] = round(np.std(vals), 4)

    # Success criteria summary
    criteria = {
        "factual_mono_rates": [],
        "halluc_violation_rates": [],
        "speccheck_beats_selfcheck": 0,
        "total_datasets": 0,
    }
    for key, result in all_results.items():
        if "success_criteria" in result:
            sc = result["success_criteria"]
            if sc.get("factual_monotonicity_rate") is not None:
                criteria["factual_mono_rates"].append(sc["factual_monotonicity_rate"])
            if sc.get("halluc_violation_rate") is not None:
                criteria["halluc_violation_rates"].append(sc["halluc_violation_rate"])
        if "significance_tests" in result:
            criteria["total_datasets"] += 1
            if result["significance_tests"].get("speccheck_vs_selfcheck", {}).get("significant_at_005"):
                criteria["speccheck_beats_selfcheck"] += 1

    if criteria["factual_mono_rates"]:
        criteria["avg_factual_mono_rate"] = round(np.mean(criteria["factual_mono_rates"]), 4)
    if criteria["halluc_violation_rates"]:
        criteria["avg_halluc_violation_rate"] = round(np.mean(criteria["halluc_violation_rates"]), 4)

    aggregate["success_criteria_summary"] = criteria

    save_json(aggregate, os.path.join(BASE_DIR, "results.json"))
    print(f"\n\nResults saved to results.json")
    print(f"Summary:")
    for method, vals in aggregate["summary"].items():
        print(f"  {method:15s}: AUC-ROC={vals['auc_roc_mean']:.4f}±{vals['auc_roc_std']:.4f}  "
              f"AUC-PR={vals['auc_pr_mean']:.4f}±{vals['auc_pr_std']:.4f}")

    return aggregate


if __name__ == "__main__":
    run_full_evaluation()
