"""Run all experiments (Exp 1-6) and generate results + figures."""
import os
import sys
import json
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from exp.shared.metrics import auroc, auprc, ece, selective_accuracy, selective_accuracy_curve

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ["nq", "triviaqa", "popqa"]
SEEDS = [42, 43, 44]
MODELS = ["llama8b", "mistral7b"]

# Signal definitions: name -> (key_in_results, higher_is_more_confident)
SIGNALS = {
    "P(answer)": ("token_prob_rag", True),
    "Neg-Entropy": ("neg_entropy_rag", True),
    "Verb. Conf.": ("verbalized_conf", True),
    "Self-Consist.": ("self_consistency", True),
    "TPD Baseline": ("tpd_baseline", True),
    "PRA-EM": ("pra_em", True),
    "PRA-F1": ("pra_f1", True),
    "PRA-NLI": ("pra_nli", True),
    "PRA-TPD": ("pra_tpd", True),
}


def load_results(model, dataset, seed):
    path = os.path.join(RESULTS_DIR, f"generations_{model}_{dataset}_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_signal_metrics(results, signal_key, higher_confident=True):
    """Compute AUROC, AUPRC, ECE, selective accuracy for a signal."""
    scores = [r[signal_key] for r in results]
    labels = [r["rag_correct_em"] for r in results]

    if not higher_confident:
        scores = [-s for s in scores]

    metrics = {
        "auroc": auroc(scores, labels),
        "auprc": auprc(scores, labels),
        "ece": ece(scores, labels),
    }
    metrics.update(selective_accuracy(scores, labels))
    return metrics


def exp1_rag_calibration():
    """Exp 1: Compare calibration of parametric vs RAG."""
    print("\n" + "="*60)
    print("EXP 1: RAG Calibration Diagnostic")
    print("="*60)

    results_exp1 = {}
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                data = load_results(model, dataset, seed)
                if data is None:
                    continue

                key = f"{model}_{dataset}_seed{seed}"
                param_scores = [r["parametric_logprob_mean"] for r in data]
                rag_scores = [r["token_prob_rag"] for r in data]
                param_labels = [r["parametric_correct_em"] for r in data]
                rag_labels = [r["rag_correct_em"] for r in data]

                results_exp1[key] = {
                    "accuracy_parametric": float(np.mean(param_labels)),
                    "accuracy_rag": float(np.mean(rag_labels)),
                    "auroc_parametric": auroc(param_scores, param_labels),
                    "auroc_rag": auroc(rag_scores, rag_labels),
                    "ece_parametric": ece(param_scores, param_labels),
                    "ece_rag": ece(rag_scores, rag_labels),
                }

    # Aggregate by model/dataset
    agg = {}
    for model in MODELS:
        for dataset in DATASETS:
            vals = [results_exp1.get(f"{model}_{dataset}_seed{s}") for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            k = f"{model}_{dataset}"
            agg[k] = {}
            for metric in vals[0]:
                values = [v[metric] for v in vals]
                agg[k][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }

    # Print summary
    print(f"\n{'Model':<12} {'Dataset':<12} {'Acc-Param':<12} {'Acc-RAG':<12} {'AUROC-Param':<14} {'AUROC-RAG':<14} {'ECE-Param':<12} {'ECE-RAG':<12}")
    for k, v in agg.items():
        parts = k.split("_", 1)
        print(f"{parts[0]:<12} {parts[1]:<12} "
              f"{v['accuracy_parametric']['mean']:.3f}±{v['accuracy_parametric']['std']:.3f}  "
              f"{v['accuracy_rag']['mean']:.3f}±{v['accuracy_rag']['std']:.3f}  "
              f"{v['auroc_parametric']['mean']:.3f}±{v['auroc_parametric']['std']:.3f}  "
              f"{v['auroc_rag']['mean']:.3f}±{v['auroc_rag']['std']:.3f}  "
              f"{v['ece_parametric']['mean']:.3f}±{v['ece_parametric']['std']:.3f}  "
              f"{v['ece_rag']['mean']:.3f}±{v['ece_rag']['std']:.3f}")

    with open(os.path.join(RESULTS_DIR, "exp1_rag_calibration.json"), "w") as f:
        json.dump({"per_seed": results_exp1, "aggregated": agg}, f, indent=2)

    return agg


def exp2_individual_signals():
    """Exp 2: Compare all individual confidence signals."""
    print("\n" + "="*60)
    print("EXP 2: Individual Signal Comparison")
    print("="*60)

    results_exp2 = {}
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                data = load_results(model, dataset, seed)
                if data is None:
                    continue

                key = f"{model}_{dataset}_seed{seed}"
                results_exp2[key] = {}
                for signal_name, (signal_key, higher) in SIGNALS.items():
                    try:
                        metrics = compute_signal_metrics(data, signal_key, higher)
                        results_exp2[key][signal_name] = metrics
                    except Exception as e:
                        print(f"  Warning: {signal_name} failed for {key}: {e}")

    # Aggregate
    agg = {}
    for model in MODELS:
        for dataset in DATASETS:
            vals = [results_exp2.get(f"{model}_{dataset}_seed{s}") for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            k = f"{model}_{dataset}"
            agg[k] = {}
            for signal_name in SIGNALS:
                signal_vals = [v.get(signal_name) for v in vals if signal_name in v]
                if not signal_vals:
                    continue
                agg[k][signal_name] = {}
                for metric in signal_vals[0]:
                    m_vals = [sv[metric] for sv in signal_vals]
                    agg[k][signal_name][metric] = {
                        "mean": float(np.mean(m_vals)),
                        "std": float(np.std(m_vals))
                    }

    # Print AUROC table
    print(f"\n{'Signal':<16}", end="")
    for model in MODELS:
        for dataset in DATASETS:
            print(f"{model[:6]}_{dataset:<8}", end="  ")
    print()

    for signal_name in SIGNALS:
        print(f"{signal_name:<16}", end="")
        for model in MODELS:
            for dataset in DATASETS:
                k = f"{model}_{dataset}"
                if k in agg and signal_name in agg[k]:
                    m = agg[k][signal_name]["auroc"]["mean"]
                    s = agg[k][signal_name]["auroc"]["std"]
                    print(f"{m:.3f}±{s:.3f}   ", end="")
                else:
                    print(f"  N/A        ", end="")
        print()

    with open(os.path.join(RESULTS_DIR, "exp2_individual_signals.json"), "w") as f:
        json.dump({"per_seed": results_exp2, "aggregated": agg}, f, indent=2)

    return agg


def exp3_signal_combination():
    """Exp 3: Logistic regression signal combination."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("\n" + "="*60)
    print("EXP 3: Signal Combination (Logistic Regression)")
    print("="*60)

    # Define feature combinations
    all_baselines = ["P(answer)", "Neg-Entropy", "Verb. Conf.", "Self-Consist.", "TPD Baseline"]
    all_pra = ["PRA-EM", "PRA-F1", "PRA-NLI", "PRA-TPD"]

    combinations = {
        "A: Best baseline": None,  # Will be filled
        "B: PRA-NLI only": ["PRA-NLI"],
        "C: Best+PRA-NLI": None,
        "D: All baselines": all_baselines,
        "E: All baselines+PRA-NLI": all_baselines + ["PRA-NLI"],
        "F: All baselines+All PRA": all_baselines + all_pra,
        "G: All signals": all_baselines + all_pra,
    }

    results_exp3 = {}
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                data = load_results(model, dataset, seed)
                if data is None:
                    continue

                key = f"{model}_{dataset}_seed{seed}"

                # Find best individual baseline (by AUROC on calibration set)
                n = len(data)
                cal_idx = list(range(n // 2))
                test_idx = list(range(n // 2, n))
                cal_data = [data[i] for i in cal_idx]
                test_data = [data[i] for i in test_idx]

                best_baseline = None
                best_auroc = -1
                for signal_name in all_baselines:
                    signal_key, higher = SIGNALS[signal_name]
                    try:
                        scores = [r[signal_key] for r in cal_data]
                        labels = [r["rag_correct_em"] for r in cal_data]
                        if not higher:
                            scores = [-s for s in scores]
                        a = auroc(scores, labels)
                        if a > best_auroc:
                            best_auroc = a
                            best_baseline = signal_name
                    except:
                        pass

                if best_baseline is None:
                    continue

                combinations["A: Best baseline"] = [best_baseline]
                combinations["C: Best+PRA-NLI"] = [best_baseline, "PRA-NLI"]

                results_exp3[key] = {"best_baseline": best_baseline}

                for combo_name, feature_names in combinations.items():
                    if feature_names is None:
                        continue

                    try:
                        # Build feature matrices
                        X_cal = np.array([[r[SIGNALS[fn][0]] for fn in feature_names] for r in cal_data])
                        y_cal = np.array([r["rag_correct_em"] for r in cal_data])
                        X_test = np.array([[r[SIGNALS[fn][0]] for fn in feature_names] for r in test_data])
                        y_test = np.array([r["rag_correct_em"] for r in test_data])

                        # Handle NaN
                        X_cal = np.nan_to_num(X_cal, nan=0.0)
                        X_test = np.nan_to_num(X_test, nan=0.0)

                        # Scale
                        scaler = StandardScaler()
                        X_cal_s = scaler.fit_transform(X_cal)
                        X_test_s = scaler.transform(X_test)

                        # Train
                        if len(set(y_cal)) < 2:
                            continue
                        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                        clf.fit(X_cal_s, y_cal)
                        probs = clf.predict_proba(X_test_s)[:, 1]

                        results_exp3[key][combo_name] = {
                            "auroc": auroc(probs.tolist(), y_test.tolist()),
                            "auprc": auprc(probs.tolist(), y_test.tolist()),
                            "ece": ece(probs.tolist(), y_test.tolist()),
                            "features": feature_names,
                        }
                        results_exp3[key][combo_name].update(
                            selective_accuracy(probs.tolist(), y_test.tolist())
                        )

                        # Extract coefficients
                        if combo_name == "G: All signals":
                            coefs = dict(zip(feature_names, clf.coef_[0].tolist()))
                            results_exp3[key]["feature_importance"] = coefs

                    except Exception as e:
                        print(f"  Warning: {combo_name} failed for {key}: {e}")

    # Aggregate
    agg = {}
    for model in MODELS:
        for dataset in DATASETS:
            vals = [results_exp3.get(f"{model}_{dataset}_seed{s}") for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            k = f"{model}_{dataset}"
            agg[k] = {}
            for combo_name in combinations:
                combo_vals = [v.get(combo_name) for v in vals if combo_name in v]
                if not combo_vals:
                    continue
                agg[k][combo_name] = {}
                for metric in ["auroc", "auprc", "ece", "acc@50", "acc@70", "acc@90"]:
                    m_vals = [cv[metric] for cv in combo_vals if metric in cv]
                    if m_vals:
                        agg[k][combo_name][metric] = {
                            "mean": float(np.mean(m_vals)),
                            "std": float(np.std(m_vals))
                        }

    # Print combination AUROC
    print(f"\n{'Combination':<30}", end="")
    for model in MODELS:
        for dataset in DATASETS:
            print(f"{model[:6]}_{dataset:<8}", end="  ")
    print()

    for combo_name in combinations:
        print(f"{combo_name:<30}", end="")
        for model in MODELS:
            for dataset in DATASETS:
                k = f"{model}_{dataset}"
                if k in agg and combo_name in agg[k] and "auroc" in agg[k][combo_name]:
                    m = agg[k][combo_name]["auroc"]["mean"]
                    s = agg[k][combo_name]["auroc"]["std"]
                    print(f"{m:.3f}±{s:.3f}   ", end="")
                else:
                    print(f"  N/A        ", end="")
        print()

    with open(os.path.join(RESULTS_DIR, "exp3_signal_combination.json"), "w") as f:
        json.dump({"per_seed": results_exp3, "aggregated": agg}, f, indent=2)

    return agg


def exp4_popularity_analysis():
    """Exp 4: PRA-Score by entity popularity (PopQA)."""
    print("\n" + "="*60)
    print("EXP 4: Popularity Analysis (PopQA)")
    print("="*60)

    results_exp4 = {}
    for model in MODELS:
        all_data = []
        for seed in SEEDS:
            data = load_results(model, "popqa", seed)
            if data:
                for r in data:
                    r["_seed"] = seed
                all_data.extend(data)

        if not all_data:
            continue

        # Compute quartiles based on s_pop
        pops = [r.get("s_pop", 0) for r in all_data]
        if all(p == 0 for p in pops):
            print(f"  No popularity data for {model}")
            continue

        quartiles = np.percentile(pops, [25, 50, 75])
        bins = ["Q1 (rare)", "Q2", "Q3", "Q4 (popular)"]

        for i, r in enumerate(all_data):
            pop = r.get("s_pop", 0)
            if pop <= quartiles[0]:
                r["_pop_bin"] = 0
            elif pop <= quartiles[1]:
                r["_pop_bin"] = 1
            elif pop <= quartiles[2]:
                r["_pop_bin"] = 2
            else:
                r["_pop_bin"] = 3

        results_exp4[model] = {
            "quartile_boundaries": quartiles.tolist(),
            "bins": {}
        }

        for bin_idx, bin_name in enumerate(bins):
            bin_data = [r for r in all_data if r.get("_pop_bin") == bin_idx]
            if not bin_data:
                continue

            bin_results = {
                "n": len(bin_data),
                "accuracy_parametric": float(np.mean([r["parametric_correct_em"] for r in bin_data])),
                "accuracy_rag": float(np.mean([r["rag_correct_em"] for r in bin_data])),
                "pra_agreement_rate": float(np.mean([r.get("pra_em", 0) for r in bin_data])),
            }

            for signal_name, (signal_key, higher) in SIGNALS.items():
                try:
                    scores = [r[signal_key] for r in bin_data]
                    labels = [r["rag_correct_em"] for r in bin_data]
                    if not higher:
                        scores = [-s for s in scores]
                    bin_results[f"auroc_{signal_name}"] = auroc(scores, labels)
                except:
                    pass

            results_exp4[model]["bins"][bin_name] = bin_results

        # Print
        print(f"\n{model}:")
        print(f"{'Bin':<16} {'N':>4} {'Acc-P':>6} {'Acc-R':>6} {'Agree':>6} {'AUROC-NLI':>10} {'AUROC-P(a)':>10}")
        for bin_name, v in results_exp4[model]["bins"].items():
            print(f"{bin_name:<16} {v['n']:>4} {v['accuracy_parametric']:>6.3f} {v['accuracy_rag']:>6.3f} "
                  f"{v['pra_agreement_rate']:>6.3f} "
                  f"{v.get('auroc_PRA-NLI', 'N/A'):>10.3f} "
                  f"{v.get('auroc_P(answer)', 'N/A'):>10.3f}")

    with open(os.path.join(RESULTS_DIR, "exp4_popularity_analysis.json"), "w") as f:
        json.dump(results_exp4, f, indent=2)

    return results_exp4


def exp5c_agreement_ablation():
    """Exp 5c: Compare PRA variants."""
    print("\n" + "="*60)
    print("EXP 5c: Agreement Measure Comparison")
    print("="*60)

    pra_signals = ["PRA-EM", "PRA-F1", "PRA-NLI", "PRA-TPD", "TPD Baseline"]
    results_exp5c = {}

    for model in MODELS:
        all_data = []
        for dataset in DATASETS:
            for seed in SEEDS:
                data = load_results(model, dataset, seed)
                if data:
                    all_data.extend(data)

        if not all_data:
            continue

        # Pairwise correlations
        from scipy.stats import pearsonr
        corr_matrix = {}
        for s1 in pra_signals:
            for s2 in pra_signals:
                try:
                    k1, _ = SIGNALS[s1]
                    k2, _ = SIGNALS[s2]
                    v1 = [r[k1] for r in all_data]
                    v2 = [r[k2] for r in all_data]
                    corr, _ = pearsonr(v1, v2)
                    corr_matrix[f"{s1}_vs_{s2}"] = float(corr)
                except:
                    pass

        results_exp5c[model] = {"correlations": corr_matrix}

        # Qualitative: EM=0 but NLI>0.8
        interesting = [r for r in all_data
                      if r.get("pra_em", 1) == 0 and r.get("pra_nli", 0) > 0.8]
        examples = []
        for r in interesting[:10]:
            examples.append({
                "question": r["question"],
                "parametric": r["parametric_answer"],
                "rag": r["rag_answer"],
                "pra_em": r["pra_em"],
                "pra_nli": r["pra_nli"],
                "correct": r["rag_correct_em"]
            })
        results_exp5c[model]["semantic_agreement_examples"] = examples
        print(f"\n{model}: Found {len(interesting)} cases where EM=0 but NLI>0.8 (out of {len(all_data)})")
        for ex in examples[:5]:
            print(f"  Q: {ex['question'][:60]}...")
            print(f"    Param: '{ex['parametric'][:40]}' | RAG: '{ex['rag'][:40]}' | NLI: {ex['pra_nli']:.3f}")

    with open(os.path.join(RESULTS_DIR, "exp5c_agreement_ablation.json"), "w") as f:
        json.dump(results_exp5c, f, indent=2)

    return results_exp5c


def exp6_cost_benefit():
    """Exp 6: Cost-benefit analysis."""
    print("\n" + "="*60)
    print("EXP 6: Cost-Benefit Analysis")
    print("="*60)

    results_exp6 = {}
    for model in MODELS:
        timing_path = os.path.join(RESULTS_DIR, f"timing_{model}.json")
        if not os.path.exists(timing_path):
            print(f"  No timing data for {model}")
            continue

        with open(timing_path) as f:
            timing = json.load(f)

        # Average timing across datasets/seeds
        param_ms = np.mean([v["per_query_param_ms"] for v in timing.values()])
        rag_ms = np.mean([v["per_query_rag_ms"] for v in timing.values()])
        sc_ms = np.mean([v["per_query_sc_ms"] for v in timing.values()])

        # Get AUROC for each method (averaged)
        method_auroc = {}
        count = 0
        for dataset in DATASETS:
            for seed in SEEDS:
                data = load_results(model, dataset, seed)
                if data is None:
                    continue
                count += 1
                for signal_name, (signal_key, higher) in SIGNALS.items():
                    try:
                        scores = [r[signal_key] for r in data]
                        labels = [r["rag_correct_em"] for r in data]
                        if not higher:
                            scores = [-s for s in scores]
                        a = auroc(scores, labels)
                        method_auroc.setdefault(signal_name, []).append(a)
                    except:
                        pass

        # Free baselines (0 extra passes)
        free_baselines = ["P(answer)", "Neg-Entropy", "Verb. Conf."]
        best_free_auroc = max(np.mean(method_auroc.get(s, [0.5])) for s in free_baselines)

        methods_info = {
            "P(answer)": {"extra_passes": 0, "latency_ms": 0},
            "Neg-Entropy": {"extra_passes": 0, "latency_ms": 0},
            "Verb. Conf.": {"extra_passes": 0, "latency_ms": 0},
            "Self-Consist.": {"extra_passes": 4, "latency_ms": sc_ms},
            "TPD Baseline": {"extra_passes": 1, "latency_ms": param_ms},
            "PRA-EM": {"extra_passes": 1, "latency_ms": param_ms},
            "PRA-F1": {"extra_passes": 1, "latency_ms": param_ms},
            "PRA-NLI": {"extra_passes": 1, "latency_ms": param_ms},
            "PRA-TPD": {"extra_passes": 1, "latency_ms": param_ms},
        }

        results_exp6[model] = {
            "timing": {
                "per_query_param_ms": float(param_ms),
                "per_query_rag_ms": float(rag_ms),
                "per_query_sc_ms": float(sc_ms),
            },
            "best_free_auroc": float(best_free_auroc),
            "methods": {}
        }

        print(f"\n{model}:")
        print(f"{'Method':<16} {'Extra Passes':>12} {'Latency (ms)':>12} {'AUROC':>8} {'AUROC/Pass':>10}")
        for method, info in methods_info.items():
            mean_auroc = float(np.mean(method_auroc.get(method, [0.5])))
            improvement = mean_auroc - best_free_auroc
            passes = max(info["extra_passes"], 1)  # avoid div by zero
            auroc_per_pass = improvement / passes if info["extra_passes"] > 0 else 0

            results_exp6[model]["methods"][method] = {
                "extra_passes": info["extra_passes"],
                "latency_ms": float(info["latency_ms"]),
                "auroc": mean_auroc,
                "auroc_improvement": float(improvement),
                "auroc_per_pass": float(auroc_per_pass),
            }
            print(f"{method:<16} {info['extra_passes']:>12} {info['latency_ms']:>12.1f} {mean_auroc:>8.3f} {auroc_per_pass:>10.4f}")

    with open(os.path.join(RESULTS_DIR, "exp6_cost_benefit.json"), "w") as f:
        json.dump(results_exp6, f, indent=2)

    return results_exp6


def test_success_criteria(exp2_agg, exp3_agg):
    """Test success criteria from proposal."""
    print("\n" + "="*60)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*60)

    # Strong: PRA-NLI in top-2 on >= 2/3 datasets, combination improves >= 2 AUROC points
    top2_count = 0
    total_checks = 0
    for model in MODELS:
        for dataset in DATASETS:
            k = f"{model}_{dataset}"
            if k not in exp2_agg:
                continue
            total_checks += 1
            # Get all signal AUROCs
            signal_aurocs = {}
            for signal_name in SIGNALS:
                if signal_name in exp2_agg[k] and "auroc" in exp2_agg[k][signal_name]:
                    signal_aurocs[signal_name] = exp2_agg[k][signal_name]["auroc"]["mean"]

            if not signal_aurocs:
                continue

            sorted_signals = sorted(signal_aurocs.items(), key=lambda x: x[1], reverse=True)
            top2_signals = [s[0] for s in sorted_signals[:2]]
            if "PRA-NLI" in top2_signals:
                top2_count += 1

    # Check combination improvement
    combo_improvements = []
    for model in MODELS:
        for dataset in DATASETS:
            k = f"{model}_{dataset}"
            if k not in exp3_agg:
                continue
            a_auroc = exp3_agg[k].get("A: Best baseline", {}).get("auroc", {}).get("mean")
            c_auroc = exp3_agg[k].get("C: Best+PRA-NLI", {}).get("auroc", {}).get("mean")
            if a_auroc is not None and c_auroc is not None:
                combo_improvements.append(c_auroc - a_auroc)

    avg_improvement = np.mean(combo_improvements) * 100 if combo_improvements else 0  # in AUROC points

    pra_top2_rate = top2_count / total_checks if total_checks > 0 else 0
    strong = pra_top2_rate >= 2/3 and avg_improvement >= 2
    moderate = avg_improvement > 0  # combination improves on average
    refutation = avg_improvement <= 0

    print(f"\nPRA-NLI in top-2 AUROC: {top2_count}/{total_checks} = {pra_top2_rate:.1%}")
    print(f"Average combination improvement (C vs A): {avg_improvement:.2f} AUROC points")
    print(f"Individual improvements: {[f'{x*100:.2f}' for x in combo_improvements]}")
    print(f"\nSTRONG confirmation: {'YES' if strong else 'NO'}")
    print(f"MODERATE confirmation: {'YES' if moderate else 'NO'}")
    print(f"REFUTATION: {'YES' if refutation else 'NO'}")

    verdict = "strong" if strong else ("moderate" if moderate else "refutation")

    return {
        "pra_nli_top2_rate": pra_top2_rate,
        "avg_combination_improvement_auroc_points": float(avg_improvement),
        "individual_improvements": [float(x*100) for x in combo_improvements],
        "strong_confirmation": strong,
        "moderate_confirmation": moderate,
        "refutation": refutation,
        "verdict": verdict,
    }


def generate_figures(exp1_agg, exp2_agg, exp3_agg, exp4_results, exp6_results):
    """Generate all publication figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 300})
    colors = sns.color_palette("colorblind", 10)

    # Figure 1: RAG Calibration Diagnostic
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models_present = [m for m in MODELS if any(f"{m}_{d}" in exp1_agg for d in DATASETS)]
    if models_present:
        x_labels = []
        param_accs = []
        rag_accs = []
        param_eces = []
        rag_eces = []
        for model in models_present:
            for dataset in DATASETS:
                k = f"{model}_{dataset}"
                if k in exp1_agg:
                    x_labels.append(f"{model[:6]}\n{dataset}")
                    param_accs.append(exp1_agg[k]["accuracy_parametric"]["mean"])
                    rag_accs.append(exp1_agg[k]["accuracy_rag"]["mean"])
                    param_eces.append(exp1_agg[k]["ece_parametric"]["mean"])
                    rag_eces.append(exp1_agg[k]["ece_rag"]["mean"])

        x = np.arange(len(x_labels))
        w = 0.35
        axes[0].bar(x - w/2, param_accs, w, label='Parametric', color=colors[0])
        axes[0].bar(x + w/2, rag_accs, w, label='RAG', color=colors[1])
        axes[0].set_ylabel('Accuracy (EM)')
        axes[0].set_title('(a) Accuracy: Parametric vs RAG')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels, fontsize=8)
        axes[0].legend()

        axes[1].bar(x - w/2, param_eces, w, label='Parametric', color=colors[0])
        axes[1].bar(x + w/2, rag_eces, w, label='RAG', color=colors[1])
        axes[1].set_ylabel('ECE (lower is better)')
        axes[1].set_title('(b) Calibration Error: Parametric vs RAG')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, fontsize=8)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_rag_calibration.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_rag_calibration.png"), bbox_inches='tight')
    plt.close()

    # Figure 2: Individual Signal AUROC
    fig, axes = plt.subplots(1, len(models_present), figsize=(7*len(models_present), 6), squeeze=False)
    for mi, model in enumerate(models_present):
        ax = axes[0, mi]
        signal_names = list(SIGNALS.keys())
        x = np.arange(len(DATASETS))
        width = 0.08
        for si, sn in enumerate(signal_names):
            means = []
            stds = []
            for dataset in DATASETS:
                k = f"{model}_{dataset}"
                if k in exp2_agg and sn in exp2_agg[k] and "auroc" in exp2_agg[k][sn]:
                    means.append(exp2_agg[k][sn]["auroc"]["mean"])
                    stds.append(exp2_agg[k][sn]["auroc"]["std"])
                else:
                    means.append(0)
                    stds.append(0)
            offset = (si - len(signal_names)/2) * width
            c = colors[3] if "PRA" in sn else colors[si % len(colors)]
            ax.bar(x + offset, means, width, yerr=stds, label=sn, color=c, alpha=0.8)
        ax.set_ylabel('AUROC')
        ax.set_title(f'{model}')
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylim(0.4, 0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure2_signal_auroc.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "figure2_signal_auroc.png"), bbox_inches='tight')
    plt.close()

    # Figure 3: Signal Combination
    if exp3_agg:
        combo_names = ["A: Best baseline", "B: PRA-NLI only", "C: Best+PRA-NLI",
                       "D: All baselines", "E: All baselines+PRA-NLI", "F: All baselines+All PRA", "G: All signals"]
        fig, axes = plt.subplots(1, len(models_present), figsize=(7*len(models_present), 5), squeeze=False)
        for mi, model in enumerate(models_present):
            ax = axes[0, mi]
            combo_aurocs = []
            combo_labels = []
            for cn in combo_names:
                vals = []
                for dataset in DATASETS:
                    k = f"{model}_{dataset}"
                    if k in exp3_agg and cn in exp3_agg[k] and "auroc" in exp3_agg[k][cn]:
                        vals.append(exp3_agg[k][cn]["auroc"]["mean"])
                if vals:
                    combo_aurocs.append(np.mean(vals))
                    combo_labels.append(cn.split(":")[0])

            pra_mask = [("PRA" in cn or cn.startswith("B") or cn.startswith("C") or cn.startswith("E") or cn.startswith("F") or cn.startswith("G"))
                       for cn in combo_names[:len(combo_labels)]]
            bar_colors = [colors[3] if pm else colors[0] for pm in pra_mask]
            ax.bar(range(len(combo_aurocs)), combo_aurocs, color=bar_colors)
            ax.set_ylabel('AUROC (avg across datasets)')
            ax.set_title(f'{model}')
            ax.set_xticks(range(len(combo_labels)))
            ax.set_xticklabels(combo_labels, rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure3_combination.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, "figure3_combination.png"), bbox_inches='tight')
        plt.close()

    # Figure 4: Selective Accuracy Curves
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5*len(DATASETS), 5), squeeze=False)
    model = models_present[0] if models_present else None
    if model:
        for di, dataset in enumerate(DATASETS):
            ax = axes[0, di]
            data = load_results(model, dataset, 42)
            if data is None:
                continue

            for signal_name in ["PRA-NLI", "P(answer)", "Self-Consist."]:
                signal_key, higher = SIGNALS[signal_name]
                try:
                    scores = [r[signal_key] for r in data]
                    labels = [r["rag_correct_em"] for r in data]
                    if not higher:
                        scores = [-s for s in scores]
                    covs, accs = selective_accuracy_curve(scores, labels)
                    ax.plot(covs, accs, label=signal_name)
                except:
                    pass

            ax.set_xlabel('Coverage')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{dataset}')
            ax.legend()
            ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure4_selective_accuracy.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, "figure4_selective_accuracy.png"), bbox_inches='tight')
    plt.close()

    # Figure 5: Popularity Analysis
    if exp4_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in exp4_results:
            bins_data = exp4_results[model].get("bins", {})
            bin_names = list(bins_data.keys())
            for signal_name in ["PRA-NLI", "P(answer)", "TPD Baseline"]:
                auroc_key = f"auroc_{signal_name}"
                vals = [bins_data[b].get(auroc_key, 0.5) for b in bin_names]
                if any(v != 0.5 for v in vals):
                    ax.plot(range(len(bin_names)), vals, 'o-', label=f"{model[:6]}-{signal_name}")

        ax.set_xticks(range(len(bin_names)))
        ax.set_xticklabels(bin_names)
        ax.set_ylabel('AUROC')
        ax.set_xlabel('Entity Popularity Quartile')
        ax.set_title('PRA-Score AUROC by Entity Popularity')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure5_popularity.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, "figure5_popularity.png"), bbox_inches='tight')
        plt.close()

    # Figure 6: Cost-Benefit Pareto
    if exp6_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in exp6_results:
            methods = exp6_results[model].get("methods", {})
            for method, info in methods.items():
                total_passes = 1 + info["extra_passes"]  # 1 for RAG baseline
                ax.scatter(total_passes, info["auroc"], s=80, zorder=5)
                ax.annotate(method, (total_passes, info["auroc"]),
                           textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax.set_xlabel('Total Inference Passes')
        ax.set_ylabel('AUROC')
        ax.set_title('Cost-Benefit: AUROC vs Inference Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure6_cost_benefit.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, "figure6_cost_benefit.png"), bbox_inches='tight')
        plt.close()

    print(f"\nFigures saved to {FIGURES_DIR}/")


def generate_latex_tables(exp2_agg, exp3_agg, exp6_results):
    """Generate LaTeX tables."""
    # Table 1: Main results
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AUROC for individual confidence signals across datasets and models (mean $\pm$ std over 3 seeds). \textbf{Bold}: best, \underline{underline}: second best.}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}",
        r"\toprule",
    ]

    # Header
    header = "Signal"
    for model in MODELS:
        for dataset in DATASETS:
            header += f" & {model[:6]}-{dataset}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for signal_name in SIGNALS:
        row = signal_name.replace("_", r"\_")
        for model in MODELS:
            for dataset in DATASETS:
                k = f"{model}_{dataset}"
                if k in exp2_agg and signal_name in exp2_agg[k] and "auroc" in exp2_agg[k][signal_name]:
                    m = exp2_agg[k][signal_name]["auroc"]["mean"]
                    s = exp2_agg[k][signal_name]["auroc"]["std"]
                    row += f" & {m:.3f}$\\pm${s:.3f}"
                else:
                    row += " & --"
        row += r" \\"
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"])

    with open(os.path.join(FIGURES_DIR, "table1_main_results.tex"), "w") as f:
        f.write("\n".join(lines))

    # Also save as CSV
    import csv
    csv_path = os.path.join(FIGURES_DIR, "table1_main_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = ["Signal"]
        for model in MODELS:
            for dataset in DATASETS:
                header_row.append(f"{model}_{dataset}")
        writer.writerow(header_row)
        for signal_name in SIGNALS:
            row = [signal_name]
            for model in MODELS:
                for dataset in DATASETS:
                    k = f"{model}_{dataset}"
                    if k in exp2_agg and signal_name in exp2_agg[k] and "auroc" in exp2_agg[k][signal_name]:
                        m = exp2_agg[k][signal_name]["auroc"]["mean"]
                        s = exp2_agg[k][signal_name]["auroc"]["std"]
                        row.append(f"{m:.3f}±{s:.3f}")
                    else:
                        row.append("--")
            writer.writerow(row)

    print(f"Tables saved to {FIGURES_DIR}/")


def save_final_results(exp1, exp2, exp3, exp4, exp5c, exp6, criteria):
    """Save aggregated results.json at workspace root."""
    root = os.path.dirname(os.path.dirname(__file__))
    root = os.path.dirname(RESULTS_DIR)  # workspace root

    final = {
        "experiment": "PRA-Score: Parametric-Retrieval Agreement for RAG Calibration",
        "models": MODELS,
        "datasets": DATASETS,
        "seeds": SEEDS,
        "n_questions_per_seed": 500,
        "success_criteria": criteria,
        "exp1_rag_calibration": exp1,
        "exp2_individual_signals": exp2,
        "exp3_signal_combination": exp3,
        "exp4_popularity_analysis": exp4,
        "exp5c_agreement_ablation": exp5c,
        "exp6_cost_benefit": exp6,
    }

    results_path = os.path.join(root, "results.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(results_path, "w") as f:
        json.dump(final, f, indent=2, cls=NumpyEncoder)
    print(f"\nFinal results saved to {results_path}")


def main():
    print("="*60)
    print("PRA-Score Experiment Analysis")
    print("="*60)

    # Check what data we have
    available = []
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                if load_results(model, dataset, seed) is not None:
                    available.append(f"{model}_{dataset}_seed{seed}")
    print(f"\nAvailable results: {len(available)}/18")
    if not available:
        print("No results found! Run inference first.")
        return

    exp1_agg = exp1_rag_calibration()
    exp2_agg = exp2_individual_signals()
    exp3_agg = exp3_signal_combination()
    exp4_results = exp4_popularity_analysis()
    exp5c_results = exp5c_agreement_ablation()
    exp6_results = exp6_cost_benefit()
    criteria = test_success_criteria(exp2_agg, exp3_agg)

    generate_figures(exp1_agg, exp2_agg, exp3_agg, exp4_results, exp6_results)
    generate_latex_tables(exp2_agg, exp3_agg, exp6_results)
    save_final_results(exp1_agg, exp2_agg, exp3_agg, exp4_results, exp5c_results, exp6_results, criteria)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
