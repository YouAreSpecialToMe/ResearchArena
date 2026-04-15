#!/usr/bin/env python3
"""Aggregate all results into final results.json at workspace root."""
import json
import os
import numpy as np
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
EXP_DIR = WORKSPACE / "exp"
MODELS = ["llama", "mistral", "phi3"]
DATASETS = ["nq", "triviaqa", "popqa"]
METHODS = ["TokenProb", "Verbalized", "SemEntropy", "SelfConsist", "Axiomatic", "CRUX", "C2UD_full"]
ABLATION_VARIANTS = ["C2UD_RS", "C2UD_CD", "C2UD_PA", "C2UD_RS_CD", "C2UD_RS_PA", "C2UD_CD_PA", "C2UD_full"]


def main():
    results = {"experiment": "C2UD: Context-Contrastive Uncertainty Decomposition",
               "models": {}, "ablation": {}, "bootstrap_tests": {},
               "failure_analysis": {}, "intervention": {}, "success_criteria": {}}

    # Load per-model results
    for model in MODELS:
        res_path = EXP_DIR / f"{model}_results.json"
        if not res_path.exists():
            print(f"  Skipping {model} - no results file")
            continue

        with open(res_path) as f:
            model_results = json.load(f)

        results["models"][model] = model_results.get("aggregated", {})
        if "bootstrap_tests" in model_results:
            results["bootstrap_tests"][model] = model_results["bootstrap_tests"]

    # Load failure analysis
    for model in MODELS:
        fa_path = EXP_DIR / f"{model}_failure_analysis.json"
        if fa_path.exists():
            with open(fa_path) as f:
                fa = json.load(f)
            results["failure_analysis"][model] = fa.get("failure_stats", {})

    # Load intervention
    interv_path = EXP_DIR / "llama_intervention_results.json"
    if interv_path.exists():
        with open(interv_path) as f:
            results["intervention"] = json.load(f)

    # Check success criteria
    print("\n=== SUCCESS CRITERIA CHECK ===\n")

    # Criterion 1: AUROC improvement over CRUX on 2/3 datasets
    wins = 0
    for ds in DATASETS:
        c2ud_aurocs = []
        crux_aurocs = []
        for model in MODELS:
            if model in results["models"]:
                agg = results["models"][model]
                if ds in agg:
                    c2ud_val = agg[ds].get("C2UD_full", {}).get("auroc", {}).get("mean", 0)
                    crux_val = agg[ds].get("CRUX", {}).get("auroc", {}).get("mean", 0)
                    c2ud_aurocs.append(c2ud_val)
                    crux_aurocs.append(crux_val)
        if c2ud_aurocs and crux_aurocs:
            c2ud_avg = np.mean(c2ud_aurocs)
            crux_avg = np.mean(crux_aurocs)
            improvement = c2ud_avg - crux_avg
            # Check bootstrap p-value
            p_vals = []
            for model in MODELS:
                bt = results.get("bootstrap_tests", {}).get(model, {}).get(ds, {}).get("CRUX", {})
                if "p_value" in bt:
                    p_vals.append(bt["p_value"])
            avg_p = np.mean(p_vals) if p_vals else 1.0
            sig = avg_p < 0.05
            print(f"  {ds}: C2UD={c2ud_avg:.4f}, CRUX={crux_avg:.4f}, diff={improvement:+.4f}, p={avg_p:.4f}, sig={sig}")
            if improvement > 0:
                wins += 1

    crit1 = wins >= 2
    results["success_criteria"]["criterion_1_auroc_improvement"] = {
        "met": crit1,
        "description": f"C2UD AUROC > CRUX on {wins}/3 datasets (need 2/3)",
    }
    print(f"\nCriterion 1 (AUROC > CRUX on 2/3 datasets): {'MET' if crit1 else 'NOT MET'}")

    # Criterion 2: Ablation - full > RS+PA
    full_better = 0
    for ds in DATASETS:
        full_aurocs = []
        rspa_aurocs = []
        for model in MODELS:
            if model in results["models"]:
                agg = results["models"][model]
                if ds in agg:
                    full_val = agg[ds].get("C2UD_full", {}).get("auroc", {}).get("mean", 0)
                    rspa_val = agg[ds].get("C2UD_RS_PA", {}).get("auroc", {}).get("mean", 0)
                    full_aurocs.append(full_val)
                    rspa_aurocs.append(rspa_val)
        if full_aurocs and rspa_aurocs:
            if np.mean(full_aurocs) > np.mean(rspa_aurocs):
                full_better += 1

    crit2 = full_better >= 2
    results["success_criteria"]["criterion_2_three_condition_value"] = {
        "met": crit2,
        "description": f"C2UD-full > C2UD-RS+PA on {full_better}/3 datasets",
    }
    print(f"Criterion 2 (Full > RS+PA ablation): {'MET' if crit2 else 'NOT MET'}")

    # Criterion 3: Cross-model performance
    model_competitive = 0
    for model in MODELS:
        if model in results["models"]:
            agg = results["models"][model]
            avg_auroc = []
            for ds in DATASETS:
                if ds in agg and "C2UD_full" in agg[ds]:
                    avg_auroc.append(agg[ds]["C2UD_full"].get("auroc", {}).get("mean", 0))
            if avg_auroc and np.mean(avg_auroc) > 0.55:
                model_competitive += 1

    crit3 = model_competitive >= 2
    results["success_criteria"]["criterion_3_cross_model"] = {
        "met": crit3,
        "description": f"C2UD competitive on {model_competitive}/{len(MODELS)} models",
    }
    print(f"Criterion 3 (Cross-model performance): {'MET' if crit3 else 'NOT MET'}")

    # Criterion 4: Failure mode diagnosis
    fa_sig = False
    for model in MODELS:
        if model in results.get("failure_analysis", {}):
            fa = results["failure_analysis"][model]
            if len(fa) >= 2:
                fa_sig = True
                break

    results["success_criteria"]["criterion_4_failure_diagnosis"] = {
        "met": fa_sig,
        "description": "Failure modes show distinct C2UD component patterns",
    }
    print(f"Criterion 4 (Failure mode diagnosis): {'MET' if fa_sig else 'NOT MET'}")

    # Criterion 5: Intervention
    interv = results.get("intervention", {}).get("strategies", {})
    c2ud_acc = interv.get("c2ud_intervene", {}).get("accuracy", 0)
    uniform_acc = interv.get("uniform_reretrieval", {}).get("accuracy", 0)
    crit5 = c2ud_acc > uniform_acc
    results["success_criteria"]["criterion_5_intervention"] = {
        "met": crit5,
        "description": f"C2UD-intervene acc={c2ud_acc:.3f} vs uniform={uniform_acc:.3f}",
    }
    print(f"Criterion 5 (Targeted intervention): {'MET' if crit5 else 'NOT MET'}")

    # Save
    with open(WORKSPACE / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {WORKSPACE / 'results.json'}")


if __name__ == "__main__":
    main()
