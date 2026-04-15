"""Aggregate all results into the final results.json at workspace root."""

import sys
import os
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.config import *


def aggregate():
    """Combine all experimental results into a single results.json."""
    base = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(base)

    results = {
        "title": "Faithful by Consensus: Identifying Causally Important Features Through Multi-Seed SAE Agreement",
        "model": MODEL_NAME,
        "layers": LAYERS,
        "n_seeds": N_SEEDS_PRIMARY,
        "dict_size": DICT_SIZE,
        "n_training_tokens": N_TRAINING_TOKENS,
    }

    # 1. Feature matching results
    matching_path = os.path.join(base, "feature_matching", "matching_results.json")
    if os.path.exists(matching_path):
        with open(matching_path) as f:
            results["feature_matching"] = json.load(f)

    # 2. Causal importance results
    ci_path = os.path.join(base, "evaluation", "causal_importance_results.json")
    if os.path.exists(ci_path):
        with open(ci_path) as f:
            results["causal_importance"] = json.load(f)

    # 3. Sparse probing results
    sp_path = os.path.join(base, "evaluation", "sparse_probing_results.json")
    if os.path.exists(sp_path):
        with open(sp_path) as f:
            results["sparse_probing"] = json.load(f)

    # 4. Steering results
    steer_path = os.path.join(base, "evaluation", "steering_results.json")
    if os.path.exists(steer_path):
        with open(steer_path) as f:
            results["steering"] = json.load(f)

    # 5. Manifold analysis
    manifold_path = os.path.join(base, "evaluation", "manifold_analysis_results.json")
    if os.path.exists(manifold_path):
        with open(manifold_path) as f:
            results["manifold_analysis"] = json.load(f)

    # 6. Consensus dictionary
    dict_path = os.path.join(base, "evaluation", "consensus_dictionary_results.json")
    if os.path.exists(dict_path):
        with open(dict_path) as f:
            results["consensus_dictionary"] = json.load(f)

    # 7. Ablations
    ablation_path = os.path.join(base, "ablation_studies", "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            results["ablations"] = json.load(f)

    # 8. Success criteria evaluation
    success = evaluate_success_criteria(results)
    results["success_criteria"] = success

    # Save
    output_path = os.path.join(workspace, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved aggregated results to {output_path}")
    print("\n=== Success Criteria Summary ===")
    for criterion in success:
        status = "PASS" if criterion["passed"] else "FAIL"
        print(f"  [{status}] {criterion['criterion']}")
        print(f"         {criterion['detail']}")

    return results


def evaluate_success_criteria(results):
    """Evaluate each success criterion."""
    criteria = []

    # Criterion 1: Consensus predicts causal importance
    ci = results.get("causal_importance", {})
    best_layer = None
    best_d = 0
    best_p = 1
    for layer_key, layer_data in ci.items():
        d = layer_data.get("cohens_d", 0)
        p = layer_data.get("mann_whitney_p", 1)
        if d > best_d:
            best_d = d
            best_p = p
            best_layer = layer_key

    criteria.append({
        "criterion": "Consensus score predicts causal importance (p<0.01, d>0.5)",
        "passed": best_p < 0.01 and best_d > 0.5,
        "detail": f"Best layer {best_layer}: Cohen's d={best_d:.3f}, p={best_p:.2e}",
        "cohens_d": best_d,
        "p_value": best_p,
    })

    # Criterion 2: Sparse probing superiority
    sp = results.get("sparse_probing", {})
    n_better = 0
    n_total = 0
    for task, task_data in sp.items():
        for k_str, k_data in task_data.items():
            cons_acc = k_data.get("consensus", {}).get("accuracy", 0)
            sing_acc = k_data.get("singleton", {}).get("accuracy", 0)
            if cons_acc > 0 and sing_acc > 0:
                n_total += 1
                if cons_acc > sing_acc:
                    n_better += 1

    frac = n_better / max(n_total, 1)
    criteria.append({
        "criterion": "Consensus features achieve higher probing accuracy (>50% of comparisons)",
        "passed": frac > 0.5 and n_total >= 3,
        "detail": f"Consensus better in {n_better}/{n_total} comparisons ({frac:.1%})",
    })

    # Criterion 3: Consensus dictionary quality
    cd = results.get("consensus_dictionary", {})
    cons_cos = cd.get("consensus", {}).get("mean_cosine_sim", 0)
    sing_cos = cd.get("singleton", {}).get("mean_cosine_sim", 0)
    cons_n = cd.get("consensus", {}).get("n_features", 0)
    full_n = cd.get("full_reference", {}).get("n_features", 0)

    criteria.append({
        "criterion": "Consensus dictionary has better per-feature utility",
        "passed": cons_cos > sing_cos,
        "detail": f"Consensus cosine sim: {cons_cos:.4f} ({cons_n} features), "
                  f"Singleton: {sing_cos:.4f}",
    })

    # Criterion 4: Manifold tiling signatures
    ma = results.get("manifold_analysis", {})
    tiling_found = False
    for layer_key, layer_data in ma.items():
        sing_density = layer_data.get("tier_density", {}).get("singleton", {}).get("mean", 0)
        cons_density = layer_data.get("tier_density", {}).get("consensus", {}).get("mean", 0)
        if sing_density > cons_density:
            tiling_found = True

    criteria.append({
        "criterion": "Singleton features show manifold tiling signatures",
        "passed": tiling_found,
        "detail": f"Singletons have {'higher' if tiling_found else 'lower'} neighborhood density",
    })

    # Criterion 5: Results replicate across layers
    n_layers_significant = 0
    for layer_key, layer_data in ci.items():
        if layer_data.get("spearman_p", 1) < 0.05:
            n_layers_significant += 1

    criteria.append({
        "criterion": "Results replicate across layers (p<0.05 in >=2 layers)",
        "passed": n_layers_significant >= 2,
        "detail": f"Significant in {n_layers_significant}/{len(ci)} layers",
    })

    return criteria


if __name__ == "__main__":
    aggregate()
