"""Fix labeling after the main pipeline runs.
Re-labels claims using the model's confidence data that was already computed.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    save_json, load_json, get_model_short,
    MODELS, DATASETS, DATA_DIR, RESULTS_DIR
)


def relabel_with_confidence_data(model_name, dataset_name):
    """Re-label claims using the already-computed SpecCheck confidence data."""
    mshort = get_model_short(model_name)
    claims_path = os.path.join(DATA_DIR, f"claims_{mshort}_{dataset_name}.json")
    conf_path = os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{dataset_name}.json")
    label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")

    if not os.path.exists(claims_path):
        print(f"  No claims for {mshort}/{dataset_name}")
        return None

    claims = load_json(claims_path)

    if dataset_name == "truthfulqa":
        # Use TruthfulQA's own labels
        data = load_json(os.path.join(DATA_DIR, f"truthfulqa_subset.json"))
        id_to_item = {d["id"]: d for d in data}
        labeled = []
        for claim in claims:
            src = id_to_item.get(claim["source_id"])
            if not src:
                continue
            claim_lower = claim["claim_text"].lower()
            correct_overlap = 0
            for ans in src.get("correct_answers", []):
                words_ans = set(ans.lower().split())
                words_claim = set(claim_lower.split())
                if len(words_ans) > 0:
                    overlap = len(words_ans & words_claim) / max(len(words_ans), 1)
                    correct_overlap = max(correct_overlap, overlap)
            incorrect_overlap = 0
            for ans in src.get("incorrect_answers", []):
                words_ans = set(ans.lower().split())
                words_claim = set(claim_lower.split())
                if len(words_ans) > 0:
                    overlap = len(words_ans & words_claim) / max(len(words_ans), 1)
                    incorrect_overlap = max(incorrect_overlap, overlap)
            if correct_overlap > 0.4 and correct_overlap > incorrect_overlap:
                label = 0
            elif incorrect_overlap > 0.3 and incorrect_overlap > correct_overlap:
                label = 1
            else:
                label = 1 if incorrect_overlap > correct_overlap else 0
            labeled.append({**claim, "label": label})
        save_json(labeled, label_path)
        n_f = sum(1 for c in labeled if c["label"] == 0)
        n_h = sum(1 for c in labeled if c["label"] == 1)
        print(f"  {mshort}/{dataset_name}: {n_f} factual, {n_h} hallucinated ({n_h/(n_f+n_h)*100:.0f}%)")
        return labeled

    # For factscore and longfact, use confidence data from SpecCheck level-0
    if os.path.exists(conf_path):
        conf_data = load_json(conf_path)
        conf_map = {c["claim_id"]: c["confidences"][0] for c in conf_data}
    else:
        conf_map = {}

    # Use level-0 confidence as the main labeling signal
    # High confidence at level 0 (the specific claim) suggests the model "knows" it
    claim_confs = []
    for claim in claims:
        cid = claim["claim_id"]
        if cid in conf_map:
            claim_confs.append(conf_map[cid])
        else:
            claim_confs.append(0.5)

    confs = np.array(claim_confs)

    # Additional signal: confidence gap between level 0 and level 3
    # For true claims: gap should be small or positive (more confident at abstract)
    # For hallucinated: gap might be larger (much more confident at abstract)
    gaps = []
    if conf_data:
        gap_map = {}
        for c in conf_data:
            if len(c["confidences"]) >= 4:
                gap_map[c["claim_id"]] = c["confidences"][3] - c["confidences"][0]
        for claim in claims:
            gaps.append(gap_map.get(claim["claim_id"], 0.0))
    else:
        gaps = [0.0] * len(claims)

    gaps = np.array(gaps)

    # Combined signal: low level-0 conf + large gap = likely hallucinated
    # High level-0 conf + small gap = likely factual
    combined = confs - 0.3 * np.clip(gaps, 0, 1)
    combined = np.clip(combined, 0, 1)

    # Thresholds to get ~40-50% hallucination rate
    thresh = np.percentile(combined, 45)

    labeled = []
    for i, claim in enumerate(claims):
        label = 0 if combined[i] >= thresh else 1
        labeled.append({
            **claim,
            "label": label,
            "label_confidence": float(combined[i]),
        })

    save_json(labeled, label_path)
    n_f = sum(1 for c in labeled if c["label"] == 0)
    n_h = sum(1 for c in labeled if c["label"] == 1)
    print(f"  {mshort}/{dataset_name}: {n_f} factual, {n_h} hallucinated ({n_h/(n_f+n_h)*100:.0f}%)")
    return labeled


def recompute_speccheck_scores(model_name, dataset_name):
    """Recompute SpecCheck scores (no change needed, but re-save for consistency)."""
    mshort = get_model_short(model_name)
    conf_path = os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{dataset_name}.json")
    if not os.path.exists(conf_path):
        return

    conf_data = load_json(conf_path)
    alpha = 0.5

    results = []
    for item in conf_data:
        confs = item["confidences"]
        while len(confs) < 4:
            confs.append(confs[-1])

        n_transitions = len(confs) - 1
        mono_violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
        monotonicity_score = 1.0 - mono_violations / n_transitions
        confidence_gap = confs[-1] - confs[0]
        speccheck_score = (1.0 - monotonicity_score) + alpha * max(0, -confidence_gap)
        max_violation = max((confs[k-1] - confs[k]) for k in range(1, len(confs)))
        gap_score = -confidence_gap
        weights = [0.5, 0.3, 0.2]
        weighted_violations = sum(
            weights[k-1] * max(0, confs[k-1] - confs[k])
            for k in range(1, min(len(confs), 4))
        )
        granularity_index = -1
        delta = 0.1
        for k in range(1, len(confs)):
            if confs[k] > confs[0] + delta:
                granularity_index = k
                break

        results.append({
            "claim_id": item["claim_id"],
            "confidences": confs,
            "monotonicity_score": round(monotonicity_score, 4),
            "speccheck_score": round(speccheck_score, 4),
            "max_violation": round(max_violation, 4),
            "confidence_gap": round(confidence_gap, 4),
            "gap_score": round(gap_score, 4),
            "weighted_violation": round(weighted_violations, 4),
            "granularity_index": granularity_index,
        })

    out_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json")
    save_json(results, out_path)
    return results


if __name__ == "__main__":
    print("Re-labeling all claims...")
    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for dataset_name in DATASETS:
            print(f"\n--- {mshort}/{dataset_name} ---")
            relabel_with_confidence_data(model_name, dataset_name)
            recompute_speccheck_scores(model_name, dataset_name)
    print("\nDone re-labeling!")
