"""
Ablation 2: Sampling-based confidence estimation.
Runs on Llama-3.1-8B + FActScore only (most expensive ablation).
Compares N=5, 10, 20 samples.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json, get_model_short,
    DATA_DIR, RESULTS_DIR
)
from shared.model_utils import load_model, unload_model, generate_text
from shared.metrics import compute_auc_roc, compute_auc_pr

DEVICE = "cuda"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "factscore"


def sampling_confidence(model, tokenizer, claims, N=5, max_claims=200):
    """Estimate confidence via sampling: generate N yes/no responses, count fraction of Yes."""
    prompts_base = [
        f'Is the following statement true? "{c["claim_text"]}" Answer with Yes or No.'
        for c in claims[:max_claims]
    ]

    all_confidences = []
    for level_key in ["levels"]:
        pass  # We process per-level below

    return None  # Handled below


def run_sampling_ablation():
    """Run sampling-based confidence ablation."""
    mshort = get_model_short(MODEL_NAME)
    ladders_path = os.path.join(DATA_DIR, f"ladders_{mshort}_{DATASET_NAME}.json")
    label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{DATASET_NAME}.json")

    if not os.path.exists(ladders_path) or not os.path.exists(label_path):
        print("Missing ladder or label data. Run main pipeline first.")
        return

    ladders = load_json(ladders_path)
    labeled = load_json(label_path)
    label_map = {c["claim_id"]: c["label"] for c in labeled}

    # Load model
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE)

    results = {}
    for N in [5, 10, 20]:
        max_claims = 200 if N <= 10 else 100
        subset = ladders[:max_claims]

        print(f"\nSampling N={N} on {len(subset)} claims...")
        t0 = time.time()

        claim_confidences = []
        for ladder in subset:
            level_confs = []
            for level_info in ladder["levels"][:4]:
                prompt = f'Is the following statement true? "{level_info["text"]}" Answer with Yes or No.'
                # Generate N responses
                responses = generate_text(
                    model, tokenizer, [prompt] * N,
                    max_new_tokens=5, temperature=0.8, top_p=0.95, batch_size=N
                )
                yes_count = sum(1 for r in responses if r.strip().lower().startswith("yes"))
                level_confs.append(yes_count / N)
            claim_confidences.append({
                "claim_id": ladder["claim_id"],
                "confidences": level_confs,
            })

        elapsed = time.time() - t0
        print(f"  N={N}: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Compute SpecCheck scores
        scores = []
        labels = []
        for item in claim_confidences:
            cid = item["claim_id"]
            if cid not in label_map:
                continue
            confs = item["confidences"]
            while len(confs) < 4:
                confs.append(confs[-1])
            n_trans = len(confs) - 1
            violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
            mono = 1.0 - violations / n_trans
            gap = confs[-1] - confs[0]
            spec_score = (1.0 - mono) + 0.5 * max(0, -gap)
            scores.append(spec_score)
            labels.append(label_map[cid])

        if len(scores) > 10 and len(set(labels)) >= 2:
            results[f"N={N}"] = {
                "auc_roc": round(compute_auc_roc(labels, scores), 4),
                "auc_pr": round(compute_auc_pr(labels, scores), 4),
                "n_claims": len(scores),
                "time_seconds": round(elapsed, 1),
            }

    # Also include logprob-based result for comparison
    logprob_path = os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{DATASET_NAME}.json")
    if os.path.exists(logprob_path):
        logprob_data = load_json(logprob_path)
        lp_scores = []
        lp_labels = []
        for item in logprob_data:
            cid = item["claim_id"]
            if cid not in label_map:
                continue
            confs = item["confidences"]
            while len(confs) < 4:
                confs.append(confs[-1])
            n_trans = len(confs) - 1
            violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
            mono = 1.0 - violations / n_trans
            gap = confs[-1] - confs[0]
            spec_score = (1.0 - mono) + 0.5 * max(0, -gap)
            lp_scores.append(spec_score)
            lp_labels.append(label_map[cid])

        if len(lp_scores) > 10:
            results["logprob"] = {
                "auc_roc": round(compute_auc_roc(lp_labels, lp_scores), 4),
                "auc_pr": round(compute_auc_pr(lp_labels, lp_scores), 4),
                "n_claims": len(lp_scores),
                "time_seconds": 0,  # Already computed
            }

    unload_model(model, tokenizer)

    out_path = os.path.join(RESULTS_DIR, f"ablation_confidence_method_{mshort}_{DATASET_NAME}.json")
    save_json(results, out_path)
    print(f"\nSampling ablation results: {results}")
    return results


if __name__ == "__main__":
    run_sampling_ablation()
