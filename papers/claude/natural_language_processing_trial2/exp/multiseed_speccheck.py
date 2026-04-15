"""
Multi-seed SpecCheck evaluation using sampling-based confidence.
For each seed, we sample N=10 yes/no responses per claim-level pair
and compute confidence as the fraction of "Yes" answers.
This adds stochasticity to SpecCheck, enabling proper error bars.
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json,
    SEEDS, MODELS, MODEL_SHORT, DATASETS, DATA_DIR, RESULTS_DIR
)
from shared.model_utils import load_model, unload_model, generate_text

N_SAMPLES = 10  # Samples per claim-level pair


def sampling_confidence(model, tokenizer, prompts, n_samples=N_SAMPLES, temperature=0.8, batch_size=8):
    """Estimate confidence by sampling n_samples yes/no responses."""
    confidences = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_confs = [[] for _ in batch]

        for _ in range(n_samples):
            responses = generate_text(model, tokenizer, batch,
                                     max_new_tokens=8, temperature=temperature,
                                     top_p=0.95, do_sample=True, batch_size=batch_size)
            for j, resp in enumerate(responses):
                resp_lower = resp.strip().lower()
                if resp_lower.startswith("yes"):
                    batch_confs[j].append(1)
                elif resp_lower.startswith("no"):
                    batch_confs[j].append(0)
                else:
                    # Try to parse
                    if "yes" in resp_lower[:10]:
                        batch_confs[j].append(1)
                    elif "no" in resp_lower[:10]:
                        batch_confs[j].append(0)
                    # else skip this sample

        for j in range(len(batch)):
            if batch_confs[j]:
                confidences.append(sum(batch_confs[j]) / len(batch_confs[j]))
            else:
                confidences.append(0.5)

    return confidences


def compute_speccheck_from_confs(confidences_per_claim, alpha=0.5):
    """Compute SpecCheck scores from confidence sequences."""
    results = []
    for claim_id, confs in confidences_per_claim.items():
        while len(confs) < 4:
            confs.append(confs[-1])

        n_trans = len(confs) - 1
        mono_violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
        monotonicity_score = 1.0 - mono_violations / n_trans
        confidence_gap = confs[-1] - confs[0]
        speccheck_score = (1.0 - monotonicity_score) + alpha * max(0, -confidence_gap)
        max_violation = max((confs[k-1] - confs[k]) for k in range(1, len(confs)))

        results.append({
            "claim_id": claim_id,
            "confidences": [round(c, 4) for c in confs],
            "monotonicity_score": round(monotonicity_score, 4),
            "speccheck_score": round(speccheck_score, 4),
            "max_violation": round(max_violation, 4),
            "confidence_gap": round(confidence_gap, 4),
        })
    return results


def run_multiseed_for_model(model_name, model, tokenizer):
    """Run multi-seed sampling-based SpecCheck for one model."""
    mshort = MODEL_SHORT[model_name]

    for dataset_name in DATASETS:
        ladders_path = os.path.join(DATA_DIR, f"ladders_{mshort}_{dataset_name}.json")
        if not os.path.exists(ladders_path):
            print(f"  No ladders for {mshort}/{dataset_name}, skipping")
            continue

        ladders = load_json(ladders_path)

        for seed in SEEDS:
            out_path = os.path.join(RESULTS_DIR,
                                    f"speccheck_sampling_seed{seed}_{mshort}_{dataset_name}.json")
            if os.path.exists(out_path):
                print(f"  Already exists: {out_path}")
                continue

            print(f"\n  Seed {seed}: {mshort}/{dataset_name} ({len(ladders)} claims)...")
            set_seed(seed)

            # Build all prompts
            all_prompts = []
            prompt_map = []
            for li, ladder in enumerate(ladders):
                for level_item in ladder["levels"]:
                    text = level_item["text"]
                    prompt = f'Is the following statement true? "{text}" Answer with Yes or No.'
                    all_prompts.append(prompt)
                    prompt_map.append((li, level_item["level"]))

            # Run sampling-based confidence
            confidences = sampling_confidence(model, tokenizer, all_prompts,
                                            n_samples=N_SAMPLES, batch_size=8)

            # Assemble per-claim
            conf_by_ladder = {}
            for (li, level), conf in zip(prompt_map, confidences):
                if li not in conf_by_ladder:
                    conf_by_ladder[li] = {}
                conf_by_ladder[li][level] = conf

            confidences_per_claim = {}
            for li, ladder in enumerate(ladders):
                confs = [conf_by_ladder.get(li, {}).get(k, 0.5) for k in range(4)]
                confidences_per_claim[ladder["claim_id"]] = confs

            # Compute scores
            results = compute_speccheck_from_confs(confidences_per_claim)
            save_json(results, out_path)
            print(f"    Saved {len(results)} scores")


def main():
    start = time.time()

    for model_name in MODELS:
        mshort = MODEL_SHORT[model_name]
        print(f"\n{'='*60}")
        print(f"Multi-seed SpecCheck for {mshort}")
        print(f"{'='*60}")

        # Check if all seeds already done
        all_done = True
        for ds in DATASETS:
            for seed in SEEDS:
                p = os.path.join(RESULTS_DIR, f"speccheck_sampling_seed{seed}_{mshort}_{ds}.json")
                if not os.path.exists(p):
                    ladders_p = os.path.join(DATA_DIR, f"ladders_{mshort}_{ds}.json")
                    if os.path.exists(ladders_p):
                        all_done = False
                        break
            if not all_done:
                break

        if all_done:
            print(f"  All multi-seed results exist for {mshort}, skipping")
            continue

        model, tokenizer = load_model(model_name)
        run_multiseed_for_model(model_name, model, tokenizer)
        unload_model(model, tokenizer)

    elapsed = time.time() - start
    print(f"\nMulti-seed SpecCheck done in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
