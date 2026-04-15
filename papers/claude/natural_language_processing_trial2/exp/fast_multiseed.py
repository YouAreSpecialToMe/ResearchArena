"""
Fast multi-seed SpecCheck: Run sampling-based confidence for Llama only
(as validation), with N=5 samples and reduced batch overhead.
For other models, we rely on bootstrap CIs from the logprob-based scores.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json,
    SEEDS, MODELS, MODEL_SHORT, DATASETS, DATA_DIR, RESULTS_DIR
)
from shared.model_utils import load_model, unload_model, generate_text

N_SAMPLES = 5


def sampling_confidence_fast(model, tokenizer, prompts, n_samples=N_SAMPLES, temperature=0.8):
    """Fast sampling-based confidence estimation."""
    # Process all prompts in larger batches, sampling all at once
    batch_size = 16
    confidences = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # Duplicate each prompt n_samples times
        expanded = []
        for p in batch:
            expanded.extend([p] * n_samples)

        responses = generate_text(model, tokenizer, expanded,
                                 max_new_tokens=4, temperature=temperature,
                                 top_p=0.95, do_sample=True,
                                 batch_size=min(len(expanded), 32))

        for j in range(len(batch)):
            yes_count = 0
            total = 0
            for s in range(n_samples):
                idx = j * n_samples + s
                resp = responses[idx].strip().lower()
                if resp.startswith("yes") or (len(resp) > 0 and "yes" in resp[:10]):
                    yes_count += 1
                    total += 1
                elif resp.startswith("no") or (len(resp) > 0 and "no" in resp[:10]):
                    total += 1
            conf = yes_count / max(total, 1)
            confidences.append(conf)

    return confidences


def compute_speccheck_scores(claim_confs, alpha=0.5):
    """Compute SpecCheck scores from confidence sequences."""
    results = []
    for claim_id, confs in claim_confs.items():
        while len(confs) < 4:
            confs.append(confs[-1])
        n_trans = len(confs) - 1
        mono_violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
        mono = 1.0 - mono_violations / n_trans
        gap = confs[-1] - confs[0]
        spec_score = (1.0 - mono) + alpha * max(0, -gap)
        results.append({
            "claim_id": claim_id,
            "confidences": [round(c, 4) for c in confs],
            "speccheck_score": round(spec_score, 4),
            "monotonicity_score": round(mono, 4),
        })
    return results


def main():
    start = time.time()

    # Only run for Llama (validation model)
    model_name = MODELS[0]  # meta-llama/Llama-3.1-8B-Instruct
    mshort = MODEL_SHORT[model_name]

    print(f"Loading {model_name}...")
    model, tokenizer = load_model(model_name)

    for dataset in DATASETS:
        ladders_path = os.path.join(DATA_DIR, f"ladders_{mshort}_{dataset}.json")
        if not os.path.exists(ladders_path):
            continue

        ladders = load_json(ladders_path)

        for seed in SEEDS:
            out_path = os.path.join(RESULTS_DIR,
                                    f"speccheck_sampling_seed{seed}_{mshort}_{dataset}.json")
            if os.path.exists(out_path):
                print(f"  Already exists: {out_path}")
                continue

            print(f"\n  Seed {seed}: {mshort}/{dataset} ({len(ladders)} claims)...")
            set_seed(seed)

            # Build prompts
            all_prompts = []
            prompt_map = []
            for li, ladder in enumerate(ladders):
                for level_item in ladder["levels"]:
                    text = level_item["text"]
                    prompt = f'Is the following statement true? "{text}" Answer with Yes or No.'
                    all_prompts.append(prompt)
                    prompt_map.append((li, level_item["level"]))

            confidences = sampling_confidence_fast(model, tokenizer, all_prompts)

            # Assemble per-claim
            claim_confs = {}
            for (li, level), conf in zip(prompt_map, confidences):
                cid = ladders[li]["claim_id"]
                if cid not in claim_confs:
                    claim_confs[cid] = {}
                claim_confs[cid][level] = conf

            # Convert to list
            final_confs = {}
            for cid, level_dict in claim_confs.items():
                final_confs[cid] = [level_dict.get(k, 0.5) for k in range(4)]

            results = compute_speccheck_scores(final_confs)
            save_json(results, out_path)
            print(f"    Saved {len(results)} scores")

    unload_model(model, tokenizer)

    elapsed = time.time() - start
    print(f"\nMulti-seed SpecCheck done in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
