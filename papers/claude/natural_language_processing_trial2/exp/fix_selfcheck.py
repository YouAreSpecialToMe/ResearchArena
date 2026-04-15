"""
Fix and re-run SelfCheckGPT baseline.

The original SelfCheck code looked for 'source_prompt' or 'prompt' in claims,
but claims have 'source_id' linking to the dataset entries. This script
properly maps claims back to their source prompts.
"""
import os
import sys
import json
import time
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json, get_model_short,
    SEEDS, MODELS, DATASETS, DATA_DIR, RESULTS_DIR
)
from shared.model_utils import (
    load_model, unload_model, generate_text, get_yes_no_logprobs
)

N_SAMPLES = 5


def build_prompt_map(ds):
    """Build a map from source_id -> prompt using the dataset files."""
    if ds == "factscore":
        data = load_json(os.path.join(DATA_DIR, "factscore_subset.json"))
        return {d["id"]: d["prompt"] for d in data}
    elif ds == "longfact":
        data = load_json(os.path.join(DATA_DIR, "longfact_subset.json"))
        return {d["id"]: d["prompt"] for d in data}
    elif ds == "truthfulqa":
        data = load_json(os.path.join(DATA_DIR, "truthfulqa_subset.json"))
        return {d["id"]: d["prompt"] for d in data}
    return {}


def run_selfcheck(model, tokenizer, model_name):
    mshort = get_model_short(model_name)
    print(f"\n{'='*60}")
    print(f"SelfCheck baseline (fixed): {mshort}")
    print(f"{'='*60}")

    for ds in DATASETS:
        print(f"\n--- {mshort}/{ds} ---")
        claims = load_json(os.path.join(DATA_DIR, f"claims_{mshort}_{ds}.json"))
        labels = load_json(os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{ds}.json"))
        prompt_map = build_prompt_map(ds)

        # Group claims by source_id
        groups = {}
        for i, claim in enumerate(claims):
            sid = claim.get("source_id", f"unknown_{i}")
            if sid not in groups:
                groups[sid] = []
            groups[sid].append((i, claim))

        print(f"  {len(claims)} claims, {len(groups)} source groups, {len(prompt_map)} prompts")

        # Generate N alternative responses per prompt
        alt_responses = {}
        gen_count = 0
        max_tokens = {"factscore": 256, "longfact": 512, "truthfulqa": 128}[ds]

        for sid in tqdm(groups.keys(), desc=f"Generating alternatives for {ds}"):
            prompt = prompt_map.get(sid, "")
            if not prompt:
                continue

            alts = generate_text(
                model, tokenizer, [prompt] * N_SAMPLES,
                max_new_tokens=max_tokens, temperature=1.0, top_p=0.95,
                batch_size=N_SAMPLES, do_sample=True
            )
            alt_responses[sid] = alts
            gen_count += 1

        print(f"  Generated alternatives for {gen_count}/{len(groups)} groups")

        # Check each claim against alternatives
        results = []
        for sid, items in tqdm(groups.items(), desc=f"Checking claims for {ds}"):
            alts = alt_responses.get(sid, [])

            for idx, claim in items:
                cid = claim.get("claim_id", claim.get("id", f"{mshort}_{ds}_{idx}"))
                label = labels[idx].get("label", 0) if idx < len(labels) else 0
                claim_text = claim.get("claim_text", claim.get("text", ""))

                if not alts:
                    results.append({
                        "claim_id": cid,
                        "label": label,
                        "hallucination_score": 0.5,
                        "support_scores": [],
                    })
                    continue

                # NLI: check if each alternative supports the claim
                nli_prompts = []
                for alt in alts:
                    alt_truncated = alt[:500]
                    p = (f'Does the following passage support the claim "{claim_text}"? '
                         f'Answer with only Yes or No.\n'
                         f'Passage: "{alt_truncated}"')
                    nli_prompts.append(p)

                support_confs = get_yes_no_logprobs(
                    model, tokenizer, nli_prompts, batch_size=N_SAMPLES
                )
                avg_support = float(np.mean(support_confs))
                hall_score = 1.0 - avg_support

                results.append({
                    "claim_id": cid,
                    "label": label,
                    "hallucination_score": hall_score,
                    "support_scores": support_confs,
                })

        save_json(results, os.path.join(RESULTS_DIR, f"baseline_selfcheck_{mshort}_{ds}.json"))
        print(f"  {mshort}/{ds}: {len(results)} claims scored")

        # Quick check
        scores = [r["hallucination_score"] for r in results]
        print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}], mean={np.mean(scores):.3f}")
        n_05 = sum(1 for s in scores if abs(s - 0.5) < 0.01)
        print(f"  Fraction at 0.5: {n_05}/{len(scores)}")


def main():
    start = time.time()
    for model_name in MODELS:
        model, tokenizer = load_model(model_name)
        run_selfcheck(model, tokenizer, model_name)
        unload_model(model, tokenizer)
        print(f"Elapsed: {(time.time()-start)/60:.1f} min")
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
