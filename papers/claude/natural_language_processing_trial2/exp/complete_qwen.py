"""
Complete Qwen experiments for longfact and truthfulqa.
Qwen/factscore is already done. Need: claims + ladders + confidence + baselines
for longfact and truthfulqa.
Also need truthfulqa outputs (not yet generated).
"""
import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json,
    SEEDS, DATA_DIR, RESULTS_DIR
)
from shared.model_utils import (
    load_model, unload_model, generate_text,
    get_yes_no_logprobs, get_token_logprobs, parse_claims
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "qwen"


def generate_truthfulqa_outputs(model, tokenizer):
    """Generate TruthfulQA outputs for Qwen."""
    out_path = os.path.join(DATA_DIR, f"outputs_{MODEL_SHORT}_truthfulqa.json")
    if os.path.exists(out_path):
        print(f"  Outputs already exist: {out_path}")
        return load_json(out_path)

    data = load_json(os.path.join(DATA_DIR, "truthfulqa_subset.json"))
    prompts = [d["prompt"] for d in data]

    print(f"  Generating TruthfulQA outputs ({len(prompts)} prompts)...")
    outputs = generate_text(model, tokenizer, prompts,
                           max_new_tokens=128, temperature=0.7, top_p=0.9, batch_size=4)

    results = [{**d, "output": out} for d, out in zip(data, outputs)]
    save_json(results, out_path)
    print(f"  Generated {len(results)} outputs")
    return results


def decompose_claims(model, tokenizer, dataset_name, outputs):
    """Decompose outputs into atomic claims."""
    out_path = os.path.join(DATA_DIR, f"claims_{MODEL_SHORT}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Claims already exist: {out_path}")
        return load_json(out_path)

    prompt_template = (
        "Break the following text into independent atomic facts. "
        "Each fact should be a single, self-contained sentence that can be "
        "verified as true or false independently. Output one fact per line, "
        "numbered. Do not include opinions or subjective statements.\n\n"
        "Text: {text}\n\nAtomic facts:"
    )

    all_claims = []
    claim_id = 0
    for item in outputs:
        text = item.get("output", "")
        if not text or len(text.strip()) < 20:
            continue

        prompt = prompt_template.format(text=text[:1000])
        result = generate_text(model, tokenizer, [prompt],
                              max_new_tokens=512, temperature=0.0, do_sample=False, batch_size=1)[0]

        claims = parse_claims(result)
        for claim in claims[:10]:
            all_claims.append({
                "claim_id": f"{MODEL_SHORT}_{dataset_name}_{claim_id}",
                "claim_text": claim,
                "source_id": item["id"],
                "source_text": text[:500],
            })
            claim_id += 1

    save_json(all_claims, out_path)
    print(f"  Extracted {len(all_claims)} claims for {dataset_name}")
    return all_claims


def generate_ladders(model, tokenizer, dataset_name, claims):
    """Generate specificity ladders."""
    out_path = os.path.join(DATA_DIR, f"ladders_{MODEL_SHORT}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Ladders already exist: {out_path}")
        return load_json(out_path)

    ladder_prompt = (
        "Given the following factual claim, rewrite it at three levels of decreasing specificity:\n"
        "Level 0 (Original): The exact claim as stated.\n"
        "Level 1 (Approximate): Replace exact values with ranges or approximate terms.\n"
        "Level 2 (Category): Replace specific instances with their general category.\n"
        "Level 3 (Abstract): The most general true version of this claim.\n\n"
        "Examples:\n"
        "Claim: Marie Curie won the Nobel Prize in Physics in 1903.\n"
        "Level 1: Marie Curie won the Nobel Prize in Physics in the early 1900s.\n"
        "Level 2: Marie Curie won a Nobel Prize in a science field.\n"
        "Level 3: Marie Curie won a major scientific award.\n\n"
        "Claim: The Great Wall of China is 13,171 miles long.\n"
        "Level 1: The Great Wall of China is approximately 13,000 miles long.\n"
        "Level 2: The Great Wall of China is thousands of miles long.\n"
        "Level 3: The Great Wall of China is very long.\n\n"
        "Claim: {claim}\n"
        "Level 1:"
    )

    ladders = []
    for i in range(0, len(claims), 4):
        batch = claims[i:i+4]
        prompts = [ladder_prompt.format(claim=c["claim_text"]) for c in batch]

        results = generate_text(model, tokenizer, prompts,
                               max_new_tokens=256, temperature=0.0, do_sample=False, batch_size=4)

        for claim, result in zip(batch, results):
            levels = [claim["claim_text"]]  # Level 0
            lines = result.strip().split("\n")
            for line in lines:
                line = line.strip()
                # Remove "Level X:" prefix
                for prefix in ["Level 1:", "Level 2:", "Level 3:", "1:", "2:", "3:"]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                if len(line) > 10:
                    levels.append(line)
                if len(levels) >= 4:
                    break

            # Pad if needed
            while len(levels) < 4:
                levels.append(levels[-1])

            ladders.append({
                "claim_id": claim["claim_id"],
                "levels": [{"level": k, "text": t} for k, t in enumerate(levels)]
            })

    save_json(ladders, out_path)
    print(f"  Generated {len(ladders)} ladders for {dataset_name}")
    return ladders


def compute_confidence(model, tokenizer, dataset_name, ladders):
    """Compute logprob confidence at each specificity level."""
    out_path = os.path.join(RESULTS_DIR, f"confidence_logprob_{MODEL_SHORT}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Confidence already exists: {out_path}")
        return load_json(out_path)

    all_prompts = []
    prompt_map = []  # (ladder_idx, level_idx)

    for li, ladder in enumerate(ladders):
        for level_item in ladder["levels"]:
            text = level_item["text"]
            prompt = f'Is the following statement true? "{text}" Answer with Yes or No.'
            all_prompts.append(prompt)
            prompt_map.append((li, level_item["level"]))

    print(f"  Computing confidence for {len(all_prompts)} prompts...")
    confidences = get_yes_no_logprobs(model, tokenizer, all_prompts, batch_size=16)

    # Assemble per-claim
    results = []
    conf_by_ladder = {}
    for (li, level), conf in zip(prompt_map, confidences):
        if li not in conf_by_ladder:
            conf_by_ladder[li] = {}
        conf_by_ladder[li][level] = conf

    for li, ladder in enumerate(ladders):
        confs = [conf_by_ladder.get(li, {}).get(k, 0.5) for k in range(4)]
        results.append({
            "claim_id": ladder["claim_id"],
            "confidences": [round(c, 4) for c in confs],
        })

    save_json(results, out_path)
    print(f"  Computed confidence for {len(results)} claims")
    return results


def compute_speccheck_scores(dataset_name, confidence_data):
    """Compute SpecCheck scores from confidence data."""
    out_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{MODEL_SHORT}_{dataset_name}.json")
    alpha = 0.5

    results = []
    for item in confidence_data:
        confs = item["confidences"]
        while len(confs) < 4:
            confs.append(confs[-1])

        n_trans = len(confs) - 1
        mono_violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
        monotonicity_score = 1.0 - mono_violations / n_trans
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

    save_json(results, out_path)
    return results


def run_baselines(model, tokenizer, dataset_name, claims):
    """Run verbalized, logprob, selfcheck, and random baselines."""
    import numpy as np

    # Verbalized confidence
    verb_path = os.path.join(RESULTS_DIR, f"baseline_verbalized_{MODEL_SHORT}_{dataset_name}.json")
    if not os.path.exists(verb_path):
        print(f"  Running verbalized baseline...")
        prompts = [
            f'Rate your confidence that the following statement is true on a scale of 0 to 100, where 0 means certainly false and 100 means certainly true. Only output the number.\nStatement: "{c["claim_text"]}"'
            for c in claims
        ]
        outputs = generate_text(model, tokenizer, prompts,
                               max_new_tokens=16, temperature=0.0, do_sample=False, batch_size=8)

        results = []
        for claim, out in zip(claims, outputs):
            import re
            nums = re.findall(r'\d+', out)
            score = int(nums[0]) if nums else 50
            score = min(100, max(0, score))
            results.append({
                "claim_id": claim["claim_id"],
                "hallucination_score": round(1.0 - score / 100.0, 4),
            })
        save_json(results, verb_path)

    # Logprob baseline
    logp_path = os.path.join(RESULTS_DIR, f"baseline_logprob_{MODEL_SHORT}_{dataset_name}.json")
    if not os.path.exists(logp_path):
        print(f"  Running logprob baseline...")
        texts = [c["claim_text"] for c in claims]
        avg_logprobs = get_token_logprobs(model, tokenizer, texts, batch_size=16)

        # Also get yes/no logprob (level-0 confidence)
        yn_prompts = [f'Is the following statement true? "{c["claim_text"]}" Answer with Yes or No.' for c in claims]
        yn_confs = get_yes_no_logprobs(model, tokenizer, yn_prompts, batch_size=16)

        results = []
        for claim, lp, yn in zip(claims, avg_logprobs, yn_confs):
            # Combine token logprob and yes/no confidence
            # Normalize: higher = more hallucinated
            norm_lp = 1.0 - min(1.0, max(0.0, (lp + 5.0) / 5.0))  # map [-5, 0] to [1, 0]
            yn_score = 1.0 - yn  # invert: low confidence = hallucinated
            combined = 0.5 * norm_lp + 0.5 * yn_score
            results.append({
                "claim_id": claim["claim_id"],
                "hallucination_score": round(combined, 4),
                "avg_logprob": round(float(lp), 4),
                "yesno_confidence": round(float(yn), 4),
            })
        save_json(results, logp_path)

    # SelfCheck baseline
    sc_path = os.path.join(RESULTS_DIR, f"baseline_selfcheck_{MODEL_SHORT}_{dataset_name}.json")
    if not os.path.exists(sc_path):
        print(f"  Running selfcheck baseline...")
        # Group claims by source_id
        source_ids = list(set(c["source_id"] for c in claims))
        source_claims = {}
        for c in claims:
            source_claims.setdefault(c["source_id"], []).append(c)

        # Load original prompts
        dataset_file = os.path.join(DATA_DIR, f"{dataset_name}_subset.json")
        dataset = load_json(dataset_file)
        id_to_prompt = {d["id"]: d["prompt"] for d in dataset}

        # Generate N=5 alternative responses per prompt
        N_SAMPLES = 5
        alt_responses = {}
        for sid in source_ids:
            prompt = id_to_prompt.get(sid, "")
            if not prompt:
                continue
            max_tok = {"factscore": 256, "longfact": 512, "truthfulqa": 128}.get(dataset_name, 256)
            alts = generate_text(model, tokenizer, [prompt] * N_SAMPLES,
                                max_new_tokens=max_tok, temperature=1.0, top_p=0.95, batch_size=4)
            alt_responses[sid] = alts

        # Check support for each claim
        results = []
        for claim in claims:
            alts = alt_responses.get(claim["source_id"], [])
            if not alts:
                results.append({"claim_id": claim["claim_id"], "hallucination_score": 0.5})
                continue

            check_prompts = [
                f'Does the following passage support the claim "{claim["claim_text"]}"? Answer Yes or No.\nPassage: {alt[:500]}'
                for alt in alts
            ]
            support_confs = get_yes_no_logprobs(model, tokenizer, check_prompts, batch_size=8)
            support_rate = sum(1 for c in support_confs if c > 0.5) / len(support_confs)
            results.append({
                "claim_id": claim["claim_id"],
                "hallucination_score": round(1.0 - support_rate, 4),
            })
        save_json(results, sc_path)

    # Random baseline (3 seeds)
    for seed in [42, 123, 456]:
        rand_path = os.path.join(RESULTS_DIR, f"baseline_random_{MODEL_SHORT}_{dataset_name}_seed{seed}.json")
        if not os.path.exists(rand_path):
            np.random.seed(seed)
            results = [{"claim_id": c["claim_id"], "hallucination_score": round(float(np.random.random()), 4)}
                       for c in claims]
            save_json(results, rand_path)


def main():
    start = time.time()
    set_seed(42)

    print("Loading Qwen model...")
    model, tokenizer = load_model(MODEL_NAME)

    for dataset_name in ["longfact", "truthfulqa"]:
        print(f"\n{'='*60}")
        print(f"Processing {MODEL_SHORT}/{dataset_name}")
        print(f"{'='*60}")

        # Generate outputs if needed (truthfulqa)
        if dataset_name == "truthfulqa":
            outputs = generate_truthfulqa_outputs(model, tokenizer)
        else:
            out_path = os.path.join(DATA_DIR, f"outputs_{MODEL_SHORT}_{dataset_name}.json")
            outputs = load_json(out_path)

        # Decompose claims
        claims = decompose_claims(model, tokenizer, dataset_name, outputs)

        # Generate ladders
        ladders = generate_ladders(model, tokenizer, dataset_name, claims)

        # Compute confidence
        conf_data = compute_confidence(model, tokenizer, dataset_name, ladders)

        # Compute SpecCheck scores
        compute_speccheck_scores(dataset_name, conf_data)

        # Run baselines
        run_baselines(model, tokenizer, dataset_name, claims)

    unload_model(model, tokenizer)

    elapsed = time.time() - start
    print(f"\nQwen completion done in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
