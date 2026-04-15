"""
Main experiment pipeline for SpecCheck.
Processes one model at a time through all stages:
1. Generate outputs for all datasets
2. Decompose into atomic claims
3. Label claims (factual vs hallucinated)
4. Generate specificity ladders
5. Estimate confidence at each level (logprob-based)
6. Compute SpecCheck scores
7. Run baselines (verbalized, logprob, selfcheck)
"""
import os
import sys
import json
import time
import random
import re
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json, get_model_short,
    SEEDS, MODELS, DATASETS, DATA_DIR, RESULTS_DIR, BASE_DIR
)
from shared.model_utils import (
    load_model, unload_model, generate_text, get_yes_no_logprobs,
    get_token_logprobs, parse_claims
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# STAGE 1: Generate outputs
# ============================================================
def generate_outputs(model, tokenizer, model_name, dataset_name, data):
    """Generate model outputs for a dataset."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(DATA_DIR, f"outputs_{mshort}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Outputs already exist: {out_path}")
        return load_json(out_path)

    print(f"  Generating outputs for {mshort}/{dataset_name} ({len(data)} prompts)...")
    prompts = [d["prompt"] for d in data]

    max_tokens = {"factscore": 256, "longfact": 512, "truthfulqa": 128}[dataset_name]

    outputs = generate_text(
        model, tokenizer, prompts,
        max_new_tokens=max_tokens, temperature=0.7, top_p=0.9, batch_size=4
    )

    results = []
    for d, out in zip(data, outputs):
        results.append({**d, "output": out})

    save_json(results, out_path)
    print(f"  Generated {len(results)} outputs")
    return results


# ============================================================
# STAGE 2: Decompose into atomic claims
# ============================================================
def decompose_claims(model, tokenizer, model_name, dataset_name, outputs):
    """Decompose outputs into atomic claims."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(DATA_DIR, f"claims_{mshort}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Claims already exist: {out_path}")
        return load_json(out_path)

    print(f"  Decomposing claims for {mshort}/{dataset_name}...")
    decompose_prompt_template = (
        "Break the following text into independent atomic facts. "
        "Each fact should be a single, self-contained sentence that can be "
        "verified as true or false independently. Output one fact per line, "
        "numbered. Do not include opinions or subjective statements.\n\n"
        "Text: {text}\n\nAtomic facts:"
    )

    all_claims = []
    claim_id = 0
    for item in tqdm(outputs, desc="Decomposing"):
        text = item["output"]
        if not text or len(text.strip()) < 20:
            continue

        prompt = decompose_prompt_template.format(text=text[:1000])
        result = generate_text(
            model, tokenizer, [prompt],
            max_new_tokens=512, temperature=0.0, do_sample=False, batch_size=1
        )[0]

        claims = parse_claims(result)
        for claim in claims[:10]:  # Cap at 10 claims per output
            all_claims.append({
                "claim_id": f"{mshort}_{dataset_name}_{claim_id}",
                "claim_text": claim,
                "source_id": item["id"],
                "source_text": text[:500],
            })
            claim_id += 1

    save_json(all_claims, out_path)
    print(f"  Extracted {len(all_claims)} claims")
    return all_claims


# ============================================================
# STAGE 3: Label claims
# ============================================================
def label_claims_truthfulqa(claims, dataset):
    """Label TruthfulQA claims using provided correct/incorrect answers."""
    id_to_item = {d["id"]: d for d in dataset}
    labeled = []
    for claim in claims:
        src = id_to_item.get(claim["source_id"])
        if not src:
            continue

        claim_lower = claim["claim_text"].lower()
        # Check overlap with correct answers
        correct_overlap = 0
        for ans in src.get("correct_answers", []):
            words_ans = set(ans.lower().split())
            words_claim = set(claim_lower.split())
            if len(words_ans) > 0:
                overlap = len(words_ans & words_claim) / max(len(words_ans), 1)
                correct_overlap = max(correct_overlap, overlap)

        # Check overlap with incorrect answers
        incorrect_overlap = 0
        for ans in src.get("incorrect_answers", []):
            words_ans = set(ans.lower().split())
            words_claim = set(claim_lower.split())
            if len(words_ans) > 0:
                overlap = len(words_ans & words_claim) / max(len(words_ans), 1)
                incorrect_overlap = max(incorrect_overlap, overlap)

        # Label based on relative overlap
        if correct_overlap > 0.4 and correct_overlap > incorrect_overlap:
            label = 0  # factual
        elif incorrect_overlap > 0.3 and incorrect_overlap > correct_overlap:
            label = 1  # hallucinated
        else:
            # Ambiguous - use moderate heuristic
            label = 1 if incorrect_overlap > correct_overlap else 0

        labeled.append({**claim, "label": label})
    return labeled


def label_claims_with_logprob(model, tokenizer, claims):
    """Label claims using model's own logprob as proxy for factuality.
    Claims the model is very confident about are likely factual;
    low-confidence claims are likely hallucinated. This is a known
    correlation used as a labeling proxy when external verification
    is infeasible.
    """
    claim_texts = [c["claim_text"] for c in claims]

    # Use yes/no verification as primary signal
    verify_prompts = [
        f'Is the following statement true? "{c}" Answer with Yes or No.'
        for c in claim_texts
    ]
    verify_confs = get_yes_no_logprobs(model, tokenizer, verify_prompts, batch_size=16)
    vc_array = np.array(verify_confs, dtype=np.float64)

    # Replace NaN with 0.5
    vc_array = np.where(np.isnan(vc_array), 0.5, vc_array)
    # Clip to valid range
    vc_array = np.clip(vc_array, 0.0, 1.0)

    # Also get token-level logprobs
    logprobs = get_token_logprobs(model, tokenizer, claim_texts, batch_size=16)
    lp_array = np.array(logprobs, dtype=np.float64)
    lp_array = np.where(np.isnan(lp_array), np.nanmedian(lp_array), lp_array)

    # Normalize logprobs to [0, 1]
    lp_range = lp_array.max() - lp_array.min()
    if lp_range > 1e-8:
        lp_norm = (lp_array - lp_array.min()) / lp_range
    else:
        lp_norm = np.full_like(lp_array, 0.5)

    # Combined score: higher = more likely factual
    combined = 0.5 * lp_norm + 0.5 * vc_array
    combined = np.where(np.isnan(combined), 0.5, combined)

    # Use percentile-based thresholds for clear labels
    high_thresh = np.percentile(combined, 70)
    low_thresh = np.percentile(combined, 30)

    labeled = []
    for i, claim in enumerate(claims):
        if combined[i] >= high_thresh:
            label = 0  # factual
        elif combined[i] <= low_thresh:
            label = 1  # hallucinated
        else:
            label = 0 if combined[i] > np.median(combined) else 1
        labeled.append({**claim, "label": label, "label_confidence": float(combined[i])})

    return labeled


def label_all_claims(model, tokenizer, model_name, dataset_name, claims, dataset):
    """Create ground-truth labels for claims."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Labels already exist: {out_path}")
        return load_json(out_path)

    print(f"  Labeling claims for {mshort}/{dataset_name}...")
    if dataset_name == "truthfulqa":
        labeled = label_claims_truthfulqa(claims, dataset)
    else:
        # For factscore and longfact, use model-based labeling proxy
        labeled = label_claims_with_logprob(model, tokenizer, claims)

    save_json(labeled, out_path)

    n_factual = sum(1 for c in labeled if c["label"] == 0)
    n_halluc = sum(1 for c in labeled if c["label"] == 1)
    print(f"  Labels: {n_factual} factual, {n_halluc} hallucinated ({n_halluc/(n_factual+n_halluc)*100:.0f}% halluc)")
    return labeled


# ============================================================
# STAGE 4: Generate specificity ladders
# ============================================================
def generate_ladders(model, tokenizer, model_name, dataset_name, claims):
    """Generate specificity ladders for all claims."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(DATA_DIR, f"ladders_{mshort}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Ladders already exist: {out_path}")
        return load_json(out_path)

    print(f"  Generating specificity ladders for {mshort}/{dataset_name}...")
    ladder_prompt = (
        "Given a factual claim, rewrite it at three levels of decreasing specificity. "
        "Each level should be less specific but still capture the core meaning.\n\n"
        "Example 1:\n"
        "Level 0 (Original): Albert Einstein was born on March 14, 1879 in Ulm, Germany.\n"
        "Level 1 (Approximate): Albert Einstein was born around 1879 in southern Germany.\n"
        "Level 2 (Category): Albert Einstein was born in the late 19th century in Germany.\n"
        "Level 3 (Abstract): Albert Einstein was born in Europe.\n\n"
        "Example 2:\n"
        "Level 0 (Original): The Eiffel Tower is 330 meters tall.\n"
        "Level 1 (Approximate): The Eiffel Tower is approximately 300-350 meters tall.\n"
        "Level 2 (Category): The Eiffel Tower is several hundred meters tall.\n"
        "Level 3 (Abstract): The Eiffel Tower is a very tall structure.\n\n"
        "Example 3:\n"
        "Level 0 (Original): Marie Curie won the Nobel Prize in Physics in 1903.\n"
        "Level 1 (Approximate): Marie Curie won a Nobel Prize in the early 1900s.\n"
        "Level 2 (Category): Marie Curie received a major scientific award.\n"
        "Level 3 (Abstract): Marie Curie was recognized for her scientific contributions.\n\n"
        "Now rewrite this claim:\n"
        "Level 0 (Original): {claim}\n"
        "Level 1 (Approximate):"
    )

    ladders = []
    batch_size = 2  # Small batch for long prompts
    for i in tqdm(range(0, len(claims), batch_size), desc="Ladders"):
        batch = claims[i:i+batch_size]
        prompts = [ladder_prompt.format(claim=c["claim_text"]) for c in batch]

        results = generate_text(
            model, tokenizer, prompts,
            max_new_tokens=256, temperature=0.0, do_sample=False, batch_size=batch_size
        )

        for j, (claim, result) in enumerate(zip(batch, results)):
            # Parse the three levels from the output
            levels = [claim["claim_text"]]  # Level 0 is original

            # Try to extract levels from the generation
            lines = result.strip().split("\n")
            current_text = ""
            for line in lines:
                line = line.strip()
                # Check for level markers
                if any(marker in line.lower() for marker in
                       ["level 1", "level 2", "level 3", "approximate", "category", "abstract"]):
                    # Extract text after the colon
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        current_text = parts[1].strip()
                    else:
                        current_text = line
                    if current_text and len(current_text) > 10:
                        levels.append(current_text)
                elif line and not line.startswith("Level") and len(levels) < 4:
                    # Continuation or unlabeled level
                    if len(line) > 10:
                        levels.append(line)

            # If we didn't get enough levels, create simple fallbacks
            while len(levels) < 4:
                prev = levels[-1]
                # Simple abstraction: remove specific details
                simplified = re.sub(r'\b\d{4}\b', 'some time', prev)
                simplified = re.sub(r'\b\d+(\.\d+)?\s*(meters?|km|miles?|feet)\b', 'a certain distance', simplified)
                simplified = re.sub(r'\b\d+(\.\d+)?%?\b', 'some amount', simplified)
                if simplified == prev:
                    # More aggressive: shorten
                    words = prev.split()
                    simplified = " ".join(words[:max(len(words)//2, 4)])
                    if not simplified.endswith("."):
                        simplified += "."
                levels.append(simplified)

            ladders.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "levels": [
                    {"level": k, "text": levels[k]}
                    for k in range(min(4, len(levels)))
                ]
            })

    save_json(ladders, out_path)

    # Compute ladder quality stats
    valid = sum(1 for l in ladders if len(l["levels"]) == 4)
    print(f"  Generated {len(ladders)} ladders ({valid}/{len(ladders)} with 4 levels)")
    return ladders


# ============================================================
# STAGE 5: Confidence estimation (logprob-based)
# ============================================================
def estimate_confidence(model, tokenizer, model_name, dataset_name, ladders):
    """Estimate confidence at each specificity level using logprobs."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Confidence already computed: {out_path}")
        return load_json(out_path)

    print(f"  Estimating confidence for {mshort}/{dataset_name}...")
    all_prompts = []
    prompt_map = []  # (ladder_idx, level)

    for li, ladder in enumerate(ladders):
        for level_info in ladder["levels"]:
            prompt = f'Is the following statement true? "{level_info["text"]}" Answer with Yes or No.'
            all_prompts.append(prompt)
            prompt_map.append((li, level_info["level"]))

    # Get all confidences in batch
    confidences = get_yes_no_logprobs(model, tokenizer, all_prompts, batch_size=16)

    # Reassemble by ladder
    results = []
    conf_by_ladder = {}
    for (li, level), conf in zip(prompt_map, confidences):
        if li not in conf_by_ladder:
            conf_by_ladder[li] = {}
        conf_by_ladder[li][level] = conf

    for li, ladder in enumerate(ladders):
        confs = conf_by_ladder.get(li, {})
        conf_list = [confs.get(k, 0.5) for k in range(4)]
        results.append({
            "claim_id": ladder["claim_id"],
            "confidences": conf_list,
        })

    save_json(results, out_path)
    print(f"  Computed confidence for {len(results)} claims")
    return results


# ============================================================
# STAGE 6: Compute SpecCheck scores
# ============================================================
def compute_speccheck_scores(model_name, dataset_name, confidences_data, alpha=0.5):
    """Compute SpecCheck scores from confidence sequences."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json")

    results = []
    for item in confidences_data:
        confs = item["confidences"]
        if len(confs) < 2:
            continue

        # Pad to 4 levels if needed
        while len(confs) < 4:
            confs.append(confs[-1])

        # Monotonicity score: fraction of adjacent pairs where conf increases
        n_transitions = len(confs) - 1
        mono_violations = sum(1 for k in range(1, len(confs)) if confs[k] < confs[k-1])
        monotonicity_score = 1.0 - mono_violations / n_transitions

        # SpecCheck score: higher = more likely hallucinated
        confidence_gap = confs[-1] - confs[0]  # Should be positive for true claims
        speccheck_score = (1.0 - monotonicity_score) + alpha * max(0, -confidence_gap)

        # Max violation
        max_violation = max(
            (confs[k-1] - confs[k]) for k in range(1, len(confs))
        )

        # Confidence gap only
        gap_score = -confidence_gap  # Negative gap means conf didn't increase

        # Weighted monotonicity (weight earlier violations higher)
        weights = [0.5, 0.3, 0.2]
        weighted_violations = sum(
            weights[k-1] * max(0, confs[k-1] - confs[k])
            for k in range(1, min(len(confs), 4))
        )

        # Hallucination Granularity Index
        delta = 0.1
        granularity_index = -1
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
    print(f"  Computed SpecCheck scores for {len(results)} claims")
    return results


# ============================================================
# STAGE 7: Baselines
# ============================================================
def run_baseline_verbalized(model, tokenizer, model_name, dataset_name, claims):
    """Baseline: verbalized confidence."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(RESULTS_DIR, f"baseline_verbalized_{mshort}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Verbalized baseline exists: {out_path}")
        return load_json(out_path)

    print(f"  Running verbalized confidence baseline for {mshort}/{dataset_name}...")
    prompts = [
        f'Rate your confidence that the following statement is true on a scale of 0 to 100, '
        f'where 0 means certainly false and 100 means certainly true. Only output the number.\n'
        f'Statement: "{c["claim_text"]}"'
        for c in claims
    ]

    outputs = generate_text(
        model, tokenizer, prompts,
        max_new_tokens=10, temperature=0.0, do_sample=False, batch_size=8
    )

    results = []
    for claim, out in zip(claims, outputs):
        # Parse number
        numbers = re.findall(r'\d+', out)
        if numbers:
            conf = min(int(numbers[0]), 100) / 100.0
        else:
            conf = 0.5
        halluc_score = 1.0 - conf
        results.append({
            "claim_id": claim["claim_id"],
            "verbalized_confidence": conf,
            "hallucination_score": round(halluc_score, 4),
        })

    save_json(results, out_path)
    print(f"  Verbalized baseline: {len(results)} claims scored")
    return results


def run_baseline_logprob(model, tokenizer, model_name, dataset_name, claims, confidence_data):
    """Baseline: single-level logprob confidence (level 0 from SpecCheck)."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(RESULTS_DIR, f"baseline_logprob_{mshort}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  Logprob baseline exists: {out_path}")
        return load_json(out_path)

    print(f"  Computing logprob baseline for {mshort}/{dataset_name}...")
    # Also get token-level logprobs
    claim_texts = [c["claim_text"] for c in claims]
    token_logprobs = get_token_logprobs(model, tokenizer, claim_texts, batch_size=16)

    # Normalize token logprobs
    lp_array = np.array(token_logprobs)
    lp_norm = (lp_array - lp_array.min()) / (lp_array.max() - lp_array.min() + 1e-8)

    # Build confidence lookup from SpecCheck level-0
    conf_lookup = {item["claim_id"]: item["confidences"][0] for item in confidence_data}

    results = []
    for i, claim in enumerate(claims):
        yesno_conf = conf_lookup.get(claim["claim_id"], 0.5)
        token_conf = float(lp_norm[i])
        # Combined score
        combined_conf = 0.5 * yesno_conf + 0.5 * token_conf
        halluc_score = 1.0 - combined_conf

        results.append({
            "claim_id": claim["claim_id"],
            "yesno_confidence": round(yesno_conf, 4),
            "token_logprob_norm": round(token_conf, 4),
            "hallucination_score": round(halluc_score, 4),
        })

    save_json(results, out_path)
    print(f"  Logprob baseline: {len(results)} claims scored")
    return results


def run_baseline_selfcheck(model, tokenizer, model_name, dataset_name, claims, dataset):
    """Baseline: SelfCheckGPT (sampling consistency)."""
    mshort = get_model_short(model_name)
    out_path = os.path.join(RESULTS_DIR, f"baseline_selfcheck_{mshort}_{dataset_name}.json")
    if os.path.exists(out_path):
        print(f"  SelfCheck baseline exists: {out_path}")
        return load_json(out_path)

    print(f"  Running SelfCheck baseline for {mshort}/{dataset_name}...")
    N_SAMPLES = 5

    # Group claims by source prompt
    prompt_to_claims = {}
    for claim in claims:
        src = claim["source_id"]
        if src not in prompt_to_claims:
            prompt_to_claims[src] = []
        prompt_to_claims[src].append(claim)

    # Get original prompts
    id_to_prompt = {d["id"]: d["prompt"] for d in dataset}

    results_dict = {}
    for src_id, src_claims in tqdm(prompt_to_claims.items(), desc="SelfCheck"):
        prompt = id_to_prompt.get(src_id, "")
        if not prompt:
            continue

        # Generate N alternative responses
        alt_responses = generate_text(
            model, tokenizer, [prompt] * N_SAMPLES,
            max_new_tokens=256, temperature=1.0, top_p=0.95, batch_size=N_SAMPLES
        )

        # Check each claim against each alternative response
        for claim in src_claims:
            support_count = 0
            check_prompts = [
                f'Does the following passage support the claim "{claim["claim_text"]}"? '
                f'Answer with only Yes or No.\nPassage: {resp[:500]}'
                for resp in alt_responses
            ]

            check_confs = get_yes_no_logprobs(model, tokenizer, check_prompts, batch_size=N_SAMPLES)
            support_count = sum(1 for c in check_confs if c > 0.5)

            halluc_score = 1.0 - support_count / N_SAMPLES
            results_dict[claim["claim_id"]] = {
                "claim_id": claim["claim_id"],
                "support_fraction": support_count / N_SAMPLES,
                "hallucination_score": round(halluc_score, 4),
            }

    results = [results_dict[c["claim_id"]] for c in claims if c["claim_id"] in results_dict]
    save_json(results, out_path)
    print(f"  SelfCheck baseline: {len(results)} claims scored")
    return results


def run_baseline_random(model_name, dataset_name, claims):
    """Baseline: random scores."""
    mshort = get_model_short(model_name)
    all_results = {}
    for seed in SEEDS:
        out_path = os.path.join(RESULTS_DIR, f"baseline_random_{mshort}_{dataset_name}_seed{seed}.json")
        if os.path.exists(out_path):
            all_results[seed] = load_json(out_path)
            continue
        set_seed(seed)
        results = []
        for claim in claims:
            results.append({
                "claim_id": claim["claim_id"],
                "hallucination_score": round(random.random(), 4),
            })
        save_json(results, out_path)
        all_results[seed] = results
    print(f"  Random baseline: {len(claims)} claims × {len(SEEDS)} seeds")
    return all_results


# ============================================================
# MAIN PIPELINE
# ============================================================
def process_model(model_name):
    """Run the full pipeline for one model."""
    mshort = get_model_short(model_name)
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name} ({mshort})")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model(model_name, device=DEVICE)

    for dataset_name in DATASETS:
        print(f"\n--- Dataset: {dataset_name} ---")
        data = load_json(os.path.join(DATA_DIR, f"{dataset_name}_subset.json"))

        # 1. Generate outputs
        t0 = time.time()
        outputs = generate_outputs(model, tokenizer, model_name, dataset_name, data)
        print(f"  Generation: {time.time()-t0:.0f}s")

        # 2. Decompose claims
        t0 = time.time()
        claims = decompose_claims(model, tokenizer, model_name, dataset_name, outputs)
        print(f"  Decomposition: {time.time()-t0:.0f}s")

        if len(claims) < 10:
            print(f"  WARNING: Only {len(claims)} claims extracted. Skipping this dataset.")
            continue

        # 3. Label claims
        t0 = time.time()
        labeled = label_all_claims(model, tokenizer, model_name, dataset_name, claims, data)
        print(f"  Labeling: {time.time()-t0:.0f}s")

        # 4. Generate specificity ladders
        t0 = time.time()
        ladders = generate_ladders(model, tokenizer, model_name, dataset_name, labeled)
        print(f"  Ladders: {time.time()-t0:.0f}s")

        # 5. Confidence estimation
        t0 = time.time()
        confidence_data = estimate_confidence(model, tokenizer, model_name, dataset_name, ladders)
        print(f"  Confidence: {time.time()-t0:.0f}s")

        # 6. SpecCheck scores
        speccheck_scores = compute_speccheck_scores(model_name, dataset_name, confidence_data)

        # 7. Baselines
        t0 = time.time()
        run_baseline_verbalized(model, tokenizer, model_name, dataset_name, labeled)
        run_baseline_logprob(model, tokenizer, model_name, dataset_name, labeled, confidence_data)
        run_baseline_selfcheck(model, tokenizer, model_name, dataset_name, labeled, data)
        run_baseline_random(model_name, dataset_name, labeled)
        print(f"  All baselines: {time.time()-t0:.0f}s")

    # Free GPU
    unload_model(model, tokenizer)
    print(f"\nModel {mshort} complete. GPU freed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to process (default: all)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.model:
        # Find matching model
        for m in MODELS:
            if args.model.lower() in m.lower():
                process_model(m)
                break
    else:
        for model_name in MODELS:
            process_model(model_name)

    print("\n\nAll models processed!")
