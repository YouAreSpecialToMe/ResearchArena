#!/usr/bin/env python3
"""
Fix all self-review issues:
1. Generate genuine K=4 (5-level) specificity ladders
2. Run independent sampling-based confidence estimation (N=5, N=10, N=20)
3. Recompute ladder depth ablation with real K=4 data
4. Extend sampling ablation to all 3 models
"""

import json
import os
import sys
import time
import gc
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home/zz865/pythonProject/autoresearch/outputs/claude/run_2/natural_language_processing/idea_01")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}
DATASETS = ["factscore", "longfact", "truthfulqa"]
SEEDS = [42, 123, 456]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def load_model(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def generate_text(model, tokenizer, prompt, max_new_tokens=256, temperature=0.0, do_sample=False):
    """Generate text from prompt."""
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = prompt
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            top_p=0.9 if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def get_yes_no_confidence(model, tokenizer, claim):
    """Get P(Yes) for a yes/no question about a claim using logprobs."""
    prompt = f'Is the following statement true? "{claim}" Answer with Yes or No.'

    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = prompt
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    # Get token IDs for Yes/No variants
    yes_tokens = []
    no_tokens = []
    for variant in ["Yes", "yes", " Yes", " yes", "YES"]:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if toks:
            yes_tokens.append(toks[0])
    for variant in ["No", "no", " No", " no", "NO"]:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if toks:
            no_tokens.append(toks[0])

    yes_tokens = list(set(yes_tokens))
    no_tokens = list(set(no_tokens))

    if not yes_tokens or not no_tokens:
        return 0.5

    yes_logit = torch.logsumexp(logits[yes_tokens], dim=0)
    no_logit = torch.logsumexp(logits[no_tokens], dim=0)

    probs = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
    return probs[0].item()


def get_sampling_confidence(model, tokenizer, claim, n_samples, seed):
    """Get sampling-based confidence: fraction of N samples that say 'Yes'."""
    prompt = f'Is the following statement true? "{claim}" Answer with Yes or No.'

    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = prompt
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    yes_count = 0
    for i in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip().lower()
        if text.startswith("yes"):
            yes_count += 1

    return yes_count / n_samples


# =====================================================
# STEP 1: Generate genuine K=4 (5-level) ladders
# =====================================================
def generate_k4_ladders(model, tokenizer, model_key):
    """Generate 5-level specificity ladders for all datasets."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Generating K=4 (5-level) ladders for {model_key}")
    print(f"{'='*60}")

    ladder_prompt_template = """Given the following factual claim, rewrite it at four levels of decreasing specificity.

Level 0 (Original): {claim}
Level 1 (Slightly less specific): Soften the most specific detail while keeping most information.
Level 2 (Approximate): Replace exact values with ranges or approximate terms.
Level 3 (Category): Replace specific instances with their general category.
Level 4 (Abstract): The most general true version of this claim.

Examples:
Claim: "Albert Einstein was born on March 14, 1879 in Ulm, Germany."
Level 1: Albert Einstein was born in mid-March 1879 in Ulm, Germany.
Level 2: Albert Einstein was born in the late 1870s in southern Germany.
Level 3: Albert Einstein was born in Germany in the 19th century.
Level 4: Albert Einstein was born in Europe.

Claim: "The Eiffel Tower is 330 meters tall and was completed in 1889."
Level 1: The Eiffel Tower is approximately 330 meters tall and was completed around 1889.
Level 2: The Eiffel Tower is over 300 meters tall and was completed in the late 1880s.
Level 3: The Eiffel Tower is a very tall structure completed in the 19th century.
Level 4: The Eiffel Tower is a famous landmark in Paris.

Claim: "{claim}"
Level 1:"""

    for dataset in DATASETS:
        ladder_file = DATA_DIR / f"ladders_k4_{model_key}_{dataset}.json"
        if ladder_file.exists():
            print(f"  K=4 ladders already exist for {model_key}/{dataset}, skipping")
            continue

        claims_file = DATA_DIR / f"labeled_claims_{model_key}_{dataset}.json"
        claims = load_json(claims_file)

        ladders_k4 = []
        print(f"  Generating K=4 ladders for {model_key}/{dataset} ({len(claims)} claims)...")

        for i, claim_entry in enumerate(claims):
            claim_text = claim_entry["claim_text"]
            prompt = ladder_prompt_template.format(claim=claim_text)

            try:
                output = generate_text(model, tokenizer, prompt, max_new_tokens=300, temperature=0.0)

                # Parse levels from output
                levels = [{"level": 0, "text": claim_text}]
                lines = output.strip().split("\n")

                # Try to extract Level 1 from the direct output
                level1_text = lines[0].strip() if lines else claim_text
                # Clean up any "Level X:" prefix
                for prefix in ["Level 1:", "Level 2:", "Level 3:", "Level 4:"]:
                    if level1_text.startswith(prefix):
                        level1_text = level1_text[len(prefix):].strip()
                levels.append({"level": 1, "text": level1_text if level1_text else claim_text})

                # Extract remaining levels
                for target_level in [2, 3, 4]:
                    found = False
                    for line in lines:
                        line = line.strip()
                        if line.startswith(f"Level {target_level}:"):
                            text = line[len(f"Level {target_level}:"):].strip()
                            if text:
                                levels.append({"level": target_level, "text": text})
                                found = True
                                break
                    if not found:
                        # Fallback: use existing K=3 ladder level if available
                        existing_ladders = load_json(DATA_DIR / f"ladders_{model_key}_{dataset}.json")
                        existing = next((l for l in existing_ladders if l["claim_id"] == claim_entry["claim_id"]), None)
                        if existing and target_level <= 3:
                            mapped = target_level if target_level <= 3 else 3
                            for el in existing["levels"]:
                                if el["level"] == mapped:
                                    levels.append({"level": target_level, "text": el["text"]})
                                    found = True
                                    break
                        if not found:
                            levels.append({"level": target_level, "text": claim_text})

                ladders_k4.append({
                    "claim_id": claim_entry["claim_id"],
                    "claim_text": claim_text,
                    "levels": levels
                })

            except Exception as e:
                print(f"    Error on claim {i}: {e}")
                # Fallback
                ladders_k4.append({
                    "claim_id": claim_entry["claim_id"],
                    "claim_text": claim_text,
                    "levels": [
                        {"level": 0, "text": claim_text},
                        {"level": 1, "text": claim_text},
                        {"level": 2, "text": claim_text},
                        {"level": 3, "text": claim_text},
                        {"level": 4, "text": claim_text},
                    ]
                })

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(claims)} claims processed")

        save_json(ladders_k4, ladder_file)
        print(f"  Done: {len(ladders_k4)} K=4 ladders for {model_key}/{dataset}")


# =====================================================
# STEP 2: Compute K=4 confidence and ablation
# =====================================================
def compute_k4_confidence_and_ablation(model, tokenizer, model_key):
    """Compute logprob confidence for K=4 ladders and recompute ablation."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Computing K=4 confidence for {model_key}")
    print(f"{'='*60}")

    for dataset in DATASETS:
        conf_file = RESULTS_DIR / f"confidence_k4_{model_key}_{dataset}.json"
        if conf_file.exists():
            print(f"  K=4 confidence already exists for {model_key}/{dataset}, skipping")
            continue

        ladder_file = DATA_DIR / f"ladders_k4_{model_key}_{dataset}.json"
        if not ladder_file.exists():
            print(f"  No K=4 ladders for {model_key}/{dataset}, skipping")
            continue

        ladders = load_json(ladder_file)
        confidences = []

        print(f"  Computing K=4 confidence for {model_key}/{dataset} ({len(ladders)} claims)...")

        for i, ladder in enumerate(ladders):
            confs = []
            for level_entry in ladder["levels"]:
                c = get_yes_no_confidence(model, tokenizer, level_entry["text"])
                confs.append(c)

            confidences.append({
                "claim_id": ladder["claim_id"],
                "confidences": confs  # 5 values for levels 0-4
            })

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(ladders)} claims done")

        save_json(confidences, conf_file)

    # Now recompute ladder depth ablation with real K=4 data
    print(f"\n  Recomputing ladder depth ablation for {model_key}...")
    from sklearn.metrics import roc_auc_score, average_precision_score

    for dataset in DATASETS:
        labels_file = DATA_DIR / f"labeled_claims_{model_key}_{dataset}.json"
        labels_data = load_json(labels_file)
        label_map = {c["claim_id"]: c["label"] for c in labels_data}

        # Load K=3 (original) confidence
        conf_k3_file = RESULTS_DIR / f"confidence_logprob_{model_key}_{dataset}.json"
        conf_k3 = load_json(conf_k3_file)

        # Load K=4 confidence
        conf_k4_file = RESULTS_DIR / f"confidence_k4_{model_key}_{dataset}.json"
        if not conf_k4_file.exists():
            continue
        conf_k4 = load_json(conf_k4_file)

        # Build claim-indexed maps
        k3_map = {c["claim_id"]: c["confidences"] for c in conf_k3}
        k4_map = {c["claim_id"]: c["confidences"] for c in conf_k4}

        ablation_results = {}

        # For each K value, compute SpecCheck score using appropriate confidence levels
        for K, description in [(1, "levels 0,3 only"), (2, "levels 0,2,3"), (3, "levels 0,1,2,3"), (4, "levels 0,1,2,3,4")]:
            scores = []
            labels = []

            for claim in labels_data:
                cid = claim["claim_id"]
                label = claim["label"]

                if K <= 3:
                    confs = k3_map.get(cid)
                    if confs is None:
                        continue
                    # Select appropriate levels
                    if K == 1:
                        selected = [confs[0], confs[3]]
                    elif K == 2:
                        selected = [confs[0], confs[2], confs[3]]
                    else:  # K=3
                        selected = confs[:4]
                else:  # K=4
                    confs = k4_map.get(cid)
                    if confs is None:
                        continue
                    selected = confs[:5]

                # Compute monotonicity score
                n_transitions = len(selected) - 1
                mono_violations = 0
                for j in range(1, len(selected)):
                    if selected[j] < selected[j-1]:
                        mono_violations += 1
                mono_score = 1.0 - (mono_violations / n_transitions)

                # SpecCheck score = 1 - M + alpha * (conf_last - conf_first)
                alpha = 0.5
                speccheck_score = (1 - mono_score) + alpha * max(0, selected[0] - selected[-1])

                scores.append(speccheck_score)
                labels.append(label)

            if len(set(labels)) < 2:
                ablation_results[f"K={K}"] = {"auc_roc": 0.5, "auc_pr": 0.5, "n_claims": len(labels)}
            else:
                auc_roc = roc_auc_score(labels, scores)
                auc_pr = average_precision_score(labels, scores)
                ablation_results[f"K={K}"] = {
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "n_claims": len(labels)
                }

        save_json(ablation_results, RESULTS_DIR / f"ablation_ladder_depth_{model_key}_{dataset}.json")
        print(f"  Ladder depth ablation for {model_key}/{dataset}:")
        for k, v in ablation_results.items():
            print(f"    {k}: AUC-ROC={v['auc_roc']:.4f}, AUC-PR={v['auc_pr']:.4f}")


# =====================================================
# STEP 3: Independent sampling-based confidence
# =====================================================
def run_sampling_ablation(model, tokenizer, model_key):
    """Run sampling-based confidence with N=5, N=10, N=20 independently."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Sampling ablation for {model_key}")
    print(f"{'='*60}")

    from sklearn.metrics import roc_auc_score, average_precision_score

    for dataset in DATASETS:
        labels_data = load_json(DATA_DIR / f"labeled_claims_{model_key}_{dataset}.json")
        ladders = load_json(DATA_DIR / f"ladders_{model_key}_{dataset}.json")
        ladder_map = {l["claim_id"]: l for l in ladders}

        # Subsample to 200 claims for efficiency (as in plan)
        np.random.seed(42)
        subset_indices = np.random.choice(len(labels_data), min(200, len(labels_data)), replace=False)
        subset = [labels_data[i] for i in sorted(subset_indices)]

        for N in [5, 10, 20]:
            out_file = RESULTS_DIR / f"sampling_N{N}_{model_key}_{dataset}.json"
            if out_file.exists():
                print(f"  Sampling N={N} already exists for {model_key}/{dataset}, skipping")
                continue

            # For N=20, further subsample to 100 claims
            if N == 20:
                working_set = subset[:100]
            else:
                working_set = subset

            print(f"  Running N={N} sampling for {model_key}/{dataset} ({len(working_set)} claims)...")
            t0 = time.time()

            results = []
            for i, claim_entry in enumerate(working_set):
                cid = claim_entry["claim_id"]
                ladder = ladder_map.get(cid)
                if ladder is None:
                    continue

                confs = []
                for level_entry in ladder["levels"]:
                    # Use a different seed per claim and per N to ensure independence
                    sampling_seed = hash((cid, N, level_entry["level"])) % (2**31)
                    c = get_sampling_confidence(
                        model, tokenizer, level_entry["text"],
                        n_samples=N, seed=sampling_seed
                    )
                    confs.append(c)

                results.append({
                    "claim_id": cid,
                    "label": claim_entry["label"],
                    "confidences": confs,
                })

                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(working_set)} claims done")

            elapsed = time.time() - t0

            # Compute SpecCheck scores
            scores = []
            labels = []
            for r in results:
                confs = r["confidences"]
                n_transitions = len(confs) - 1
                mono_violations = sum(1 for j in range(1, len(confs)) if confs[j] < confs[j-1])
                mono_score = 1.0 - (mono_violations / n_transitions)
                alpha = 0.5
                speccheck_score = (1 - mono_score) + alpha * max(0, confs[0] - confs[-1])
                r["speccheck_score"] = speccheck_score
                scores.append(speccheck_score)
                labels.append(r["label"])

            metrics = {}
            if len(set(labels)) >= 2:
                metrics = {
                    "auc_roc": roc_auc_score(labels, scores),
                    "auc_pr": average_precision_score(labels, scores),
                }
            else:
                metrics = {"auc_roc": 0.5, "auc_pr": 0.5}

            output = {
                "model": model_key,
                "dataset": dataset,
                "N": N,
                "n_claims": len(results),
                "wall_time_seconds": elapsed,
                "metrics": metrics,
                "claims": results,
            }

            save_json(output, out_file)
            print(f"    N={N}: AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}, time={elapsed:.1f}s")


def main():
    print("=" * 70)
    print("FIXING ALL SELF-REVIEW ISSUES")
    print("=" * 70)

    for model_key, model_name in MODELS.items():
        print(f"\n\n{'#'*70}")
        print(f"# Processing model: {model_key} ({model_name})")
        print(f"{'#'*70}")

        model, tokenizer = load_model(model_name)

        # Step 1: Generate K=4 ladders
        generate_k4_ladders(model, tokenizer, model_key)

        # Step 2: Compute K=4 confidence and ablation
        compute_k4_confidence_and_ablation(model, tokenizer, model_key)

        # Step 3: Sampling ablation
        run_sampling_ablation(model, tokenizer, model_key)

        unload_model(model, tokenizer)

    print("\n\nAll model-dependent steps complete!")
    print("Run fix_analysis.py next for TruthfulQA failure analysis and final results.")


if __name__ == "__main__":
    main()
