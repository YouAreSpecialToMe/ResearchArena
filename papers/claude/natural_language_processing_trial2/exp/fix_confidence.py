"""
Fix NaN values in confidence data by re-computing with a more robust method.
This runs after the main pipeline to fix any NaN confidence values.
"""
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    save_json, load_json, get_model_short,
    MODELS, DATASETS, DATA_DIR, RESULTS_DIR
)

DEVICE = "cuda"


def robust_yes_no_confidence(model, tokenizer, prompt):
    """Get P(Yes) for a single prompt (no batching issues)."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            text = prompt
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position logits

    # Find Yes/No tokens
    yes_ids = set()
    no_ids = set()
    for w in ["Yes", "yes", " Yes", " yes", "YES"]:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for w in ["No", "no", " No", " no", "NO"]:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])

    if not yes_ids or not no_ids:
        return 0.5

    yes_logits = logits[list(yes_ids)]
    no_logits = logits[list(no_ids)]

    yes_prob = torch.logsumexp(yes_logits, dim=0).exp().item()
    no_prob = torch.logsumexp(no_logits, dim=0).exp().item()

    total = yes_prob + no_prob
    if total <= 0 or np.isnan(total):
        return 0.5

    return yes_prob / total


def fix_confidence_data(model_name):
    """Fix NaN values in confidence data for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mshort = get_model_short(model_name)

    # Check which datasets need fixing
    need_fix = []
    for dataset_name in DATASETS:
        conf_path = os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{dataset_name}.json")
        if not os.path.exists(conf_path):
            continue
        data = load_json(conf_path)
        nan_count = sum(1 for d in data for c in d["confidences"] if c is None or (isinstance(c, float) and np.isnan(c)))
        if nan_count > 0:
            need_fix.append((dataset_name, conf_path, data, nan_count))
            print(f"  {mshort}/{dataset_name}: {nan_count} NaN values to fix")

    if not need_fix:
        print(f"  {mshort}: No NaN values found!")
        return

    # Load model
    print(f"Loading {model_name} for confidence fix...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()

    for dataset_name, conf_path, data, nan_count in need_fix:
        ladder_path = os.path.join(DATA_DIR, f"ladders_{mshort}_{dataset_name}.json")
        if not os.path.exists(ladder_path):
            continue
        ladders = load_json(ladder_path)
        ladder_map = {l["claim_id"]: l for l in ladders}

        fixed = 0
        for item in tqdm(data, desc=f"Fixing {mshort}/{dataset_name}"):
            confs = item["confidences"]
            cid = item["claim_id"]
            ladder = ladder_map.get(cid)
            if ladder is None:
                continue

            for level_idx in range(len(confs)):
                if confs[level_idx] is None or (isinstance(confs[level_idx], float) and np.isnan(confs[level_idx])):
                    # Re-compute this confidence
                    if level_idx < len(ladder["levels"]):
                        text = ladder["levels"][level_idx]["text"]
                        prompt = f'Is the following statement true? "{text}" Answer with Yes or No.'
                        conf = robust_yes_no_confidence(model, tokenizer, prompt)
                        confs[level_idx] = round(conf, 6)
                        fixed += 1
                    else:
                        confs[level_idx] = 0.5
                        fixed += 1

            item["confidences"] = confs

        save_json(data, conf_path)
        print(f"  Fixed {fixed} NaN values in {mshort}/{dataset_name}")

        # Also recompute SpecCheck scores
        from run_pipeline import compute_speccheck_scores
        compute_speccheck_scores(model_name, dataset_name, data)

    # Free model
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.model:
        for m in MODELS:
            if args.model.lower() in m.lower():
                fix_confidence_data(m)
                break
    else:
        for m in MODELS:
            fix_confidence_data(m)
