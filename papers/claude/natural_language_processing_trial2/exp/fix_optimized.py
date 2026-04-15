#!/usr/bin/env python3
"""
Optimized fix for all self-review issues:
1. K=4 ladders: only llama/factscore (per plan), batched generation
2. Sampling ablation: N=5,10,20 with independent seeds, all 3 models but only factscore
3. Runs much faster by being targeted

Then calls fix_analysis.py for figures and final results.
"""

import json, os, sys, time, gc, torch
import numpy as np
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path("/home/zz865/pythonProject/autoresearch/outputs/claude/run_2/natural_language_processing/idea_01")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}
DATASETS = ["factscore", "longfact", "truthfulqa"]

def load_json(p):
    with open(p) as f: return json.load(f)

def save_json(d, p):
    with open(p, "w") as f: json.dump(d, f, indent=2)
    print(f"  Saved: {p}")

def load_model(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

def unload(model, tok):
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

def format_prompt(tok, user_msg):
    if hasattr(tok, 'apply_chat_template'):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return user_msg

def get_yes_no_conf(model, tok, claim):
    """Logprob P(Yes) for a claim."""
    prompt = format_prompt(tok, f'Is the following statement true? "{claim}" Answer with Yes or No.')
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    yes_ids, no_ids = set(), set()
    for v in ["Yes", "yes", " Yes", " yes", "YES"]:
        t = tok.encode(v, add_special_tokens=False)
        if t: yes_ids.add(t[0])
    for v in ["No", "no", " No", " no", "NO"]:
        t = tok.encode(v, add_special_tokens=False)
        if t: no_ids.add(t[0])

    if not yes_ids or not no_ids: return 0.5
    y = torch.logsumexp(logits[list(yes_ids)], 0)
    n = torch.logsumexp(logits[list(no_ids)], 0)
    return torch.softmax(torch.stack([y, n]), 0)[0].item()


def sampling_conf(model, tok, claim, n_samples, seed):
    """Sampling-based P(Yes): generate N answers, count Yes fraction."""
    prompt = format_prompt(tok, f'Is the following statement true? "{claim}" Answer with Yes or No.')
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    yes_count = 0
    for i in range(n_samples):
        # Different seed for each sample to ensure independence
        torch.manual_seed(seed + i * 7919)
        torch.cuda.manual_seed(seed + i * 7919)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, temperature=0.8,
                                 do_sample=True, top_p=0.95, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        if text.startswith("yes"):
            yes_count += 1
    return yes_count / n_samples


# =====================================================
# PART 1: K=4 Ladders (Llama/FActScore only)
# =====================================================
def generate_k4_ladders_llama(model, tok):
    print("\n" + "="*60)
    print("PART 1: K=4 Ladders (Llama/FActScore)")
    print("="*60)

    ladder_file = DATA_DIR / "ladders_k4_llama_factscore.json"
    if ladder_file.exists():
        print("  Already exists, skipping.")
        return

    claims = load_json(DATA_DIR / "labeled_claims_llama_factscore.json")

    prompt_template = """Given the claim, rewrite it at four levels of decreasing specificity.

Claim: "Albert Einstein was born on March 14, 1879 in Ulm, Germany."
Level 1 (Slightly less specific): Albert Einstein was born in mid-March 1879 in Ulm, Germany.
Level 2 (Approximate): Albert Einstein was born in the late 1870s in southern Germany.
Level 3 (Category): Albert Einstein was born in Germany in the 19th century.
Level 4 (Abstract): Albert Einstein was born in Europe.

Claim: "The population of Tokyo is approximately 13.96 million as of 2023."
Level 1 (Slightly less specific): The population of Tokyo is approximately 14 million.
Level 2 (Approximate): The population of Tokyo is over 10 million.
Level 3 (Category): Tokyo is one of the most populous cities in the world.
Level 4 (Abstract): Tokyo is a large city.

Claim: "{claim}"
Level 1 (Slightly less specific):"""

    ladders = []
    print(f"  Generating K=4 ladders for {len(claims)} claims...")

    for i, c in enumerate(claims):
        ct = c["claim_text"]
        prompt = prompt_template.format(claim=ct)
        full_prompt = format_prompt(tok, prompt)

        inputs = tok(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, temperature=0.0,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        levels = [{"level": 0, "text": ct}]
        lines = text.split("\n")

        # Parse the output
        current_level = 1
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove "Level X (...): " prefix
            for prefix_pat in [f"Level {current_level}", f"level {current_level}"]:
                if line.lower().startswith(prefix_pat.lower()):
                    # Find the colon after the prefix
                    colon_idx = line.find(":", len(prefix_pat))
                    if colon_idx >= 0:
                        line = line[colon_idx+1:].strip()
                    break

            if line and current_level <= 4:
                levels.append({"level": current_level, "text": line})
                current_level += 1

            if current_level > 4:
                break

        # Pad missing levels
        while len(levels) < 5:
            levels.append({"level": len(levels), "text": levels[-1]["text"]})

        ladders.append({"claim_id": c["claim_id"], "claim_text": ct, "levels": levels[:5]})

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(claims)} done")

    save_json(ladders, ladder_file)
    print(f"  Generated {len(ladders)} K=4 ladders")


# =====================================================
# PART 2: K=4 Confidence + Ablation
# =====================================================
def compute_k4_ablation(model, tok):
    print("\n" + "="*60)
    print("PART 2: K=4 Confidence + Ladder Depth Ablation")
    print("="*60)

    from sklearn.metrics import roc_auc_score, average_precision_score

    ladder_file = DATA_DIR / "ladders_k4_llama_factscore.json"
    conf_file = RESULTS_DIR / "confidence_k4_llama_factscore.json"

    if not ladder_file.exists():
        print("  No K=4 ladders found!")
        return

    if not conf_file.exists():
        ladders = load_json(ladder_file)
        confs_out = []
        print(f"  Computing K=4 confidence for {len(ladders)} claims...")

        for i, ladder in enumerate(ladders):
            claim_confs = []
            for lev in ladder["levels"]:
                c = get_yes_no_conf(model, tok, lev["text"])
                claim_confs.append(c)
            confs_out.append({"claim_id": ladder["claim_id"], "confidences": claim_confs})
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(ladders)} done")

        save_json(confs_out, conf_file)
    else:
        confs_out = load_json(conf_file)
        print("  K=4 confidence already computed")

    # Recompute ladder depth ablation for all model/dataset combos
    # K=1,2,3 use existing 4-level data; K=4 only available for llama/factscore
    for mk in MODELS:
        for ds in DATASETS:
            labels = load_json(DATA_DIR / f"labeled_claims_{mk}_{ds}.json")
            label_map = {c["claim_id"]: c["label"] for c in labels}

            # K=3 confidence (original 4 levels)
            k3_data = load_json(RESULTS_DIR / f"confidence_logprob_{mk}_{ds}.json")
            k3_map = {c["claim_id"]: c["confidences"] for c in k3_data}

            # K=4 only for llama/factscore
            k4_map = {}
            if mk == "llama" and ds == "factscore":
                k4_data = load_json(conf_file)
                k4_map = {c["claim_id"]: c["confidences"] for c in k4_data}

            abl = {}
            for K in [1, 2, 3] + ([4] if k4_map else []):
                scores, labs = [], []
                for c in labels:
                    cid = c["claim_id"]
                    if K <= 3:
                        confs = k3_map.get(cid)
                        if confs is None or len(confs) < 4:
                            continue
                        if K == 1: sel = [confs[0], confs[3]]
                        elif K == 2: sel = [confs[0], confs[2], confs[3]]
                        else: sel = confs[:4]
                    else:
                        confs = k4_map.get(cid)
                        if confs is None or len(confs) < 5:
                            continue
                        sel = confs[:5]

                    n_tr = len(sel) - 1
                    viol = sum(1 for j in range(1, len(sel)) if sel[j] < sel[j-1])
                    mono = 1.0 - viol / n_tr
                    alpha = 0.5
                    spec_score = (1 - mono) + alpha * max(0, sel[0] - sel[-1])
                    scores.append(spec_score)
                    labs.append(c["label"])

                if len(set(labs)) >= 2:
                    abl[f"K={K}"] = {
                        "auc_roc": roc_auc_score(labs, scores),
                        "auc_pr": average_precision_score(labs, scores),
                        "n_claims": len(labs),
                    }
                else:
                    abl[f"K={K}"] = {"auc_roc": 0.5, "auc_pr": 0.5, "n_claims": len(labs)}

            save_json(abl, RESULTS_DIR / f"ablation_ladder_depth_{mk}_{ds}.json")
            print(f"  Ladder depth {mk}/{ds}: " + ", ".join(f"{k}={v['auc_roc']:.4f}" for k,v in abl.items()))


# =====================================================
# PART 3: Sampling Ablation (N=5,10,20)
# =====================================================
def run_sampling_ablation(model, tok, model_key):
    """Run independent sampling for N=5,10,20 on factscore."""
    print(f"\n{'='*60}")
    print(f"PART 3: Sampling Ablation for {model_key}")
    print(f"{'='*60}")

    from sklearn.metrics import roc_auc_score, average_precision_score

    dataset = "factscore"  # Focus on factscore per plan
    labels = load_json(DATA_DIR / f"labeled_claims_{model_key}_{dataset}.json")
    ladders = load_json(DATA_DIR / f"ladders_{model_key}_{dataset}.json")
    lad_map = {l["claim_id"]: l for l in ladders}

    # Subsample 200 claims
    np.random.seed(42)
    idx = np.random.choice(len(labels), min(200, len(labels)), replace=False)
    subset = [labels[i] for i in sorted(idx)]

    for N in [5, 10, 20]:
        out_file = RESULTS_DIR / f"sampling_N{N}_{model_key}_{dataset}.json"
        if out_file.exists():
            print(f"  N={N} already exists, skipping")
            continue

        # N=20: only 100 claims
        working = subset[:100] if N == 20 else subset

        print(f"  N={N}, {len(working)} claims...")
        t0 = time.time()

        results = []
        for i, c in enumerate(working):
            cid = c["claim_id"]
            lad = lad_map.get(cid)
            if lad is None:
                continue

            confs = []
            for lev in lad["levels"]:
                # Unique seed per claim × level × N to ensure independence
                seed = abs(hash((cid, lev["level"], N))) % (2**31)
                conf = sampling_conf(model, tok, lev["text"], N, seed)
                confs.append(conf)

            # Compute speccheck score
            n_tr = len(confs) - 1
            viol = sum(1 for j in range(1, len(confs)) if confs[j] < confs[j-1])
            mono = 1.0 - viol / n_tr
            spec_score = (1 - mono) + 0.5 * max(0, confs[0] - confs[-1])

            results.append({
                "claim_id": cid, "label": c["label"],
                "confidences": confs, "speccheck_score": spec_score,
            })

            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(working)} done")

        elapsed = time.time() - t0

        labs = [r["label"] for r in results]
        scs = [r["speccheck_score"] for r in results]
        metrics = {}
        if len(set(labs)) >= 2:
            metrics = {"auc_roc": roc_auc_score(labs, scs), "auc_pr": average_precision_score(labs, scs)}
        else:
            metrics = {"auc_roc": 0.5, "auc_pr": 0.5}

        save_json({
            "model": model_key, "dataset": dataset, "N": N,
            "n_claims": len(results), "wall_time_seconds": elapsed,
            "metrics": metrics, "claims": results,
        }, out_file)

        print(f"    N={N}: AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}, {elapsed:.0f}s")

    # Also run on truthfulqa for cross-dataset comparison (just N=10)
    for extra_ds in ["truthfulqa", "longfact"]:
        out_file = RESULTS_DIR / f"sampling_N10_{model_key}_{extra_ds}.json"
        if out_file.exists():
            print(f"  N=10/{extra_ds} already exists, skipping")
            continue

        labels_e = load_json(DATA_DIR / f"labeled_claims_{model_key}_{extra_ds}.json")
        ladders_e = load_json(DATA_DIR / f"ladders_{model_key}_{extra_ds}.json")
        lad_map_e = {l["claim_id"]: l for l in ladders_e}

        np.random.seed(42)
        idx_e = np.random.choice(len(labels_e), min(200, len(labels_e)), replace=False)
        subset_e = [labels_e[i] for i in sorted(idx_e)]

        print(f"  N=10/{extra_ds}, {len(subset_e)} claims...")
        t0 = time.time()

        results_e = []
        for i, c in enumerate(subset_e):
            cid = c["claim_id"]
            lad = lad_map_e.get(cid)
            if lad is None:
                continue

            confs = []
            for lev in lad["levels"]:
                seed = abs(hash((cid, lev["level"], 10))) % (2**31)
                conf = sampling_conf(model, tok, lev["text"], 10, seed)
                confs.append(conf)

            n_tr = len(confs) - 1
            viol = sum(1 for j in range(1, len(confs)) if confs[j] < confs[j-1])
            mono = 1.0 - viol / n_tr
            spec_score = (1 - mono) + 0.5 * max(0, confs[0] - confs[-1])

            results_e.append({
                "claim_id": cid, "label": c["label"],
                "confidences": confs, "speccheck_score": spec_score,
            })

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(subset_e)} done")

        elapsed = time.time() - t0
        labs = [r["label"] for r in results_e]
        scs = [r["speccheck_score"] for r in results_e]
        metrics = {}
        if len(set(labs)) >= 2:
            metrics = {"auc_roc": roc_auc_score(labs, scs), "auc_pr": average_precision_score(labs, scs)}
        else:
            metrics = {"auc_roc": 0.5, "auc_pr": 0.5}

        save_json({
            "model": model_key, "dataset": extra_ds, "N": 10,
            "n_claims": len(results_e), "wall_time_seconds": elapsed,
            "metrics": metrics, "claims": results_e,
        }, out_file)

        print(f"    N=10/{extra_ds}: AUC-ROC={metrics['auc_roc']:.4f}, {elapsed:.0f}s")


def main():
    print("="*70, flush=True)
    print("OPTIMIZED FIX FOR ALL SELF-REVIEW ISSUES", flush=True)
    print("="*70, flush=True)

    # === LLAMA: K=4 ladders + K=4 confidence + sampling ===
    model, tok = load_model(MODELS["llama"])
    generate_k4_ladders_llama(model, tok)
    compute_k4_ablation(model, tok)
    run_sampling_ablation(model, tok, "llama")
    unload(model, tok)

    # === MISTRAL: sampling only ===
    model, tok = load_model(MODELS["mistral"])
    run_sampling_ablation(model, tok, "mistral")

    # Also fix ladder depth ablation (K=1,2,3 only since K=4 is llama-only)
    # Already done in compute_k4_ablation for all models
    unload(model, tok)

    # === QWEN: sampling only ===
    model, tok = load_model(MODELS["qwen"])
    run_sampling_ablation(model, tok, "qwen")
    unload(model, tok)

    print("\n" + "="*70)
    print("MODEL-DEPENDENT STEPS COMPLETE")
    print("="*70)
    print("Now run: python exp/fix_analysis.py")


if __name__ == "__main__":
    main()
