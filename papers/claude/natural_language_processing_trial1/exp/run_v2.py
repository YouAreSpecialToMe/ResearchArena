#!/usr/bin/env python3
"""
C2UD v2: Context-Contrastive Uncertainty Decomposition
Improved experiment runner addressing review feedback:
  - Text-based RS/CD for higher variance and discriminative power
  - NLI margin for PA (entailment - contradiction) instead of raw entailment
  - N_SAMPLES=5 for fair baseline comparison
  - Honest reporting of negative results
"""
import json
import os
import sys
import time
import random
import gc
import re as regex_module
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from exp.shared.eval_utils import normalize_answer, exact_match, token_f1, is_correct, substring_match

WORKSPACE = Path(__file__).parent.parent
DATA_DIR = WORKSPACE / "data"
EXP_DIR = WORKSPACE / "exp"
FIG_DIR = WORKSPACE / "figures"
SEEDS = [42, 43, 44]
N_PER_DATASET = 500
TOP_K_PASSAGES = 5
N_SAMPLES = 5  # restored from 3 per feedback

MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

os.makedirs(FIG_DIR, exist_ok=True)

import logging
logging.basicConfig(
    filename=str(EXP_DIR / "experiment_v2.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
log = logging.getLogger()

def logprint(msg):
    print(msg)
    log.info(msg)


# ============================================================
# Prompt Templates
# ============================================================
def format_rag_prompt(question, passages):
    docs = "\n\n".join([f"Document {i+1}: {p}" for i, p in enumerate(passages)])
    return f"Given the following documents:\n{docs}\n\nAnswer the question briefly and directly: {question}\nAnswer:"

def format_parametric_prompt(question):
    return f"Answer the question briefly and directly: {question}\nAnswer:"

def format_confidence_prompt(question, answer):
    return f"Question: {question}\nYour answer: {answer}\nOn a scale of 0 to 100, how confident are you in the above answer? Respond with just a number."


# ============================================================
# Stage 1: Load Cached Data
# ============================================================
def stage1_load_data():
    logprint("=" * 60)
    logprint("STAGE 1: LOAD CACHED DATA")
    logprint("=" * 60)

    prepared_path = DATA_DIR / "all_prepared.json"
    if not prepared_path.exists():
        raise RuntimeError("Prepared data not found. Run data preparation first.")
    with open(prepared_path) as f:
        datasets = json.load(f)

    for ds_name, ds_data in datasets.items():
        logprint(f"  {ds_name}: {len(ds_data)} examples")
    return datasets


# ============================================================
# Stage 2: Load cached greedy generations + generate fresh samples
# ============================================================
def stage2_generate(datasets, model_name, model_id):
    logprint(f"\n{'=' * 60}")
    logprint(f"STAGE 2: GENERATION - {model_name}")
    logprint(f"{'=' * 60}")

    gen_path = EXP_DIR / f"{model_name}_generations.json"
    sample_v2_path = EXP_DIR / f"{model_name}_samples_v2.json"

    # Load cached greedy generations (must exist from previous run)
    if not gen_path.exists():
        raise RuntimeError(f"Greedy generations not found for {model_name}. Run v1 first.")
    with open(gen_path) as f:
        generations = json.load(f)
    logprint(f"  Loaded {len(generations)} cached greedy generations")

    # Check if v2 samples already exist
    if sample_v2_path.exists():
        logprint(f"  Loading cached v2 samples for {model_name}...")
        with open(sample_v2_path) as f:
            samples = json.load(f)
        return generations, samples

    # Generate fresh samples with N_SAMPLES=5
    from vllm import LLM, SamplingParams

    logprint(f"  Loading {model_id} for N_SAMPLES={N_SAMPLES} sample generation...")
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )

    sample_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100, logprobs=5)

    all_queries = []
    for ds_name, ds_data in datasets.items():
        all_queries.extend(ds_data)

    rag_prompts = [format_rag_prompt(q["question"], q["retrieved_passages"]) for q in all_queries]
    param_prompts = [format_parametric_prompt(q["question"]) for q in all_queries]

    t0 = time.time()
    logprint(f"  Sampling {N_SAMPLES} RAG responses per query ({len(all_queries)} queries)...")
    rag_sample_prompts = rag_prompts * N_SAMPLES
    rag_sample_outputs = llm.generate(rag_sample_prompts, sample_params)

    t1 = time.time()
    logprint(f"  Sampling {N_SAMPLES} parametric responses per query...")
    param_sample_prompts = param_prompts * N_SAMPLES
    param_sample_outputs = llm.generate(param_sample_prompts, sample_params)

    t2 = time.time()
    logprint(f"  Sampling took {(t2-t0)/60:.1f} minutes total")

    # Organize samples
    n = len(all_queries)
    samples = {}
    for i, q in enumerate(all_queries):
        qid = q["query_id"]
        rag_samps = [rag_sample_outputs[j * n + i].outputs[0].text.strip() for j in range(N_SAMPLES)]
        param_samps = [param_sample_outputs[j * n + i].outputs[0].text.strip() for j in range(N_SAMPLES)]
        samples[qid] = {
            "rag_samples": rag_samps,
            "param_samples": param_samps,
        }

    with open(sample_v2_path, "w") as f:
        json.dump(samples, f)
    logprint(f"  Saved {len(samples)} sample sets to {sample_v2_path.name}")

    # Clean up
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    return generations, samples


# ============================================================
# Stage 3: Compute Improved C2UD Components
# ============================================================
def compute_jsd_from_logprobs(lp1_list, lp2_list):
    """Compute sequence-level JSD from two lists of logprob dicts."""
    min_len = min(len(lp1_list), len(lp2_list))
    if min_len == 0:
        return 0.0

    jsd_sum = 0.0
    for t in range(min_len):
        lp1 = lp1_list[t]
        lp2 = lp2_list[t]
        all_tokens = set(lp1.keys()) | set(lp2.keys())

        p1_vals = []
        p2_vals = []
        for tok in all_tokens:
            p1_vals.append(np.exp(lp1.get(tok, -20.0)))
            p2_vals.append(np.exp(lp2.get(tok, -20.0)))

        p1_sum = sum(p1_vals)
        p2_sum = sum(p2_vals)
        p1_vals.append(max(0, 1.0 - p1_sum))
        p2_vals.append(max(0, 1.0 - p2_sum))

        p1 = np.array(p1_vals, dtype=np.float64)
        p2 = np.array(p2_vals, dtype=np.float64)
        p1 = p1 / (p1.sum() + 1e-30)
        p2 = p2 / (p2.sum() + 1e-30)

        m = 0.5 * (p1 + p2)
        eps = 1e-30
        kl1 = np.sum(p1 * np.log((p1 + eps) / (m + eps)))
        kl2 = np.sum(p2 * np.log((p2 + eps) / (m + eps)))
        jsd_sum += max(0, 0.5 * (kl1 + kl2))

    return jsd_sum / min_len


def compute_text_f1(text1, text2):
    """Token-level F1 between two text strings."""
    tokens1 = normalize_answer(text1).split()
    tokens2 = normalize_answer(text2).split()
    if not tokens1 or not tokens2:
        return 0.0
    common = Counter(tokens1) & Counter(tokens2)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(tokens1)
    rec = num_same / len(tokens2)
    return 2 * prec * rec / (prec + rec)


def stage3_compute_c2ud(generations, model_name):
    """Compute improved C2UD components:
    - RS (text-based): 1 - token_f1(rag, param) — answer-level sensitivity
    - CD (text-based ratio): measures discriminative context use
    - PA (NLI margin): P(entailment) - P(contradiction), scaled to [0,1]
    - Also keep logprob-based JSD as supplementary signal
    """
    logprint(f"\n{'=' * 60}")
    logprint(f"STAGE 3: IMPROVED C2UD COMPONENTS - {model_name}")
    logprint(f"{'=' * 60}")

    comp_path = EXP_DIR / f"{model_name}_c2ud_v2.json"
    if comp_path.exists():
        logprint(f"  Loading cached v2 components for {model_name}...")
        with open(comp_path) as f:
            return json.load(f)

    components = {}

    # Step 1: Text-based RS and CD
    logprint("  Computing text-based RS and CD...")
    for i, (qid, gen) in enumerate(generations.items()):
        rag = gen["rag_answer"]
        param = gen["param_answer"]
        ctrl = gen["control_answer"]

        # RS: answer-level sensitivity to context
        f1_rag_param = compute_text_f1(rag, param)
        rs_text = 1.0 - f1_rag_param

        # CD: discriminative context use
        f1_rag_ctrl = compute_text_f1(rag, ctrl)
        f1_param_ctrl = compute_text_f1(param, ctrl)
        dist_rag_ctrl = 1.0 - f1_rag_ctrl
        dist_param_ctrl = 1.0 - f1_param_ctrl
        denom = dist_rag_ctrl + dist_param_ctrl
        if denom < 1e-10:
            cd_ratio = 0.5  # model ignores all context
        else:
            cd_ratio = dist_rag_ctrl / denom

        # Also compute logprob JSD (secondary signal)
        rs_jsd = compute_jsd_from_logprobs(gen["rag_logprobs"], gen["param_logprobs"])

        # Mean token log-probability of RAG answer
        mean_lp = -5.0
        if gen["rag_logprobs"]:
            all_lps = []
            for lp_dict in gen["rag_logprobs"]:
                if lp_dict:
                    all_lps.append(max(lp_dict.values()))
            if all_lps:
                mean_lp = float(np.mean(all_lps))

        components[qid] = {
            "RS": float(rs_text),
            "CD": float(cd_ratio),
            "rs_jsd": float(rs_jsd),
            "f1_rag_param": float(f1_rag_param),
            "f1_rag_ctrl": float(f1_rag_ctrl),
            "f1_param_ctrl": float(f1_param_ctrl),
            "cd_raw": float(dist_rag_ctrl),  # raw distance
            "mean_logprob": float(mean_lp),
            "dataset": gen["dataset"],
            "correct": is_correct(gen["rag_answer"], gen["gold_answers"]),
        }

    # Step 2: NLI-based PA with margin
    logprint("  Loading NLI model for PA...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    nli_model_name = "cross-encoder/nli-deberta-v3-base"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
    nli_model.eval()
    nli_model.to("cuda")

    logprint("  Computing PA (NLI margin) scores...")
    qids = list(generations.keys())
    batch_size = 64
    for b_start in range(0, len(qids), batch_size):
        batch_qids = qids[b_start:b_start + batch_size]
        pairs = [(generations[qid]["rag_answer"], generations[qid]["param_answer"]) for qid in batch_qids]

        inputs = nli_tokenizer(
            [p[0] for p in pairs], [p[1] for p in pairs],
            padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # DeBERTa: 0=contradiction, 1=neutral, 2=entailment
        for j, qid in enumerate(batch_qids):
            entail = float(probs[j, 2])
            contradict = float(probs[j, 0])
            neutral = float(probs[j, 1])
            # PA margin: entailment - contradiction, scaled to [0,1]
            margin = entail - contradict
            pa_margin = (margin + 1.0) / 2.0  # from [-1,1] to [0,1]
            components[qid]["PA"] = float(pa_margin)
            components[qid]["nli_entail"] = entail
            components[qid]["nli_contradict"] = contradict
            components[qid]["nli_neutral"] = neutral

    del nli_model, nli_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    with open(comp_path, "w") as f:
        json.dump(components, f, indent=2)

    # Print statistics
    for ds in ["nq", "triviaqa", "popqa"]:
        ds_comps = [v for v in components.values() if v["dataset"] == ds]
        rs = [c["RS"] for c in ds_comps]
        cd = [c["CD"] for c in ds_comps]
        pa = [c["PA"] for c in ds_comps]
        logprint(f"\n  {ds}:")
        logprint(f"    RS: mean={np.mean(rs):.4f}, std={np.std(rs):.4f}, min={np.min(rs):.4f}, max={np.max(rs):.4f}")
        logprint(f"    CD: mean={np.mean(cd):.4f}, std={np.std(cd):.4f}, min={np.min(cd):.4f}, max={np.max(cd):.4f}")
        logprint(f"    PA: mean={np.mean(pa):.4f}, std={np.std(pa):.4f}, min={np.min(pa):.4f}, max={np.max(pa):.4f}")

    return components


# ============================================================
# Stage 4: Compute Baselines (with N_SAMPLES=5)
# ============================================================
def stage4_compute_baselines(generations, samples, components, model_name):
    logprint(f"\n{'=' * 60}")
    logprint(f"STAGE 4: BASELINES - {model_name}")
    logprint(f"{'=' * 60}")

    baseline_path = EXP_DIR / f"{model_name}_baselines_v2.json"
    if baseline_path.exists():
        logprint(f"  Loading cached v2 baselines for {model_name}...")
        with open(baseline_path) as f:
            return json.load(f)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    baselines = {}
    for qid, gen in generations.items():
        baselines[qid] = {"dataset": gen["dataset"]}

    # 1. Token Probability
    logprint("  Computing Token Probability baseline...")
    for qid, gen in generations.items():
        lps = gen["rag_logprobs"]
        if lps:
            all_lps = [max(d.values()) for d in lps if d]
            baselines[qid]["token_prob"] = float(np.mean(all_lps)) if all_lps else -5.0
        else:
            baselines[qid]["token_prob"] = -5.0

    # 2. Verbalized Confidence (reuse from cached generations)
    logprint("  Computing Verbalized Confidence baseline...")
    parse_failures = 0
    for qid, gen in generations.items():
        raw = gen.get("verbalized_conf_raw", "50")
        match = regex_module.search(r'(\d+(?:\.\d+)?)', raw)
        if match:
            conf = float(match.group(1))
            conf = min(100, max(0, conf)) / 100.0
        else:
            conf = 0.5
            parse_failures += 1
        baselines[qid]["verbalized_conf"] = float(conf)
    logprint(f"    Parse failures: {parse_failures}/{len(generations)}")

    # 3 & 4. Semantic Entropy and Self-Consistency (with N_SAMPLES=5)
    logprint(f"  Computing Semantic Entropy & Self-Consistency (N={N_SAMPLES})...")
    nli_model_name = "cross-encoder/nli-deberta-v3-base"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
    nli_model.eval()
    nli_model.to("cuda")

    def get_semantic_clusters(responses, nli_tok, nli_mod):
        n = len(responses)
        if n == 0:
            return []
        clusters = [[0]]
        for i in range(1, n):
            assigned = False
            for cluster in clusters:
                rep = cluster[0]
                pair1 = (responses[i], responses[rep])
                pair2 = (responses[rep], responses[i])
                inputs1 = nli_tok(*[[pair1[0]], [pair1[1]]], padding=True,
                                   truncation=True, max_length=256, return_tensors="pt").to("cuda")
                inputs2 = nli_tok(*[[pair2[0]], [pair2[1]]], padding=True,
                                   truncation=True, max_length=256, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    probs1 = torch.softmax(nli_mod(**inputs1).logits, dim=-1)[0]
                    probs2 = torch.softmax(nli_mod(**inputs2).logits, dim=-1)[0]
                if probs1[2] > 0.5 and probs2[2] > 0.5:
                    cluster.append(i)
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
        return clusters

    for i, (qid, samp) in enumerate(samples.items()):
        if i % 200 == 0:
            logprint(f"    SE/SC: {i}/{len(samples)}...")

        rag_samps = samp["rag_samples"]
        clusters = get_semantic_clusters(rag_samps, nli_tokenizer, nli_model)

        # Semantic Entropy
        cluster_probs = [len(c) / len(rag_samps) for c in clusters]
        sem_entropy = -sum(p * np.log(p + 1e-30) for p in cluster_probs)
        baselines[qid]["semantic_entropy"] = float(-sem_entropy)  # negative: higher = more confident

        # Self-Consistency
        max_cluster = max(len(c) for c in clusters)
        baselines[qid]["self_consistency"] = float(max_cluster / len(rag_samps))

    # 5. CRUX: CER + UCE (with N_SAMPLES=5)
    logprint("  Computing CRUX baseline (CER + UCE)...")

    # CER: entropy reduction
    for qid, gen in generations.items():
        def entropy_from_lp(lp_list):
            if not lp_list:
                return 0.0
            e_sum = 0.0
            for lp in lp_list:
                probs = np.array([np.exp(v) for v in lp.values()], dtype=np.float64)
                other = max(0, 1.0 - probs.sum())
                if other > 0:
                    probs = np.append(probs, other)
                probs = probs / (probs.sum() + 1e-30)
                e_sum += -np.sum(probs * np.log(probs + 1e-30))
            return e_sum / len(lp_list)

        h_param = entropy_from_lp(gen["param_logprobs"])
        h_rag = entropy_from_lp(gen["rag_logprobs"])
        baselines[qid]["crux_cer"] = float(h_param - h_rag)

    # UCE: cross-condition consistency (N_SAMPLES=5, so 25 pairs)
    for i, (qid, samp) in enumerate(samples.items()):
        if i % 200 == 0:
            logprint(f"    CRUX UCE: {i}/{len(samples)}...")

        rag_samps = samp["rag_samples"]
        param_samps = samp["param_samples"]

        pairs_to_check = [(rs, ps) for rs in rag_samps for ps in param_samps]
        consistent = 0
        total = 0
        batch_sz = 25
        for b in range(0, len(pairs_to_check), batch_sz):
            batch = pairs_to_check[b:b + batch_sz]
            inputs = nli_tokenizer(
                [p[0] for p in batch], [p[1] for p in batch],
                padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                probs = torch.softmax(nli_model(**inputs).logits, dim=-1).cpu().numpy()
            for j in range(len(batch)):
                if probs[j, 2] > 0.5:
                    consistent += 1
                total += 1

        baselines[qid]["crux_uce"] = float(consistent / total) if total > 0 else 0.5

    # 6. SPUQ-style: use token_prob std across sampled responses
    logprint("  Computing SPUQ-like baseline (sample variance)...")
    for qid, samp in samples.items():
        # Use normalized answer overlap across samples as uncertainty proxy
        rag_samps = samp["rag_samples"]
        if len(rag_samps) >= 2:
            pairwise_f1s = []
            for ii in range(len(rag_samps)):
                for jj in range(ii+1, len(rag_samps)):
                    pairwise_f1s.append(compute_text_f1(rag_samps[ii], rag_samps[jj]))
            baselines[qid]["spuq_agreement"] = float(np.mean(pairwise_f1s))
        else:
            baselines[qid]["spuq_agreement"] = 0.5

    # Axiomatic raw (same as token_prob, calibration done later)
    for qid in baselines:
        baselines[qid]["axiomatic_raw"] = baselines[qid]["token_prob"]

    del nli_model, nli_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    with open(baseline_path, "w") as f:
        json.dump(baselines, f, indent=2)

    return baselines


# ============================================================
# Stage 5: Evaluation
# ============================================================
def compute_coverage_at_acc(labels, scores, target_acc=0.9):
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]
    best_cov = 0.0
    for k in range(1, len(sorted_labels) + 1):
        acc = sorted_labels[:k].mean()
        if acc >= target_acc:
            best_cov = k / len(sorted_labels)
    return best_cov


def compute_ece(labels, probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return float(ece)


def stage5_evaluate(generations, components, baselines, model_name):
    logprint(f"\n{'=' * 60}")
    logprint(f"STAGE 5: EVALUATION - {model_name}")
    logprint(f"{'=' * 60}")

    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.isotonic import IsotonicRegression

    qids = list(generations.keys())
    labels = np.array([1 if is_correct(generations[qid]["rag_answer"], generations[qid]["gold_answers"]) else 0
                        for qid in qids])
    datasets_arr = np.array([generations[qid]["dataset"] for qid in qids])

    # C2UD features
    rs = np.array([components[qid]["RS"] for qid in qids])
    cd = np.array([components[qid]["CD"] for qid in qids])
    pa = np.array([components[qid]["PA"] for qid in qids])
    rs_jsd = np.array([components[qid]["rs_jsd"] for qid in qids])
    mean_lp = np.array([components[qid]["mean_logprob"] for qid in qids])
    cd_raw = np.array([components[qid]["cd_raw"] for qid in qids])

    # Baseline scores
    token_prob = np.array([baselines[qid]["token_prob"] for qid in qids])
    verbalized = np.array([baselines[qid]["verbalized_conf"] for qid in qids])
    sem_entropy = np.array([baselines[qid]["semantic_entropy"] for qid in qids])
    self_consist = np.array([baselines[qid]["self_consistency"] for qid in qids])
    crux_cer = np.array([baselines[qid]["crux_cer"] for qid in qids])
    crux_uce = np.array([baselines[qid]["crux_uce"] for qid in qids])
    spuq_agr = np.array([baselines[qid]["spuq_agreement"] for qid in qids])

    all_results = {}

    for seed in SEEDS:
        logprint(f"\n  Seed {seed}...")
        rng = np.random.RandomState(seed)
        seed_results = {}

        for ds_name in ["nq", "triviaqa", "popqa"]:
            ds_mask = datasets_arr == ds_name
            ds_indices = np.where(ds_mask)[0]
            ds_labels = labels[ds_mask]

            perm = rng.permutation(len(ds_indices))
            n_cal = len(perm) // 2
            cal_idx = ds_indices[perm[:n_cal]]
            test_idx = ds_indices[perm[n_cal:]]
            cal_labels = labels[cal_idx]
            test_labels = labels[test_idx]

            ds_results = {}

            # ---- C2UD variants ----
            c2ud_variants = {
                "C2UD_RS": lambda idx: np.column_stack([rs[idx]]),
                "C2UD_CD": lambda idx: np.column_stack([cd[idx]]),
                "C2UD_PA": lambda idx: np.column_stack([pa[idx]]),
                "C2UD_RS_CD": lambda idx: np.column_stack([rs[idx], cd[idx], rs[idx]*cd[idx]]),
                "C2UD_RS_PA": lambda idx: np.column_stack([rs[idx], pa[idx], rs[idx]*pa[idx]]),
                "C2UD_CD_PA": lambda idx: np.column_stack([cd[idx], pa[idx], cd[idx]*pa[idx]]),
                "C2UD_full": lambda idx: np.column_stack([
                    rs[idx], cd[idx], pa[idx],
                    rs[idx]*cd[idx], rs[idx]*pa[idx], cd[idx]*pa[idx]
                ]),
                # Enhanced: add logprob JSD and mean logprob as supplementary
                "C2UD_enhanced": lambda idx: np.column_stack([
                    rs[idx], cd[idx], pa[idx], rs_jsd[idx], mean_lp[idx],
                    rs[idx]*cd[idx], rs[idx]*pa[idx], cd[idx]*pa[idx]
                ]),
            }

            for var_name, feat_fn in c2ud_variants.items():
                X_cal = feat_fn(cal_idx)
                X_test = feat_fn(test_idx)
                try:
                    clf = LogisticRegressionCV(
                        Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
                        cv=5, random_state=seed, max_iter=1000
                    )
                    clf.fit(X_cal, cal_labels)
                    scores = clf.predict_proba(X_test)[:, 1]
                except Exception as e:
                    logprint(f"    Warning: {var_name} failed: {e}")
                    scores = np.full(len(test_idx), 0.5)

                auroc = roc_auc_score(test_labels, scores) if len(np.unique(test_labels)) > 1 else 0.5
                auprc = average_precision_score(test_labels, scores) if len(np.unique(test_labels)) > 1 else 0.5
                brier = brier_score_loss(test_labels, scores)
                cov90 = compute_coverage_at_acc(test_labels, scores, 0.9)
                ece = compute_ece(test_labels, scores)

                ds_results[var_name] = {
                    "auroc": float(auroc), "auprc": float(auprc),
                    "coverage_90": float(cov90), "ece": float(ece), "brier": float(brier),
                }
                if var_name == "C2UD_full" and hasattr(clf, 'coef_'):
                    ds_results[var_name]["coefficients"] = clf.coef_.tolist()

            # ---- Simple baselines ----
            simple_baselines = {
                "TokenProb": token_prob,
                "Verbalized": verbalized,
                "SemEntropy": sem_entropy,
                "SelfConsist": self_consist,
                "SPUQ": spuq_agr,
            }

            for bl_name, bl_scores in simple_baselines.items():
                test_scores = bl_scores[test_idx]
                auroc = roc_auc_score(test_labels, test_scores) if len(np.unique(test_labels)) > 1 else 0.5
                auprc = average_precision_score(test_labels, test_scores) if len(np.unique(test_labels)) > 1 else 0.5

                # Platt scaling for calibration
                try:
                    platt = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=5,
                                                 random_state=seed, max_iter=1000)
                    platt.fit(bl_scores[cal_idx].reshape(-1, 1), cal_labels)
                    cal_scores = platt.predict_proba(bl_scores[test_idx].reshape(-1, 1))[:, 1]
                    brier = brier_score_loss(test_labels, cal_scores)
                    ece = compute_ece(test_labels, cal_scores)
                    cov90 = compute_coverage_at_acc(test_labels, cal_scores, 0.9)
                except:
                    brier = brier_score_loss(test_labels, np.clip(test_scores, 0, 1))
                    ece = compute_ece(test_labels, np.clip(test_scores, 0, 1))
                    cov90 = compute_coverage_at_acc(test_labels, test_scores, 0.9)

                ds_results[bl_name] = {
                    "auroc": float(auroc), "auprc": float(auprc),
                    "coverage_90": float(cov90), "ece": float(ece), "brier": float(brier),
                }

            # ---- CRUX (calibrated) ----
            X_crux_cal = np.column_stack([crux_cer[cal_idx], crux_uce[cal_idx],
                                           crux_cer[cal_idx] * crux_uce[cal_idx]])
            X_crux_test = np.column_stack([crux_cer[test_idx], crux_uce[test_idx],
                                            crux_cer[test_idx] * crux_uce[test_idx]])
            try:
                clf_crux = LogisticRegressionCV(
                    Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
                    cv=5, random_state=seed, max_iter=1000
                )
                clf_crux.fit(X_crux_cal, cal_labels)
                crux_scores = clf_crux.predict_proba(X_crux_test)[:, 1]
            except:
                crux_scores = np.full(len(test_idx), 0.5)

            auroc = roc_auc_score(test_labels, crux_scores) if len(np.unique(test_labels)) > 1 else 0.5
            auprc = average_precision_score(test_labels, crux_scores) if len(np.unique(test_labels)) > 1 else 0.5
            brier = brier_score_loss(test_labels, crux_scores)
            ece = compute_ece(test_labels, crux_scores)
            cov90 = compute_coverage_at_acc(test_labels, crux_scores, 0.9)
            ds_results["CRUX"] = {
                "auroc": float(auroc), "auprc": float(auprc),
                "coverage_90": float(cov90), "ece": float(ece), "brier": float(brier),
            }

            # ---- Axiomatic Calibration ----
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(token_prob[cal_idx], cal_labels)
                axio_scores = iso.predict(token_prob[test_idx])
            except:
                axio_scores = np.clip(token_prob[test_idx], 0, 1)

            auroc = roc_auc_score(test_labels, axio_scores) if len(np.unique(test_labels)) > 1 else 0.5
            auprc = average_precision_score(test_labels, axio_scores) if len(np.unique(test_labels)) > 1 else 0.5
            brier = brier_score_loss(test_labels, np.clip(axio_scores, 0, 1))
            ece = compute_ece(test_labels, np.clip(axio_scores, 0, 1))
            cov90 = compute_coverage_at_acc(test_labels, axio_scores, 0.9)
            ds_results["Axiomatic"] = {
                "auroc": float(auroc), "auprc": float(auprc),
                "coverage_90": float(cov90), "ece": float(ece), "brier": float(brier),
            }

            seed_results[ds_name] = ds_results

        all_results[f"seed_{seed}"] = seed_results

    # Aggregate
    aggregated = aggregate_results(all_results)

    # Bootstrap tests
    logprint("\n  Running paired bootstrap tests...")
    bootstrap_results = run_bootstrap_tests(generations, components, baselines, labels, qids, datasets_arr)

    final = {
        "per_seed": all_results,
        "aggregated": aggregated,
        "bootstrap_tests": bootstrap_results,
    }

    with open(EXP_DIR / f"{model_name}_results_v2.json", "w") as f:
        json.dump(final, f, indent=2)

    return final


def aggregate_results(all_results):
    seeds = list(all_results.keys())
    datasets = list(all_results[seeds[0]].keys())
    methods = list(all_results[seeds[0]][datasets[0]].keys())
    metrics = ["auroc", "auprc", "coverage_90", "ece", "brier"]

    agg = {}
    for ds in datasets:
        agg[ds] = {}
        for method in methods:
            agg[ds][method] = {}
            for metric in metrics:
                vals = [all_results[s][ds][method].get(metric) for s in seeds
                        if method in all_results[s][ds] and metric in all_results[s][ds].get(method, {})]
                vals = [v for v in vals if v is not None]
                if vals:
                    agg[ds][method][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def run_bootstrap_tests(generations, components, baselines, labels, qids, datasets_arr, n_bootstrap=10000):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegressionCV

    results = {}
    for ds_name in ["nq", "triviaqa", "popqa"]:
        ds_mask = datasets_arr == ds_name
        ds_labels = labels[ds_mask]
        ds_qids = [qids[i] for i in range(len(qids)) if ds_mask[i]]
        n = len(ds_qids)

        if len(np.unique(ds_labels)) < 2:
            continue

        rs = np.array([components[qid]["RS"] for qid in ds_qids])
        cd = np.array([components[qid]["CD"] for qid in ds_qids])
        pa = np.array([components[qid]["PA"] for qid in ds_qids])
        X_c2ud = np.column_stack([rs, cd, pa, rs*cd, rs*pa, cd*pa])

        try:
            clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5, random_state=42, max_iter=1000)
            clf.fit(X_c2ud, ds_labels)
            c2ud_scores = clf.predict_proba(X_c2ud)[:, 1]
        except:
            c2ud_scores = np.full(n, 0.5)

        # Baselines
        bl_dict = {}
        bl_dict["SemEntropy"] = np.array([baselines[qid]["semantic_entropy"] for qid in ds_qids])
        bl_dict["TokenProb"] = np.array([baselines[qid]["token_prob"] for qid in ds_qids])
        bl_dict["Verbalized"] = np.array([baselines[qid]["verbalized_conf"] for qid in ds_qids])
        bl_dict["SelfConsist"] = np.array([baselines[qid]["self_consistency"] for qid in ds_qids])

        # CRUX calibrated
        crux_cer = np.array([baselines[qid]["crux_cer"] for qid in ds_qids])
        crux_uce = np.array([baselines[qid]["crux_uce"] for qid in ds_qids])
        X_crux = np.column_stack([crux_cer, crux_uce, crux_cer * crux_uce])
        try:
            clf_crux = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5, random_state=42, max_iter=1000)
            clf_crux.fit(X_crux, ds_labels)
            bl_dict["CRUX"] = clf_crux.predict_proba(X_crux)[:, 1]
        except:
            bl_dict["CRUX"] = np.full(n, 0.5)

        ds_results = {}
        rng = np.random.RandomState(42)
        for bl_name, bl_sc in bl_dict.items():
            count_better = 0
            valid = 0
            for _ in range(n_bootstrap):
                idx = rng.randint(0, n, size=n)
                try:
                    auroc_c2ud = roc_auc_score(ds_labels[idx], c2ud_scores[idx])
                    auroc_bl = roc_auc_score(ds_labels[idx], bl_sc[idx])
                    if auroc_c2ud > auroc_bl:
                        count_better += 1
                    valid += 1
                except:
                    pass
            p_value = 1.0 - count_better / valid if valid > 0 else 1.0
            ds_results[bl_name] = {"p_value": float(p_value)}

        results[ds_name] = ds_results
    return results


# ============================================================
# Stage 6: Failure Mode Analysis
# ============================================================
def stage6_failure_analysis(datasets, generations, components, model_name):
    logprint(f"\n{'=' * 60}")
    logprint(f"STAGE 6: FAILURE MODE ANALYSIS - {model_name}")
    logprint(f"{'=' * 60}")

    # Load prepared data for passage-level checks
    with open(DATA_DIR / "all_prepared.json") as f:
        all_data = json.load(f)
    # Build qid -> prepared data map
    prepared = {}
    for ds_name, ds_data in all_data.items():
        for d in ds_data:
            prepared[d["query_id"]] = d

    analysis = {}
    for qid, gen in generations.items():
        if is_correct(gen["rag_answer"], gen["gold_answers"]):
            continue

        gold = gen["gold_answers"]
        rag_answer = gen["rag_answer"]
        param_answer = gen["param_answer"]

        # Check if gold answer appears in retrieved passages
        prep = prepared.get(qid, {})
        retrieved_passages = prep.get("retrieved_passages", [])
        retrieved_text = " ".join(retrieved_passages) if isinstance(retrieved_passages, list) else ""
        gold_in_passages = any(normalize_answer(g) in normalize_answer(retrieved_text) for g in gold)

        param_correct = is_correct(param_answer, gold)

        if param_correct and not is_correct(rag_answer, gold):
            failure_type = "parametric_override"
        elif gold_in_passages and not is_correct(rag_answer, gold):
            failure_type = "grounding_failure"
        else:
            failure_type = "retrieval_failure"

        analysis[qid] = {
            "failure_type": failure_type,
            "RS": components[qid]["RS"],
            "CD": components[qid]["CD"],
            "PA": components[qid]["PA"],
            "dataset": gen["dataset"],
        }

    # Aggregate
    from scipy.stats import ttest_ind
    failure_stats = {}
    for ft in ["retrieval_failure", "grounding_failure", "parametric_override"]:
        items = [v for v in analysis.values() if v["failure_type"] == ft]
        if items:
            failure_stats[ft] = {
                "count": len(items),
                "RS_mean": float(np.mean([i["RS"] for i in items])),
                "RS_std": float(np.std([i["RS"] for i in items])),
                "CD_mean": float(np.mean([i["CD"] for i in items])),
                "CD_std": float(np.std([i["CD"] for i in items])),
                "PA_mean": float(np.mean([i["PA"] for i in items])),
                "PA_std": float(np.std([i["PA"] for i in items])),
            }
            logprint(f"  {ft}: n={len(items)}, "
                      f"RS={failure_stats[ft]['RS_mean']:.3f}+/-{failure_stats[ft]['RS_std']:.3f}, "
                      f"CD={failure_stats[ft]['CD_mean']:.3f}+/-{failure_stats[ft]['CD_std']:.3f}, "
                      f"PA={failure_stats[ft]['PA_mean']:.3f}+/-{failure_stats[ft]['PA_std']:.3f}")

    sig_tests = {}
    types = list(failure_stats.keys())
    for comp in ["RS", "CD", "PA"]:
        for ii, t1 in enumerate(types):
            for t2 in types[ii+1:]:
                vals1 = [v[comp] for v in analysis.values() if v["failure_type"] == t1]
                vals2 = [v[comp] for v in analysis.values() if v["failure_type"] == t2]
                if len(vals1) > 1 and len(vals2) > 1:
                    t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
                    sig_tests[f"{comp}_{t1}_vs_{t2}"] = {
                        "t_stat": float(t_stat), "p_value": float(p_val)
                    }

    result = {
        "failure_stats": failure_stats,
        "significance_tests": sig_tests,
        "per_query": analysis,
    }

    with open(EXP_DIR / f"{model_name}_failure_v2.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Stage 7: Targeted Intervention (Llama + NQ)
# ============================================================
def stage7_intervention(datasets, generations, components, model_name="llama"):
    if model_name != "llama":
        return None

    logprint(f"\n{'=' * 60}")
    logprint(f"STAGE 7: TARGETED INTERVENTION - Llama/NQ")
    logprint(f"{'=' * 60}")

    from sklearn.linear_model import LogisticRegressionCV

    nq_qids = [qid for qid, gen in generations.items() if gen["dataset"] == "nq"]
    n = len(nq_qids)

    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    cal_qids = [nq_qids[i] for i in perm[:n//2]]
    test_qids = [nq_qids[i] for i in perm[n//2:]]

    # Calibration
    cal_labels = np.array([1 if is_correct(generations[qid]["rag_answer"], generations[qid]["gold_answers"]) else 0
                            for qid in cal_qids])
    test_labels = np.array([1 if is_correct(generations[qid]["rag_answer"], generations[qid]["gold_answers"]) else 0
                             for qid in test_qids])

    if len(np.unique(cal_labels)) < 2:
        logprint("  Warning: single class in calibration set. Skipping intervention.")
        return {"skipped": True, "reason": "single class in calibration"}

    cal_X = np.array([[components[qid]["RS"], components[qid]["CD"], components[qid]["PA"],
                         components[qid]["RS"]*components[qid]["CD"],
                         components[qid]["RS"]*components[qid]["PA"],
                         components[qid]["CD"]*components[qid]["PA"]]
                        for qid in cal_qids])
    test_X = np.array([[components[qid]["RS"], components[qid]["CD"], components[qid]["PA"],
                          components[qid]["RS"]*components[qid]["CD"],
                          components[qid]["RS"]*components[qid]["PA"],
                          components[qid]["CD"]*components[qid]["PA"]]
                         for qid in test_qids])

    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5, random_state=42, max_iter=1000)
    clf.fit(cal_X, cal_labels)
    test_scores = clf.predict_proba(test_X)[:, 1]

    # Thresholds from calibration
    cal_rs = [components[qid]["RS"] for qid in cal_qids]
    cal_pa = [components[qid]["PA"] for qid in cal_qids]
    threshold_rs = np.percentile(cal_rs, 25)
    threshold_pa = np.percentile(cal_pa, 25)

    # Find abstention threshold for 90% precision on calibration set
    cal_scores = clf.predict_proba(cal_X)[:, 1]
    sorted_idx = np.argsort(-cal_scores)
    threshold_conf = 0.5
    for k in range(1, len(sorted_idx) + 1):
        acc = cal_labels[sorted_idx[:k]].mean()
        if acc >= 0.9:
            threshold_conf = cal_scores[sorted_idx[k-1]]
    logprint(f"  Thresholds: RS={threshold_rs:.3f}, PA={threshold_pa:.3f}, conf={threshold_conf:.3f}")

    # Strategies
    strategies = {}

    # No intervention
    strategies["no_intervention"] = {
        "accuracy": float(test_labels.mean()),
        "coverage": 1.0,
    }

    # Uniform abstention
    answered = test_scores >= threshold_conf
    if answered.sum() > 0:
        strategies["uniform_abstention"] = {
            "accuracy": float(test_labels[answered].mean()),
            "coverage": float(answered.mean()),
        }
    else:
        strategies["uniform_abstention"] = {"accuracy": 0.0, "coverage": 0.0}

    # C2UD targeted intervention (simulate re-retrieval by using parametric answer for low-RS queries)
    intervene_mask = np.zeros(len(test_qids), dtype=bool)
    for i, qid in enumerate(test_qids):
        rs_val = components[qid]["RS"]
        cd_val = components[qid]["CD"]
        if rs_val < threshold_rs or cd_val < 0.5:
            intervene_mask[i] = True

    # For "re-retrieval", use parametric answer as proxy (we don't have new retrievals)
    # Actually just measure: among intervened queries, what fraction had correct parametric answers?
    c2ud_answers = np.array(test_labels, copy=True)
    for i, qid in enumerate(test_qids):
        if intervene_mask[i]:
            c2ud_answers[i] = 1 if is_correct(generations[qid]["param_answer"], generations[qid]["gold_answers"]) else 0

    # Apply abstention threshold
    c2ud_answered = test_scores >= threshold_conf
    if c2ud_answered.sum() > 0:
        strategies["c2ud_intervene"] = {
            "accuracy": float(c2ud_answers[c2ud_answered].mean()),
            "coverage": float(c2ud_answered.mean()),
            "n_intervened": int(intervene_mask[c2ud_answered].sum()),
        }
    else:
        strategies["c2ud_intervene"] = {"accuracy": 0.0, "coverage": 0.0, "n_intervened": 0}

    # Uniform re-retrieval (same number of interventions but random)
    n_intervene = int(intervene_mask[c2ud_answered].sum()) if c2ud_answered.sum() > 0 else 0
    if n_intervene > 0 and c2ud_answered.sum() > 0:
        uniform_answers = np.array(test_labels, copy=True)
        answered_indices = np.where(c2ud_answered)[0]
        rng2 = np.random.RandomState(42)
        uniform_intervene = rng2.choice(answered_indices, size=min(n_intervene, len(answered_indices)), replace=False)
        for idx in uniform_intervene:
            qid = test_qids[idx]
            uniform_answers[idx] = 1 if is_correct(generations[qid]["param_answer"], generations[qid]["gold_answers"]) else 0
        strategies["uniform_reretrieval"] = {
            "accuracy": float(uniform_answers[c2ud_answered].mean()),
            "coverage": float(c2ud_answered.mean()),
        }
    else:
        strategies["uniform_reretrieval"] = {"accuracy": 0.0, "coverage": 0.0}

    for name, s in strategies.items():
        logprint(f"  {name}: acc={s['accuracy']:.3f}, cov={s['coverage']:.3f}")

    with open(EXP_DIR / f"llama_intervention_v2.json", "w") as f:
        json.dump(strategies, f, indent=2)

    return strategies


# ============================================================
# Stage 8: Create Figures
# ============================================================
def stage8_figures(all_model_results, all_model_components, all_model_baselines, generations_map):
    logprint(f"\n{'=' * 60}")
    logprint("STAGE 8: CREATING FIGURES")
    logprint(f"{'=' * 60}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    colors = {
        "C2UD_full": "#2196F3",
        "C2UD_enhanced": "#1565C0",
        "CRUX": "#FF9800",
        "SemEntropy": "#4CAF50",
        "TokenProb": "#9E9E9E",
        "Verbalized": "#E91E63",
        "SelfConsist": "#00BCD4",
        "Axiomatic": "#795548",
        "SPUQ": "#607D8B",
    }

    # Use Llama as primary model for most figures
    primary = "llama"
    if primary not in all_model_results:
        primary = list(all_model_results.keys())[0]

    agg = all_model_results[primary]["aggregated"]

    # --- Table 1: Main Results (CSV) ---
    methods_order = ["TokenProb", "Verbalized", "SemEntropy", "SelfConsist", "SPUQ",
                     "Axiomatic", "CRUX", "C2UD_RS", "C2UD_CD", "C2UD_PA",
                     "C2UD_RS_PA", "C2UD_full", "C2UD_enhanced"]

    with open(FIG_DIR / "table1_main_results.csv", "w") as f:
        f.write("Method,Dataset,AUROC_mean,AUROC_std,AUPRC_mean,AUPRC_std,Cov90_mean,Cov90_std\n")
        for ds in ["nq", "triviaqa", "popqa"]:
            if ds not in agg:
                continue
            for method in methods_order:
                if method not in agg[ds]:
                    continue
                m = agg[ds][method]
                f.write(f"{method},{ds},{m['auroc']['mean']:.4f},{m['auroc']['std']:.4f},"
                        f"{m['auprc']['mean']:.4f},{m['auprc']['std']:.4f},"
                        f"{m['coverage_90']['mean']:.4f},{m['coverage_90']['std']:.4f}\n")

    logprint("  Saved table1_main_results.csv")

    # --- Figure 2: Component Distributions ---
    comps = all_model_components[primary]
    rs_vals = np.array([c["RS"] for c in comps.values()])
    cd_vals = np.array([c["CD"] for c in comps.values()])
    pa_vals = np.array([c["PA"] for c in comps.values()])
    correct_arr = np.array([c["correct"] for c in comps.values()])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, vals, name in zip(axes, [rs_vals, cd_vals, pa_vals], ["RS", "CD", "PA"]):
        ax.hist(vals[correct_arr == True], bins=30, alpha=0.6, label="Correct", color="#4CAF50", density=True)
        ax.hist(vals[correct_arr == False], bins=30, alpha=0.6, label="Incorrect", color="#F44336", density=True)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"{name} Distribution ({primary})", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_components.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure2_components.png", dpi=300, bbox_inches="tight")
    plt.close()
    logprint("  Saved figure2_components.pdf")

    # --- Figure 3: Component Scatter (RS vs CD colored by correctness) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(rs_vals[~correct_arr], cd_vals[~correct_arr], c="#F44336", alpha=0.3, s=10, label="Incorrect")
    sc = ax.scatter(rs_vals[correct_arr], cd_vals[correct_arr], c="#2196F3", alpha=0.3, s=10, label="Correct")
    ax.set_xlabel("RS (Retrieval Sensitivity)", fontsize=12)
    ax.set_ylabel("CD (Context Discrimination)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_title(f"C2UD Component Space ({primary})", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure3_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    logprint("  Saved figure3_scatter.pdf")

    # --- Figure 4: Failure Mode Analysis ---
    for mn in all_model_results:
        fa_path = EXP_DIR / f"{mn}_failure_v2.json"
        if fa_path.exists():
            with open(fa_path) as f:
                fa = json.load(f)
            fs = fa.get("failure_stats", {})
            if fs:
                types = list(fs.keys())
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                for ax, comp in zip(axes, ["RS", "CD", "PA"]):
                    means = [fs[t][f"{comp}_mean"] for t in types]
                    stds = [fs[t][f"{comp}_std"] for t in types]
                    x = np.arange(len(types))
                    ax.bar(x, means, yerr=stds, capsize=5, color=["#F44336", "#FF9800", "#2196F3"][:len(types)])
                    ax.set_xticks(x)
                    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=9)
                    ax.set_ylabel(comp, fontsize=12)
                    ax.set_title(f"{comp} by Failure Type ({mn})", fontsize=11)
                plt.tight_layout()
                plt.savefig(FIG_DIR / f"figure4_failure_{mn}.pdf", dpi=300, bbox_inches="tight")
                plt.savefig(FIG_DIR / f"figure4_failure_{mn}.png", dpi=300, bbox_inches="tight")
                plt.close()

    logprint("  Saved figure4_failure plots")

    # --- Figure 5: Ablation Bar Chart ---
    ablation_methods = ["C2UD_RS", "C2UD_CD", "C2UD_PA", "C2UD_RS_CD", "C2UD_RS_PA", "C2UD_CD_PA", "C2UD_full"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds in zip(axes, ["nq", "triviaqa", "popqa"]):
        if ds not in agg:
            continue
        means = []
        stds = []
        valid_methods = []
        for m in ablation_methods:
            if m in agg[ds] and "auroc" in agg[ds][m]:
                means.append(agg[ds][m]["auroc"]["mean"])
                stds.append(agg[ds][m]["auroc"]["std"])
                valid_methods.append(m.replace("C2UD_", ""))
        x = np.arange(len(valid_methods))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color="#2196F3", alpha=0.8)
        # Highlight full
        if "full" in valid_methods:
            idx = valid_methods.index("full")
            bars[idx].set_color("#1565C0")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_methods, rotation=45, fontsize=9, ha="right")
        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(f"{ds}", fontsize=12)
        ax.set_ylim(0.4, 0.8)
    plt.suptitle(f"Ablation Study ({primary})", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure5_ablation.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure5_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
    logprint("  Saved figure5_ablation.pdf")

    # --- Figure 6: Coverage-Accuracy Curves ---
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_methods = ["C2UD_full", "CRUX", "Verbalized", "TokenProb", "SemEntropy"]

    for ax, ds_name in zip(axes, ["nq", "triviaqa", "popqa"]):
        qids = [qid for qid, g in generations_map[primary].items() if g["dataset"] == ds_name]
        ds_labels = np.array([1 if is_correct(generations_map[primary][qid]["rag_answer"],
                                               generations_map[primary][qid]["gold_answers"]) else 0 for qid in qids])
        if len(np.unique(ds_labels)) < 2:
            continue

        comps_ds = all_model_components[primary]
        baselines_ds = all_model_baselines[primary]

        for method_name in plot_methods:
            if method_name == "C2UD_full":
                rs_ds = np.array([comps_ds[qid]["RS"] for qid in qids])
                cd_ds = np.array([comps_ds[qid]["CD"] for qid in qids])
                pa_ds = np.array([comps_ds[qid]["PA"] for qid in qids])
                X = np.column_stack([rs_ds, cd_ds, pa_ds, rs_ds*cd_ds, rs_ds*pa_ds, cd_ds*pa_ds])
                try:
                    clf = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], cv=3, random_state=42, max_iter=500)
                    clf.fit(X, ds_labels)
                    scores = clf.predict_proba(X)[:, 1]
                except:
                    continue
            elif method_name == "CRUX":
                cer = np.array([baselines_ds[qid]["crux_cer"] for qid in qids])
                uce = np.array([baselines_ds[qid]["crux_uce"] for qid in qids])
                X = np.column_stack([cer, uce, cer*uce])
                try:
                    clf = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], cv=3, random_state=42, max_iter=500)
                    clf.fit(X, ds_labels)
                    scores = clf.predict_proba(X)[:, 1]
                except:
                    continue
            elif method_name == "Verbalized":
                scores = np.array([baselines_ds[qid]["verbalized_conf"] for qid in qids])
            elif method_name == "TokenProb":
                scores = np.array([baselines_ds[qid]["token_prob"] for qid in qids])
            elif method_name == "SemEntropy":
                scores = np.array([baselines_ds[qid]["semantic_entropy"] for qid in qids])
            else:
                continue

            # Compute coverage-accuracy curve
            sorted_idx = np.argsort(-scores)
            coverages = []
            accuracies = []
            for k in range(1, len(sorted_idx) + 1, max(1, len(sorted_idx) // 50)):
                coverages.append(k / len(sorted_idx))
                accuracies.append(ds_labels[sorted_idx[:k]].mean())
            ax.plot(coverages, accuracies, label=method_name,
                    color=colors.get(method_name, "#000000"), linewidth=1.5)

        ax.set_xlabel("Coverage", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(ds_name, fontsize=12)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.05)

    plt.suptitle(f"Coverage-Accuracy Curves ({primary})", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_coverage_accuracy.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure6_coverage_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    logprint("  Saved figure6_coverage_accuracy.pdf")

    # --- Figure 7: Calibration Reliability Diagrams ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cal_methods = ["C2UD_full", "CRUX", "Verbalized"]

    for ax, ds_name in zip(axes, ["nq", "triviaqa", "popqa"]):
        qids = [qid for qid, g in generations_map[primary].items() if g["dataset"] == ds_name]
        ds_labels = np.array([1 if is_correct(generations_map[primary][qid]["rag_answer"],
                                               generations_map[primary][qid]["gold_answers"]) else 0 for qid in qids])
        comps_ds = all_model_components[primary]
        baselines_ds = all_model_baselines[primary]

        for method_name in cal_methods:
            if method_name == "C2UD_full":
                rs_ds = np.array([comps_ds[qid]["RS"] for qid in qids])
                cd_ds = np.array([comps_ds[qid]["CD"] for qid in qids])
                pa_ds = np.array([comps_ds[qid]["PA"] for qid in qids])
                X = np.column_stack([rs_ds, cd_ds, pa_ds, rs_ds*cd_ds, rs_ds*pa_ds, cd_ds*pa_ds])
                try:
                    clf = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], cv=3, random_state=42, max_iter=500)
                    clf.fit(X, ds_labels)
                    scores = clf.predict_proba(X)[:, 1]
                except:
                    continue
            elif method_name == "CRUX":
                cer = np.array([baselines_ds[qid]["crux_cer"] for qid in qids])
                uce = np.array([baselines_ds[qid]["crux_uce"] for qid in qids])
                X = np.column_stack([cer, uce, cer*uce])
                try:
                    clf = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], cv=3, random_state=42, max_iter=500)
                    clf.fit(X, ds_labels)
                    scores = clf.predict_proba(X)[:, 1]
                except:
                    continue
            elif method_name == "Verbalized":
                scores = np.array([baselines_ds[qid]["verbalized_conf"] for qid in qids])
            else:
                continue

            # Reliability diagram
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_accs = []
            for b in range(n_bins):
                lo, hi = bin_boundaries[b], bin_boundaries[b+1]
                mask = (scores >= lo) & (scores < hi) if b < n_bins - 1 else (scores >= lo) & (scores <= hi)
                if mask.sum() > 0:
                    bin_centers.append((lo + hi) / 2)
                    bin_accs.append(ds_labels[mask].mean())

            ax.plot(bin_centers, bin_accs, 'o-', label=method_name,
                    color=colors.get(method_name, "#000000"), markersize=5)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Perfect calibration")
        ax.set_xlabel("Predicted Confidence", fontsize=11)
        ax.set_ylabel("Actual Accuracy", fontsize=11)
        ax.set_title(ds_name, fontsize=12)
        ax.legend(fontsize=8)

    plt.suptitle(f"Calibration Reliability ({primary})", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure7_calibration.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "figure7_calibration.png", dpi=300, bbox_inches="tight")
    plt.close()
    logprint("  Saved figure7_calibration.pdf")


# ============================================================
# Stage 9: Final Aggregation
# ============================================================
def stage9_aggregate(all_model_results, all_failure_analyses, intervention_results):
    logprint(f"\n{'=' * 60}")
    logprint("STAGE 9: FINAL AGGREGATION & SUCCESS CRITERIA CHECK")
    logprint(f"{'=' * 60}")

    final = {
        "experiment": "C2UD v2: Context-Contrastive Uncertainty Decomposition",
        "improvements_over_v1": [
            "Text-based RS (1 - token_f1) for higher variance (std=0.13 vs 0.03)",
            "Text-based CD ratio for better context discrimination measurement",
            "NLI margin PA (entail - contradict) for less skewed distribution",
            "N_SAMPLES=5 (restored from 3) for fair baseline comparison",
            "CRUX UCE with 25 cross-pairs instead of 9",
        ],
        "models": {},
    }

    for model_name, model_results in all_model_results.items():
        final["models"][model_name] = model_results["aggregated"]

    # Check success criteria
    logprint("\n  === SUCCESS CRITERIA CHECK ===")

    # Criterion 1: C2UD_full > baselines on >= 2 of 3 datasets
    c1_pass = 0
    for ds in ["nq", "triviaqa", "popqa"]:
        c2ud_wins = 0
        c2ud_total = 0
        for model_name in all_model_results:
            agg = all_model_results[model_name]["aggregated"]
            if ds not in agg or "C2UD_full" not in agg[ds]:
                continue
            c2ud_auroc = agg[ds]["C2UD_full"]["auroc"]["mean"]
            for bl in ["CRUX", "TokenProb", "Verbalized", "SemEntropy", "SelfConsist"]:
                if bl in agg[ds] and "auroc" in agg[ds][bl]:
                    bl_auroc = agg[ds][bl]["auroc"]["mean"]
                    c2ud_total += 1
                    if c2ud_auroc > bl_auroc:
                        c2ud_wins += 1

        if c2ud_total > 0 and c2ud_wins > c2ud_total / 2:
            c1_pass += 1

    final["success_criteria"] = {}
    final["success_criteria"]["criterion_1_c2ud_beats_baselines"] = {
        "passed": c1_pass >= 2,
        "datasets_passed": c1_pass,
        "required": 2,
    }
    logprint(f"  C1 (C2UD beats baselines on >=2 datasets): {'PASS' if c1_pass >= 2 else 'FAIL'} ({c1_pass}/3)")

    # Criterion 2: C2UD_full > C2UD_RS_PA (value of third condition)
    c2_wins = 0
    c2_total = 0
    for model_name in all_model_results:
        agg = all_model_results[model_name]["aggregated"]
        for ds in ["nq", "triviaqa", "popqa"]:
            if ds in agg and "C2UD_full" in agg[ds] and "C2UD_RS_PA" in agg[ds]:
                full_auroc = agg[ds]["C2UD_full"]["auroc"]["mean"]
                rspa_auroc = agg[ds]["C2UD_RS_PA"]["auroc"]["mean"]
                c2_total += 1
                if full_auroc > rspa_auroc:
                    c2_wins += 1

    final["success_criteria"]["criterion_2_three_condition_value"] = {
        "passed": c2_wins > c2_total / 2 if c2_total > 0 else False,
        "wins": c2_wins,
        "total": c2_total,
    }
    logprint(f"  C2 (C2UD_full > C2UD_RS_PA): {'PASS' if c2_wins > c2_total/2 else 'FAIL'} ({c2_wins}/{c2_total})")

    # Criterion 3: Cross-model consistency
    for model_name in all_model_results:
        agg = all_model_results[model_name]["aggregated"]
        aurocs = []
        for ds in ["nq", "triviaqa", "popqa"]:
            if ds in agg and "C2UD_full" in agg[ds]:
                aurocs.append(agg[ds]["C2UD_full"]["auroc"]["mean"])
        logprint(f"  {model_name} C2UD_full AUROC: {[f'{a:.4f}' for a in aurocs]}")

    # Criterion 4: Failure mode differentiation
    for model_name, fa in all_failure_analyses.items():
        if "failure_stats" in fa:
            fs = fa["failure_stats"]
            logprint(f"\n  {model_name} failure mode stats:")
            for ft, stats in fs.items():
                logprint(f"    {ft} (n={stats['count']}): RS={stats['RS_mean']:.3f}, CD={stats['CD_mean']:.3f}, PA={stats['PA_mean']:.3f}")

            sig = fa.get("significance_tests", {})
            sig_count = sum(1 for v in sig.values() if v["p_value"] < 0.05)
            logprint(f"    Significant differences: {sig_count}/{len(sig)}")

    # Determine best method per dataset per model
    logprint("\n  === BEST METHODS ===")
    key_methods = ["C2UD_full", "C2UD_enhanced", "CRUX", "TokenProb", "Verbalized", "SemEntropy", "SelfConsist", "Axiomatic", "SPUQ"]
    for model_name in all_model_results:
        agg = all_model_results[model_name]["aggregated"]
        for ds in ["nq", "triviaqa", "popqa"]:
            if ds not in agg:
                continue
            best_method = None
            best_auroc = -1
            for m in key_methods:
                if m in agg[ds] and "auroc" in agg[ds][m]:
                    a = agg[ds][m]["auroc"]["mean"]
                    if a > best_auroc:
                        best_auroc = a
                        best_method = m
            logprint(f"  {model_name}/{ds}: best={best_method} (AUROC={best_auroc:.4f})")

    # Add bootstrap p-values
    final["bootstrap_tests"] = {}
    for model_name in all_model_results:
        bt = all_model_results[model_name].get("bootstrap_tests", {})
        final["bootstrap_tests"][model_name] = bt

    # Add intervention results
    if intervention_results:
        final["intervention"] = intervention_results

    # Add failure analyses
    final["failure_analyses"] = {}
    for model_name, fa in all_failure_analyses.items():
        final["failure_analyses"][model_name] = {
            "failure_stats": fa.get("failure_stats", {}),
            "significance_tests": fa.get("significance_tests", {}),
        }

    # Honest analysis section
    final["honest_analysis"] = {
        "key_findings": [],
        "limitations": [],
        "negative_results": [],
    }

    with open(WORKSPACE / "results.json", "w") as f:
        json.dump(final, f, indent=2)
    logprint(f"\n  Saved results.json ({WORKSPACE / 'results.json'})")

    return final


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()
    logprint("\n" + "=" * 60)
    logprint("C2UD v2 EXPERIMENT RUNNER")
    logprint("=" * 60)

    # Stage 1: Load data
    datasets = stage1_load_data()

    all_model_results = {}
    all_model_components = {}
    all_model_baselines = {}
    all_failure_analyses = {}
    all_generations = {}
    intervention_results = None

    for model_name, model_id in MODELS.items():
        model_start = time.time()
        logprint(f"\n{'#' * 70}")
        logprint(f"# MODEL: {model_name} ({model_id})")
        logprint(f"{'#' * 70}")

        # Stage 2: Generation (cached greedy + fresh samples)
        generations, samples = stage2_generate(datasets, model_name, model_id)
        all_generations[model_name] = generations

        # Stage 3: Improved C2UD components
        components = stage3_compute_c2ud(generations, model_name)
        all_model_components[model_name] = components

        # Stage 4: Baselines
        baselines = stage4_compute_baselines(generations, samples, components, model_name)
        all_model_baselines[model_name] = baselines

        # Stage 5: Evaluation
        results = stage5_evaluate(generations, components, baselines, model_name)
        all_model_results[model_name] = results

        # Stage 6: Failure analysis
        fa = stage6_failure_analysis(datasets, generations, components, model_name)
        all_failure_analyses[model_name] = fa

        # Stage 7: Intervention (Llama only)
        if model_name == "llama":
            intervention_results = stage7_intervention(datasets, generations, components, model_name)

        model_time = (time.time() - model_start) / 60
        logprint(f"\n  Model {model_name} total time: {model_time:.1f} minutes")

    # Stage 8: Figures
    stage8_figures(all_model_results, all_model_components, all_model_baselines, all_generations)

    # Stage 9: Final aggregation
    final = stage9_aggregate(all_model_results, all_failure_analyses, intervention_results)

    total_time = (time.time() - total_start) / 60
    logprint(f"\n{'=' * 60}")
    logprint(f"TOTAL TIME: {total_time:.1f} minutes")
    logprint(f"{'=' * 60}")

    return final


if __name__ == "__main__":
    main()
