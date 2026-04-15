"""
Complete re-run of SpecCheck experiments with fixed confidence extraction.

Key fix: The get_yes_no_logprobs function had a bug where it used seq_len-1
as the position index, but with left-padding the last real token is at
position total_length-1. This caused ~29% of confidence values to be 0.5
(the NaN fallback) because the code was reading logits at padding positions.

This script:
1. Re-runs confidence estimation for all 3 models x 3 datasets
2. Re-computes SpecCheck scores
3. Re-runs all baselines (verbalized, logprob, selfcheck, random)
4. Re-runs ablations (ladder depth, score variants, claim types, combinations)
5. Runs statistical evaluation
6. Generates all figures
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    set_seed, save_json, load_json, get_model_short,
    SEEDS, MODELS, DATASETS, DATA_DIR, RESULTS_DIR, BASE_DIR, FIGURES_DIR
)
from shared.model_utils import (
    load_model, unload_model, generate_text, get_yes_no_logprobs,
    get_token_logprobs
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# STAGE 1: Confidence estimation (FIXED)
# ============================================================
def run_confidence_estimation(model, tokenizer, model_name):
    """Re-run logprob-based confidence estimation with the fixed position bug."""
    mshort = get_model_short(model_name)
    print(f"\n{'='*60}")
    print(f"CONFIDENCE ESTIMATION: {mshort}")
    print(f"{'='*60}")

    for ds in DATASETS:
        print(f"\n--- {mshort}/{ds} ---")
        ladders = load_json(os.path.join(DATA_DIR, f"ladders_{mshort}_{ds}.json"))
        labels = load_json(os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{ds}.json"))

        # Build prompts for all levels
        all_prompts = []
        claim_ids = []
        for ladder in ladders:
            levels = ladder.get("levels", [])
            if not levels:
                continue
            cid = ladder.get("claim_id", ladder.get("id", ""))
            for level_info in levels:
                text = level_info.get("text", "")
                prompt = f'Is the following statement true? "{text}" Answer with only Yes or No.'
                all_prompts.append(prompt)
            claim_ids.append((cid, len(levels)))

        print(f"  Total prompts: {len(all_prompts)} ({len(claim_ids)} claims)")

        # Run confidence estimation in batches
        all_confs = get_yes_no_logprobs(model, tokenizer, all_prompts, batch_size=32)

        # Re-assemble per-claim confidence sequences
        results = []
        idx = 0
        label_map = {l.get("claim_id", l.get("id", i)): l.get("label", 0)
                     for i, l in enumerate(labels)}
        for cid, n_levels in claim_ids:
            confs = all_confs[idx:idx+n_levels]
            idx += n_levels
            results.append({
                "claim_id": cid,
                "confidences": confs,
            })

        # Check quality: how many 0.5 values?
        total_vals = sum(len(r["confidences"]) for r in results)
        n_05 = sum(1 for r in results for c in r["confidences"] if abs(c - 0.5) < 0.01)
        print(f"  Fraction of 0.5 values: {n_05}/{total_vals} = {n_05/total_vals:.4f}")

        # Show some examples
        for r in results[:3]:
            print(f"    {r['claim_id']}: {[round(c, 3) for c in r['confidences']]}")

        save_json(results, os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{ds}.json"))
    print(f"  Done with {mshort}")


# ============================================================
# STAGE 2: SpecCheck scores
# ============================================================
def compute_speccheck_scores(mshort, ds):
    """Compute SpecCheck scores from confidence sequences."""
    confs = load_json(os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{ds}.json"))
    labels = load_json(os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{ds}.json"))

    label_map = {}
    for i, l in enumerate(labels):
        cid = l.get("claim_id", l.get("id", f"{mshort}_{ds}_{i}"))
        label_map[cid] = l.get("label", 0)

    results = []
    for item in confs:
        cid = item["claim_id"]
        c = item["confidences"]
        if len(c) < 2:
            continue

        # Monotonicity: fraction of adjacent pairs where conf increases or stays same
        n_transitions = len(c) - 1
        n_monotonic = sum(1 for k in range(1, len(c)) if c[k] >= c[k-1] - 0.01)
        mono_score = n_monotonic / n_transitions

        # SpecCheck score variants
        conf_gap = c[-1] - c[0] if len(c) >= 2 else 0

        # Max violation: largest drop in confidence
        max_violation = max((c[k-1] - c[k]) for k in range(1, len(c))) if len(c) > 1 else 0
        max_violation = max(0, max_violation)

        # Default SpecCheck: 1 - monotonicity + alpha * |conf_gap| (negative gap means drop)
        alpha = 0.5
        speccheck_default = (1 - mono_score) + alpha * max_violation

        # Hallucination Granularity Index
        delta = 0.05
        granularity = -1
        for k in range(1, len(c)):
            if c[k] > c[0] + delta:
                granularity = k
                break

        results.append({
            "claim_id": cid,
            "label": label_map.get(cid, 0),
            "confidences": c,
            "monotonicity_score": mono_score,
            "speccheck_score": speccheck_default,
            "max_violation": max_violation,
            "confidence_gap": conf_gap,
            "granularity_index": granularity,
        })

    save_json(results, os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json"))
    return results


# ============================================================
# STAGE 3: Baselines
# ============================================================
def run_verbalized_baseline(model, tokenizer, model_name):
    """Verbalized confidence baseline."""
    mshort = get_model_short(model_name)
    print(f"\n--- Verbalized baseline: {mshort} ---")

    for ds in DATASETS:
        claims = load_json(os.path.join(DATA_DIR, f"claims_{mshort}_{ds}.json"))
        labels = load_json(os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{ds}.json"))

        prompts = []
        for claim in claims:
            text = claim.get("claim_text", claim.get("text", ""))
            p = (f'Rate your confidence that the following statement is true '
                 f'on a scale of 0 to 100, where 0 means certainly false and '
                 f'100 means certainly true. Only output the number.\n'
                 f'Statement: "{text}"')
            prompts.append(p)

        outputs = generate_text(model, tokenizer, prompts,
                                max_new_tokens=16, temperature=0.0, do_sample=False, batch_size=16)

        results = []
        for i, (claim, output) in enumerate(zip(claims, outputs)):
            cid = claim.get("claim_id", claim.get("id", f"{mshort}_{ds}_{i}"))
            # Parse number from output
            import re
            nums = re.findall(r'\d+(?:\.\d+)?', output)
            if nums:
                conf = float(nums[0])
                conf = min(100, max(0, conf))
            else:
                conf = 50.0
            # Hallucination score = 1 - confidence/100
            hall_score = 1.0 - conf / 100.0
            label = labels[i].get("label", 0) if i < len(labels) else 0
            results.append({
                "claim_id": cid,
                "label": label,
                "verbalized_confidence": conf,
                "hallucination_score": hall_score,
            })

        save_json(results, os.path.join(RESULTS_DIR, f"baseline_verbalized_{mshort}_{ds}.json"))
        print(f"  {mshort}/{ds}: {len(results)} claims scored")


def run_logprob_baseline(model, tokenizer, model_name):
    """Logprob confidence baseline: average token logprob of the claim."""
    mshort = get_model_short(model_name)
    print(f"\n--- Logprob baseline: {mshort} ---")

    for ds in DATASETS:
        claims = load_json(os.path.join(DATA_DIR, f"claims_{mshort}_{ds}.json"))
        labels = load_json(os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{ds}.json"))

        texts = [c.get("claim_text", c.get("text", "")) for c in claims]
        logprobs = get_token_logprobs(model, tokenizer, texts, batch_size=32)

        # Also use the level-0 yes/no confidence from SpecCheck
        conf_file = os.path.join(RESULTS_DIR, f"confidence_logprob_{mshort}_{ds}.json")
        level0_confs = {}
        if os.path.exists(conf_file):
            conf_data = load_json(conf_file)
            for item in conf_data:
                if item["confidences"]:
                    level0_confs[item["claim_id"]] = item["confidences"][0]

        results = []
        for i, (claim, lp) in enumerate(zip(claims, logprobs)):
            cid = claim.get("claim_id", claim.get("id", f"{mshort}_{ds}_{i}"))
            label = labels[i].get("label", 0) if i < len(labels) else 0
            # Normalize logprob to [0, 1] range: exp(avg_logprob)
            norm_lp = np.exp(max(lp, -20))  # Clip extreme values
            # Hallucination score = 1 - confidence
            hall_score = 1.0 - norm_lp
            # Yes/No level-0 confidence
            yesno_conf = level0_confs.get(cid, 0.5)
            results.append({
                "claim_id": cid,
                "label": label,
                "avg_logprob": lp,
                "norm_logprob": norm_lp,
                "yesno_confidence": yesno_conf,
                "hallucination_score": 1.0 - yesno_conf,
            })

        save_json(results, os.path.join(RESULTS_DIR, f"baseline_logprob_{mshort}_{ds}.json"))
        print(f"  {mshort}/{ds}: {len(results)} claims scored")


def run_selfcheck_baseline(model, tokenizer, model_name):
    """SelfCheckGPT-style sampling consistency baseline."""
    mshort = get_model_short(model_name)
    print(f"\n--- SelfCheck baseline: {mshort} ---")

    for ds in DATASETS:
        claims = load_json(os.path.join(DATA_DIR, f"claims_{mshort}_{ds}.json"))
        labels = load_json(os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{ds}.json"))

        # Group claims by source prompt
        prompt_map = {}
        for i, claim in enumerate(claims):
            pid = claim.get("prompt_id", claim.get("entity_id", claim.get("question_id", f"p{i}")))
            if pid not in prompt_map:
                prompt_map[pid] = {"prompt": claim.get("source_prompt", claim.get("prompt", "")), "claims": []}
            prompt_map[pid]["claims"].append((i, claim))

        # Generate N=5 alternative responses per prompt
        N_SAMPLES = 5
        alt_responses = {}
        prompts_list = [(pid, info["prompt"]) for pid, info in prompt_map.items() if info["prompt"]]
        print(f"  Generating {N_SAMPLES} alternatives for {len(prompts_list)} prompts...")

        for pid, prompt in tqdm(prompts_list, desc=f"SelfCheck {mshort}/{ds}"):
            max_tokens = {"factscore": 256, "longfact": 512, "truthfulqa": 128}[ds]
            alts = generate_text(
                model, tokenizer, [prompt] * N_SAMPLES,
                max_new_tokens=max_tokens, temperature=1.0, top_p=0.95,
                batch_size=N_SAMPLES, do_sample=True
            )
            alt_responses[pid] = alts

        # Check each claim against alternatives using yes/no NLI
        print(f"  Checking claim support across alternatives...")
        results = []
        for pid, info in prompt_map.items():
            alts = alt_responses.get(pid, [])
            if not alts:
                for idx, claim in info["claims"]:
                    cid = claim.get("claim_id", claim.get("id", f"{mshort}_{ds}_{idx}"))
                    label = labels[idx].get("label", 0) if idx < len(labels) else 0
                    results.append({"claim_id": cid, "label": label, "hallucination_score": 0.5})
                continue

            for idx, claim in info["claims"]:
                cid = claim.get("claim_id", claim.get("id", f"{mshort}_{ds}_{idx}"))
                label = labels[idx].get("label", 0) if idx < len(labels) else 0
                claim_text = claim.get("claim_text", claim.get("text", ""))

                # Check support in each alternative
                nli_prompts = []
                for alt in alts:
                    alt_truncated = alt[:500]
                    p = (f'Does the following passage support the claim "{claim_text}"? '
                         f'Answer with only Yes or No.\n'
                         f'Passage: "{alt_truncated}"')
                    nli_prompts.append(p)

                support_confs = get_yes_no_logprobs(model, tokenizer, nli_prompts, batch_size=N_SAMPLES)
                # SelfCheck score = 1 - fraction supported
                avg_support = np.mean(support_confs)
                hall_score = 1.0 - avg_support

                results.append({
                    "claim_id": cid,
                    "label": label,
                    "support_scores": support_confs,
                    "hallucination_score": hall_score,
                })

        save_json(results, os.path.join(RESULTS_DIR, f"baseline_selfcheck_{mshort}_{ds}.json"))
        print(f"  {mshort}/{ds}: {len(results)} claims scored")


def run_random_baseline():
    """Random baseline for all models/datasets/seeds."""
    print("\n--- Random baseline ---")
    for mshort in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            labels = load_json(os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{ds}.json"))
            for seed in SEEDS:
                set_seed(seed)
                results = []
                for i, l in enumerate(labels):
                    cid = l.get("claim_id", l.get("id", f"{mshort}_{ds}_{i}"))
                    results.append({
                        "claim_id": cid,
                        "label": l.get("label", 0),
                        "hallucination_score": random.random(),
                    })
                save_json(results, os.path.join(RESULTS_DIR,
                          f"baseline_random_{mshort}_{ds}_seed{seed}.json"))
    print("  Done")


# ============================================================
# STAGE 4: Ablations
# ============================================================
def run_ablation_ladder_depth(mshort, ds):
    """Test ladder depth K=1,2,3,4."""
    scores = load_json(os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json"))

    results = {}
    for K in [1, 2, 3, 4]:
        y_true = []
        y_score = []
        for item in scores:
            confs = item["confidences"]
            label = item["label"]
            if len(confs) < 2:
                continue

            # Select subset of levels based on K
            if K == 1:
                subset = [confs[0], confs[-1]]
            elif K == 2:
                indices = [0, len(confs)//2, len(confs)-1]
                subset = [confs[min(i, len(confs)-1)] for i in indices]
            elif K == 3:
                subset = confs[:4] if len(confs) >= 4 else confs
            else:  # K=4
                subset = confs  # Use all available

            # Compute monotonicity for this subset
            n_trans = len(subset) - 1
            if n_trans == 0:
                continue
            n_mono = sum(1 for k in range(1, len(subset)) if subset[k] >= subset[k-1] - 0.01)
            mono = n_mono / n_trans
            max_viol = max((subset[k-1] - subset[k]) for k in range(1, len(subset)))
            max_viol = max(0, max_viol)
            spec_score = (1 - mono) + 0.5 * max_viol

            y_true.append(label)
            y_score.append(spec_score)

        if len(set(y_true)) < 2:
            continue
        auc_roc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)
        results[f"K={K}"] = {"auc_roc": auc_roc, "auc_pr": auc_pr, "n_claims": len(y_true)}

    save_json(results, os.path.join(RESULTS_DIR, f"ablation_ladder_depth_{mshort}_{ds}.json"))
    return results


def run_ablation_score_variants(mshort, ds):
    """Compare different SpecCheck scoring methods."""
    scores = load_json(os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json"))

    y_true = [s["label"] for s in scores]
    if len(set(y_true)) < 2:
        return {}

    variants = {}

    # Variant 1: Pure monotonicity
    y_mono = [1 - s["monotonicity_score"] for s in scores]
    variants["pure_monotonicity"] = {
        "auc_roc": roc_auc_score(y_true, y_mono),
        "auc_pr": average_precision_score(y_true, y_mono),
    }

    # Variant 2: Max violation
    y_maxv = [s["max_violation"] for s in scores]
    variants["max_violation"] = {
        "auc_roc": roc_auc_score(y_true, y_maxv),
        "auc_pr": average_precision_score(y_true, y_maxv),
    }

    # Variant 3: Confidence gap
    y_gap = [-s["confidence_gap"] for s in scores]  # Negative gap = hallucination
    variants["confidence_gap"] = {
        "auc_roc": roc_auc_score(y_true, y_gap),
        "auc_pr": average_precision_score(y_true, y_gap),
    }

    # Variant 4: Default SpecCheck (alpha=0.5)
    y_default = [s["speccheck_score"] for s in scores]
    variants["speccheck_default"] = {
        "auc_roc": roc_auc_score(y_true, y_default),
        "auc_pr": average_precision_score(y_true, y_default),
    }

    # Sweep alpha
    for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
        y_alpha = [(1 - s["monotonicity_score"]) + alpha * s["max_violation"] for s in scores]
        variants[f"alpha={alpha}"] = {
            "auc_roc": roc_auc_score(y_true, y_alpha),
            "auc_pr": average_precision_score(y_true, y_alpha),
        }

    save_json(variants, os.path.join(RESULTS_DIR, f"ablation_score_variants_{mshort}_{ds}.json"))
    return variants


def run_claim_type_analysis(mshort, ds):
    """Analyze SpecCheck performance by claim type."""
    import re as re_mod
    claims = load_json(os.path.join(DATA_DIR, f"claims_{mshort}_{ds}.json"))
    scores = load_json(os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json"))

    score_map = {s["claim_id"]: s for s in scores}

    # Categorize claims
    categories = {}
    for i, claim in enumerate(claims):
        cid = claim.get("claim_id", claim.get("id", f"{mshort}_{ds}_{i}"))
        text = claim.get("claim_text", claim.get("text", ""))

        if re_mod.search(r'\d{4}|\d+\.\d+|\d+%|\$\d+', text):
            cat = "numerical"
        elif re_mod.search(r'(?:born|died|founded|established|year|century|decade|era|period)', text, re_mod.I):
            cat = "temporal"
        elif re_mod.search(r'[A-Z][a-z]+ [A-Z][a-z]+|University|Company|City|Country', text):
            cat = "entity_specific"
        else:
            cat = "general"

        if cat not in categories:
            categories[cat] = {"y_true": [], "y_speccheck": []}
        if cid in score_map:
            s = score_map[cid]
            categories[cat]["y_true"].append(s["label"])
            categories[cat]["y_speccheck"].append(s["speccheck_score"])

    results = {}
    for cat, data in categories.items():
        if len(set(data["y_true"])) < 2 or len(data["y_true"]) < 10:
            continue
        results[cat] = {
            "n_claims": len(data["y_true"]),
            "n_hallucinated": sum(data["y_true"]),
            "auc_roc": roc_auc_score(data["y_true"], data["y_speccheck"]),
            "auc_pr": average_precision_score(data["y_true"], data["y_speccheck"]),
        }

    save_json(results, os.path.join(RESULTS_DIR, f"analysis_claim_types_{mshort}_{ds}.json"))
    return results


def run_combination_analysis(mshort, ds):
    """Test SpecCheck + baseline combinations via logistic regression."""
    speccheck = load_json(os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json"))
    verbalized_path = os.path.join(RESULTS_DIR, f"baseline_verbalized_{mshort}_{ds}.json")
    logprob_path = os.path.join(RESULTS_DIR, f"baseline_logprob_{mshort}_{ds}.json")
    selfcheck_path = os.path.join(RESULTS_DIR, f"baseline_selfcheck_{mshort}_{ds}.json")

    if not all(os.path.exists(p) for p in [verbalized_path, logprob_path, selfcheck_path]):
        return {}

    verb = {v["claim_id"]: v for v in load_json(verbalized_path)}
    lp = {v["claim_id"]: v for v in load_json(logprob_path)}
    sc = {v["claim_id"]: v for v in load_json(selfcheck_path)}

    # Build feature matrix
    X_all = []
    y_all = []
    for s in speccheck:
        cid = s["claim_id"]
        if cid not in verb or cid not in lp or cid not in sc:
            continue
        features = {
            "speccheck": s["speccheck_score"],
            "verbalized": verb[cid].get("hallucination_score", 0.5),
            "logprob": lp[cid].get("hallucination_score", 0.5),
            "selfcheck": sc[cid].get("hallucination_score", 0.5),
        }
        X_all.append(features)
        y_all.append(s["label"])

    if len(set(y_all)) < 2 or len(y_all) < 20:
        return {}

    y_all = np.array(y_all)

    # Test different feature combinations
    combos = {
        "speccheck_only": ["speccheck"],
        "verbalized_only": ["verbalized"],
        "logprob_only": ["logprob"],
        "selfcheck_only": ["selfcheck"],
        "all_baselines": ["verbalized", "logprob", "selfcheck"],
        "all_baselines_plus_speccheck": ["speccheck", "verbalized", "logprob", "selfcheck"],
        "speccheck_plus_selfcheck": ["speccheck", "selfcheck"],
        "speccheck_plus_logprob": ["speccheck", "logprob"],
    }

    results = {}
    for combo_name, feature_names in combos.items():
        X = np.array([[x[f] for f in feature_names] for x in X_all])

        auc_rocs = []
        auc_prs = []
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_preds = np.zeros(len(y_all))
            for train_idx, test_idx in skf.split(X, y_all):
                clf = LogisticRegression(random_state=seed, max_iter=1000)
                clf.fit(X[train_idx], y_all[train_idx])
                fold_preds[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
            auc_rocs.append(roc_auc_score(y_all, fold_preds))
            auc_prs.append(average_precision_score(y_all, fold_preds))

        results[combo_name] = {
            "auc_roc_mean": float(np.mean(auc_rocs)),
            "auc_roc_std": float(np.std(auc_rocs)),
            "auc_pr_mean": float(np.mean(auc_prs)),
            "auc_pr_std": float(np.std(auc_prs)),
            "n_claims": len(y_all),
        }

    save_json(results, os.path.join(RESULTS_DIR, f"analysis_combination_{mshort}_{ds}.json"))
    return results


# ============================================================
# STAGE 5: Evaluation
# ============================================================
def compute_all_metrics():
    """Compute comprehensive evaluation metrics for all methods."""
    print(f"\n{'='*60}")
    print("COMPUTING EVALUATION METRICS")
    print(f"{'='*60}")

    all_results = {}

    for mshort in ["llama", "mistral", "qwen"]:
        all_results[mshort] = {}
        for ds in DATASETS:
            print(f"\n--- {mshort}/{ds} ---")

            # Load SpecCheck scores
            spec_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json")
            if not os.path.exists(spec_path):
                print(f"  SKIPPED: {spec_path} not found")
                continue
            speccheck = load_json(spec_path)
            y_true = [s["label"] for s in speccheck]
            y_speccheck = [s["speccheck_score"] for s in speccheck]

            if len(set(y_true)) < 2:
                print(f"  SKIPPED: only one class")
                continue

            methods = {"speccheck": y_speccheck}

            # Load baselines
            for bname in ["verbalized", "logprob", "selfcheck"]:
                bpath = os.path.join(RESULTS_DIR, f"baseline_{bname}_{mshort}_{ds}.json")
                if os.path.exists(bpath):
                    bdata = load_json(bpath)
                    bmap = {b["claim_id"]: b.get("hallucination_score", 0.5) for b in bdata}
                    y_base = [bmap.get(s["claim_id"], 0.5) for s in speccheck]
                    methods[bname] = y_base

            # Random baseline (averaged over seeds)
            rand_scores = []
            for seed in SEEDS:
                rpath = os.path.join(RESULTS_DIR, f"baseline_random_{mshort}_{ds}_seed{seed}.json")
                if os.path.exists(rpath):
                    rdata = load_json(rpath)
                    rmap = {r["claim_id"]: r["hallucination_score"] for r in rdata}
                    rand_scores.append([rmap.get(s["claim_id"], 0.5) for s in speccheck])
            if rand_scores:
                methods["random"] = [float(np.mean([rs[i] for rs in rand_scores]))
                                     for i in range(len(speccheck))]

            # Also include monotonicity-only and max-violation scores
            methods["mono_only"] = [1 - s["monotonicity_score"] for s in speccheck]
            methods["max_violation"] = [s["max_violation"] for s in speccheck]
            methods["conf_gap"] = [-s["confidence_gap"] for s in speccheck]

            ds_results = {}
            for method_name, y_pred in methods.items():
                try:
                    auc_roc = roc_auc_score(y_true, y_pred)
                    auc_pr = average_precision_score(y_true, y_pred)

                    # Bootstrap CI
                    n_boot = 1000
                    boot_rocs = []
                    boot_prs = []
                    rng = np.random.RandomState(42)
                    n = len(y_true)
                    for _ in range(n_boot):
                        idx = rng.randint(0, n, n)
                        yt = [y_true[i] for i in idx]
                        yp = [y_pred[i] for i in idx]
                        if len(set(yt)) < 2:
                            continue
                        boot_rocs.append(roc_auc_score(yt, yp))
                        boot_prs.append(average_precision_score(yt, yp))

                    ci_roc = (float(np.percentile(boot_rocs, 2.5)), float(np.percentile(boot_rocs, 97.5))) if boot_rocs else (0, 0)
                    ci_pr = (float(np.percentile(boot_prs, 2.5)), float(np.percentile(boot_prs, 97.5))) if boot_prs else (0, 0)

                    ds_results[method_name] = {
                        "auc_roc": float(auc_roc),
                        "auc_pr": float(auc_pr),
                        "auc_roc_ci": ci_roc,
                        "auc_pr_ci": ci_pr,
                        "n_claims": len(y_true),
                        "n_hallucinated": sum(y_true),
                        "prevalence": float(np.mean(y_true)),
                    }
                    print(f"  {method_name}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")
                except Exception as e:
                    print(f"  {method_name}: ERROR - {e}")

            # Monotonicity analysis
            factual_mono = [s["monotonicity_score"] for s in speccheck if s["label"] == 0]
            halluc_mono = [s["monotonicity_score"] for s in speccheck if s["label"] == 1]
            factual_perfect_mono = sum(1 for m in factual_mono if m >= 0.99) / max(len(factual_mono), 1)
            halluc_perfect_mono = sum(1 for m in halluc_mono if m >= 0.99) / max(len(halluc_mono), 1)
            halluc_violated = sum(1 for m in halluc_mono if m < 0.99) / max(len(halluc_mono), 1)

            ds_results["monotonicity_analysis"] = {
                "factual_mean_monotonicity": float(np.mean(factual_mono)) if factual_mono else 0,
                "halluc_mean_monotonicity": float(np.mean(halluc_mono)) if halluc_mono else 0,
                "factual_perfect_monotonicity_rate": float(factual_perfect_mono),
                "halluc_perfect_monotonicity_rate": float(halluc_perfect_mono),
                "halluc_violation_rate": float(halluc_violated),
                "mono_diff": float(np.mean(factual_mono) - np.mean(halluc_mono)) if factual_mono and halluc_mono else 0,
                "n_factual": len(factual_mono),
                "n_hallucinated": len(halluc_mono),
            }

            # Statistical test: SpecCheck vs best baseline
            if "selfcheck" in ds_results and "speccheck" in ds_results:
                # Paired bootstrap test
                n_better = 0
                rng = np.random.RandomState(42)
                n = len(y_true)
                y_spec = methods["speccheck"]
                y_self = methods["selfcheck"]
                for _ in range(10000):
                    idx = rng.randint(0, n, n)
                    yt = [y_true[i] for i in idx]
                    if len(set(yt)) < 2:
                        continue
                    auc_s = average_precision_score(yt, [y_spec[i] for i in idx])
                    auc_b = average_precision_score(yt, [y_self[i] for i in idx])
                    if auc_s > auc_b:
                        n_better += 1
                ds_results["speccheck_vs_selfcheck_p"] = 1 - n_better / 10000

            all_results[mshort][ds] = ds_results

    # Save evaluation statistics
    save_json(all_results, os.path.join(RESULTS_DIR, "evaluation_statistics.json"))

    # Check success criteria
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")

    # Criterion 1: SpecCheck > SelfCheckGPT AUC-PR on >= 2/3 benchmarks
    wins = 0
    for ds in DATASETS:
        spec_better = 0
        for m in ["llama", "mistral", "qwen"]:
            r = all_results.get(m, {}).get(ds, {})
            if "speccheck" in r and "selfcheck" in r:
                if r["speccheck"]["auc_pr"] > r["selfcheck"]["auc_pr"]:
                    spec_better += 1
        if spec_better >= 2:  # majority of models
            wins += 1
    print(f"Criterion 1: SpecCheck > SelfCheck on {wins}/3 benchmarks (need >= 2)")

    # Criterion 2: Factual monotonicity > 85%
    factual_rates = []
    for m in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            r = all_results.get(m, {}).get(ds, {}).get("monotonicity_analysis", {})
            if r:
                factual_rates.append(r.get("factual_perfect_monotonicity_rate", 0))
    avg_factual = np.mean(factual_rates) if factual_rates else 0
    print(f"Criterion 2: Factual perfect monotonicity = {avg_factual:.1%} (need > 85%)")

    # Criterion 3: Hallucinated monotonicity violated > 40%
    halluc_rates = []
    for m in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            r = all_results.get(m, {}).get(ds, {}).get("monotonicity_analysis", {})
            if r:
                halluc_rates.append(r.get("halluc_violation_rate", 0))
    avg_halluc = np.mean(halluc_rates) if halluc_rates else 0
    print(f"Criterion 3: Hallucinated violation rate = {avg_halluc:.1%} (need > 40%)")

    return all_results


# ============================================================
# STAGE 6: Sampling-based confidence ablation
# ============================================================
def run_sampling_ablation(model, tokenizer, model_name):
    """Ablation: sampling-based confidence vs logprob-based."""
    mshort = get_model_short(model_name)
    ds = "factscore"  # Only run on one dataset to save compute
    print(f"\n--- Sampling ablation: {mshort}/{ds} ---")

    ladders = load_json(os.path.join(DATA_DIR, f"ladders_{mshort}_{ds}.json"))
    labels = load_json(os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{ds}.json"))

    # Take a subset for efficiency
    subset = ladders[:200]

    for N_SAMPLES in [5, 10]:
        print(f"  N={N_SAMPLES} samples...")
        for seed in SEEDS:
            set_seed(seed)
            results = []
            for ladder in tqdm(subset, desc=f"N={N_SAMPLES}, seed={seed}"):
                cid = ladder.get("claim_id", ladder.get("id", ""))
                levels = ladder.get("levels", [])
                confs = []
                for level in levels:
                    text = level.get("text", "")
                    prompt = f'Is the following statement true? "{text}" Answer with only Yes or No.'
                    outputs = generate_text(
                        model, tokenizer, [prompt] * N_SAMPLES,
                        max_new_tokens=8, temperature=0.8, do_sample=True,
                        batch_size=N_SAMPLES
                    )
                    yes_count = sum(1 for o in outputs if o.strip().lower().startswith("yes"))
                    confs.append(yes_count / N_SAMPLES)

                results.append({"claim_id": cid, "confidences": confs})

            save_json(results, os.path.join(RESULTS_DIR,
                      f"speccheck_sampling_seed{seed}_{mshort}_{ds}.json"))

    print(f"  Done with sampling ablation for {mshort}")


# ============================================================
# STAGE 7: Figures
# ============================================================
def generate_figures(eval_results):
    """Generate all paper figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        'font.size': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    # Colorblind-friendly palette
    colors = {
        'speccheck': '#0072B2',
        'selfcheck': '#D55E00',
        'verbalized': '#009E73',
        'logprob': '#CC79A7',
        'random': '#999999',
        'mono_only': '#56B4E9',
        'max_violation': '#E69F00',
        'conf_gap': '#F0E442',
    }
    method_labels = {
        'speccheck': 'SpecCheck',
        'selfcheck': 'SelfCheckGPT',
        'verbalized': 'Verbalized Conf.',
        'logprob': 'Logprob Conf.',
        'random': 'Random',
    }

    # ---- Figure 2: Main Results Bar Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    main_methods = ['speccheck', 'selfcheck', 'verbalized', 'logprob', 'random']

    for ax_idx, ds in enumerate(DATASETS):
        ax = axes[ax_idx]
        x_pos = np.arange(len(main_methods))
        width = 0.25

        for m_idx, mshort in enumerate(["llama", "mistral", "qwen"]):
            vals = []
            for method in main_methods:
                r = eval_results.get(mshort, {}).get(ds, {}).get(method, {})
                vals.append(r.get("auc_pr", 0))

            offset = (m_idx - 1) * width
            bars = ax.bar(x_pos + offset, vals, width, label=mshort.capitalize() if ax_idx == 0 else "")

        ax.set_title(ds.replace("_", " ").title(), fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([method_labels.get(m, m) for m in main_methods], rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        if ax_idx == 0:
            ax.set_ylabel("AUC-PR")

    axes[0].legend(title="Model", fontsize=9)
    fig.suptitle("Hallucination Detection: AUC-PR Across Datasets and Models", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_2_main_results.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "figure_2_main_results.png"))
    plt.close()

    # ---- Figure 3: Monotonicity Analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram of monotonicity scores
    all_factual_mono = []
    all_halluc_mono = []
    for mshort in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            spec_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json")
            if not os.path.exists(spec_path):
                continue
            scores = load_json(spec_path)
            for s in scores:
                if s["label"] == 0:
                    all_factual_mono.append(s["monotonicity_score"])
                else:
                    all_halluc_mono.append(s["monotonicity_score"])

    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(all_factual_mono, bins=bins, alpha=0.6, label="Factual", color='#0072B2', density=True)
    ax.hist(all_halluc_mono, bins=bins, alpha=0.6, label="Hallucinated", color='#D55E00', density=True)
    ax.set_xlabel("Monotonicity Score")
    ax.set_ylabel("Density")
    ax.set_title("Monotonicity Score Distribution")
    ax.legend()

    # Right: average confidence profiles
    ax = axes[1]
    factual_profiles = []
    halluc_profiles = []
    for mshort in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            spec_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json")
            if not os.path.exists(spec_path):
                continue
            scores = load_json(spec_path)
            for s in scores:
                confs = s["confidences"]
                if len(confs) == 4:
                    if s["label"] == 0:
                        factual_profiles.append(confs)
                    else:
                        halluc_profiles.append(confs)

    if factual_profiles and halluc_profiles:
        factual_mean = np.mean(factual_profiles, axis=0)
        factual_std = np.std(factual_profiles, axis=0)
        halluc_mean = np.mean(halluc_profiles, axis=0)
        halluc_std = np.std(halluc_profiles, axis=0)
        levels = [0, 1, 2, 3]
        ax.plot(levels, factual_mean, 'o-', color='#0072B2', label='Factual', linewidth=2)
        ax.fill_between(levels, factual_mean - factual_std, factual_mean + factual_std, alpha=0.2, color='#0072B2')
        ax.plot(levels, halluc_mean, 's-', color='#D55E00', label='Hallucinated', linewidth=2)
        ax.fill_between(levels, halluc_mean - halluc_std, halluc_mean + halluc_std, alpha=0.2, color='#D55E00')
        ax.set_xlabel("Specificity Level (0=Original, 3=Abstract)")
        ax.set_ylabel("Confidence P(Yes)")
        ax.set_title("Confidence Profiles: Factual vs Hallucinated")
        ax.legend()
        ax.set_xticks(levels)
        ax.set_xticklabels(["Original\n(Level 0)", "Approx.\n(Level 1)",
                            "Category\n(Level 2)", "Abstract\n(Level 3)"])

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_3_monotonicity.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "figure_3_monotonicity.png"))
    plt.close()

    # ---- Figure 4: Claim Type Analysis ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    claim_type_data = {}
    for mshort in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            ct_path = os.path.join(RESULTS_DIR, f"analysis_claim_types_{mshort}_{ds}.json")
            if not os.path.exists(ct_path):
                continue
            ct = load_json(ct_path)
            for cat, metrics in ct.items():
                if cat not in claim_type_data:
                    claim_type_data[cat] = []
                claim_type_data[cat].append(metrics.get("auc_pr", 0))

    if claim_type_data:
        cats = sorted(claim_type_data.keys())
        means = [np.mean(claim_type_data[c]) for c in cats]
        stds = [np.std(claim_type_data[c]) for c in cats]
        ax.bar(range(len(cats)), means, yerr=stds, color='#0072B2', alpha=0.7, capsize=3)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.replace("_", " ").title() for c in cats], rotation=20)
        ax.set_ylabel("AUC-PR")
        ax.set_title("SpecCheck Performance by Claim Type")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_4_claim_types.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "figure_4_claim_types.png"))
    plt.close()

    # ---- Figure 5: Ablation Studies ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Ladder depth
    ax = axes[0, 0]
    for mshort in ["llama", "mistral", "qwen"]:
        ks = []
        auc_prs = []
        for ds in DATASETS:
            ld_path = os.path.join(RESULTS_DIR, f"ablation_ladder_depth_{mshort}_{ds}.json")
            if not os.path.exists(ld_path):
                continue
            ld = load_json(ld_path)
            for k_str, metrics in sorted(ld.items()):
                k = int(k_str.split("=")[1])
                ks.append(k)
                auc_prs.append(metrics["auc_pr"])
        if ks:
            # Average across datasets
            k_vals = sorted(set(ks))
            avg_by_k = {k: np.mean([auc_prs[i] for i in range(len(ks)) if ks[i] == k]) for k in k_vals}
            ax.plot(list(avg_by_k.keys()), list(avg_by_k.values()), 'o-', label=mshort.capitalize())
    ax.set_xlabel("Ladder Depth K")
    ax.set_ylabel("AUC-PR")
    ax.set_title("(a) Effect of Ladder Depth")
    ax.legend(fontsize=9)

    # (b) Score variants
    ax = axes[0, 1]
    variant_data = {}
    for mshort in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            sv_path = os.path.join(RESULTS_DIR, f"ablation_score_variants_{mshort}_{ds}.json")
            if not os.path.exists(sv_path):
                continue
            sv = load_json(sv_path)
            for vname, metrics in sv.items():
                if vname.startswith("alpha="):
                    continue
                if vname not in variant_data:
                    variant_data[vname] = []
                variant_data[vname].append(metrics.get("auc_pr", 0))

    if variant_data:
        vnames = sorted(variant_data.keys())
        means = [np.mean(variant_data[v]) for v in vnames]
        stds = [np.std(variant_data[v]) for v in vnames]
        ax.bar(range(len(vnames)), means, yerr=stds, color='#009E73', alpha=0.7, capsize=3)
        ax.set_xticks(range(len(vnames)))
        ax.set_xticklabels([v.replace("_", " ").title() for v in vnames], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("AUC-PR")
        ax.set_title("(b) Score Variant Comparison")
        ax.set_ylim(0, 1.05)

    # (c) Alpha sweep
    ax = axes[1, 0]
    for mshort in ["llama", "mistral", "qwen"]:
        alphas = []
        auc_prs = []
        for ds in DATASETS:
            sv_path = os.path.join(RESULTS_DIR, f"ablation_score_variants_{mshort}_{ds}.json")
            if not os.path.exists(sv_path):
                continue
            sv = load_json(sv_path)
            for vname, metrics in sv.items():
                if vname.startswith("alpha="):
                    a = float(vname.split("=")[1])
                    alphas.append(a)
                    auc_prs.append(metrics.get("auc_pr", 0))
        if alphas:
            a_vals = sorted(set(alphas))
            avg_by_a = {a: np.mean([auc_prs[i] for i in range(len(alphas)) if alphas[i] == a]) for a in a_vals}
            ax.plot(list(avg_by_a.keys()), list(avg_by_a.values()), 'o-', label=mshort.capitalize())
    ax.set_xlabel("Alpha (violation weight)")
    ax.set_ylabel("AUC-PR")
    ax.set_title("(c) Alpha Parameter Sweep")
    ax.legend(fontsize=9)

    # (d) Combination analysis
    ax = axes[1, 1]
    combo_data = {}
    for mshort in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            co_path = os.path.join(RESULTS_DIR, f"analysis_combination_{mshort}_{ds}.json")
            if not os.path.exists(co_path):
                continue
            co = load_json(co_path)
            for cname, metrics in co.items():
                if cname not in combo_data:
                    combo_data[cname] = []
                combo_data[cname].append(metrics.get("auc_pr_mean", 0))

    if combo_data:
        # Show key combos
        show_combos = ["speccheck_only", "selfcheck_only", "logprob_only",
                       "all_baselines", "all_baselines_plus_speccheck"]
        show_combos = [c for c in show_combos if c in combo_data]
        means = [np.mean(combo_data[c]) for c in show_combos]
        stds = [np.std(combo_data[c]) for c in show_combos]
        short_labels = {
            "speccheck_only": "SpecCheck",
            "selfcheck_only": "SelfCheck",
            "logprob_only": "Logprob",
            "all_baselines": "All Base.",
            "all_baselines_plus_speccheck": "All+Spec.",
        }
        ax.bar(range(len(show_combos)), means, yerr=stds, color='#CC79A7', alpha=0.7, capsize=3)
        ax.set_xticks(range(len(show_combos)))
        ax.set_xticklabels([short_labels.get(c, c) for c in show_combos], rotation=20)
        ax.set_ylabel("AUC-PR (5-fold CV)")
        ax.set_title("(d) Method Combinations")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_5_ablations.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "figure_5_ablations.png"))
    plt.close()

    # ---- Supplementary: ROC Curves ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax_idx, ds in enumerate(DATASETS):
        ax = axes[ax_idx]
        for method in ['speccheck', 'selfcheck', 'logprob']:
            all_yt = []
            all_yp = []
            for mshort in ["llama", "mistral", "qwen"]:
                spec_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{ds}.json")
                if not os.path.exists(spec_path):
                    continue
                speccheck = load_json(spec_path)
                yt = [s["label"] for s in speccheck]

                if method == 'speccheck':
                    yp = [s["speccheck_score"] for s in speccheck]
                else:
                    bpath = os.path.join(RESULTS_DIR, f"baseline_{method}_{mshort}_{ds}.json")
                    if not os.path.exists(bpath):
                        continue
                    bdata = load_json(bpath)
                    bmap = {b["claim_id"]: b.get("hallucination_score", 0.5) for b in bdata}
                    yp = [bmap.get(s["claim_id"], 0.5) for s in speccheck]

                all_yt.extend(yt)
                all_yp.extend(yp)

            if all_yt and len(set(all_yt)) >= 2:
                fpr, tpr, _ = roc_curve(all_yt, all_yp)
                auc = roc_auc_score(all_yt, all_yp)
                ax.plot(fpr, tpr, label=f"{method_labels.get(method, method)} (AUC={auc:.3f})",
                        color=colors.get(method, '#333333'))

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel("FPR")
        if ax_idx == 0:
            ax.set_ylabel("TPR")
        ax.set_title(f"{ds.title()}")
        ax.legend(fontsize=8)

    fig.suptitle("ROC Curves (All Models Pooled)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "supp_roc_curves.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "supp_roc_curves.png"))
    plt.close()

    # ---- Supplementary: Heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for fig_idx, method in enumerate(["speccheck", "selfcheck"]):
        ax = axes[fig_idx]
        matrix = np.zeros((3, 3))
        for i, mshort in enumerate(["llama", "mistral", "qwen"]):
            for j, ds in enumerate(DATASETS):
                r = eval_results.get(mshort, {}).get(ds, {}).get(method, {})
                matrix[i, j] = r.get("auc_pr", 0)

        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0.3, vmax=1.0, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels([d.title() for d in DATASETS])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Llama", "Mistral", "Qwen"])
        ax.set_title(method_labels.get(method, method))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha='center', va='center',
                        color='white' if matrix[i,j] > 0.65 else 'black', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("AUC-PR Heatmap: SpecCheck vs SelfCheckGPT", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "supp_heatmap.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "supp_heatmap.png"))
    plt.close()

    # ---- Figure 1: Method schematic with example ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Show example of factual vs hallucinated confidence profiles
    ax.text(0.5, 0.95, "SpecCheck: Confidence Monotonicity Across Specificity Levels",
            transform=ax.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

    # Factual example
    levels_x = [0, 1, 2, 3]
    factual_ex = [0.75, 0.82, 0.90, 0.95]
    halluc_ex = [0.70, 0.55, 0.85, 0.60]

    ax.plot(levels_x, factual_ex, 'o-', color='#0072B2', linewidth=2.5, markersize=10,
            label='Factual claim (monotonic ✓)')
    ax.plot(levels_x, halluc_ex, 's-', color='#D55E00', linewidth=2.5, markersize=10,
            label='Hallucinated claim (non-monotonic ✗)')

    ax.set_xlabel("Specificity Level", fontsize=12)
    ax.set_ylabel("Confidence P(True)", fontsize=12)
    ax.set_xticks(levels_x)
    ax.set_xticklabels([
        'Level 0\n"Born in 1856\nin Fremont, OH"',
        'Level 1\n"Born in the\n1850s in Ohio"',
        'Level 2\n"Born in the\nmid-19th century"',
        'Level 3\n"Was born in\nthe United States"'
    ], fontsize=8)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "figure_1_method_overview.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "figure_1_method_overview.png"))
    plt.close()

    print(f"\nAll figures saved to {FIGURES_DIR}")


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()

    # Step 0: Random baseline (no GPU needed)
    run_random_baseline()

    # Process each model sequentially
    for model_name in MODELS:
        mshort = get_model_short(model_name)
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL: {mshort} ({model_name})")
        print(f"{'='*60}")

        model, tokenizer = load_model(model_name)

        # Step 1: Re-run confidence estimation (FIXED)
        run_confidence_estimation(model, tokenizer, model_name)

        # Step 2: Re-compute SpecCheck scores
        for ds in DATASETS:
            compute_speccheck_scores(mshort, ds)

        # Step 3: Run baselines
        run_verbalized_baseline(model, tokenizer, model_name)
        run_logprob_baseline(model, tokenizer, model_name)
        run_selfcheck_baseline(model, tokenizer, model_name)

        # Step 4: Sampling ablation (only for llama to save compute)
        if mshort == "llama":
            run_sampling_ablation(model, tokenizer, model_name)

        # Unload model to free GPU
        unload_model(model, tokenizer)

        # Step 5: Ablations (CPU-only, use saved scores)
        for ds in DATASETS:
            run_ablation_ladder_depth(mshort, ds)
            run_ablation_score_variants(mshort, ds)
            run_claim_type_analysis(mshort, ds)
            run_combination_analysis(mshort, ds)

        elapsed = time.time() - start_time
        print(f"\nElapsed so far: {elapsed/60:.1f} min")

    # Step 6: Evaluation
    eval_results = compute_all_metrics()

    # Step 7: Figures
    generate_figures(eval_results)

    # Step 8: Save canonical results.json
    save_canonical_results(eval_results)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"{'='*60}")


def save_canonical_results(eval_results):
    """Save a single canonical results.json at workspace root."""
    print("\nSaving canonical results.json...")

    results = {
        "experiment": "SpecCheck: Detecting LLM Hallucinations via Confidence Monotonicity",
        "models": ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-7B-Instruct"],
        "datasets": ["FActScore", "LongFact", "TruthfulQA"],
        "seeds": SEEDS,
        "main_results": {},
        "monotonicity_analysis": {},
        "ablations": {},
        "combination_analysis": {},
        "success_criteria": {},
    }

    # Main results
    for mshort in ["llama", "mistral", "qwen"]:
        results["main_results"][mshort] = {}
        results["monotonicity_analysis"][mshort] = {}
        for ds in DATASETS:
            ds_res = eval_results.get(mshort, {}).get(ds, {})
            main_methods = {}
            for method in ["speccheck", "selfcheck", "verbalized", "logprob", "random"]:
                if method in ds_res:
                    main_methods[method] = {
                        "auc_roc": ds_res[method]["auc_roc"],
                        "auc_pr": ds_res[method]["auc_pr"],
                        "auc_roc_ci": ds_res[method].get("auc_roc_ci", []),
                        "auc_pr_ci": ds_res[method].get("auc_pr_ci", []),
                    }
            results["main_results"][mshort][ds] = main_methods

            mono = ds_res.get("monotonicity_analysis", {})
            if mono:
                results["monotonicity_analysis"][mshort][ds] = mono

    # Ablations
    for mshort in ["llama", "mistral", "qwen"]:
        results["ablations"][mshort] = {}
        for ds in DATASETS:
            results["ablations"][mshort][ds] = {}
            # Ladder depth
            ld_path = os.path.join(RESULTS_DIR, f"ablation_ladder_depth_{mshort}_{ds}.json")
            if os.path.exists(ld_path):
                results["ablations"][mshort][ds]["ladder_depth"] = load_json(ld_path)
            # Score variants
            sv_path = os.path.join(RESULTS_DIR, f"ablation_score_variants_{mshort}_{ds}.json")
            if os.path.exists(sv_path):
                results["ablations"][mshort][ds]["score_variants"] = load_json(sv_path)
            # Claim types
            ct_path = os.path.join(RESULTS_DIR, f"analysis_claim_types_{mshort}_{ds}.json")
            if os.path.exists(ct_path):
                results["ablations"][mshort][ds]["claim_types"] = load_json(ct_path)

    # Combinations
    for mshort in ["llama", "mistral", "qwen"]:
        results["combination_analysis"][mshort] = {}
        for ds in DATASETS:
            co_path = os.path.join(RESULTS_DIR, f"analysis_combination_{mshort}_{ds}.json")
            if os.path.exists(co_path):
                results["combination_analysis"][mshort][ds] = load_json(co_path)

    # Success criteria evaluation
    # Criterion 1: SpecCheck > SelfCheck on >= 2/3 benchmarks
    wins_by_ds = {}
    for ds in DATASETS:
        n_wins = 0
        n_total = 0
        for m in ["llama", "mistral", "qwen"]:
            r = eval_results.get(m, {}).get(ds, {})
            if "speccheck" in r and "selfcheck" in r:
                n_total += 1
                if r["speccheck"]["auc_pr"] > r["selfcheck"]["auc_pr"]:
                    n_wins += 1
        wins_by_ds[ds] = {"wins": n_wins, "total": n_total}
    n_ds_wins = sum(1 for ds, v in wins_by_ds.items() if v["wins"] > v["total"] / 2)
    results["success_criteria"]["criterion_1_speccheck_beats_selfcheck"] = {
        "target": "SpecCheck > SelfCheck AUC-PR on >= 2/3 benchmarks",
        "result": wins_by_ds,
        "datasets_won": n_ds_wins,
        "met": n_ds_wins >= 2,
    }

    # Criterion 2: Factual monotonicity > 85%
    factual_rates = []
    for m in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            mono = eval_results.get(m, {}).get(ds, {}).get("monotonicity_analysis", {})
            if mono:
                factual_rates.append(mono.get("factual_perfect_monotonicity_rate", 0))
    avg_factual = float(np.mean(factual_rates)) if factual_rates else 0
    results["success_criteria"]["criterion_2_factual_monotonicity"] = {
        "target": ">85% factual claims with perfect monotonicity",
        "result": avg_factual,
        "per_model_dataset": factual_rates,
        "met": avg_factual > 0.85,
    }

    # Criterion 3: Hallucinated violation > 40%
    halluc_rates = []
    for m in ["llama", "mistral", "qwen"]:
        for ds in DATASETS:
            mono = eval_results.get(m, {}).get(ds, {}).get("monotonicity_analysis", {})
            if mono:
                halluc_rates.append(mono.get("halluc_violation_rate", 0))
    avg_halluc = float(np.mean(halluc_rates)) if halluc_rates else 0
    results["success_criteria"]["criterion_3_halluc_violation"] = {
        "target": ">40% hallucinated claims violate monotonicity",
        "result": avg_halluc,
        "per_model_dataset": halluc_rates,
        "met": avg_halluc > 0.40,
    }

    save_json(results, os.path.join(BASE_DIR, "results.json"))
    print("Canonical results.json saved.")


if __name__ == "__main__":
    main()
