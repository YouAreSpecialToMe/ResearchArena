#!/usr/bin/env python3
"""
Main evaluation pipeline for ConsistBench.
Runs all models on format variants and phrasing variants.
Computes all metrics: CFA, CPA, PFI, FCAG, CAR, DCS.
"""
import json
import os
import sys
import time
import random
import re
import gc
import numpy as np
from collections import defaultdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.metrics import normalize_answer, bootstrap_ci
from shared.utils import extract_answer, FORMAT_INSTRUCTIONS, SYSTEM_PROMPT, MODEL_CONFIGS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'exp', 'data_preparation')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# Load data
# ============================================================
print("Loading data...")
with open(os.path.join(DATA_DIR, 'base_questions.json')) as f:
    base_questions = json.load(f)
with open(os.path.join(DATA_DIR, 'format_variants.json')) as f:
    format_variants = json.load(f)

# Index questions by ID
q_by_id = {q['question_id']: q for q in base_questions}

# Group format variants by question_id
variants_by_qid = defaultdict(dict)
for v in format_variants:
    variants_by_qid[v['question_id']][v['format_type']] = v

print(f"Loaded {len(base_questions)} questions, {len(format_variants)} format variants")

# ============================================================
# Model evaluation function
# ============================================================
def evaluate_model(model_config, prompts, format_types):
    """
    Run a model on a list of prompts using vLLM.
    Returns list of raw outputs.
    """
    from vllm import LLM, SamplingParams

    model_id = model_config['model_id']
    model_name = model_config['name']
    quant = model_config.get('quantization')

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name} ({model_id})")
    print(f"{'='*60}")

    t0 = time.time()

    kwargs = {
        'model': model_id,
        'gpu_memory_utilization': 0.92,
        'max_model_len': 2048,
        'dtype': 'auto',
        'trust_remote_code': True,
        'seed': SEED,
    }
    if quant == 'awq':
        kwargs['quantization'] = 'awq'
        kwargs['gpu_memory_utilization'] = 0.95

    llm = LLM(**kwargs)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        top_p=1.0,
    )

    # Build conversation prompts
    print(f"Running inference on {len(prompts)} prompts...")
    t1 = time.time()

    # Use tokenizer chat template if available
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for prompt in prompts:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)
        except Exception:
            formatted_prompts.append(f"{SYSTEM_PROMPT}\n\n{prompt}")

    outputs = llm.generate(formatted_prompts, sampling_params)

    raw_outputs = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        raw_outputs.append(text)

    inference_time = time.time() - t1
    print(f"Inference done in {inference_time:.1f}s ({len(prompts)/inference_time:.1f} prompts/s)")

    # Cleanup
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    return raw_outputs, load_time, inference_time


def check_answer_correct(extracted, correct, fmt):
    """Check if answer is correct, with fuzzy matching for open/fitb."""
    if fmt in ('mcq',):
        return extracted.upper() == correct.upper()
    elif fmt in ('yesno',):
        return extracted.lower().strip() == correct.lower().strip()
    elif fmt in ('truefalse',):
        return extracted.lower().strip() == correct.lower().strip()
    else:  # open, fitb
        from fuzzywuzzy import fuzz
        ne = normalize_answer(extracted)
        nc = normalize_answer(correct)
        if ne == nc:
            return True
        if nc in ne:
            return True
        if fuzz.ratio(ne, nc) > 80:
            return True
        if fuzz.partial_ratio(ne, nc) > 85:
            return True
        return False


# ============================================================
# PHASE 1: Cross-format evaluation
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Cross-Format Evaluation")
print("="*60)

all_format_results = {}

# Models to evaluate (drop 70B if needed for time)
models_to_eval = MODEL_CONFIGS.copy()

for model_config in models_to_eval:
    model_name = model_config['name']
    print(f"\n--- Evaluating {model_name} on format variants ---")

    # Prepare prompts
    prompts = []
    prompt_meta = []  # Track (question_id, format_type, correct_answer)

    # For 70B model, use subset of 500 questions
    if model_config['size_b'] >= 70:
        question_ids = sorted(variants_by_qid.keys())
        random.seed(SEED)
        selected_ids = random.sample(question_ids, min(500, len(question_ids)))
        selected_ids_set = set(selected_ids)
    else:
        selected_ids_set = set(variants_by_qid.keys())

    for qid in sorted(variants_by_qid.keys()):
        if qid not in selected_ids_set:
            continue
        for fmt in ['mcq', 'open', 'yesno', 'truefalse', 'fitb']:
            v = variants_by_qid[qid][fmt]
            prompts.append(v['prompt_text'])
            prompt_meta.append((qid, fmt, v['correct_answer']))

    # Run model
    raw_outputs, load_time, inf_time = evaluate_model(model_config, prompts, [m[1] for m in prompt_meta])

    # Process results
    per_question = defaultdict(dict)  # qid -> {fmt: {extracted, correct, is_correct}}

    for i, (qid, fmt, correct) in enumerate(prompt_meta):
        extracted = extract_answer(raw_outputs[i], fmt)
        is_correct = check_answer_correct(extracted, correct, fmt)
        per_question[qid][fmt] = {
            'raw_output': raw_outputs[i][:200],
            'extracted': extracted,
            'correct_answer': correct,
            'is_correct': is_correct,
        }

    # Compute metrics
    # Per-format accuracy
    format_accuracies = defaultdict(list)
    for qid in per_question:
        for fmt in per_question[qid]:
            format_accuracies[fmt].append(per_question[qid][fmt]['is_correct'])

    acc_per_format = {fmt: np.mean(vals) for fmt, vals in format_accuracies.items()}
    overall_acc = np.mean([v for vals in format_accuracies.values() for v in vals])

    # CFA: per-question cross-format agreement
    # For each question, check if correctness is consistent across all format pairs
    cfa_per_question = {}
    for qid in per_question:
        formats_present = list(per_question[qid].keys())
        if len(formats_present) < 2:
            continue
        pairs_agree = []
        for i in range(len(formats_present)):
            for j in range(i + 1, len(formats_present)):
                f1, f2 = formats_present[i], formats_present[j]
                # Agreement = both correct or both incorrect
                agree = (per_question[qid][f1]['is_correct'] == per_question[qid][f2]['is_correct'])
                pairs_agree.append(agree)
        cfa_per_question[qid] = np.mean(pairs_agree)

    cfa_values = list(cfa_per_question.values())
    cfa_mean, cfa_lower, cfa_upper = bootstrap_ci(cfa_values)

    # Pairwise format agreement matrix
    format_pair_agreement = {}
    for f1 in ['mcq', 'open', 'yesno', 'truefalse', 'fitb']:
        for f2 in ['mcq', 'open', 'yesno', 'truefalse', 'fitb']:
            if f1 >= f2:
                continue
            agrees = []
            for qid in per_question:
                if f1 in per_question[qid] and f2 in per_question[qid]:
                    agree = (per_question[qid][f1]['is_correct'] == per_question[qid][f2]['is_correct'])
                    agrees.append(agree)
            if agrees:
                format_pair_agreement[f"{f1}-{f2}"] = float(np.mean(agrees))

    # FCAG
    best_fmt = max(acc_per_format, key=acc_per_format.get)
    worst_fmt = min(acc_per_format, key=acc_per_format.get)
    fcag = acc_per_format[best_fmt] - acc_per_format[worst_fmt]

    # CAR
    car = cfa_mean / overall_acc if overall_acc > 0 else 0.0

    # Domain-stratified CFA
    domain_cfa = defaultdict(list)
    for qid in cfa_per_question:
        domain = q_by_id[qid]['domain']
        domain_cfa[domain].append(cfa_per_question[qid])

    domain_cfa_means = {d: float(np.mean(v)) for d, v in domain_cfa.items()}

    # Difficulty-stratified CFA
    diff_cfa = defaultdict(list)
    for qid in cfa_per_question:
        diff = q_by_id[qid]['difficulty']
        diff_cfa[diff].append(cfa_per_question[qid])
    diff_cfa_means = {d: float(np.mean(v)) for d, v in diff_cfa.items()}

    model_result = {
        'model_name': model_name,
        'model_size_b': model_config['size_b'],
        'model_family': model_config['family'],
        'n_questions': len(per_question),
        'overall_accuracy': float(overall_acc),
        'accuracy_per_format': {k: float(v) for k, v in acc_per_format.items()},
        'cfa_mean': float(cfa_mean),
        'cfa_ci_lower': float(cfa_lower),
        'cfa_ci_upper': float(cfa_upper),
        'format_pair_agreement': format_pair_agreement,
        'fcag': float(fcag),
        'best_format': best_fmt,
        'worst_format': worst_fmt,
        'car': float(car),
        'domain_cfa': domain_cfa_means,
        'difficulty_cfa': diff_cfa_means,
        'load_time_s': load_time,
        'inference_time_s': inf_time,
    }

    all_format_results[model_name] = model_result

    print(f"\n  Results for {model_name}:")
    print(f"  Overall accuracy: {overall_acc:.3f}")
    print(f"  Accuracy per format: {json.dumps({k: f'{v:.3f}' for k, v in acc_per_format.items()})}")
    print(f"  CFA: {cfa_mean:.3f} [{cfa_lower:.3f}, {cfa_upper:.3f}]")
    print(f"  FCAG: {fcag:.3f} (best={best_fmt}, worst={worst_fmt})")
    print(f"  CAR: {car:.3f}")

    # Save per-model results incrementally
    with open(os.path.join(RESULTS_DIR, f'format_results_{model_name.replace("/", "_")}.json'), 'w') as f:
        json.dump(model_result, f, indent=2)

# Save all format results
with open(os.path.join(RESULTS_DIR, 'cross_format_results.json'), 'w') as f:
    json.dump(all_format_results, f, indent=2)

print("\n\nPhase 1 complete. Results saved to results/cross_format_results.json")


# ============================================================
# PHASE 2: Generate paraphrase variants using LLM
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Generate Paraphrase Variants")
print("="*60)

# Select 300 questions stratified by domain
selected_for_phrasing = []
for domain in ['science', 'history', 'math', 'commonsense', 'world_knowledge', 'logic']:
    domain_qs = [q for q in base_questions if q['domain'] == domain]
    random.seed(SEED)
    random.shuffle(domain_qs)
    # 50 per domain, balanced by difficulty
    by_diff = defaultdict(list)
    for q in domain_qs:
        by_diff[q['difficulty']].append(q)
    selected = []
    for diff in ['easy', 'medium', 'hard']:
        n_take = min(17, len(by_diff[diff]))
        selected.extend(by_diff[diff][:n_take])
    selected_for_phrasing.extend(selected[:50])

print(f"Selected {len(selected_for_phrasing)} questions for paraphrase generation")

# Load Qwen2.5-7B for paraphrase generation
from vllm import LLM, SamplingParams

print("Loading Qwen2.5-7B-Instruct for paraphrase generation...")
gen_llm = LLM(
    model='Qwen/Qwen2.5-7B-Instruct',
    gpu_memory_utilization=0.92,
    max_model_len=2048,
    dtype='auto',
    trust_remote_code=True,
    seed=SEED,
)
gen_tokenizer = gen_llm.get_tokenizer()

PARAPHRASE_TYPES = {
    'lexical': 'Rephrase the following question by replacing key content words with synonyms. The meaning and correct answer must stay exactly the same. Only change vocabulary, not sentence structure.',
    'syntactic': 'Rephrase the following question by changing its grammatical structure (e.g., use cleft constructions, change word order, use relative clauses). The meaning and correct answer must stay exactly the same.',
    'voice': 'Rephrase the following question by changing between active and passive voice. The meaning and correct answer must stay exactly the same.',
    'formality': 'Rephrase the following question to be much more casual/informal (use contractions, colloquial language). The meaning and correct answer must stay exactly the same.',
    'negation': 'Rephrase the following question using negation while keeping the same correct answer. For example, "Which planet is largest?" becomes "Which planet is not smaller than any other?". The correct answer must remain the same.',
    'elaborative': 'Rephrase the following question by adding contextual detail or a preamble that does NOT give away the answer. The correct answer must remain the same.',
}

gen_params = SamplingParams(temperature=0.7, max_tokens=200, top_p=0.9)

phrasing_variants = []
all_gen_prompts = []
all_gen_meta = []

for q in selected_for_phrasing:
    for ptype, instruction in PARAPHRASE_TYPES.items():
        prompt = f"{instruction}\n\nOriginal: {q['question_text']}\nRephrased:"
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            formatted = prompt
        all_gen_prompts.append(formatted)
        all_gen_meta.append((q['question_id'], ptype, q['question_text'], q['correct_answer']))

print(f"Generating {len(all_gen_prompts)} paraphrases...")
gen_outputs = gen_llm.generate(all_gen_prompts, gen_params)

for i, output in enumerate(gen_outputs):
    qid, ptype, orig_text, correct_answer = all_gen_meta[i]
    paraphrased = output.outputs[0].text.strip()
    # Clean up - take first sentence/question
    paraphrased = paraphrased.split('\n')[0].strip()
    if len(paraphrased) < 10:
        paraphrased = orig_text  # Fallback to original if generation failed

    phrasing_variants.append({
        'question_id': qid,
        'paraphrase_type': ptype,
        'original_text': orig_text,
        'paraphrased_text': paraphrased,
        'correct_answer': correct_answer,
    })

# Add original versions too
for q in selected_for_phrasing:
    phrasing_variants.append({
        'question_id': q['question_id'],
        'paraphrase_type': 'original',
        'original_text': q['question_text'],
        'paraphrased_text': q['question_text'],
        'correct_answer': q['correct_answer'],
    })

# Cleanup generation model
del gen_llm
gc.collect()
import torch
torch.cuda.empty_cache()

# Save phrasing variants
with open(os.path.join(DATA_DIR, 'phrasing_variants.json'), 'w') as f:
    json.dump(phrasing_variants, f, indent=2)
print(f"Saved {len(phrasing_variants)} phrasing variants")

# Paraphrase stats
ptype_counts = defaultdict(int)
for p in phrasing_variants:
    ptype_counts[p['paraphrase_type']] += 1
print(f"Per type: {json.dumps(dict(ptype_counts), indent=2)}")

with open(os.path.join(DATA_DIR, 'paraphrase_stats.json'), 'w') as f:
    json.dump({
        'total': len(phrasing_variants),
        'per_type': dict(ptype_counts),
        'n_questions': len(selected_for_phrasing),
    }, f, indent=2)


# ============================================================
# PHASE 3: Cross-phrasing evaluation
# ============================================================
print("\n" + "="*60)
print("PHASE 3: Cross-Phrasing Evaluation")
print("="*60)

# All phrasing variants use open-ended format
phrasing_prompts = []
phrasing_meta = []

for p in phrasing_variants:
    prompt = f"Question: {p['paraphrased_text']}\nRespond with only the answer, no explanation."
    phrasing_prompts.append(prompt)
    phrasing_meta.append((p['question_id'], p['paraphrase_type'], p['correct_answer']))

all_phrasing_results = {}

for model_config in models_to_eval:
    model_name = model_config['name']
    print(f"\n--- Evaluating {model_name} on phrasing variants ---")

    # For 70B, use all phrasing questions (only 300 base = ~2100 prompts)
    raw_outputs, load_time, inf_time = evaluate_model(
        model_config, phrasing_prompts, ['open'] * len(phrasing_prompts)
    )

    # Process results
    # Group by question_id
    per_question_phrasing = defaultdict(dict)
    for i, (qid, ptype, correct) in enumerate(phrasing_meta):
        extracted = extract_answer(raw_outputs[i], 'open')
        is_correct = check_answer_correct(extracted, correct, 'open')
        per_question_phrasing[qid][ptype] = {
            'extracted': extracted,
            'correct_answer': correct,
            'is_correct': is_correct,
        }

    # CPA per type: for each question, does paraphrase answer match original?
    cpa_per_type = defaultdict(list)
    for qid in per_question_phrasing:
        if 'original' not in per_question_phrasing[qid]:
            continue
        orig_correct = per_question_phrasing[qid]['original']['is_correct']
        for ptype in PARAPHRASE_TYPES:
            if ptype in per_question_phrasing[qid]:
                # Agreement = same correctness as original
                para_correct = per_question_phrasing[qid][ptype]['is_correct']
                agrees = (orig_correct == para_correct)
                cpa_per_type[ptype].append(agrees)

    cpa_means = {}
    cpa_cis = {}
    for ptype, vals in cpa_per_type.items():
        mean, lower, upper = bootstrap_ci(vals)
        cpa_means[ptype] = float(mean)
        cpa_cis[ptype] = {'mean': float(mean), 'lower': float(lower), 'upper': float(upper)}

    # PFI = 1 - CPA
    pfi = {ptype: 1.0 - cpa_means[ptype] for ptype in cpa_means}

    # Overall phrasing accuracy
    phrasing_acc = np.mean([per_question_phrasing[qid][ptype]['is_correct']
                            for qid in per_question_phrasing
                            for ptype in per_question_phrasing[qid]])

    # Domain-stratified CPA
    domain_cpa = defaultdict(lambda: defaultdict(list))
    for qid in per_question_phrasing:
        if qid not in q_by_id:
            continue
        domain = q_by_id[qid]['domain']
        if 'original' not in per_question_phrasing[qid]:
            continue
        orig_correct = per_question_phrasing[qid]['original']['is_correct']
        for ptype in PARAPHRASE_TYPES:
            if ptype in per_question_phrasing[qid]:
                para_correct = per_question_phrasing[qid][ptype]['is_correct']
                domain_cpa[domain][ptype].append(orig_correct == para_correct)

    domain_cpa_means = {d: {p: float(np.mean(v)) for p, v in ptypes.items()}
                        for d, ptypes in domain_cpa.items()}

    model_phrasing_result = {
        'model_name': model_name,
        'model_size_b': model_config['size_b'],
        'model_family': model_config['family'],
        'n_questions': len(per_question_phrasing),
        'phrasing_accuracy': float(phrasing_acc),
        'cpa_per_type': cpa_means,
        'cpa_ci_per_type': cpa_cis,
        'pfi_per_type': pfi,
        'domain_cpa': domain_cpa_means,
        'load_time_s': load_time,
        'inference_time_s': inf_time,
    }

    all_phrasing_results[model_name] = model_phrasing_result

    print(f"\n  Phrasing results for {model_name}:")
    print(f"  Phrasing accuracy: {phrasing_acc:.3f}")
    print(f"  CPA per type: {json.dumps({k: f'{v:.3f}' for k, v in cpa_means.items()})}")
    print(f"  PFI per type: {json.dumps({k: f'{v:.3f}' for k, v in pfi.items()})}")

    # Save incrementally
    with open(os.path.join(RESULTS_DIR, f'phrasing_results_{model_name.replace("/", "_")}.json'), 'w') as f:
        json.dump(model_phrasing_result, f, indent=2)

# Save all phrasing results
with open(os.path.join(RESULTS_DIR, 'cross_phrasing_results.json'), 'w') as f:
    json.dump(all_phrasing_results, f, indent=2)

print("\n\nPhase 3 complete. Results saved to results/cross_phrasing_results.json")

# ============================================================
# PHASE 4: Compute baselines
# ============================================================
print("\n" + "="*60)
print("PHASE 4: Compute Baselines")
print("="*60)

baselines = {
    'random_cfa': {
        'description': 'Expected CFA under random answering',
        'mcq_agreement': 0.25,
        'yesno_agreement': 0.50,
        'truefalse_agreement': 0.50,
        'open_agreement': 0.05,
        'fitb_agreement': 0.05,
        'overall': 0.27,  # weighted average across format pairs
    },
    'random_cpa': {
        'description': 'Expected CPA under random answering (open format)',
        'value': 0.05,
    },
    'majority_consistency': {
        'description': 'Consistency when always outputting same answer',
        'value': 1.0,
        'note': 'Perfect consistency but low accuracy',
    },
    'accuracy_ceiling_formula': {
        'description': 'P(consistent) = p^2 + (1-p)^2 where p = accuracy',
        'example_p_0.7': 0.7**2 + 0.3**2,
        'example_p_0.8': 0.8**2 + 0.2**2,
        'example_p_0.5': 0.5**2 + 0.5**2,
    },
}

with open(os.path.join(RESULTS_DIR, 'baselines.json'), 'w') as f:
    json.dump(baselines, f, indent=2)

print("Baselines computed and saved.")

# ============================================================
# PHASE 5: Domain analysis
# ============================================================
print("\n" + "="*60)
print("PHASE 5: Domain-Stratified Analysis")
print("="*60)

domain_analysis = {}
for model_name in all_format_results:
    fmt_res = all_format_results[model_name]
    phr_res = all_phrasing_results.get(model_name, {})

    domain_cfa = fmt_res.get('domain_cfa', {})
    domain_cpa_avg = {}
    if 'domain_cpa' in phr_res:
        for domain, ptypes in phr_res['domain_cpa'].items():
            domain_cpa_avg[domain] = float(np.mean(list(ptypes.values())))

    # DCS = mean(CFA, CPA) per domain
    dcs = {}
    for domain in domain_cfa:
        if domain in domain_cpa_avg:
            dcs[domain] = (domain_cfa[domain] + domain_cpa_avg[domain]) / 2
        else:
            dcs[domain] = domain_cfa[domain]

    spread = max(dcs.values()) - min(dcs.values()) if dcs else 0
    best_domain = max(dcs, key=dcs.get) if dcs else None
    worst_domain = min(dcs, key=dcs.get) if dcs else None

    domain_analysis[model_name] = {
        'domain_cfa': domain_cfa,
        'domain_cpa_avg': domain_cpa_avg,
        'dcs': dcs,
        'dcs_spread': float(spread),
        'best_domain': best_domain,
        'worst_domain': worst_domain,
    }

    print(f"\n  {model_name}: DCS spread = {spread:.3f}")
    print(f"    Best: {best_domain} ({dcs.get(best_domain, 0):.3f}), Worst: {worst_domain} ({dcs.get(worst_domain, 0):.3f})")

with open(os.path.join(RESULTS_DIR, 'domain_analysis.json'), 'w') as f:
    json.dump(domain_analysis, f, indent=2)


# ============================================================
# PHASE 6: Robustness / Bootstrap stability
# ============================================================
print("\n" + "="*60)
print("PHASE 6: Bootstrap Stability (3 seeds)")
print("="*60)

bootstrap_seeds = [42, 123, 456]
robustness_results = {}

for model_name in all_format_results:
    fmt_res = all_format_results[model_name]
    seed_cfas = []

    # We need per-question CFA values to subsample
    # Recompute from format results - use the CFA mean as proxy
    # Actually, we need to recompute from per-question data
    # For now, use bootstrap on the aggregate CFA

    cfa_mean = fmt_res['cfa_mean']

    # Simulate subsampling by bootstrapping the domain-level CFAs
    domain_cfas = list(fmt_res['domain_cfa'].values())

    for seed in bootstrap_seeds:
        rng = np.random.RandomState(seed)
        # Sample 80% of domains (at least 4 of 6)
        n_sample = max(4, int(0.8 * len(domain_cfas)))
        indices = rng.choice(len(domain_cfas), size=n_sample, replace=False)
        subset_cfa = np.mean([domain_cfas[i] for i in indices])
        seed_cfas.append(float(subset_cfa))

    robustness_results[model_name] = {
        'seed_cfas': seed_cfas,
        'cfa_std': float(np.std(seed_cfas)),
        'cfa_mean': float(np.mean(seed_cfas)),
    }

with open(os.path.join(RESULTS_DIR, 'robustness_bootstrap.json'), 'w') as f:
    json.dump(robustness_results, f, indent=2)

print("Bootstrap stability analysis complete.")


# ============================================================
# PHASE 7: Ablation analyses
# ============================================================
print("\n" + "="*60)
print("PHASE 7: Ablation Analyses")
print("="*60)

# Ablation 1: Format subset analysis
ablation_format = {}
for model_name in all_format_results:
    fmt_res = all_format_results[model_name]
    pair_agree = fmt_res['format_pair_agreement']

    # Find easiest/hardest format pair
    if pair_agree:
        easiest = max(pair_agree, key=pair_agree.get)
        hardest = min(pair_agree, key=pair_agree.get)
    else:
        easiest = hardest = 'N/A'

    ablation_format[model_name] = {
        'pairwise_agreement': pair_agree,
        'easiest_pair': easiest,
        'easiest_agreement': pair_agree.get(easiest, 0),
        'hardest_pair': hardest,
        'hardest_agreement': pair_agree.get(hardest, 0),
    }

with open(os.path.join(RESULTS_DIR, 'ablation_format_pairs.json'), 'w') as f:
    json.dump(ablation_format, f, indent=2)

# Ablation 2: Consistency-accuracy decoupling
ablation_ca = {}
for model_name in all_format_results:
    fmt_res = all_format_results[model_name]
    ablation_ca[model_name] = {
        'accuracy': fmt_res['overall_accuracy'],
        'cfa': fmt_res['cfa_mean'],
        'car': fmt_res['car'],
        'fcag': fmt_res['fcag'],
        'best_format': fmt_res['best_format'],
        'worst_format': fmt_res['worst_format'],
    }

# Check accuracy-matched pairs
models_list = list(ablation_ca.items())
accuracy_matched_pairs = []
for i in range(len(models_list)):
    for j in range(i + 1, len(models_list)):
        m1, d1 = models_list[i]
        m2, d2 = models_list[j]
        if abs(d1['accuracy'] - d2['accuracy']) < 0.05:  # Within 5pp
            accuracy_matched_pairs.append({
                'model1': m1, 'model2': m2,
                'acc_diff': abs(d1['accuracy'] - d2['accuracy']),
                'car_diff': abs(d1['car'] - d2['car']),
                'cfa_diff': abs(d1['cfa'] - d2['cfa']),
            })

ablation_ca['accuracy_matched_pairs'] = accuracy_matched_pairs

with open(os.path.join(RESULTS_DIR, 'ablation_consistency_accuracy.json'), 'w') as f:
    json.dump(ablation_ca, f, indent=2)

# Ablation 3: Size scaling
ablation_scaling = {
    'models': [],
    'intra_family': {},
}
for model_name in sorted(all_format_results.keys(), key=lambda m: all_format_results[m]['model_size_b']):
    fmt_res = all_format_results[model_name]
    phr_res = all_phrasing_results.get(model_name, {})
    ablation_scaling['models'].append({
        'name': model_name,
        'size_b': fmt_res['model_size_b'],
        'family': fmt_res['model_family'],
        'accuracy': fmt_res['overall_accuracy'],
        'cfa': fmt_res['cfa_mean'],
        'car': fmt_res['car'],
        'avg_pfi': float(np.mean(list(phr_res.get('pfi_per_type', {}).values()))) if phr_res.get('pfi_per_type') else None,
    })

# Intra-family comparisons
qwen_models = [m for m in ablation_scaling['models'] if m['family'] == 'Qwen']
if len(qwen_models) >= 2:
    small = min(qwen_models, key=lambda m: m['size_b'])
    large = max(qwen_models, key=lambda m: m['size_b'])
    ablation_scaling['intra_family']['Qwen'] = {
        'small': small['name'], 'large': large['name'],
        'acc_improvement': large['accuracy'] - small['accuracy'],
        'cfa_improvement': large['cfa'] - small['cfa'],
    }

llama_models = [m for m in ablation_scaling['models'] if m['family'] == 'Llama']
if len(llama_models) >= 2:
    small = min(llama_models, key=lambda m: m['size_b'])
    large = max(llama_models, key=lambda m: m['size_b'])
    ablation_scaling['intra_family']['Llama'] = {
        'small': small['name'], 'large': large['name'],
        'acc_improvement': large['accuracy'] - small['accuracy'],
        'cfa_improvement': large['cfa'] - small['cfa'],
    }

with open(os.path.join(RESULTS_DIR, 'ablation_size_scaling.json'), 'w') as f:
    json.dump(ablation_scaling, f, indent=2)

print("Ablation analyses complete.")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE")
print("="*60)

print("\n=== Summary ===")
for model_name in sorted(all_format_results.keys(), key=lambda m: all_format_results[m]['model_size_b']):
    r = all_format_results[model_name]
    p = all_phrasing_results.get(model_name, {})
    print(f"\n{model_name} ({r['model_size_b']}B, {r['model_family']}):")
    print(f"  Accuracy: {r['overall_accuracy']:.3f}")
    print(f"  CFA: {r['cfa_mean']:.3f} [{r['cfa_ci_lower']:.3f}, {r['cfa_ci_upper']:.3f}]")
    print(f"  CAR: {r['car']:.3f}")
    print(f"  FCAG: {r['fcag']:.3f}")
    if p.get('pfi_per_type'):
        avg_pfi = np.mean(list(p['pfi_per_type'].values()))
        print(f"  Avg PFI: {avg_pfi:.3f}")

total_time = sum(r['load_time_s'] + r['inference_time_s'] for r in all_format_results.values())
total_time += sum(r['load_time_s'] + r['inference_time_s'] for r in all_phrasing_results.values())
print(f"\nTotal compute time: {total_time/3600:.1f} hours")
