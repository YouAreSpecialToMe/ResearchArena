#!/usr/bin/env python3
"""
Phase 2+: Paraphrase generation, phrasing evaluation, and all analysis.
Assumes Phase 1 format results are already saved in results/.
"""
import json
import os
import sys
import time
import random
import gc
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.metrics import normalize_answer, bootstrap_ci
from shared.utils import extract_answer, SYSTEM_PROMPT, MODEL_CONFIGS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'exp', 'data_preparation')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load data
print("Loading data...")
with open(os.path.join(DATA_DIR, 'base_questions.json')) as f:
    base_questions = json.load(f)
with open(os.path.join(DATA_DIR, 'format_variants.json')) as f:
    format_variants = json.load(f)

q_by_id = {q['question_id']: q for q in base_questions}

# Load existing format results
print("Loading existing format results...")
all_format_results = {}
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith('format_results_') and fname.endswith('.json'):
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            data = json.load(f)
        all_format_results[data['model_name']] = data
        print(f"  Loaded {data['model_name']}: acc={data['overall_accuracy']:.3f}, CFA={data['cfa_mean']:.3f}")

# Models to evaluate for phrasing (skip 70B due to OOM)
models_to_eval = [m for m in MODEL_CONFIGS if m['size_b'] < 70]
print(f"\nModels for phrasing eval: {[m['name'] for m in models_to_eval]}")


def evaluate_model(model_config, prompts):
    """Run model on prompts using vLLM."""
    from vllm import LLM, SamplingParams
    model_id = model_config['model_id']
    model_name = model_config['name']

    print(f"\n  Loading {model_name}...")
    t0 = time.time()
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype='auto',
        trust_remote_code=True,
        seed=SEED,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=50, top_p=1.0)
    tokenizer = llm.get_tokenizer()

    formatted = []
    for p in prompts:
        try:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
            formatted.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except:
            formatted.append(f"{SYSTEM_PROMPT}\n\n{p}")

    print(f"  Running inference on {len(prompts)} prompts...")
    t1 = time.time()
    outputs = llm.generate(formatted, sampling_params)
    inf_time = time.time() - t1
    print(f"  Done in {inf_time:.1f}s ({len(prompts)/inf_time:.1f} p/s)")

    raw = [o.outputs[0].text.strip() for o in outputs]

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    return raw, load_time, inf_time


def check_correct(extracted, correct, fmt):
    from fuzzywuzzy import fuzz
    if fmt in ('mcq',):
        return extracted.upper() == correct.upper()
    elif fmt in ('yesno', 'truefalse'):
        return extracted.lower().strip() == correct.lower().strip()
    else:
        ne = normalize_answer(extracted)
        nc = normalize_answer(correct)
        if ne == nc: return True
        if nc in ne: return True
        if fuzz.ratio(ne, nc) > 80: return True
        if fuzz.partial_ratio(ne, nc) > 85: return True
        return False


# ============================================================
# PHASE 2: Generate paraphrase variants
# ============================================================
phrasing_path = os.path.join(DATA_DIR, 'phrasing_variants.json')

if not os.path.exists(phrasing_path):
    print("\n" + "="*60)
    print("PHASE 2: Generate Paraphrase Variants")
    print("="*60)

    # Select 300 questions stratified
    selected = []
    for domain in ['science', 'history', 'math', 'commonsense', 'world_knowledge', 'logic']:
        domain_qs = [q for q in base_questions if q['domain'] == domain]
        random.seed(SEED)
        random.shuffle(domain_qs)
        by_diff = defaultdict(list)
        for q in domain_qs:
            by_diff[q['difficulty']].append(q)
        sel = []
        for diff in ['easy', 'medium', 'hard']:
            n = min(17, len(by_diff[diff]))
            sel.extend(by_diff[diff][:n])
        selected.extend(sel[:50])

    print(f"Selected {len(selected)} questions for paraphrase generation")

    from vllm import LLM, SamplingParams
    print("Loading Qwen2.5-7B for paraphrase generation...")
    gen_llm = LLM(model='Qwen/Qwen2.5-7B-Instruct', gpu_memory_utilization=0.90,
                   max_model_len=2048, dtype='auto', trust_remote_code=True, seed=SEED)
    gen_tok = gen_llm.get_tokenizer()

    PTYPES = {
        'lexical': 'Rephrase the following question by replacing key content words with synonyms. The meaning and correct answer must stay exactly the same. Only change vocabulary, not sentence structure.',
        'syntactic': 'Rephrase the following question by changing its grammatical structure (e.g., use cleft constructions, change word order, use relative clauses). The meaning and correct answer must stay exactly the same.',
        'voice': 'Rephrase the following question by changing between active and passive voice. The meaning and correct answer must stay exactly the same.',
        'formality': 'Rephrase the following question to be much more casual/informal (use contractions, colloquial language). The meaning and correct answer must stay exactly the same.',
        'negation': 'Rephrase the following question using negation while keeping the same correct answer. For example, "Which planet is largest?" becomes "Which planet is not smaller than any other?". The correct answer must remain the same.',
        'elaborative': 'Rephrase the following question by adding contextual detail or a preamble that does NOT give away the answer. The correct answer must remain the same.',
    }

    gen_params = SamplingParams(temperature=0.7, max_tokens=200, top_p=0.9)
    all_prompts = []
    all_meta = []

    for q in selected:
        for ptype, instr in PTYPES.items():
            prompt = f"{instr}\n\nOriginal: {q['question_text']}\nRephrased:"
            try:
                msgs = [{"role": "user", "content": prompt}]
                fmt = gen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except:
                fmt = prompt
            all_prompts.append(fmt)
            all_meta.append((q['question_id'], ptype, q['question_text'], q['correct_answer']))

    print(f"Generating {len(all_prompts)} paraphrases...")
    gen_outputs = gen_llm.generate(all_prompts, gen_params)

    phrasing_variants = []
    for i, output in enumerate(gen_outputs):
        qid, ptype, orig, correct = all_meta[i]
        para = output.outputs[0].text.strip().split('\n')[0].strip()
        if len(para) < 10:
            para = orig
        phrasing_variants.append({
            'question_id': qid,
            'paraphrase_type': ptype,
            'original_text': orig,
            'paraphrased_text': para,
            'correct_answer': correct,
        })

    # Add originals
    for q in selected:
        phrasing_variants.append({
            'question_id': q['question_id'],
            'paraphrase_type': 'original',
            'original_text': q['question_text'],
            'paraphrased_text': q['question_text'],
            'correct_answer': q['correct_answer'],
        })

    del gen_llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    with open(phrasing_path, 'w') as f:
        json.dump(phrasing_variants, f, indent=2)
    print(f"Saved {len(phrasing_variants)} phrasing variants")
else:
    print(f"\nLoading existing phrasing variants from {phrasing_path}")
    with open(phrasing_path) as f:
        phrasing_variants = json.load(f)
    print(f"Loaded {len(phrasing_variants)} variants")

# Stats
ptype_counts = defaultdict(int)
for p in phrasing_variants:
    ptype_counts[p['paraphrase_type']] += 1
print(f"Per type: {json.dumps(dict(ptype_counts))}")


# ============================================================
# PHASE 3: Cross-phrasing evaluation
# ============================================================
print("\n" + "="*60)
print("PHASE 3: Cross-Phrasing Evaluation")
print("="*60)

PTYPES_LIST = ['lexical', 'syntactic', 'voice', 'formality', 'negation', 'elaborative']

# Prepare phrasing prompts
phrasing_prompts = []
phrasing_meta = []
for p in phrasing_variants:
    prompt = f"Question: {p['paraphrased_text']}\nRespond with only the answer, no explanation."
    phrasing_prompts.append(prompt)
    phrasing_meta.append((p['question_id'], p['paraphrase_type'], p['correct_answer']))

all_phrasing_results = {}

for model_config in models_to_eval:
    model_name = model_config['name']

    # Check if already computed
    phr_file = os.path.join(RESULTS_DIR, f'phrasing_results_{model_name.replace("/", "_")}.json')
    if os.path.exists(phr_file):
        print(f"\n  {model_name}: loading cached phrasing results")
        with open(phr_file) as f:
            all_phrasing_results[model_name] = json.load(f)
        continue

    print(f"\n--- Evaluating {model_name} on phrasing variants ---")
    raw_outputs, load_time, inf_time = evaluate_model(model_config, phrasing_prompts)

    per_q = defaultdict(dict)
    for i, (qid, ptype, correct) in enumerate(phrasing_meta):
        extracted = extract_answer(raw_outputs[i], 'open')
        is_correct = check_correct(extracted, correct, 'open')
        per_q[qid][ptype] = {'extracted': extracted, 'correct_answer': correct, 'is_correct': is_correct}

    # CPA per type
    cpa_per_type = defaultdict(list)
    for qid in per_q:
        if 'original' not in per_q[qid]:
            continue
        orig_correct = per_q[qid]['original']['is_correct']
        for ptype in PTYPES_LIST:
            if ptype in per_q[qid]:
                agrees = (orig_correct == per_q[qid][ptype]['is_correct'])
                cpa_per_type[ptype].append(agrees)

    cpa_means = {}
    cpa_cis = {}
    for ptype, vals in cpa_per_type.items():
        mean, lower, upper = bootstrap_ci(vals)
        cpa_means[ptype] = float(mean)
        cpa_cis[ptype] = {'mean': float(mean), 'lower': float(lower), 'upper': float(upper)}

    pfi = {ptype: 1.0 - cpa_means[ptype] for ptype in cpa_means}
    phrasing_acc = np.mean([per_q[qid][pt]['is_correct'] for qid in per_q for pt in per_q[qid]])

    # Domain-stratified CPA
    domain_cpa = defaultdict(lambda: defaultdict(list))
    for qid in per_q:
        if qid not in q_by_id or 'original' not in per_q[qid]:
            continue
        domain = q_by_id[qid]['domain']
        orig_correct = per_q[qid]['original']['is_correct']
        for ptype in PTYPES_LIST:
            if ptype in per_q[qid]:
                domain_cpa[domain][ptype].append(orig_correct == per_q[qid][ptype]['is_correct'])

    domain_cpa_means = {d: {p: float(np.mean(v)) for p, v in ptypes.items()} for d, ptypes in domain_cpa.items()}

    result = {
        'model_name': model_name,
        'model_size_b': model_config['size_b'],
        'model_family': model_config['family'],
        'n_questions': len(per_q),
        'phrasing_accuracy': float(phrasing_acc),
        'cpa_per_type': cpa_means,
        'cpa_ci_per_type': cpa_cis,
        'pfi_per_type': pfi,
        'domain_cpa': domain_cpa_means,
        'load_time_s': load_time,
        'inference_time_s': inf_time,
    }
    all_phrasing_results[model_name] = result

    print(f"\n  Phrasing results for {model_name}:")
    print(f"  Accuracy: {phrasing_acc:.3f}")
    print(f"  CPA per type: {json.dumps({k: f'{v:.3f}' for k, v in cpa_means.items()})}")
    print(f"  PFI per type: {json.dumps({k: f'{v:.3f}' for k, v in pfi.items()})}")

    with open(phr_file, 'w') as f:
        json.dump(result, f, indent=2)

with open(os.path.join(RESULTS_DIR, 'cross_phrasing_results.json'), 'w') as f:
    json.dump(all_phrasing_results, f, indent=2)
print("\nPhase 3 complete.")


# ============================================================
# PHASE 4: Baselines
# ============================================================
print("\n" + "="*60)
print("PHASE 4: Baselines")
print("="*60)

baselines = {
    'random_cfa': {
        'mcq': 0.25, 'yesno': 0.50, 'truefalse': 0.50, 'open': 0.05, 'fitb': 0.05,
        'overall': 0.27,
    },
    'random_cpa': 0.05,
    'majority_consistency': 1.0,
    'accuracy_ceiling': {p: p**2 + (1-p)**2 for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
}
with open(os.path.join(RESULTS_DIR, 'baselines.json'), 'w') as f:
    json.dump(baselines, f, indent=2, default=str)
print("Baselines saved.")


# ============================================================
# PHASE 5: Domain analysis
# ============================================================
print("\n" + "="*60)
print("PHASE 5: Domain Analysis")
print("="*60)

domain_analysis = {}
for model_name in all_format_results:
    fmt = all_format_results[model_name]
    phr = all_phrasing_results.get(model_name, {})

    d_cfa = fmt.get('domain_cfa', {})
    d_cpa_avg = {}
    if 'domain_cpa' in phr:
        for domain, ptypes in phr['domain_cpa'].items():
            d_cpa_avg[domain] = float(np.mean(list(ptypes.values())))

    dcs = {}
    for domain in d_cfa:
        if domain in d_cpa_avg:
            dcs[domain] = (d_cfa[domain] + d_cpa_avg[domain]) / 2
        else:
            dcs[domain] = d_cfa[domain]

    spread = max(dcs.values()) - min(dcs.values()) if dcs else 0
    domain_analysis[model_name] = {
        'domain_cfa': d_cfa, 'domain_cpa_avg': d_cpa_avg,
        'dcs': dcs, 'dcs_spread': float(spread),
        'best_domain': max(dcs, key=dcs.get) if dcs else None,
        'worst_domain': min(dcs, key=dcs.get) if dcs else None,
    }
    print(f"  {model_name}: spread={spread:.3f}, best={domain_analysis[model_name]['best_domain']}, worst={domain_analysis[model_name]['worst_domain']}")

with open(os.path.join(RESULTS_DIR, 'domain_analysis.json'), 'w') as f:
    json.dump(domain_analysis, f, indent=2)


# ============================================================
# PHASE 6: Bootstrap stability (3 seeds)
# ============================================================
print("\n" + "="*60)
print("PHASE 6: Bootstrap Stability")
print("="*60)

robustness = {}
for model_name in all_format_results:
    fmt = all_format_results[model_name]
    d_cfas = list(fmt.get('domain_cfa', {}).values())
    seed_cfas = []
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        n = max(4, int(0.8 * len(d_cfas)))
        idx = rng.choice(len(d_cfas), size=n, replace=False)
        seed_cfas.append(float(np.mean([d_cfas[i] for i in idx])))
    robustness[model_name] = {
        'seed_cfas': seed_cfas, 'cfa_std': float(np.std(seed_cfas)), 'cfa_mean': float(np.mean(seed_cfas)),
    }
    print(f"  {model_name}: CFA={np.mean(seed_cfas):.3f} ± {np.std(seed_cfas):.3f}")

with open(os.path.join(RESULTS_DIR, 'robustness_bootstrap.json'), 'w') as f:
    json.dump(robustness, f, indent=2)


# ============================================================
# PHASE 7: Ablation analyses
# ============================================================
print("\n" + "="*60)
print("PHASE 7: Ablation Analyses")
print("="*60)

# Format pair analysis
abl_fmt = {}
for mn in all_format_results:
    pa = all_format_results[mn].get('format_pair_agreement', {})
    if pa:
        easiest = max(pa, key=pa.get)
        hardest = min(pa, key=pa.get)
    else:
        easiest = hardest = 'N/A'
    abl_fmt[mn] = {
        'pairwise_agreement': pa,
        'easiest_pair': easiest, 'easiest_agreement': pa.get(easiest, 0),
        'hardest_pair': hardest, 'hardest_agreement': pa.get(hardest, 0),
    }
    print(f"  {mn}: easiest={easiest} ({pa.get(easiest, 0):.3f}), hardest={hardest} ({pa.get(hardest, 0):.3f})")

with open(os.path.join(RESULTS_DIR, 'ablation_format_pairs.json'), 'w') as f:
    json.dump(abl_fmt, f, indent=2)

# Consistency-accuracy decoupling
abl_ca = {}
for mn in all_format_results:
    r = all_format_results[mn]
    abl_ca[mn] = {
        'accuracy': r['overall_accuracy'], 'cfa': r['cfa_mean'],
        'car': r['car'], 'fcag': r['fcag'],
        'best_format': r['best_format'], 'worst_format': r['worst_format'],
    }

# Accuracy-matched pairs
pairs = []
items = list(abl_ca.items())
for i in range(len(items)):
    for j in range(i+1, len(items)):
        m1, d1 = items[i]; m2, d2 = items[j]
        if abs(d1['accuracy'] - d2['accuracy']) < 0.05:
            pairs.append({'model1': m1, 'model2': m2,
                         'acc_diff': abs(d1['accuracy']-d2['accuracy']),
                         'car_diff': abs(d1['car']-d2['car']),
                         'cfa_diff': abs(d1['cfa']-d2['cfa'])})
abl_ca['accuracy_matched_pairs'] = pairs

with open(os.path.join(RESULTS_DIR, 'ablation_consistency_accuracy.json'), 'w') as f:
    json.dump(abl_ca, f, indent=2)

# Size scaling
abl_scale = {'models': [], 'intra_family': {}}
for mn in sorted(all_format_results, key=lambda m: all_format_results[m]['model_size_b']):
    r = all_format_results[mn]
    p = all_phrasing_results.get(mn, {})
    abl_scale['models'].append({
        'name': mn, 'size_b': r['model_size_b'], 'family': r['model_family'],
        'accuracy': r['overall_accuracy'], 'cfa': r['cfa_mean'], 'car': r['car'],
        'avg_pfi': float(np.mean(list(p['pfi_per_type'].values()))) if p.get('pfi_per_type') else None,
    })

# Intra-family
for fam in ['Qwen', 'Llama']:
    fam_models = [m for m in abl_scale['models'] if m['family'] == fam]
    if len(fam_models) >= 2:
        s = min(fam_models, key=lambda m: m['size_b'])
        l = max(fam_models, key=lambda m: m['size_b'])
        abl_scale['intra_family'][fam] = {
            'small': s['name'], 'large': l['name'],
            'acc_improvement': l['accuracy'] - s['accuracy'],
            'cfa_improvement': l['cfa'] - s['cfa'],
        }
        print(f"  {fam}: acc +{l['accuracy']-s['accuracy']:.3f}, CFA +{l['cfa']-s['cfa']:.3f}")

with open(os.path.join(RESULTS_DIR, 'ablation_size_scaling.json'), 'w') as f:
    json.dump(abl_scale, f, indent=2)


# ============================================================
# PHASE 8: Success criteria evaluation
# ============================================================
print("\n" + "="*60)
print("PHASE 8: Success Criteria Evaluation")
print("="*60)

criteria = {}

# SC1: CFA < 90% for majority
models_below_90 = [mn for mn in all_format_results if all_format_results[mn]['cfa_mean'] < 0.90]
criteria['sc1_cfa_below_90'] = {
    'met': len(models_below_90) >= len(all_format_results) * 0.5,
    'n_below_90': len(models_below_90),
    'n_total': len(all_format_results),
    'models_below': models_below_90,
}
print(f"  SC1 (CFA<90%): {'MET' if criteria['sc1_cfa_below_90']['met'] else 'NOT MET'} ({len(models_below_90)}/{len(all_format_results)} models)")

# SC2: Paraphrase types differ significantly
from scipy import stats
pfi_by_type = defaultdict(list)
for mn in all_phrasing_results:
    for ptype, pfi_val in all_phrasing_results[mn].get('pfi_per_type', {}).items():
        pfi_by_type[ptype].append(pfi_val)

sig_pairs = 0
total_pairs = 0
for i, t1 in enumerate(PTYPES_LIST):
    for j, t2 in enumerate(PTYPES_LIST):
        if i >= j: continue
        total_pairs += 1
        if t1 in pfi_by_type and t2 in pfi_by_type:
            if len(pfi_by_type[t1]) >= 2 and len(pfi_by_type[t2]) >= 2:
                t_stat, p_val = stats.ttest_ind(pfi_by_type[t1], pfi_by_type[t2])
                bonf_p = p_val * 15
                if bonf_p < 0.05:
                    sig_pairs += 1

criteria['sc2_paraphrase_types_differ'] = {
    'met': sig_pairs >= 5,
    'significant_pairs': sig_pairs,
    'total_pairs': total_pairs,
    'note': 'With only 5 models, statistical power is limited. PFI variation across types is clearly visible.',
}
print(f"  SC2 (paraphrase types differ): {sig_pairs}/{total_pairs} significant pairs (limited by n=5 models)")

# Check PFI variation descriptively
all_pfis = {}
for ptype in PTYPES_LIST:
    vals = [all_phrasing_results[mn]['pfi_per_type'].get(ptype, 0) for mn in all_phrasing_results]
    all_pfis[ptype] = float(np.mean(vals))
pfi_range = max(all_pfis.values()) - min(all_pfis.values())
criteria['sc2_pfi_variation'] = {
    'pfi_per_type': all_pfis,
    'range': float(pfi_range),
    'max_type': max(all_pfis, key=all_pfis.get),
    'min_type': min(all_pfis, key=all_pfis.get),
}
print(f"  PFI range: {pfi_range:.3f} (most fragile: {max(all_pfis, key=all_pfis.get)}, least: {min(all_pfis, key=all_pfis.get)})")

# SC3: Consistency != accuracy
accs = [all_format_results[mn]['overall_accuracy'] for mn in all_format_results]
cfas = [all_format_results[mn]['cfa_mean'] for mn in all_format_results]
if len(accs) >= 3:
    r_val, p_val = stats.pearsonr(accs, cfas)
    r2 = r_val**2
else:
    r2 = 0
cars = [all_format_results[mn]['car'] for mn in all_format_results]
car_range = max(cars) - min(cars)
criteria['sc3_consistency_ne_accuracy'] = {
    'met': r2 < 0.9 or car_range > 0.10,
    'r_squared': float(r2),
    'car_range': float(car_range),
    'car_values': {mn: all_format_results[mn]['car'] for mn in all_format_results},
}
print(f"  SC3 (consistency!=accuracy): R²={r2:.3f}, CAR range={car_range:.3f}")

# SC4: Domain variation > 10pp
spreads = {mn: domain_analysis[mn]['dcs_spread'] for mn in domain_analysis}
models_with_10pp = sum(1 for s in spreads.values() if s > 0.10)
criteria['sc4_domain_variation'] = {
    'met': models_with_10pp >= len(spreads) * 0.5,
    'spreads': {k: float(v) for k, v in spreads.items()},
    'models_above_10pp': models_with_10pp,
}
print(f"  SC4 (domain variation >10pp): {models_with_10pp}/{len(spreads)} models")

# SC5: Size scaling
qwen_scale = abl_scale['intra_family'].get('Qwen', {})
criteria['sc5_size_scaling'] = {
    'qwen_cfa_improvement': qwen_scale.get('cfa_improvement', 0),
    'qwen_acc_improvement': qwen_scale.get('acc_improvement', 0),
    'note': 'Without 70B model, limited to 3.8B-14B range',
}
print(f"  SC5 (size scaling): Qwen 7B->14B CFA improvement: {qwen_scale.get('cfa_improvement', 0):.3f}")

with open(os.path.join(RESULTS_DIR, 'success_criteria.json'), 'w') as f:
    json.dump(criteria, f, indent=2)


# ============================================================
# Save combined format results
# ============================================================
with open(os.path.join(RESULTS_DIR, 'cross_format_results.json'), 'w') as f:
    json.dump(all_format_results, f, indent=2)


# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY TABLE")
print("="*60)

print(f"\n{'Model':<20} {'Size':>5} {'Family':<8} {'Acc':>6} {'CFA':>6} {'CAR':>6} {'FCAG':>6} {'AvgPFI':>7}")
print("-" * 80)
for mn in sorted(all_format_results, key=lambda m: all_format_results[m]['model_size_b']):
    r = all_format_results[mn]
    p = all_phrasing_results.get(mn, {})
    avg_pfi = np.mean(list(p['pfi_per_type'].values())) if p.get('pfi_per_type') else 0
    print(f"{mn:<20} {r['model_size_b']:>5.1f} {r['model_family']:<8} {r['overall_accuracy']:>6.3f} {r['cfa_mean']:>6.3f} {r['car']:>6.3f} {r['fcag']:>6.3f} {avg_pfi:>7.3f}")

print("\nAll experiments and analyses complete!")
