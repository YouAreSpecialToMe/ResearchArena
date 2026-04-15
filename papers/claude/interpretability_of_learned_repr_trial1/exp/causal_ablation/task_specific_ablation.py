"""Task-specific causal evaluations: IOI, Greater-Than, Subject-Verb Agreement.

For each task, we ablate core vs peripheral features and measure task-specific
accuracy changes, testing whether the aggregate KL finding (peripheral > core
importance) holds for specific circuits.
"""
import sys, os, json, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from shared.sae import TopKSAE
import gc

DEVICE = "cuda"
SAE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sae_training')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

MODELS = {
    "gpt2_small": {"name": "gpt2", "layer": 6, "d_model": 768},
    "pythia_160m": {"name": "EleutherAI/pythia-160m", "layer": 6, "d_model": 768},
    "pythia_410m": {"name": "EleutherAI/pythia-410m", "layer": 12, "d_model": 1024},
}
N_FEATURES = 16384


def load_sae(model_key, seed):
    path = os.path.join(SAE_DIR, f"{model_key}_seed{seed}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    sae = TopKSAE(cfg["d_model"], cfg["n_features"], cfg["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    return sae.eval().to(DEVICE)


def get_hook_name(model_key, layer_idx):
    if model_key == "gpt2_small":
        return f"transformer.h.{layer_idx}"
    else:
        return f"gpt_neox.layers.{layer_idx}"


# --- IOI Task ---
NAMES = ["Mary", "John", "Alice", "Bob", "Sarah", "James", "Emma", "David",
         "Lisa", "Michael", "Kate", "Tom", "Anna", "Chris", "Laura", "Mark"]
PLACES = ["store", "park", "office", "school", "library", "restaurant",
          "hospital", "museum", "theater", "market"]
OBJECTS = ["drink", "book", "letter", "gift", "key", "phone", "bag", "hat"]

def generate_ioi_examples(n=500):
    """Generate IOI prompts: 'When {A} and {B} went to the {place}, {B} gave a {obj} to'
    Answer should be A (the indirect object)."""
    examples = []
    rng = random.Random(42)
    for _ in range(n):
        names = rng.sample(NAMES, 2)
        a, b = names[0], names[1]
        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)
        prompt = f"When {a} and {b} went to the {place}, {b} gave a {obj} to"
        examples.append({"prompt": prompt, "correct": a, "incorrect": b})
    return examples


# --- Greater-Than Task ---
def generate_greater_than_examples(n=500):
    """Generate greater-than prompts: 'The war lasted from {year} to {year+}'
    The model should predict a year > start_year."""
    examples = []
    rng = random.Random(42)
    for _ in range(n):
        century = rng.choice([17, 18, 19])
        start = rng.randint(0, 80)
        start_year = century * 100 + start
        prompt = f"The war lasted from {start_year} to {century}"
        # Valid completions are two-digit numbers > start
        valid_ends = [str(d).zfill(2) for d in range(start + 1, 100)]
        invalid_ends = [str(d).zfill(2) for d in range(0, start + 1)]
        examples.append({
            "prompt": prompt,
            "valid_ends": valid_ends,
            "invalid_ends": invalid_ends,
            "start": start,
        })
    return examples


# --- Subject-Verb Agreement ---
def generate_sva_examples(n=500):
    """Generate SVA prompts with attractors.
    'The key(s) to the cabinet(s) is/are'"""
    singular_nouns = ["key", "book", "letter", "cat", "dog", "boy", "girl", "man", "woman", "child"]
    plural_nouns = ["keys", "books", "letters", "cats", "dogs", "boys", "girls", "men", "women", "children"]
    singular_verbs = [" is", " was", " has"]
    plural_verbs = [" are", " were", " have"]
    prepositions = ["to the", "near the", "beside the", "behind the", "above the"]

    examples = []
    rng = random.Random(42)
    for _ in range(n):
        # Subject is singular, attractor is plural (or vice versa)
        if rng.random() < 0.5:
            # Singular subject, plural attractor
            subj_idx = rng.randint(0, len(singular_nouns) - 1)
            attr_idx = rng.randint(0, len(plural_nouns) - 1)
            subject = singular_nouns[subj_idx]
            attractor = plural_nouns[attr_idx]
            prep = rng.choice(prepositions)
            prompt = f"The {subject} {prep} {attractor}"
            correct_verbs = singular_verbs
            incorrect_verbs = plural_verbs
        else:
            # Plural subject, singular attractor
            subj_idx = rng.randint(0, len(plural_nouns) - 1)
            attr_idx = rng.randint(0, len(singular_nouns) - 1)
            subject = plural_nouns[subj_idx]
            attractor = singular_nouns[attr_idx]
            prep = rng.choice(prepositions)
            prompt = f"The {subject} {prep} {attractor}"
            correct_verbs = plural_verbs
            incorrect_verbs = singular_verbs

        examples.append({
            "prompt": prompt,
            "correct_verbs": correct_verbs,
            "incorrect_verbs": incorrect_verbs,
        })
    return examples


def compute_task_feature_importance(model, sae, tokenizer, examples, task_type,
                                     model_key, layer_idx, feature_indices):
    """Compute per-feature importance for a specific task via zero-ablation.

    For efficiency, we evaluate a subset of features (core + sample of peripheral).
    Returns dict mapping feature_idx -> task_metric_change.
    """
    model.eval()
    sae.eval()
    hook_name = get_hook_name(model_key, layer_idx)
    W_dec = sae.W_dec.data

    # Get baseline task performance
    baseline_score = evaluate_task(model, tokenizer, examples, task_type)
    print(f"    Baseline {task_type} score: {baseline_score:.4f}")

    feature_importance = {}

    # Batch features for efficiency
    for f_idx in tqdm(feature_indices, desc=f"    Ablating features ({task_type})"):
        # For each example, run model with this feature ablated
        score = evaluate_task_with_ablation(
            model, sae, tokenizer, examples, task_type,
            model_key, layer_idx, f_idx
        )
        feature_importance[int(f_idx)] = baseline_score - score  # positive = feature helps task

    return feature_importance, baseline_score


def evaluate_task(model, tokenizer, examples, task_type):
    """Evaluate baseline task performance."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for ex in examples:
            input_ids = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].to(DEVICE)
            logits = model(input_ids).logits[0, -1]  # last token logits

            if task_type == "ioi":
                correct_id = tokenizer.encode(" " + ex["correct"])[0]
                incorrect_id = tokenizer.encode(" " + ex["incorrect"])[0]
                if logits[correct_id] > logits[incorrect_id]:
                    correct += 1
                total += 1

            elif task_type == "greater_than":
                # Check if model assigns higher prob to valid years
                valid_prob = 0
                invalid_prob = 0
                for ve in ex["valid_ends"][:10]:  # sample for speed
                    tok = tokenizer.encode(ve)
                    if len(tok) > 0:
                        valid_prob += logits[tok[0]].exp().item()
                for ie in ex["invalid_ends"][:10]:
                    tok = tokenizer.encode(ie)
                    if len(tok) > 0:
                        invalid_prob += logits[tok[0]].exp().item()
                if valid_prob > invalid_prob:
                    correct += 1
                total += 1

            elif task_type == "sva":
                correct_prob = 0
                incorrect_prob = 0
                for cv in ex["correct_verbs"]:
                    tok = tokenizer.encode(cv)
                    if len(tok) > 0:
                        correct_prob += logits[tok[0]].exp().item()
                for iv in ex["incorrect_verbs"]:
                    tok = tokenizer.encode(iv)
                    if len(tok) > 0:
                        incorrect_prob += logits[tok[0]].exp().item()
                if correct_prob > incorrect_prob:
                    correct += 1
                total += 1

    return correct / max(total, 1)


def evaluate_task_with_ablation(model, sae, tokenizer, examples, task_type,
                                  model_key, layer_idx, feature_idx):
    """Evaluate task with a single feature ablated."""
    model.eval()
    sae.eval()
    hook_name = get_hook_name(model_key, layer_idx)
    W_dec = sae.W_dec.data
    correct = 0
    total = 0

    # Sample examples for speed
    sample_examples = examples[:100]

    for ex in sample_examples:
        input_ids = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].to(DEVICE)

        # Get original hidden states and SAE activations
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, d_model]
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            sae_acts = sae.get_activations(flat_hidden)

            # Compute delta from zeroing this feature
            delta = sae_acts[:, feature_idx:feature_idx+1] * W_dec[feature_idx:feature_idx+1]
            delta_reshaped = delta.reshape(hidden.shape)

        # Hook to subtract this feature's contribution
        def make_hook(d):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (output[0] - d,) + output[1:]
                return output - d
            return hook_fn

        target_module = dict(model.named_modules())[hook_name]
        handle = target_module.register_forward_hook(make_hook(delta_reshaped))

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]

        handle.remove()

        if task_type == "ioi":
            correct_id = tokenizer.encode(" " + ex["correct"])[0]
            incorrect_id = tokenizer.encode(" " + ex["incorrect"])[0]
            if logits[correct_id] > logits[incorrect_id]:
                correct += 1
            total += 1
        elif task_type == "greater_than":
            valid_prob = 0
            invalid_prob = 0
            for ve in ex["valid_ends"][:10]:
                tok = tokenizer.encode(ve)
                if len(tok) > 0:
                    valid_prob += logits[tok[0]].exp().item()
            for ie in ex["invalid_ends"][:10]:
                tok = tokenizer.encode(ie)
                if len(tok) > 0:
                    invalid_prob += logits[tok[0]].exp().item()
            if valid_prob > invalid_prob:
                correct += 1
            total += 1
        elif task_type == "sva":
            correct_prob = 0
            incorrect_prob = 0
            for cv in ex["correct_verbs"]:
                tok = tokenizer.encode(cv)
                if len(tok) > 0:
                    correct_prob += logits[tok[0]].exp().item()
            for iv in ex["incorrect_verbs"]:
                tok = tokenizer.encode(iv)
                if len(tok) > 0:
                    incorrect_prob += logits[tok[0]].exp().item()
            if correct_prob > incorrect_prob:
                correct += 1
            total += 1

    return correct / max(total, 1)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate task datasets
    ioi_examples = generate_ioi_examples(500)
    gt_examples = generate_greater_than_examples(500)
    sva_examples = generate_sva_examples(500)

    all_task_results = {}

    for model_key, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Task-specific ablation: {model_key}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"]).to(DEVICE).eval()
        sae = load_sae(model_key, 42)

        # Load convergence scores and core mask
        conv_path = os.path.join(RESULTS_DIR, f"convergence_scores_{model_key}.json")
        with open(conv_path) as f:
            conv_data = json.load(f)
        conv_scores = np.array(conv_data["convergence_score_combined"])
        core_mask = np.array(conv_data["core_mask_08_75"])

        # Load activations to identify active features
        act_path = os.path.join(SAE_DIR, f"activations_{model_key}_seed42.pt")
        sae_acts = torch.load(act_path, map_location="cpu", weights_only=True)
        act_freq = (sae_acts > 0).float().mean(dim=0).numpy()
        active_mask = act_freq > 0.001  # features that activate on >0.1% of tokens

        # Select features to evaluate: all active core + sample of active peripheral
        core_active = np.where(core_mask & active_mask)[0]
        periph_active = np.where(~core_mask & active_mask)[0]

        # Sample up to 200 from each group for tractability
        rng = np.random.RandomState(42)
        n_sample = min(150, len(core_active), len(periph_active))
        if len(core_active) > n_sample:
            core_sample = rng.choice(core_active, n_sample, replace=False)
        else:
            core_sample = core_active
        if len(periph_active) > n_sample:
            periph_sample = rng.choice(periph_active, n_sample, replace=False)
        else:
            periph_sample = periph_active

        all_features = np.concatenate([core_sample, periph_sample])
        is_core = np.array([True]*len(core_sample) + [False]*len(periph_sample))

        print(f"  Evaluating {len(core_sample)} core + {len(periph_sample)} peripheral features")

        model_results = {}

        for task_type, examples in [("ioi", ioi_examples), ("greater_than", gt_examples), ("sva", sva_examples)]:
            print(f"\n  Task: {task_type}")
            t0 = time.time()

            importance_dict, baseline = compute_task_feature_importance(
                model, sae, tokenizer, examples, task_type,
                model_key, model_cfg["layer"], all_features
            )

            elapsed = time.time() - t0
            print(f"    Took {elapsed/60:.1f} min")

            # Split into core vs peripheral
            core_imp = [importance_dict[int(f)] for f in core_sample if int(f) in importance_dict]
            periph_imp = [importance_dict[int(f)] for f in periph_sample if int(f) in importance_dict]

            core_imp = np.array(core_imp)
            periph_imp = np.array(periph_imp)

            d = cohens_d(core_imp, periph_imp)

            if len(core_imp) > 1 and len(periph_imp) > 1:
                try:
                    u_stat, u_p = mannwhitneyu(core_imp, periph_imp, alternative='two-sided')
                except:
                    u_stat, u_p = 0, 1.0
            else:
                u_stat, u_p = 0, 1.0

            # Spearman correlation with convergence score
            all_imp = np.array([importance_dict.get(int(f), 0) for f in all_features])
            all_conv = np.array([conv_scores[f] for f in all_features])
            all_freq = np.array([act_freq[f] for f in all_features])

            rho_conv, p_conv = spearmanr(all_conv, all_imp)
            rho_freq, p_freq = spearmanr(all_freq, all_imp)

            task_result = {
                "baseline_accuracy": float(baseline),
                "n_core_evaluated": len(core_imp),
                "n_periph_evaluated": len(periph_imp),
                "core_mean_importance": float(core_imp.mean()) if len(core_imp) > 0 else 0,
                "periph_mean_importance": float(periph_imp.mean()) if len(periph_imp) > 0 else 0,
                "cohens_d": float(d),
                "mann_whitney_p": float(u_p),
                "spearman_convergence": float(rho_conv),
                "spearman_convergence_p": float(p_conv),
                "spearman_frequency": float(rho_freq),
                "spearman_frequency_p": float(p_freq),
                "runtime_minutes": elapsed / 60,
            }
            model_results[task_type] = task_result
            print(f"    Core mean imp: {core_imp.mean():.4f}, Periph mean imp: {periph_imp.mean():.4f}")
            print(f"    Cohen's d: {d:.4f}, p={u_p:.2e}")
            print(f"    ρ(conv,imp)={rho_conv:.4f}, ρ(freq,imp)={rho_freq:.4f}")

        all_task_results[model_key] = model_results

        del model, sae, sae_acts
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    path = os.path.join(RESULTS_DIR, "task_specific_causal.json")
    with open(path, "w") as f:
        json.dump(all_task_results, f, indent=2)
    print(f"\nSaved task-specific results to {path}")
    print(json.dumps(all_task_results, indent=2))


if __name__ == "__main__":
    main()
