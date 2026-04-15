"""Measure cross-model feature universality via activation correlation."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
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
K = 64
N_EVAL_SEQS = 2000  # sequences for cross-model eval
SEQ_LEN = 128


def load_sae(model_key, seed):
    """Load trained SAE."""
    path = os.path.join(SAE_DIR, f"{model_key}_seed{seed}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    sae = TopKSAE(cfg["d_model"], cfg["n_features"], cfg["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    return sae.eval().to(DEVICE)


def collect_sae_activations_on_text(model, sae, token_ids, layer_idx, batch_size=32, seq_len=SEQ_LEN):
    """Run text through model+SAE and collect feature activations."""
    model.eval()
    sae.eval()
    n_seqs = len(token_ids) // seq_len
    token_ids = token_ids[:n_seqs * seq_len].reshape(n_seqs, seq_len)

    all_acts = []
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1]
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            sae_acts = sae.get_activations(flat_hidden)
            all_acts.append(sae_acts.cpu())

    return torch.cat(all_acts, dim=0)


def compute_cross_model_correlation(acts1, acts2, chunk_size=512, top_k=1):
    """Compute per-feature cross-model alignment score.
    For each feature in model 1, find its top-k correlated feature in model 2.
    """
    n1, n2 = acts1.shape[1], acts2.shape[1]
    n_samples = min(acts1.shape[0], acts2.shape[0])
    acts1 = acts1[:n_samples].float()
    acts2 = acts2[:n_samples].float()

    # Normalize for correlation computation
    acts1_centered = acts1 - acts1.mean(dim=0, keepdim=True)
    acts2_centered = acts2 - acts2.mean(dim=0, keepdim=True)
    acts1_norm = acts1_centered / (acts1_centered.norm(dim=0, keepdim=True) + 1e-8)
    acts2_norm = acts2_centered / (acts2_centered.norm(dim=0, keepdim=True) + 1e-8)

    # Compute correlation matrix in chunks
    alignment_scores = np.zeros(n1)
    for i in tqdm(range(0, n1, chunk_size), desc="Cross-model correlation"):
        chunk = acts1_norm[:, i:i+chunk_size]  # [n_samples, chunk]
        corr = (chunk.T @ acts2_norm).numpy()  # [chunk, n2]
        # Top-1 absolute correlation
        alignment_scores[i:i+chunk_size] = np.abs(corr).max(axis=1)

    return alignment_scores


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load shared evaluation text
    print("Loading evaluation text...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    # Use validation set (different from training) for cross-model eval

    model_keys = list(MODELS.keys())
    model_pairs = [(model_keys[i], model_keys[j])
                   for i in range(len(model_keys))
                   for j in range(i+1, len(model_keys))]

    # Collect activations for each model (seed=42)
    model_acts = {}
    for model_key, model_cfg in MODELS.items():
        print(f"\nCollecting activations for {model_key}...")
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize eval text
        all_ids = []
        total = 0
        for ex in ds:
            text = ex["text"].strip()
            if len(text) < 50:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"][0]
            all_ids.append(ids)
            total += len(ids)
            if total >= N_EVAL_SEQS * SEQ_LEN:
                break
        token_ids = torch.cat(all_ids)[:N_EVAL_SEQS * SEQ_LEN]

        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"]).to(DEVICE).eval()
        sae = load_sae(model_key, 42)

        acts = collect_sae_activations_on_text(
            model, sae, token_ids, model_cfg["layer"]
        )
        model_acts[model_key] = acts
        print(f"  Activations shape: {acts.shape}")

        del model, sae
        gc.collect()
        torch.cuda.empty_cache()

    # Compute cross-model alignment for each pair
    alignment_scores = {}
    for m1, m2 in model_pairs:
        print(f"\nComputing alignment: {m1} <-> {m2}")
        # Align by position (both use same text, different tokenizers -> use min length)
        n_common = min(model_acts[m1].shape[0], model_acts[m2].shape[0])
        acts1 = model_acts[m1][:n_common]
        acts2 = model_acts[m2][:n_common]

        scores_1to2 = compute_cross_model_correlation(acts1, acts2)
        scores_2to1 = compute_cross_model_correlation(acts2, acts1)

        alignment_scores[f"{m1}_to_{m2}"] = scores_1to2
        alignment_scores[f"{m2}_to_{m1}"] = scores_2to1

        print(f"  {m1}→{m2}: mean={scores_1to2.mean():.4f}, median={np.median(scores_1to2):.4f}")
        print(f"  {m2}→{m1}: mean={scores_2to1.mean():.4f}, median={np.median(scores_2to1):.4f}")

    # For each model, compute mean cross-model alignment across all partners
    for model_key in model_keys:
        keys = [k for k in alignment_scores if k.startswith(model_key)]
        if keys:
            scores = np.mean([alignment_scores[k] for k in keys], axis=0)
        else:
            scores = np.zeros(N_FEATURES)
        alignment_scores[f"{model_key}_mean"] = scores

    # Test hypothesis: convergence score predicts universality
    results_all = {}
    for model_key in model_keys:
        conv_path = os.path.join(RESULTS_DIR, f"convergence_scores_{model_key}.json")
        if not os.path.exists(conv_path):
            print(f"  Skipping {model_key} - no convergence scores found")
            continue

        with open(conv_path) as f:
            conv_data = json.load(f)

        conv_scores = np.array(conv_data["convergence_score_combined"])
        univ_scores = alignment_scores[f"{model_key}_mean"]

        rho, p = spearmanr(conv_scores, univ_scores)
        print(f"\n  {model_key}: Convergence ↔ Universality: ρ={rho:.4f}, p={p:.2e}")

        # Also test decoder-only and activation-only convergence
        rho_dec, p_dec = spearmanr(np.array(conv_data["convergence_score_decoder"]), univ_scores)
        rho_act, p_act = spearmanr(np.array(conv_data["convergence_score_activation"]), univ_scores)

        # Core vs peripheral universality
        core_mask = np.array(conv_data["core_mask_08_75"])
        core_univ = univ_scores[core_mask].mean()
        periph_univ = univ_scores[~core_mask].mean()

        results_all[model_key] = {
            "spearman_rho_combined": float(rho),
            "spearman_p_combined": float(p),
            "spearman_rho_decoder": float(rho_dec),
            "spearman_p_decoder": float(p_dec),
            "spearman_rho_activation": float(rho_act),
            "spearman_p_activation": float(p_act),
            "core_mean_universality": float(core_univ),
            "peripheral_mean_universality": float(periph_univ),
            "universality_scores": univ_scores.tolist(),
        }

    # Save results
    for model_key in model_keys:
        if model_key in results_all:
            path = os.path.join(RESULTS_DIR, f"cross_model_alignment_{model_key}.json")
            with open(path, "w") as f:
                json.dump(results_all[model_key], f)

    summary = {k: {kk: vv for kk, vv in v.items() if kk != "universality_scores"}
               for k, v in results_all.items()}
    with open(os.path.join(RESULTS_DIR, "cross_model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\nCross-model alignment summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
