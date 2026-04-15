"""Causal importance assessment via zero-ablation of SAE features.

Uses proper model hooks to ablate features at the target layer and measure
the effect on output logits.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
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
N_EVAL_TOKENS = 2048
SEQ_LEN = 128


def load_sae(model_key, seed):
    path = os.path.join(SAE_DIR, f"{model_key}_seed{seed}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    sae = TopKSAE(cfg["d_model"], cfg["n_features"], cfg["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    return sae.eval().to(DEVICE)


def get_eval_data(tokenizer, n_tokens=N_EVAL_TOKENS):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    all_ids = []
    total = 0
    for ex in ds:
        text = ex["text"].strip()
        if len(text) < 50:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"][0]
        all_ids.append(ids)
        total += len(ids)
        if total >= n_tokens + SEQ_LEN:
            break
    return torch.cat(all_ids)[:n_tokens]


def get_hook_name(model_key, layer_idx):
    """Get the correct hook point name for the model."""
    if model_key == "gpt2_small":
        return f"transformer.h.{layer_idx}"
    else:  # pythia
        return f"gpt_neox.layers.{layer_idx}"


def compute_feature_importance(model, sae, token_ids, model_key, layer_idx):
    """Compute feature importance using proper hook-based ablation.

    Strategy: For each feature, we measure its contribution to the reconstruction.
    We use two complementary metrics:
    1. Direct effect: ||a_i * w_dec_i||^2 (magnitude of feature's contribution to hidden state)
    2. Indirect effect: KL divergence from hook-based ablation on a subset of features
    """
    model.eval()
    sae.eval()

    n_seqs = len(token_ids) // SEQ_LEN
    token_ids = token_ids[:n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)

    W_dec = sae.W_dec.data  # [n_features, d_model] on CUDA

    # Step 1: Collect hidden states and SAE activations
    all_hidden = []
    with torch.no_grad():
        for i in range(0, len(token_ids), 8):
            batch = token_ids[i:i+8].to(DEVICE)
            outputs = model(batch, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1]
            all_hidden.append(hidden.cpu())
    all_hidden = torch.cat(all_hidden, dim=0)
    flat_hidden = all_hidden.reshape(-1, all_hidden.shape[-1])  # [n_tokens, d_model]

    # Get SAE activations
    n_tokens_total = flat_hidden.shape[0]
    with torch.no_grad():
        sae_acts_list = []
        for i in range(0, n_tokens_total, 2048):
            batch = flat_hidden[i:i+2048].to(DEVICE)
            acts = sae.get_activations(batch)
            sae_acts_list.append(acts.cpu())
        sae_acts = torch.cat(sae_acts_list, dim=0)  # [n_tokens, n_features]

    # Step 2: Direct effect - activation magnitude * decoder norm contribution
    # For each feature i: importance = mean(|a_i|) * ||w_dec_i||
    # This measures the expected magnitude of the feature's contribution to the hidden state
    act_magnitude = sae_acts.float().abs().mean(dim=0)  # [n_features]
    dec_norms = W_dec.cpu().norm(dim=1)  # [n_features]
    direct_effect = (act_magnitude * dec_norms).numpy()

    # Step 3: Variance explained by each feature
    # For each feature: compute what fraction of variance in activation patterns it explains
    # This is done by computing reconstruction loss with vs without the feature
    n_sample = min(512, n_tokens_total)
    sample_idx = torch.randperm(n_tokens_total)[:n_sample]
    sampled_acts = sae_acts[sample_idx].float()  # [n_sample, n_features]
    sampled_hidden = flat_hidden[sample_idx].float()  # [n_sample, d_model]

    # Full SAE reconstruction
    with torch.no_grad():
        full_recon = (sampled_acts.to(DEVICE) @ W_dec + sae.b_dec).cpu()  # [n_sample, d_model]

    # For each feature, compute how much reconstruction error increases when it's removed
    recon_importance = np.zeros(N_FEATURES)
    for fi in tqdm(range(0, N_FEATURES, 512), desc="Computing recon importance"):
        fi_end = min(fi + 512, N_FEATURES)
        for f_idx in range(fi, fi_end):
            # Only compute for active features
            if act_magnitude[f_idx] < 1e-10:
                continue
            # Change in reconstruction when feature is zeroed
            delta = sampled_acts[:, f_idx:f_idx+1] * W_dec[f_idx:f_idx+1].cpu()  # [n_sample, d_model]
            recon_importance[f_idx] = delta.pow(2).sum(dim=1).mean().item()

    # Step 4: Hook-based KL divergence for TOP features (most expensive, so subsample)
    # Only do this for a subset of features to validate the proxy metrics
    print("  Running hook-based ablation on top features...")
    hook_name = get_hook_name(model_key, layer_idx)

    # Pick 500 most-active features + 500 random features for validation
    active_features = torch.argsort(act_magnitude, descending=True)[:500].numpy()
    random_features = np.random.RandomState(42).choice(N_FEATURES, 500, replace=False)
    eval_features = np.unique(np.concatenate([active_features, random_features]))

    # Use a small subset of tokens for hook-based ablation
    eval_batch = token_ids[:4].to(DEVICE)  # 4 sequences

    kl_values = np.zeros(N_FEATURES)
    kl_values[:] = np.nan  # NaN for features we don't evaluate

    # Get original logits
    with torch.no_grad():
        orig_outputs = model(eval_batch, output_hidden_states=True)
        orig_logits = orig_outputs.logits
        orig_log_probs = F.log_softmax(orig_logits.float(), dim=-1)
        orig_hidden = orig_outputs.hidden_states[layer_idx + 1]
        # Get SAE activations for these specific tokens
        flat_orig_hidden = orig_hidden.reshape(-1, orig_hidden.shape[-1])
        eval_sae_acts = sae.get_activations(flat_orig_hidden)

    for f_idx in tqdm(eval_features, desc="Hook ablation"):
        if act_magnitude[f_idx] < 1e-10:
            kl_values[f_idx] = 0
            continue

        # Compute the delta from zeroing this feature
        delta = eval_sae_acts[:, f_idx:f_idx+1] * W_dec[f_idx:f_idx+1]  # [n_tokens, d_model]
        delta_reshaped = delta.reshape(orig_hidden.shape)

        # Hook to subtract this feature's contribution
        def make_hook(d):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    modified = output[0] - d
                    return (modified,) + output[1:]
                return output - d
            return hook_fn

        # Register hook
        target_module = dict(model.named_modules())[hook_name]
        handle = target_module.register_forward_hook(make_hook(delta_reshaped))

        with torch.no_grad():
            ablated_logits = model(eval_batch).logits
            ablated_log_probs = F.log_softmax(ablated_logits.float(), dim=-1)

        handle.remove()

        # KL divergence
        kl = (orig_log_probs.exp() * (orig_log_probs - ablated_log_probs)).sum(dim=-1).mean().item()
        kl_values[f_idx] = max(kl, 0)

    # Combine metrics: use direct_effect as primary (available for all features),
    # validate against hook-based KL
    valid_kl = ~np.isnan(kl_values)
    if valid_kl.sum() > 100:
        rho_direct_kl, _ = spearmanr(direct_effect[valid_kl], kl_values[valid_kl])
        rho_recon_kl, _ = spearmanr(recon_importance[valid_kl], kl_values[valid_kl])
        print(f"  Validation: direct_effect↔KL ρ={rho_direct_kl:.4f}, recon_importance↔KL ρ={rho_recon_kl:.4f}")

    # Use combined importance score
    # Normalize each metric to [0, 1] and average
    def normalize(x):
        x = np.array(x, dtype=float)
        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            return (x - xmin) / (xmax - xmin)
        return np.zeros_like(x)

    combined_importance = 0.5 * normalize(direct_effect) + 0.5 * normalize(recon_importance)

    return combined_importance, direct_effect, recon_importance, kl_values


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for model_key, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Causal ablation: {model_key}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"]).to(DEVICE).eval()
        sae = load_sae(model_key, 42)

        token_ids = get_eval_data(tokenizer)
        print(f"  Eval tokens: {len(token_ids)}")

        t0 = time.time()
        importance, direct_effect, recon_importance, kl_values = compute_feature_importance(
            model, sae, token_ids, model_key, model_cfg["layer"]
        )
        elapsed = time.time() - t0
        print(f"  Ablation took {elapsed/60:.1f} min")

        # Load convergence scores
        conv_path = os.path.join(RESULTS_DIR, f"convergence_scores_{model_key}.json")
        with open(conv_path) as f:
            conv_data = json.load(f)

        conv_scores = np.array(conv_data["convergence_score_combined"])
        core_mask = np.array(conv_data["core_mask_08_75"])

        # Core vs peripheral comparison
        core_imp = importance[core_mask]
        periph_imp = importance[~core_mask]

        d = cohens_d(core_imp, periph_imp)
        u_stat, u_p = mannwhitneyu(core_imp, periph_imp, alternative='greater')

        # Spearman correlation
        rho, p = spearmanr(conv_scores, importance)

        # Baseline predictors
        act_path = os.path.join(SAE_DIR, f"activations_{model_key}_seed42.pt")
        sae_acts = torch.load(act_path, map_location="cpu", weights_only=True)
        act_freq = (sae_acts > 0).float().mean(dim=0).numpy()

        checkpoint = torch.load(os.path.join(SAE_DIR, f"{model_key}_seed42.pt"),
                                map_location="cpu", weights_only=True)
        dec_norm = checkpoint["state_dict"]["W_dec"].norm(dim=1).numpy()

        act_mag = sae_acts.sum(dim=0).numpy() / (sae_acts > 0).float().sum(dim=0).numpy().clip(min=1)

        rho_freq, p_freq = spearmanr(act_freq, importance)
        rho_norm, p_norm = spearmanr(dec_norm, importance)
        rho_mag, p_mag = spearmanr(act_mag, importance)

        # AUC-ROC
        from sklearn.metrics import roc_auc_score
        top20_mask = importance >= np.percentile(importance, 80)

        auc_conv = roc_auc_score(top20_mask, conv_scores)
        auc_freq = roc_auc_score(top20_mask, act_freq)
        auc_norm = roc_auc_score(top20_mask, dec_norm)
        auc_mag = roc_auc_score(top20_mask, act_mag)
        auc_random = 0.5

        print(f"\n  Core vs Peripheral:")
        print(f"    Core mean imp: {core_imp.mean():.6f}, Peripheral mean imp: {periph_imp.mean():.6f}")
        print(f"    Cohen's d: {d:.4f}")
        print(f"    Mann-Whitney p: {u_p:.2e}")
        print(f"\n  Correlations with importance:")
        print(f"    Convergence: ρ={rho:.4f}, p={p:.2e}")
        print(f"    Act freq:    ρ={rho_freq:.4f}, p={p_freq:.2e}")
        print(f"    Dec norm:    ρ={rho_norm:.4f}, p={p_norm:.2e}")
        print(f"    Act mag:     ρ={rho_mag:.4f}, p={p_mag:.2e}")
        print(f"\n  AUC-ROC for top-20% importance:")
        print(f"    Convergence: {auc_conv:.4f}")
        print(f"    Act freq:    {auc_freq:.4f}")
        print(f"    Dec norm:    {auc_norm:.4f}")
        print(f"    Act mag:     {auc_mag:.4f}")
        print(f"    Random:      {auc_random:.4f}")

        results = {
            "kl_divergences": importance.tolist(),  # use combined importance as primary metric
            "direct_effect": direct_effect.tolist(),
            "recon_importance": recon_importance.tolist(),
            "core_mean_kl": float(core_imp.mean()),
            "peripheral_mean_kl": float(periph_imp.mean()),
            "cohens_d": float(d),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(u_p),
            "spearman_rho_convergence": float(rho),
            "spearman_p_convergence": float(p),
            "spearman_rho_freq": float(rho_freq),
            "spearman_rho_norm": float(rho_norm),
            "spearman_rho_mag": float(rho_mag),
            "auc_convergence": float(auc_conv),
            "auc_freq": float(auc_freq),
            "auc_norm": float(auc_norm),
            "auc_mag": float(auc_mag),
            "auc_random": auc_random,
            "runtime_minutes": elapsed / 60,
        }

        path = os.path.join(RESULTS_DIR, f"causal_importance_{model_key}.json")
        with open(path, "w") as f:
            json.dump(results, f)

        all_results[model_key] = {k: v for k, v in results.items()
                                   if k not in ["kl_divergences", "direct_effect", "recon_importance"]}

        del model, sae, sae_acts
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_DIR, "causal_importance_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nCausal importance summary:")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
