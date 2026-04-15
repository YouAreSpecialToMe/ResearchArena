"""Generate Figure 7: Qualitative feature examples showing top core vs peripheral
features with their activating contexts.

This addresses reviewer feedback: add concrete examples to make the core/peripheral
distinction intuitive.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoModelForCausalLM, AutoTokenizer
from shared.sae import TopKSAE
from datasets import load_dataset

DEVICE = "cuda"
SAE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sae_training')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')

N_FEATURES = 16384


def load_sae(model_key, seed):
    path = os.path.join(SAE_DIR, f"{model_key}_seed{seed}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    sae = TopKSAE(cfg["d_model"], cfg["n_features"], cfg["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    return sae.eval().to(DEVICE)


def get_activating_contexts(model, sae, tokenizer, feature_idx, layer_idx,
                             model_key, n_contexts=10, n_texts=500):
    """Find the top-activating token contexts for a given feature."""
    # Load some text data
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    all_contexts = []
    all_activations = []

    hook_name = f"transformer.h.{layer_idx}" if model_key == "gpt2_small" else f"gpt_neox.layers.{layer_idx}"

    count = 0
    for ex in ds:
        text = ex["text"].strip()
        if len(text) < 50:
            continue

        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1]
            flat = hidden.reshape(-1, hidden.shape[-1])
            acts = sae.get_activations(flat)

        feature_acts = acts[:, feature_idx].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        for pos in range(len(tokens)):
            if feature_acts[pos] > 0:
                # Get context window
                start = max(0, pos - 5)
                end = min(len(tokens), pos + 6)
                context_tokens = tokens[start:end]
                target_pos = pos - start

                context_str = ""
                for i, t in enumerate(context_tokens):
                    t_clean = t.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
                    if i == target_pos:
                        context_str += f"[{t_clean}]"
                    else:
                        context_str += t_clean

                all_contexts.append(context_str)
                all_activations.append(feature_acts[pos])

        count += 1
        if count >= n_texts:
            break

    # Sort by activation and return top
    if len(all_activations) == 0:
        return [], []

    sorted_idx = np.argsort(all_activations)[::-1][:n_contexts]
    top_contexts = [all_contexts[i] for i in sorted_idx]
    top_acts = [all_activations[i] for i in sorted_idx]

    return top_contexts, top_acts


def compute_cross_seed_similarity(model_key, feature_idx, seeds=[42, 123, 456, 789, 1024]):
    """Compute decoder cosine similarity of a feature across all seed pairs."""
    decoders = {}
    for seed in seeds:
        path = os.path.join(SAE_DIR, f"{model_key}_seed{seed}.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        W_dec = checkpoint["state_dict"]["W_dec"]
        W_dec = torch.nn.functional.normalize(W_dec, dim=1)
        decoders[seed] = W_dec

    # For the reference feature in seed 42, find best match in each other seed
    ref_vec = decoders[42][feature_idx]  # [d_model]
    sim_matrix = np.zeros((len(seeds), len(seeds)))

    for i, s1 in enumerate(seeds):
        for j, s2 in enumerate(seeds):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i == 0:  # ref seed
                # Find best match of feature_idx in seed s2
                sims = decoders[s2] @ ref_vec
                sim_matrix[i, j] = sims.max().item()
                sim_matrix[j, i] = sims.max().item()

    return sim_matrix


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    model_key = "gpt2_small"
    model_name = "gpt2"
    layer_idx = 6

    # Load convergence scores
    conv_path = os.path.join(RESULTS_DIR, f"convergence_scores_{model_key}.json")
    with open(conv_path) as f:
        conv_data = json.load(f)
    conv_scores = np.array(conv_data["convergence_score_combined"])
    core_mask = np.array(conv_data["core_mask_08_75"])

    # Load activation frequencies
    act_path = os.path.join(SAE_DIR, f"activations_{model_key}_seed42.pt")
    sae_acts = torch.load(act_path, map_location="cpu", weights_only=True)
    act_freq = (sae_acts > 0).float().mean(dim=0).numpy()

    # Find top core and top peripheral features (that are active)
    active_mask = act_freq > 0.005

    core_active_idx = np.where(core_mask & active_mask)[0]
    periph_active_idx = np.where(~core_mask & active_mask)[0]

    # Sort by convergence score
    core_sorted = core_active_idx[np.argsort(conv_scores[core_active_idx])[::-1]]
    periph_sorted = periph_active_idx[np.argsort(conv_scores[periph_active_idx])]

    top_core = core_sorted[:5]
    top_periph = periph_sorted[:5]

    print(f"Top 5 core features: {top_core} (conv scores: {conv_scores[top_core]})")
    print(f"Top 5 peripheral features: {top_periph} (conv scores: {conv_scores[top_periph]})")

    # Load model and SAE
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()
    sae = load_sae(model_key, 42)

    # Collect contexts for each feature
    core_data = []
    for f_idx in top_core:
        contexts, acts = get_activating_contexts(model, sae, tokenizer, f_idx, layer_idx, model_key, n_contexts=5, n_texts=200)
        core_data.append({
            "feature_idx": int(f_idx),
            "convergence_score": float(conv_scores[f_idx]),
            "act_freq": float(act_freq[f_idx]),
            "contexts": contexts,
            "activations": [float(a) for a in acts],
        })

    periph_data = []
    for f_idx in top_periph:
        contexts, acts = get_activating_contexts(model, sae, tokenizer, f_idx, layer_idx, model_key, n_contexts=5, n_texts=200)
        periph_data.append({
            "feature_idx": int(f_idx),
            "convergence_score": float(conv_scores[f_idx]),
            "act_freq": float(act_freq[f_idx]),
            "contexts": contexts,
            "activations": [float(a) for a in acts],
        })

    # Save raw data
    qual_data = {"core": core_data, "peripheral": periph_data}
    with open(os.path.join(RESULTS_DIR, "qualitative_features.json"), "w") as f:
        json.dump(qual_data, f, indent=2)

    # Generate figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ax_idx, (category, data, color) in enumerate([
        ("Core Features (High Convergence)", core_data, "#2196F3"),
        ("Peripheral Features (Low Convergence)", periph_data, "#FF9800")
    ]):
        ax = axes[ax_idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(category, fontsize=14, fontweight='bold', pad=10, color=color)

        y_pos = 0.95
        for feat in data[:5]:
            if not feat["contexts"]:
                continue

            header = f"Feature #{feat['feature_idx']}  (conv={feat['convergence_score']:.3f}, freq={feat['act_freq']:.4f})"
            ax.text(0.02, y_pos, header, fontsize=9, fontweight='bold',
                    fontfamily='monospace', verticalalignment='top')
            y_pos -= 0.04

            for ctx in feat["contexts"][:3]:
                # Truncate long contexts
                display_ctx = ctx[:80] + "..." if len(ctx) > 80 else ctx
                ax.text(0.04, y_pos, display_ctx, fontsize=7.5,
                        fontfamily='monospace', verticalalignment='top',
                        color='#333333')
                y_pos -= 0.03

            y_pos -= 0.02

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_feature_examples.pdf"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_feature_examples.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nSaved Figure 7 to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
