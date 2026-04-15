"""Compare convergence score vs baselines for predicting cross-model universality.

This addresses reviewer feedback: convergence score's unique value should be
for universality prediction, but this was never compared against baselines.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SAE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sae_training')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

MODELS = ["gpt2_small", "pythia_160m", "pythia_410m"]
N_FEATURES = 16384
import torch


def main():
    results = {}

    for model_key in MODELS:
        print(f"\n{'='*60}")
        print(f"Universality baseline comparison: {model_key}")
        print(f"{'='*60}")

        # Load convergence scores
        conv_path = os.path.join(RESULTS_DIR, f"convergence_scores_{model_key}.json")
        with open(conv_path) as f:
            conv_data = json.load(f)
        conv_scores = np.array(conv_data["convergence_score_combined"])
        conv_decoder = np.array(conv_data["convergence_score_decoder"])
        conv_activation = np.array(conv_data["convergence_score_activation"])

        # Load cross-model alignment scores
        align_path = os.path.join(RESULTS_DIR, f"cross_model_alignment_{model_key}.json")
        with open(align_path) as f:
            align_data = json.load(f)
        universality = np.array(align_data["universality_scores"])

        # Load feature properties (activation frequency, decoder norm, etc.)
        act_path = os.path.join(SAE_DIR, f"activations_{model_key}_seed42.pt")
        sae_acts = torch.load(act_path, map_location="cpu", weights_only=True)
        act_freq = (sae_acts > 0).float().mean(dim=0).numpy()
        act_mag = sae_acts.float().abs().mean(dim=0).numpy()

        checkpoint = torch.load(os.path.join(SAE_DIR, f"{model_key}_seed42.pt"),
                                map_location="cpu", weights_only=True)
        dec_norm = checkpoint["state_dict"]["W_dec"].norm(dim=1).numpy()

        # Compute Spearman correlations with universality
        predictors = {
            "convergence_combined": conv_scores,
            "convergence_decoder": conv_decoder,
            "convergence_activation": conv_activation,
            "activation_frequency": act_freq,
            "activation_magnitude": act_mag,
            "decoder_norm": dec_norm,
        }

        # Top-20% universality as binary target for AUC
        top20_mask = universality >= np.percentile(universality, 80)

        model_results = {}
        for name, pred in predictors.items():
            rho, p = spearmanr(pred, universality)
            try:
                auc = roc_auc_score(top20_mask, pred)
            except:
                auc = 0.5

            model_results[name] = {
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "auc_top20": float(auc),
            }
            print(f"  {name:30s}: ρ={rho:.4f} (p={p:.2e}), AUC={auc:.4f}")

        # Also compare on active features only
        active_mask = act_freq > 0.001
        if active_mask.sum() > 100:
            print(f"\n  Active features only ({active_mask.sum()}):")
            top20_active = universality[active_mask] >= np.percentile(universality[active_mask], 80)
            for name, pred in predictors.items():
                rho, p = spearmanr(pred[active_mask], universality[active_mask])
                try:
                    auc = roc_auc_score(top20_active, pred[active_mask])
                except:
                    auc = 0.5
                model_results[f"{name}_active_only"] = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "auc_top20": float(auc),
                }
                print(f"    {name:30s}: ρ={rho:.4f} (p={p:.2e}), AUC={auc:.4f}")

        results[model_key] = model_results

    path = os.path.join(RESULTS_DIR, "universality_baselines.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
