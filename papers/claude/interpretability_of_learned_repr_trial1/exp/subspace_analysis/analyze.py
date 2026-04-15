"""Subspace analysis of convergent core vs variable periphery."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from shared.sae import TopKSAE
import gc

SAE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sae_training')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

MODELS = {
    "gpt2_small": {"d_model": 768},
    "pythia_160m": {"d_model": 768},
    "pythia_410m": {"d_model": 1024},
}
SEEDS = [42, 123, 456, 789, 1024]
N_FEATURES = 16384
N_NULL = 100  # random partitions for null distribution


def load_decoder_weights(model_key, seed):
    path = os.path.join(SAE_DIR, f"{model_key}_seed{seed}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["state_dict"]["W_dec"].numpy()  # [n_features, d_model]


def get_subspace(W, k=None):
    """Get principal subspace via SVD, keeping components for 90% variance."""
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    cum_var = np.cumsum(S**2) / np.sum(S**2)
    if k is None:
        k = np.searchsorted(cum_var, 0.9) + 1
    return Vh[:k].T  # [d_model, k] - orthonormal basis


def compute_principal_angles(W1, W2, k=None):
    """Compute principal angles between subspaces spanned by W1 and W2."""
    V1 = get_subspace(W1, k)
    V2 = get_subspace(W2, k)
    min_dim = min(V1.shape[1], V2.shape[1])
    angles = subspace_angles(V1[:, :min_dim], V2[:, :min_dim])
    return angles


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for model_key, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Subspace analysis: {model_key}")
        print(f"{'='*60}")

        # Load convergence data
        conv_path = os.path.join(RESULTS_DIR, f"convergence_scores_{model_key}.json")
        with open(conv_path) as f:
            conv_data = json.load(f)
        core_mask = np.array(conv_data["core_mask_08_75"])

        # Load all decoders
        decoders = {}
        for seed in SEEDS:
            decoders[seed] = load_decoder_weights(model_key, seed)

        d_model = model_cfg["d_model"]
        n_core = core_mask.sum()
        n_periph = (~core_mask).sum()
        print(f"  Core: {n_core}, Peripheral: {n_periph}")

        # 1. Principal angles between core subspaces across seeds
        print("  Computing core subspace consistency...")
        core_angles = []
        for i, s1 in enumerate(SEEDS):
            for s2 in SEEDS[i+1:]:
                W1_core = decoders[s1][core_mask]
                W2_core = decoders[s2][core_mask]
                angles = compute_principal_angles(W1_core, W2_core)
                core_angles.append(np.mean(angles))

        # 2. Principal angles between peripheral subspaces across seeds
        print("  Computing peripheral subspace consistency...")
        periph_angles = []
        for i, s1 in enumerate(SEEDS):
            for s2 in SEEDS[i+1:]:
                W1_periph = decoders[s1][~core_mask]
                W2_periph = decoders[s2][~core_mask]
                angles = compute_principal_angles(W1_periph, W2_periph)
                periph_angles.append(np.mean(angles))

        # 3. Null distribution: random partitions
        print(f"  Computing null distribution ({N_NULL} random partitions)...")
        null_angles = []
        rng = np.random.RandomState(42)
        for _ in tqdm(range(N_NULL), desc="Null dist"):
            random_mask = rng.rand(N_FEATURES) < (n_core / N_FEATURES)
            s1, s2 = SEEDS[0], SEEDS[1]
            W1_rand = decoders[s1][random_mask]
            W2_rand = decoders[s2][random_mask]
            if len(W1_rand) > 10 and len(W2_rand) > 10:
                angles = compute_principal_angles(W1_rand, W2_rand)
                null_angles.append(np.mean(angles))

        mean_core_angle = np.mean(core_angles)
        mean_periph_angle = np.mean(periph_angles)
        mean_null_angle = np.mean(null_angles)
        null_5th = np.percentile(null_angles, 5)

        print(f"  Core subspace mean angle: {mean_core_angle:.4f} rad")
        print(f"  Peripheral subspace mean angle: {mean_periph_angle:.4f} rad")
        print(f"  Null mean angle: {mean_null_angle:.4f} rad (5th percentile: {null_5th:.4f})")
        print(f"  Peripheral < null 5th percentile: {mean_periph_angle < null_5th}")

        # 4. Reconstruction quality analysis
        print("  Computing reconstruction quality...")
        sae_path = os.path.join(SAE_DIR, f"{model_key}_seed42.pt")
        checkpoint = torch.load(sae_path, map_location="cpu", weights_only=True)
        W_dec = checkpoint["state_dict"]["W_dec"]  # [n_features, d_model]
        b_dec = checkpoint["state_dict"]["b_dec"]
        W_enc = checkpoint["state_dict"]["W_enc"]
        b_enc = checkpoint["state_dict"]["b_enc"]

        # Load eval activations
        act_path = os.path.join(SAE_DIR, f"activations_{model_key}_seed42.pt")
        sae_acts = torch.load(act_path, map_location="cpu", weights_only=True)[:5000]

        # Original data (approximate from SAE reconstruction)
        x_recon_all = (sae_acts @ W_dec + b_dec).float()

        # Core-only reconstruction
        acts_core = sae_acts.clone()
        acts_core[:, ~torch.tensor(core_mask)] = 0
        x_recon_core = (acts_core @ W_dec + b_dec).float()

        # Peripheral-only reconstruction
        acts_periph = sae_acts.clone()
        acts_periph[:, torch.tensor(core_mask)] = 0
        x_recon_periph = (acts_periph @ W_dec + b_dec).float()

        # Random subset of same size as core
        rng = np.random.RandomState(42)
        random_mask = np.zeros(N_FEATURES, dtype=bool)
        random_idx = rng.choice(N_FEATURES, size=n_core, replace=False)
        random_mask[random_idx] = True
        acts_random = sae_acts.clone()
        acts_random[:, ~torch.tensor(random_mask)] = 0
        x_recon_random = (acts_random @ W_dec + b_dec).float()

        # Compute explained variance relative to all-features reconstruction
        var_total = x_recon_all.var(dim=0).sum().item()
        mse_core = (x_recon_all - x_recon_core).pow(2).mean().item()
        mse_periph = (x_recon_all - x_recon_periph).pow(2).mean().item()
        mse_random = (x_recon_all - x_recon_random).pow(2).mean().item()

        # Also compute actual explained variance
        ev_all = 1.0  # by definition
        ev_core = 1 - mse_core / var_total if var_total > 0 else 0
        ev_periph = 1 - mse_periph / var_total if var_total > 0 else 0
        ev_random = 1 - mse_random / var_total if var_total > 0 else 0

        print(f"  Reconstruction (relative explained variance):")
        print(f"    All features: {ev_all:.4f}")
        print(f"    Core only ({n_core}): {ev_core:.4f}")
        print(f"    Peripheral only ({n_periph}): {ev_periph:.4f}")
        print(f"    Random ({n_core}): {ev_random:.4f}")

        # 5. Feature property comparison
        print("  Comparing feature properties...")
        act_freq_core = (sae_acts[:, torch.tensor(core_mask)] > 0).float().mean(dim=0).numpy()
        act_freq_periph = (sae_acts[:, ~torch.tensor(core_mask)] > 0).float().mean(dim=0).numpy()

        act_mag_core = sae_acts[:, torch.tensor(core_mask)].sum(dim=0).numpy() / \
                       (sae_acts[:, torch.tensor(core_mask)] > 0).float().sum(dim=0).numpy().clip(min=1)
        act_mag_periph = sae_acts[:, ~torch.tensor(core_mask)].sum(dim=0).numpy() / \
                         (sae_acts[:, ~torch.tensor(core_mask)] > 0).float().sum(dim=0).numpy().clip(min=1)

        dec_norm_core = W_dec[torch.tensor(core_mask)].norm(dim=1).numpy()
        dec_norm_periph = W_dec[~torch.tensor(core_mask)].norm(dim=1).numpy()

        # Encoder-decoder alignment
        enc_dec_sim = torch.nn.functional.cosine_similarity(W_enc, W_dec, dim=1).numpy()
        ed_core = enc_dec_sim[core_mask]
        ed_periph = enc_dec_sim[~core_mask]

        # Dead features
        dead_core = (sae_acts[:, torch.tensor(core_mask)].sum(dim=0) == 0).float().mean().item()
        dead_periph = (sae_acts[:, ~torch.tensor(core_mask)].sum(dim=0) == 0).float().mean().item()

        print(f"    Act freq - core: {act_freq_core.mean():.4f}, periph: {act_freq_periph.mean():.4f}")
        print(f"    Act mag - core: {act_mag_core.mean():.4f}, periph: {act_mag_periph.mean():.4f}")
        print(f"    Dec norm - core: {dec_norm_core.mean():.4f}, periph: {dec_norm_periph.mean():.4f}")
        print(f"    Enc-Dec align - core: {ed_core.mean():.4f}, periph: {ed_periph.mean():.4f}")
        print(f"    Dead frac - core: {dead_core:.4f}, periph: {dead_periph:.4f}")

        results = {
            "core_subspace_angles": [float(a) for a in core_angles],
            "peripheral_subspace_angles": [float(a) for a in periph_angles],
            "null_angles_mean": float(mean_null_angle),
            "null_angles_5th_pct": float(null_5th),
            "mean_core_angle": float(mean_core_angle),
            "mean_peripheral_angle": float(mean_periph_angle),
            "peripheral_below_null_5th": bool(mean_periph_angle < null_5th),
            "reconstruction": {
                "ev_all": float(ev_all),
                "ev_core": float(ev_core),
                "ev_peripheral": float(ev_periph),
                "ev_random_subset": float(ev_random),
            },
            "feature_properties": {
                "act_freq_core_mean": float(act_freq_core.mean()),
                "act_freq_periph_mean": float(act_freq_periph.mean()),
                "act_mag_core_mean": float(act_mag_core.mean()),
                "act_mag_periph_mean": float(act_mag_periph.mean()),
                "dec_norm_core_mean": float(dec_norm_core.mean()),
                "dec_norm_periph_mean": float(dec_norm_periph.mean()),
                "enc_dec_align_core_mean": float(ed_core.mean()),
                "enc_dec_align_periph_mean": float(ed_periph.mean()),
                "dead_frac_core": float(dead_core),
                "dead_frac_periph": float(dead_periph),
            },
        }

        path = os.path.join(RESULTS_DIR, f"subspace_analysis_{model_key}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        all_results[model_key] = {k: v for k, v in results.items()
                                   if k not in ["core_subspace_angles", "peripheral_subspace_angles"]}

    with open(os.path.join(RESULTS_DIR, "subspace_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nSubspace analysis summary:")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
