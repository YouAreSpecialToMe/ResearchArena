#!/usr/bin/env python3
"""Generate revision artifacts: qualitative figure + statistical significance tests."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
from scipy import stats

WORKSPACE = Path(__file__).resolve().parent.parent
EXP = WORKSPACE / "exp"
FIGURES = WORKSPACE / "figures"
FIGURES.mkdir(exist_ok=True)

# ============================================================
# PART 1: Qualitative comparison figure
# ============================================================

def generate_qualitative_figure():
    """Create 4x2 grid: left=highest PCS particle, right=lowest PCS particle per prompt."""
    print("=" * 60)
    print("PART 1: Qualitative comparison figure")
    print("=" * 60)

    with open(EXP / "analysis" / "pcs_particle_data_v2.json") as f:
        particle_data = json.load(f)

    # For each prompt, find the particle with highest PCS and lowest PCS
    # PCS values are negative (log-likelihoods), so "highest" means least negative
    # We want prompts where PCS rank inversion is clear:
    # highest PCS particle has LOWER CLIP score than lowest PCS particle
    candidates = []
    for entry in particle_data:
        pcs = np.array(entry["pcs_scores"])
        clip = np.array(entry["clip_scores"])

        best_pcs_idx = int(np.argmax(pcs))  # highest PCS (least negative)
        worst_pcs_idx = int(np.argmin(pcs))  # lowest PCS (most negative)

        # PCS rank inversion: highest PCS -> lower CLIP, lowest PCS -> higher CLIP
        clip_diff = clip[worst_pcs_idx] - clip[best_pcs_idx]
        pcs_range = float(pcs[best_pcs_idx] - pcs[worst_pcs_idx])

        if clip_diff > 0 and pcs_range > 30:  # clear inversion with meaningful PCS spread
            candidates.append({
                "prompt_idx": entry["prompt_idx"],
                "prompt": entry["prompt"],
                "best_pcs_idx": best_pcs_idx,
                "worst_pcs_idx": worst_pcs_idx,
                "best_pcs_val": float(pcs[best_pcs_idx]),
                "worst_pcs_val": float(pcs[worst_pcs_idx]),
                "best_pcs_clip": float(clip[best_pcs_idx]),
                "worst_pcs_clip": float(clip[worst_pcs_idx]),
                "clip_diff": float(clip_diff),
                "pcs_range": pcs_range,
            })

    # Sort by clip_diff (how clear the inversion is)
    candidates.sort(key=lambda x: x["clip_diff"], reverse=True)

    print(f"Found {len(candidates)} prompts with PCS rank inversion")
    for i, c in enumerate(candidates[:10]):
        print(f"  {i}: prompt={c['prompt_idx']:3d} clip_diff={c['clip_diff']:.4f} "
              f"pcs_range={c['pcs_range']:.1f} \"{c['prompt'][:50]}...\"")

    # Select 4 diverse prompts (spread across different prompt indices for variety)
    selected = []
    used_first_words = set()
    for c in candidates:
        first_word = c["prompt"].split()[0].lower()
        # Skip if too similar to already selected
        if first_word in used_first_words and len(selected) < 10:
            continue
        selected.append(c)
        used_first_words.add(first_word)
        if len(selected) == 4:
            break

    # If we don't have 4 diverse ones, just take top 4
    if len(selected) < 4:
        selected = candidates[:4]

    print(f"\nSelected {len(selected)} prompts for figure:")
    for s in selected:
        print(f"  Prompt {s['prompt_idx']}: \"{s['prompt'][:60]}\"")
        print(f"    Highest PCS particle {s['best_pcs_idx']}: PCS={s['best_pcs_val']:.1f}, CLIP={s['best_pcs_clip']:.4f}")
        print(f"    Lowest PCS particle {s['worst_pcs_idx']}: PCS={s['worst_pcs_val']:.1f}, CLIP={s['worst_pcs_clip']:.4f}")

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(8, 14))
    fig.suptitle("PCS Rank Inversion: Highest-PCS vs Lowest-PCS Particles",
                 fontsize=14, fontweight="bold", y=0.98)

    col_titles = ["Highest PCS (Selected by PCS)", "Lowest PCS (Rejected by PCS)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=10)

    img_dir = EXP / "analysis" / "particle_images"

    for i, s in enumerate(selected):
        pidx = s["prompt_idx"]

        # Left column: highest PCS particle
        img_path_best = img_dir / f"prompt{pidx:04d}_particle{s['best_pcs_idx']}.png"
        # Right column: lowest PCS particle
        img_path_worst = img_dir / f"prompt{pidx:04d}_particle{s['worst_pcs_idx']}.png"

        for j, (img_path, pcs_val, clip_val, label) in enumerate([
            (img_path_best, s["best_pcs_val"], s["best_pcs_clip"], "Highest PCS"),
            (img_path_worst, s["worst_pcs_val"], s["worst_pcs_clip"], "Lowest PCS"),
        ]):
            ax = axes[i, j]
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image not found", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="red")

            ax.set_xticks([])
            ax.set_yticks([])

            # Add PCS and CLIP labels
            info_text = f"PCS = {pcs_val:.1f}\nCLIP = {clip_val:.4f}"
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                    fontsize=9, color="white", fontweight="bold",
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

        # Add prompt text on left side
        prompt_short = s["prompt"]
        if len(prompt_short) > 55:
            prompt_short = prompt_short[:52] + "..."
        axes[i, 0].set_ylabel(f"({chr(97+i)}) {prompt_short}", fontsize=9,
                               labelpad=10, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.15, wspace=0.05)

    for fmt in ["png", "pdf"]:
        out = FIGURES / f"figure_qualitative.{fmt}"
        fig.savefig(out, dpi=300 if fmt == "png" else None)
        print(f"Saved {out}")
    plt.close(fig)


# ============================================================
# PART 2: Statistical significance tests
# ============================================================

def compute_per_prompt_clip_scores():
    """Compute per-prompt CLIP scores for CoPS, Random-K, PCS-Best-of-K by scoring images."""
    print("\n" + "=" * 60)
    print("PART 2: Computing per-prompt CLIP scores")
    print("=" * 60)

    # Load CLIP model
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # Load prompts
    with open(EXP / "data" / "coco_500_prompts.json") as f:
        prompts_data = json.load(f)
    prompts = [p["prompt"] for p in prompts_data]

    methods = {
        "cops_resample": "CoPS",
        "random_k": "Random-K",
        "pcs_bestofk": "PCS-Best-of-K",
    }
    seeds = [42, 123, 456]

    results = {}  # method -> seed -> list of per-prompt CLIP scores

    for method_key, method_name in methods.items():
        results[method_key] = {}
        for seed in seeds:
            dir_path = EXP / "main" / f"{method_key}_seed{seed}" / "coco"
            if not dir_path.exists():
                print(f"  WARNING: {dir_path} not found, skipping")
                continue

            per_prompt_clips = []
            n_found = 0
            for i, prompt in enumerate(prompts[:300]):
                img_path = dir_path / f"{i:05d}.png"
                if not img_path.exists():
                    per_prompt_clips.append(float("nan"))
                    continue

                img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                text = tokenizer([prompt]).to(device)

                with torch.no_grad():
                    img_f = model.encode_image(img)
                    txt_f = model.encode_text(text)
                    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                    clip_s = float((img_f * txt_f).sum())

                per_prompt_clips.append(clip_s)
                n_found += 1

            results[method_key][seed] = per_prompt_clips
            print(f"  {method_name} seed{seed}: scored {n_found} images, "
                  f"mean CLIP = {np.nanmean(per_prompt_clips):.4f}")

    return results


def run_significance_tests(per_prompt_scores):
    """Run paired t-test and Wilcoxon signed-rank test."""
    print("\n" + "=" * 60)
    print("PART 2: Statistical Significance Tests")
    print("=" * 60)

    comparisons = [
        ("cops_resample", "random_k", "CoPS vs Random-K"),
        ("pcs_bestofk", "random_k", "PCS-Best-of-K vs Random-K"),
    ]

    seeds = [42, 123, 456]

    for method_a, method_b, comp_name in comparisons:
        print(f"\n{'=' * 50}")
        print(f"  {comp_name}")
        print(f"{'=' * 50}")

        for seed in seeds:
            if seed not in per_prompt_scores.get(method_a, {}):
                continue
            if seed not in per_prompt_scores.get(method_b, {}):
                continue

            a = np.array(per_prompt_scores[method_a][seed])
            b = np.array(per_prompt_scores[method_b][seed])

            # Remove NaN pairs
            valid = ~(np.isnan(a) | np.isnan(b))
            a = a[valid]
            b = b[valid]
            n = len(a)

            if n < 5:
                print(f"\n  Seed {seed}: Only {n} valid pairs, skipping")
                continue

            diff = a - b
            mean_a = np.mean(a)
            mean_b = np.mean(b)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)

            # Cohen's d (paired)
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(a, b)

            # Wilcoxon signed-rank test
            try:
                w_stat, w_pval = stats.wilcoxon(a, b, alternative="two-sided")
            except ValueError as e:
                w_stat, w_pval = float("nan"), float("nan")
                print(f"  Wilcoxon failed: {e}")

            print(f"\n  Seed {seed} (n={n} prompts):")
            print(f"    Mean {method_a}: {mean_a:.6f}")
            print(f"    Mean {method_b}: {mean_b:.6f}")
            print(f"    Mean difference: {mean_diff:.6f} ({'+' if mean_diff > 0 else ''}{mean_diff:.6f})")
            print(f"    Std of differences: {std_diff:.6f}")
            print(f"    Cohen's d: {cohens_d:.4f}")
            print(f"    Paired t-test: t={t_stat:.4f}, p={t_pval:.6f} {'***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else 'n.s.'}")
            print(f"    Wilcoxon signed-rank: W={w_stat:.1f}, p={w_pval:.6f} {'***' if w_pval < 0.001 else '**' if w_pval < 0.01 else '*' if w_pval < 0.05 else 'n.s.'}")

        # Aggregate across seeds
        print(f"\n  --- Aggregated across seeds ---")
        all_a, all_b = [], []
        for seed in seeds:
            if seed in per_prompt_scores.get(method_a, {}) and seed in per_prompt_scores.get(method_b, {}):
                a = np.array(per_prompt_scores[method_a][seed])
                b = np.array(per_prompt_scores[method_b][seed])
                valid = ~(np.isnan(a) | np.isnan(b))
                all_a.extend(a[valid].tolist())
                all_b.extend(b[valid].tolist())

        all_a = np.array(all_a)
        all_b = np.array(all_b)
        n = len(all_a)

        if n >= 5:
            diff = all_a - all_b
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

            t_stat, t_pval = stats.ttest_rel(all_a, all_b)
            try:
                w_stat, w_pval = stats.wilcoxon(all_a, all_b, alternative="two-sided")
            except ValueError:
                w_stat, w_pval = float("nan"), float("nan")

            print(f"    Total paired samples: {n}")
            print(f"    Mean {method_a}: {np.mean(all_a):.6f}")
            print(f"    Mean {method_b}: {np.mean(all_b):.6f}")
            print(f"    Mean difference: {mean_diff:.6f}")
            print(f"    Cohen's d: {cohens_d:.4f}")
            print(f"    Paired t-test: t={t_stat:.4f}, p={t_pval:.6f} {'***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else 'n.s.'}")
            print(f"    Wilcoxon signed-rank: W={w_stat:.1f}, p={w_pval:.6f} {'***' if w_pval < 0.001 else '**' if w_pval < 0.01 else '*' if w_pval < 0.05 else 'n.s.'}")


if __name__ == "__main__":
    # Part 1: Qualitative figure
    generate_qualitative_figure()

    # Part 2: Per-prompt CLIP scores and significance tests
    per_prompt_scores = compute_per_prompt_clip_scores()
    run_significance_tests(per_prompt_scores)

    print("\n" + "=" * 60)
    print("All revision artifacts generated successfully!")
    print("=" * 60)
