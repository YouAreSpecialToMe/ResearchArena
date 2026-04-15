"""
Step 5: Generate publication-quality figures.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

CAPS = ["factual", "syntax", "sentiment", "semantic", "ner", "reasoning"]
CAP_LABELS = {"factual": "Factual", "syntax": "Syntax",
              "sentiment": "Sentiment", "semantic": "Semantic",
              "ner": "NER", "reasoning": "Reasoning"}

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
PAL = sns.color_palette("colorblind", n_colors=8)
FIG_W, FIG_H = 6.5, 4.0
DPI = 300

def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {name}")


def load(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


# ── Figure 1: Feature–Capability Heatmap ─────────────────────────────────────

def fig1_heatmap():
    print("Figure 1: feature-capability heatmap")
    fli = load("fli_scores.json")
    peaks = load("peak_layers.json")

    layers = list(range(12))
    data = np.zeros((12, len(CAPS)))
    for j, c in enumerate(CAPS):
        for l in layers:
            d = fli.get(c, {}).get(str(l), {})
            data[l, j] = d.get("fli_mean", 0)

    fig, ax = plt.subplots(figsize=(FIG_W, 5))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(CAPS)))
    ax.set_xticklabels([CAP_LABELS[c] for c in CAPS], rotation=30, ha="right")
    ax.set_yticks(layers)
    ax.set_yticklabels(layers)
    ax.set_ylabel("Layer")
    ax.set_xlabel("Capability")
    ax.set_title("FLI Across Layers and Capabilities")
    # Annotate peak layers
    for j, c in enumerate(CAPS):
        pl = peaks[c]
        ax.scatter(j, pl, marker="*", color="red", s=120, zorder=5)
    fig.colorbar(im, ax=ax, label="FLI")
    _save(fig, "figure1_feature_capability_heatmap")


# ── Figure 2: FLI bar + line ─────────────────────────────────────────────────

def fig2_fli():
    print("Figure 2: FLI")
    fli = load("fli_scores.json")
    peaks = load("peak_layers.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W * 1.5, FIG_H))

    # (a) Bar chart at peak layer
    means, stds, labels = [], [], []
    for c in sorted(CAPS, key=lambda c: -fli.get(c, {}).get(
            str(peaks[c]), {}).get("fli_mean", 0)):
        d = fli.get(c, {}).get(str(peaks[c]), {})
        means.append(d.get("fli_mean", 0))
        stds.append(d.get("fli_std", 0))
        labels.append(CAP_LABELS[c])
    x = np.arange(len(labels))
    bars = ax1.bar(x, means, yerr=stds, capsize=4, color=PAL[:len(labels)],
                   edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("FLI")
    ax1.set_title("(a) FLI at Peak Layer")

    # (b) Line plot across layers
    for i, c in enumerate(CAPS):
        ys = [fli.get(c, {}).get(str(l), {}).get("fli_mean", 0)
              for l in range(12)]
        ax2.plot(range(12), ys, marker="o", ms=4, label=CAP_LABELS[c],
                 color=PAL[i])
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("FLI")
    ax2.set_title("(b) FLI Across Layers")
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    _save(fig, "figure2_fli")


# ── Figure 3: Causal validation ──────────────────────────────────────────────

def fig3_causal():
    print("Figure 3: causal validation")
    causal = load("causal_validation.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W * 1.5, FIG_H))

    # (a) Dose-response
    Ks = [10, 20, 50, 100]
    for i, c in enumerate(CAPS):
        sr = causal[c]["seed_results"]
        base = np.mean([s["baseline"] for s in sr])
        ys = []
        for k in Ks:
            vals = [s.get(f"ablated_top{k}", base) for s in sr]
            ys.append(np.mean(vals) / max(abs(base), 1e-10))
        ax1.plot(Ks, ys, marker="o", ms=4, label=CAP_LABELS[c],
                 color=PAL[i])
    ax1.set_xlabel("# Features Ablated")
    ax1.set_ylabel("Normalised Score")
    ax1.set_title("(a) Dose-Response")
    ax1.legend(fontsize=7)

    # (b) Top-50 vs Random-50
    caps_sorted = sorted(CAPS, key=lambda c: -causal[c].get(
        "causal_fidelity_mean", 0))
    x = np.arange(len(caps_sorted))
    w = 0.35
    dt = [abs(causal[c]["drop_top50_mean"]) for c in caps_sorted]
    dr = [abs(np.mean([s["drop_random50"] for s in causal[c]["seed_results"]]))
          for c in caps_sorted]
    ax2.bar(x - w/2, dt, w, label="Top-50", color=PAL[0])
    ax2.bar(x + w/2, dr, w, label="Random-50", color=PAL[1])
    ax2.set_xticks(x)
    ax2.set_xticklabels([CAP_LABELS[c] for c in caps_sorted],
                        rotation=30, ha="right")
    ax2.set_ylabel("|Performance Drop|")
    ax2.set_title("(b) Top vs Random Ablation")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "figure3_causal_validation")


# ── Figure 4: Overlap matrix ─────────────────────────────────────────────────

def fig4_overlap():
    print("Figure 4: overlap")
    ov = load("capability_overlap.json")
    mat = np.array(ov["overlap_at_peak"])
    labels = [CAP_LABELS[c] for c in CAPS]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(FIG_W * 1.8, FIG_H))

    # (a) Heatmap
    sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=labels,
                yticklabels=labels, cmap="YlOrRd", ax=ax1, cbar_kws={"shrink": 0.8})
    ax1.set_title("(a) Overlap at Peak Layers")

    # (b) Dendrogram
    Z = linkage(1 - mat, method="ward")
    dendrogram(Z, labels=labels, ax=ax2, leaf_rotation=45)
    ax2.set_title("(b) Hierarchical Clustering")

    # (c) Breadth distribution
    bdist = ov.get("feature_breadth_distribution", {})
    bs = sorted(bdist.keys(), key=int)
    ax3.bar([int(b) for b in bs], [bdist[b] for b in bs],
            color=PAL[2], edgecolor="black", linewidth=0.5)
    ax3.set_xlabel("# Capabilities Supported")
    ax3.set_ylabel("# Features")
    ax3.set_title("(c) Feature Breadth")

    fig.tight_layout()
    _save(fig, "figure4_overlap")


# ── Figure 5: Dark matter ────────────────────────────────────────────────────

def fig5_dark_matter():
    print("Figure 5: dark matter")
    dm = load("dark_matter_probes.json")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(CAPS))
    w = 0.2
    conditions = ["features_accuracy", "residual_accuracy",
                  "original_accuracy", "random_accuracy"]
    cond_labels = ["SAE Features", "SAE Residual",
                   "Full Activation", "Random"]
    for k, (cond, lab) in enumerate(zip(conditions, cond_labels)):
        vals = [dm.get(c, {}).get(cond, 0.5) for c in CAPS]
        ax.bar(x + k * w, vals, w, label=lab, color=PAL[k],
               edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([CAP_LABELS[c] for c in CAPS], rotation=30,
                       ha="right")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Dark Matter Analysis: Probing SAE Features vs Residuals")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="Chance")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    _save(fig, "figure5_dark_matter")


# ── Figure 6: Architecture comparison ────────────────────────────────────────

def fig6_architecture():
    print("Figure 6: architecture comparison")
    try:
        arch = load("architecture_comparison.json")
    except FileNotFoundError:
        print("  skipped (no data)")
        return

    fp = arch.get("fli_primary", {})
    fa = arch.get("fli_alt", {})
    common = sorted(set(fp) & set(fa))
    if not common:
        print("  skipped (no common keys)")
        return

    x = [fp[k] for k in common]
    y = [fa[k] for k in common]
    rho = arch.get("spearman_rho", 0)

    fig, ax = plt.subplots(figsize=(FIG_W * 0.7, FIG_W * 0.7))
    ax.scatter(x, y, s=40, c=PAL[0], edgecolors="black", linewidths=0.5)
    mn, mx = min(min(x), min(y)), max(max(x), max(y))
    ax.plot([mn, mx], [mn, mx], ls="--", color="gray", lw=0.8)
    # Regression line
    z = np.polyfit(x, y, 1)
    xs = np.linspace(mn, mx, 100)
    ax.plot(xs, np.polyval(z, xs), color=PAL[1], lw=1.5)
    ax.set_xlabel("FLI — TopK 32K")
    ax.set_ylabel("FLI — Standard 24K (JB)")
    ax.set_title(f"Architecture Comparison (ρ = {rho:.2f})")
    fig.tight_layout()
    _save(fig, "figure6_architecture_comparison")


# ── Tables ───────────────────────────────────────────────────────────────────

def table1_main():
    print("Table 1: main results")
    fli = load("fli_scores.json")
    peaks = load("peak_layers.json")
    causal = load("causal_validation.json")
    dm = load("dark_matter_probes.json")

    rows = []
    for c in CAPS:
        pl = peaks[c]
        d = fli.get(c, {}).get(str(pl), {})
        cf = causal.get(c, {})
        dmr = dm.get(c, {})
        rows.append({
            "cap": CAP_LABELS[c], "peak": pl,
            "fli": f"{d.get('fli_mean',0):.4f}±{d.get('fli_std',0):.4f}",
            "eff": f"{d.get('effective_features_mean',0):.0f}",
            "cf": f"{cf.get('causal_fidelity_mean',0):.2f}"
                  f"±{cf.get('causal_fidelity_std',0):.2f}",
            "dm": f"{dmr.get('dark_matter_ratio',0):.3f}",
            "feat_acc": f"{dmr.get('features_accuracy',0):.3f}",
        })

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main experimental results per capability.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Capability & Peak Layer & FLI & Eff.\ Features "
        r"& Causal Fidelity & DM Ratio & Feat.\ Acc \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['cap']} & {r['peak']} & {r['fli']} & {r['eff']} "
            f"& {r['cf']} & {r['dm']} & {r['feat_acc']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(FIG_DIR / "table1_main_results.tex", "w") as f:
        f.write("\n".join(lines))
    print("  → table1_main_results.tex")


def table2_ablations():
    print("Table 2: ablations")
    try:
        attr = load("attribution_ablation.json")
    except FileNotFoundError:
        print("  skipped")
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Attribution method comparison (performance drop from ablating top-50 features).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Capability & Grad$\times$Act & Act-Only & Grad-Only & Random \\",
        r"\midrule",
    ]
    methods = ["grad_x_act", "activation_only", "gradient_only", "random"]
    for c in CAPS:
        if c not in attr:
            continue
        vals = []
        for m in methods:
            d = attr[c].get(m, {}).get("drop", 0)
            vals.append(f"{d:.4f}")
        lines.append(f"{CAP_LABELS[c]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(FIG_DIR / "table2_ablations.tex", "w") as f:
        f.write("\n".join(lines))
    print("  → table2_ablations.tex")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Generating Figures ===\n")
    fig1_heatmap()
    fig2_fli()
    fig3_causal()
    fig4_overlap()
    fig5_dark_matter()
    fig6_architecture()
    table1_main()
    table2_ablations()
    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
