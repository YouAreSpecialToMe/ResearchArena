"""
Step 4: Statistical tests, success-criteria evaluation, final results.json.
CORRECTED: Uses capped causal fidelity, honest degenerate-case reporting.
"""
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats as scipy_stats

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

CAPS = ["factual", "syntax", "sentiment", "semantic", "ner", "reasoning"]


def load(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def evaluate_success_criteria():
    print("=== Evaluating Success Criteria ===\n")

    fli = load("fli_scores.json")
    peaks = load("peak_layers.json")
    causal = load("causal_validation.json")
    dm = load("dark_matter_probes.json")
    try:
        arch = load("architecture_comparison.json")
    except FileNotFoundError:
        arch = {}
    try:
        attr_abl = load("attribution_ablation.json")
    except FileNotFoundError:
        attr_abl = {}

    ev = {}

    # ── Identify degenerate capabilities ─────────────────────────────────
    # A capability is degenerate if effective_features ≈ d_sae (near-uniform)
    degenerate_caps = []
    for c in CAPS:
        pl = str(peaks[c])
        if pl in fli.get(c, {}):
            eff = fli[c][pl].get("effective_features_mean", 0)
            fli_val = fli[c][pl].get("fli_mean", 0)
            if eff > 10000 or fli_val < 0.05:
                degenerate_caps.append(c)
    if degenerate_caps:
        print(f"  ⚠ Degenerate capabilities (near-uniform attribution): {degenerate_caps}")
    non_degenerate_caps = [c for c in CAPS if c not in degenerate_caps]
    print(f"  Meaningful capabilities: {non_degenerate_caps}\n")

    # ── Criterion 1: FLI varies ≥ 2× across capabilities ────────────────
    print("--- Criterion 1: FLI variation ---")
    pfli = {}
    for c in CAPS:
        pl = str(peaks[c])
        if pl in fli.get(c, {}):
            pfli[c] = fli[c][pl]["fli_mean"]

    pfli_nd = {c: v for c, v in pfli.items() if c in non_degenerate_caps}

    if pfli_nd:
        vals = list(pfli_nd.values())
        mx, mn = max(vals), min(vals)
        ratio = mx / max(mn, 1e-10)
        # Also include all caps for context
        vals_all = list(pfli.values())
        ratio_all = max(vals_all) / max(min(vals_all), 1e-10)

        groups = []
        for c in non_degenerate_caps:
            g = [v["fli_mean"] for v in fli.get(c, {}).values()]
            if g:
                groups.append(g)
        f_stat, p_val = (scipy_stats.f_oneway(*groups)
                         if len(groups) >= 2 else (0, 1))

        status = ("confirmed" if ratio >= 2.0 and p_val < 0.05
                  else "partially_supported" if ratio >= 1.5
                  else "refuted")
        ev["criterion_1"] = {
            "description": "FLI varies ≥2× across capabilities",
            "status": status,
            "max_fli": float(mx), "min_fli": float(mn),
            "ratio_non_degenerate": float(ratio),
            "ratio_all": float(ratio_all),
            "most_localized": max(pfli_nd, key=pfli_nd.get),
            "least_localized": min(pfli_nd, key=pfli_nd.get),
            "anova_f": float(f_stat), "anova_p": float(p_val),
            "per_capability_fli": {k: float(v) for k, v in pfli.items()},
            "degenerate_caps": degenerate_caps,
            "note": ("Evaluated on non-degenerate capabilities only. "
                     f"Degenerate caps {degenerate_caps} had near-uniform "
                     "attribution (FLI ≈ 0)." if degenerate_caps else ""),
        }
        print(f"  max={mx:.4f} ({ev['criterion_1']['most_localized']}), "
              f"min={mn:.4f} ({ev['criterion_1']['least_localized']})")
        print(f"  ratio={ratio:.2f} (non-degenerate), ANOVA F={f_stat:.2f} p={p_val:.4f}")
        print(f"  → {status}")
    else:
        ev["criterion_1"] = {"status": "refuted", "reason": "all capabilities degenerate"}

    # ── Criterion 2: FLI predicts causal faithfulness ────────────────────
    print("\n--- Criterion 2: FLI → causal faithfulness ---")
    # Use capped fidelity and normalized drop difference
    cfli, cfid, cndd = [], [], []
    for c in non_degenerate_caps:
        pl = str(peaks[c])
        if pl in fli.get(c, {}) and c in causal:
            cfli.append(fli[c][pl]["fli_mean"])
            cfid.append(causal[c].get("causal_fidelity_mean", 0))
            cndd.append(causal[c].get("normalized_drop_diff_mean", 0))

    if len(cfli) >= 4:
        rho, p = scipy_stats.spearmanr(cfli, cfid)
        rho_ndd, p_ndd = scipy_stats.spearmanr(cfli, cndd)
    elif len(cfli) >= 3:
        rho, p = scipy_stats.spearmanr(cfli, cfid)
        rho_ndd, p_ndd = scipy_stats.spearmanr(cfli, cndd)
    else:
        rho, p = 0, 1
        rho_ndd, p_ndd = 0, 1

    # 3-most vs 3-least localized drop comparison (non-degenerate only)
    ranked_nd = sorted(pfli_nd, key=pfli_nd.get, reverse=True) if pfli_nd else []
    mid = len(ranked_nd) // 2
    top_half = ranked_nd[:max(mid, 1)]
    bot_half = ranked_nd[max(mid, 1):]

    drops_top = [abs(causal[c]["drop_top50_mean"]) for c in top_half if c in causal]
    drops_bot = [abs(causal[c]["drop_top50_mean"]) for c in bot_half if c in causal]
    mean_drop_top = float(np.mean(drops_top)) if drops_top else 0
    mean_drop_bot = float(np.mean(drops_bot)) if drops_bot else 0
    drop_ratio = mean_drop_top / max(mean_drop_bot, 1e-10)

    status = ("confirmed" if rho > 0.5 and p < 0.1
              else "partially_supported" if rho > 0.3 or drop_ratio > 2
              else "refuted")
    ev["criterion_2"] = {
        "description": "FLI predicts causal faithfulness (capped fidelity)",
        "status": status,
        "spearman_rho_fidelity": float(rho), "spearman_p_fidelity": float(p),
        "spearman_rho_ndd": float(rho_ndd), "spearman_p_ndd": float(p_ndd),
        "top_half_mean_drop": mean_drop_top,
        "bottom_half_mean_drop": mean_drop_bot,
        "drop_ratio": float(drop_ratio),
        "top_caps": top_half, "bottom_caps": bot_half,
        "note": f"Evaluated on non-degenerate caps only: {non_degenerate_caps}",
    }
    print(f"  Spearman ρ(FLI,fidelity)={rho:.3f} p={p:.4f}")
    print(f"  Spearman ρ(FLI,NDD)={rho_ndd:.3f} p={p_ndd:.4f}")
    print(f"  top-half drop={mean_drop_top:.4f}, bot-half drop={mean_drop_bot:.4f}, ratio={drop_ratio:.2f}")
    print(f"  → {status}")

    # ── Criterion 3: dark-matter dependence ──────────────────────────────
    print("\n--- Criterion 3: FLI → dark-matter dependence ---")
    dfli, ddm = [], []
    for c in non_degenerate_caps:
        pl = str(peaks[c])
        if pl in fli.get(c, {}) and c in dm and not dm[c].get("skipped"):
            dfli.append(fli[c][pl]["fli_mean"])
            ddm.append(dm[c].get("dark_matter_ratio", 1))
    if len(dfli) >= 4:
        rho3, p3 = scipy_stats.spearmanr(dfli, ddm)
    elif len(dfli) >= 3:
        rho3, p3 = scipy_stats.spearmanr(dfli, ddm)
    else:
        rho3, p3 = 0, 1

    # Compare probe accuracies
    if ranked_nd:
        resid_top = [dm[c]["residual_accuracy"] for c in top_half
                     if c in dm and not dm[c].get("skipped")]
        resid_bot = [dm[c]["residual_accuracy"] for c in bot_half
                     if c in dm and not dm[c].get("skipped")]
        mean_rt = float(np.mean(resid_top)) if resid_top else 0.5
        mean_rb = float(np.mean(resid_bot)) if resid_bot else 0.5
        resid_ratio = mean_rb / max(mean_rt, 0.01)
    else:
        mean_rt, mean_rb, resid_ratio = 0.5, 0.5, 1.0

    status = ("confirmed" if rho3 < -0.5 and p3 < 0.05
              else "partially_supported" if rho3 < -0.3 or resid_ratio > 1.5
              else "refuted")
    ev["criterion_3"] = {
        "description": "FLI inversely predicts dark-matter dependence",
        "status": status,
        "spearman_rho": float(rho3), "spearman_p": float(p3),
        "resid_probe_top_half": mean_rt,
        "resid_probe_bot_half": mean_rb,
        "resid_ratio": float(resid_ratio),
        "note": f"Evaluated on non-degenerate caps: {non_degenerate_caps}",
    }
    print(f"  Spearman ρ={rho3:.3f} p={p3:.4f}")
    print(f"  resid-probe top={mean_rt:.3f}, bot={mean_rb:.3f}, ratio={resid_ratio:.2f}")
    print(f"  → {status}")

    # ── Criterion 4: cross-architecture consistency ──────────────────────
    print("\n--- Criterion 4: cross-architecture consistency ---")
    if arch and "spearman_rho" in arch:
        rho4 = arch["spearman_rho"]
        p4 = arch["spearman_p"]
        status = ("confirmed" if rho4 > 0.7
                  else "partially_supported" if rho4 > 0.5
                  else "refuted")
        ev["criterion_4"] = {
            "description": "FLI consistent across SAE architectures",
            "status": status,
            "spearman_rho": float(rho4), "spearman_p": float(p4),
            "primary": arch.get("primary_type", ""),
            "alt": arch.get("alt_type", ""),
            "n_comparisons": arch.get("n_comparisons", 0),
        }
        print(f"  Spearman ρ={rho4:.3f} p={p4:.4f}")
        print(f"  → {status}")
    else:
        ev["criterion_4"] = {"status": "not_evaluated",
                             "reason": "architecture comparison not run"}
        print("  not evaluated")

    # ── Criterion 5: localization predicts interpretability ──────────────
    print("\n--- Criterion 5: localization → interpretability ---")
    try:
        from transformer_lens import HookedTransformer
        from sae_lens import SAE
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
        W_E = model.W_E.detach()

        interp_scores = {}
        for c in non_degenerate_caps:
            pl = peaks[c]
            sae = SAE.from_pretrained(
                release="gpt2-small-resid-post-v5-32k",
                sae_id=f"blocks.{pl}.hook_resid_post",
                device=str(device))
            if isinstance(sae, tuple):
                sae = sae[0]

            with open(RESULTS_DIR / "top_features.json") as f:
                tf = json.load(f)
            top20 = tf[c][str(pl)][:20]

            W_dec_top = sae.W_dec[top20]
            W_dec_n = W_dec_top / W_dec_top.norm(dim=-1, keepdim=True)
            W_E_n = W_E / W_E.norm(dim=-1, keepdim=True)
            sims = (W_dec_n @ W_E_n.T).max(dim=-1).values
            interp_scores[c] = float(sims.mean().item())
            del sae
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

        it_top = [interp_scores[c] for c in top_half if c in interp_scores]
        it_bot = [interp_scores[c] for c in bot_half if c in interp_scores]
        mit = float(np.mean(it_top)) if it_top else 0
        mib = float(np.mean(it_bot)) if it_bot else 0
        if len(it_top) >= 2 and len(it_bot) >= 2:
            t_stat, t_p = scipy_stats.ttest_ind(it_top, it_bot)
        else:
            t_stat, t_p = 0, 1

        status = ("confirmed" if mit > mib and t_p < 0.05
                  else "partially_supported" if mit > mib
                  else "refuted")
        ev["criterion_5"] = {
            "description": "Localized caps have higher token-interpretability",
            "status": status,
            "interp_scores": interp_scores,
            "top_half_mean": mit, "bot_half_mean": mib,
            "t_stat": float(t_stat), "t_p": float(t_p),
        }
        print(f"  top-half interp={mit:.4f}, bot-half interp={mib:.4f}")
        print(f"  t={t_stat:.2f} p={t_p:.4f}")
        print(f"  → {status}")

    except Exception as e:
        ev["criterion_5"] = {"status": "error", "error": str(e)}
        print(f"  error: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "success_criteria_evaluation.json", "w") as f:
        json.dump(ev, f, indent=2)

    confirmed = sum(1 for v in ev.values()
                    if isinstance(v, dict) and v.get("status") == "confirmed")
    partial = sum(1 for v in ev.values()
                  if isinstance(v, dict)
                  and v.get("status") == "partially_supported")
    print(f"\n  Summary: {confirmed} confirmed, {partial} partially supported, "
          f"{5 - confirmed - partial} other")
    return ev


def build_results_json():
    """Aggregate everything into the top-level results.json."""
    print("\n=== Building results.json ===")

    fli = load("fli_scores.json")
    peaks = load("peak_layers.json")
    causal = load("causal_validation.json")
    dm = load("dark_matter_probes.json")
    overlap = load("capability_overlap.json")
    meta = load("sae_metadata.json")
    ev = load("success_criteria_evaluation.json")

    try:
        arch = load("architecture_comparison.json")
    except FileNotFoundError:
        arch = {}
    try:
        attr_abl = load("attribution_ablation.json")
    except FileNotFoundError:
        attr_abl = {}
    try:
        topk = load("topk_sensitivity.json")
    except FileNotFoundError:
        topk = {}

    results = {
        "title": "The Functional Anatomy of Sparse Features in Language Models",
        "model": "GPT-2 Small (124M parameters)",
        "sae_release": "gpt2-small-resid-post-v5-32k",
        "sae_type": "TopK",
        "sae_d_sae": int(meta["0"]["d_sae"]),
        "capabilities": CAPS,
        "seeds": [42, 123, 456],
        "n_examples_attribution": 300,
        "n_examples_causal": 200,
        "fli_scores": fli,
        "peak_layers": peaks,
        "causal_validation": causal,
        "dark_matter_probes": dm,
        "capability_overlap": overlap,
        "architecture_comparison": arch,
        "attribution_ablation": attr_abl,
        "topk_sensitivity": topk,
        "success_criteria": ev,
    }

    with open(ROOT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  → results.json written")
    return results


def write_findings_summary(ev):
    """Write an honest narrative summary of findings."""
    lines = ["# Findings Summary\n"]

    # Check for degenerate capabilities
    c1 = ev.get("criterion_1", {})
    degenerate = c1.get("degenerate_caps", [])
    if degenerate:
        lines.append("## Important Caveats\n")
        lines.append(f"The following capabilities produced **degenerate results** "
                     f"with near-uniform feature attribution: {degenerate}.")
        lines.append("This means the capability-specific loss function did not "
                     "produce meaningful gradients for these capabilities, "
                     "making their FLI, overlap, and causal results unreliable.")
        lines.append("All success criteria are evaluated on non-degenerate "
                     "capabilities only.\n")

    lines.append("## Success Criteria Evaluation\n")
    for key in sorted(ev):
        d = ev[key]
        if not isinstance(d, dict):
            continue
        status = d.get("status", "?")
        desc = d.get("description", key)
        lines.append(f"### {key}: {desc}")
        lines.append(f"**Status: {status}**\n")

        if "ratio_non_degenerate" in d:
            lines.append(f"- FLI ratio (non-degenerate): {d['ratio_non_degenerate']:.2f}")
        elif "ratio" in d:
            lines.append(f"- FLI ratio: {d['ratio']:.2f}")
        if "spearman_rho_fidelity" in d:
            lines.append(f"- Spearman ρ(FLI, capped fidelity) = "
                         f"{d['spearman_rho_fidelity']:.3f} "
                         f"(p = {d['spearman_p_fidelity']:.4f})")
            lines.append(f"- Spearman ρ(FLI, norm. drop diff) = "
                         f"{d['spearman_rho_ndd']:.3f} "
                         f"(p = {d['spearman_p_ndd']:.4f})")
        elif "spearman_rho" in d:
            lines.append(f"- Spearman ρ = {d['spearman_rho']:.3f} "
                         f"(p = {d['spearman_p']:.4f})")
        if "drop_ratio" in d:
            lines.append(f"- Drop ratio (more/less localized): "
                         f"{d['drop_ratio']:.2f}")
        if "note" in d and d["note"]:
            lines.append(f"- Note: {d['note']}")
        lines.append("")

    lines.append("## Key Takeaways\n")
    confirmed = [k for k, v in ev.items()
                 if isinstance(v, dict) and v.get("status") == "confirmed"]
    partial = [k for k, v in ev.items()
               if isinstance(v, dict)
               and v.get("status") == "partially_supported"]
    refuted = [k for k, v in ev.items()
               if isinstance(v, dict) and v.get("status") == "refuted"]
    if confirmed:
        lines.append(f"- **Confirmed** ({len(confirmed)}): "
                     + ", ".join(confirmed))
    if partial:
        lines.append(f"- **Partially supported** ({len(partial)}): "
                     + ", ".join(partial))
    if refuted:
        lines.append(f"- **Refuted** ({len(refuted)}): "
                     + ", ".join(refuted))
    lines.append("")

    with open(RESULTS_DIR / "findings_summary.md", "w") as f:
        f.write("\n".join(lines))
    print("  → findings_summary.md written")


def main():
    ev = evaluate_success_criteria()
    build_results_json()
    write_findings_summary(ev)
    print("\nDone.")


if __name__ == "__main__":
    main()
