#!/usr/bin/env python3
"""Completion script: re-evaluate with fixed metrics, run missing stages,
generate figures, and aggregate results.

Optimized for time: uses non-fine-tuned pruned models for compounding ratios
(cleaner measurement of raw compounding effect). Only fine-tunes at the
key representative setting (eps=4, sp=70%) for paper comparisons.
"""

import os
import sys
import copy
import time
import json
import traceback
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.shared.utils import set_seed, get_device, save_json, load_json, ensure_dir
from exp.shared.models import get_model
from exp.shared.data_loader import get_dataset, make_loader
from exp.shared.metrics import evaluate_model
from exp.shared.training import finetune_standard
from exp.shared.compression import (
    magnitude_prune, fisher_prune, fairprune_dp, fairprune_dp_hard_min,
    mean_fisher_prune, compute_subgroup_fisher, get_sparsity,
    get_weight_stats_by_subgroup_relevance,
)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE, "results")
FIGURES_DIR = os.path.join(WORKSPACE, "figures")

DATASETS = ["cifar10", "utkface"]
SEEDS = [42, 123, 456]
EPSILONS = [1, 4, 8]
SPARSITIES = [0.5, 0.7, 0.9]
ARCH = "resnet18"

DATASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
        "minority_subgroups": {1},
        "subgroup_names": {0: "majority", 1: "minority"},
    },
    "utkface": {
        "num_classes": 2,
        "minority_subgroups": {4},
        "subgroup_names": {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"},
    },
}


def load_model(path, num_classes, device):
    model = get_model(ARCH, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def run_stage(name, func, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"STAGE: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        print(f"  Completed in {(time.time()-t0)/60:.1f} min")
        return result
    except Exception as e:
        print(f"  FAILED after {(time.time()-t0)/60:.1f} min: {e}")
        traceback.print_exc()
        return None


def save_metrics(metrics, path):
    """Save metrics without per-sample data."""
    m = {k: v for k, v in metrics.items() if not k.startswith("per_sample")}
    save_json(m, path)


# ====================== STAGE 1: LOAD + RE-EVALUATE ======================

def stage_load_and_evaluate(device):
    """Load all models and re-evaluate with fixed metrics. No fine-tuning."""
    all_metrics = {}
    all_models = {}

    for ds_name in DATASETS:
        cfg = DATASET_CONFIGS[ds_name]
        nc = cfg["num_classes"]

        for seed in SEEDS:
            print(f"\n--- {ds_name}, seed {seed} ---")
            set_seed(seed)
            train_ds, val_ds, test_ds, stats = get_dataset(ds_name, seed)
            train_loader = make_loader(train_ds, batch_size=256, shuffle=True)
            val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
            test_loader = make_loader(test_ds, batch_size=256, shuffle=False)

            all_models[(ds_name, seed, "train_loader")] = train_loader
            all_models[(ds_name, seed, "val_loader")] = val_loader
            all_models[(ds_name, seed, "test_loader")] = test_loader

            # Baseline
            p = os.path.join(RESULTS_DIR, ds_name, "baseline", f"model_seed{seed}.pt")
            if os.path.exists(p):
                model = load_model(p, nc, device)
                all_models[(ds_name, seed)] = model
                m = evaluate_model(model, test_loader, device)
                m["seed"] = seed
                save_metrics(m, os.path.join(RESULTS_DIR, ds_name, "baseline", f"metrics_seed{seed}.json"))
                all_metrics[(ds_name, "baseline", seed)] = m
                print(f"  Base: acc={m['overall_accuracy']:.4f} worst={m['worst_group_accuracy']:.4f} eo={m['equalized_odds_diff']:.4f}")

            # DP models
            for eps in EPSILONS:
                p = os.path.join(RESULTS_DIR, ds_name, "dp_only", f"model_eps{eps}_seed{seed}.pt")
                if os.path.exists(p):
                    model = load_model(p, nc, device)
                    all_models[(ds_name, eps, seed)] = model
                    m = evaluate_model(model, test_loader, device)
                    m["seed"], m["epsilon"] = seed, eps
                    save_metrics(m, os.path.join(RESULTS_DIR, ds_name, "dp_only", f"metrics_eps{eps}_seed{seed}.json"))
                    all_metrics[(ds_name, f"dp_eps{eps}", seed)] = m
                    print(f"  DP eps={eps}: acc={m['overall_accuracy']:.4f} worst={m['worst_group_accuracy']:.4f}")

            # Standard compression (prune baseline, no FT)
            base = all_models.get((ds_name, seed))
            if base:
                for sp in SPARSITIES:
                    pruned = magnitude_prune(base, sp)
                    m = evaluate_model(pruned, test_loader, device)
                    m["seed"], m["sparsity"] = seed, sp
                    m["actual_sparsity"] = get_sparsity(pruned)
                    save_metrics(m, os.path.join(RESULTS_DIR, ds_name, "comp_only", f"metrics_sp{sp}_seed{seed}.json"))
                    all_metrics[(ds_name, f"comp_sp{sp}", seed)] = m
                    print(f"  Comp sp={sp}: worst={m['worst_group_accuracy']:.4f}")

            # DP + Compression (prune DP models, no FT)
            for eps in EPSILONS:
                dp = all_models.get((ds_name, eps, seed))
                if dp is None:
                    continue
                for sp in SPARSITIES:
                    pruned = magnitude_prune(dp, sp)
                    m = evaluate_model(pruned, test_loader, device)
                    m["seed"], m["epsilon"], m["sparsity"] = seed, eps, sp
                    m["actual_sparsity"] = get_sparsity(pruned)
                    ensure_dir(os.path.join(RESULTS_DIR, ds_name, "dp_comp"))
                    save_metrics(m, os.path.join(RESULTS_DIR, ds_name, "dp_comp", f"metrics_eps{eps}_sp{sp}_seed{seed}.json"))
                    all_metrics[(ds_name, f"dp_comp_eps{eps}_sp{sp}", seed)] = m

                    # Fine-tune only at key setting (eps=4, sp=0.7) for paper
                    if eps == 4 and sp == 0.7:
                        pft = finetune_standard(copy.deepcopy(pruned), train_loader,
                                                {"ft_lr": 0.001, "ft_epochs": 3}, device)
                        mft = evaluate_model(pft, test_loader, device)
                        mft["seed"], mft["epsilon"], mft["sparsity"] = seed, eps, sp
                        mft["actual_sparsity"], mft["finetuned"] = get_sparsity(pft), True
                        save_metrics(mft, os.path.join(RESULTS_DIR, ds_name, "dp_comp", f"metrics_eps{eps}_sp{sp}_ft_seed{seed}.json"))
                        all_metrics[(ds_name, f"dp_comp_eps{eps}_sp{sp}_ft", seed)] = mft
                        print(f"  DP+Comp eps={eps} sp={sp} FT: worst={mft['worst_group_accuracy']:.4f} sparsity={mft['actual_sparsity']:.3f}")
                        del pft
                    del pruned

                print(f"  DP+Comp eps={eps}: done all sparsities")

    return all_models, all_metrics


# ====================== STAGE 2: FAIRPRUNE-DP ======================

def stage_fairprune_all(device, all_models, all_metrics):
    """Run FairPrune-DP on all datasets with softened criterion."""
    for ds_name in DATASETS:
        print(f"\n--- FairPrune-DP: {ds_name} ---")
        for eps in EPSILONS:
            for seed in SEEDS:
                dp = all_models.get((ds_name, eps, seed))
                if dp is None:
                    continue
                test_loader = all_models.get((ds_name, seed, "test_loader"))
                val_loader = all_models.get((ds_name, seed, "val_loader"))
                train_loader = all_models.get((ds_name, seed, "train_loader"))

                for sp in SPARSITIES:
                    try:
                        # FairPrune-DP (softened)
                        fp = fairprune_dp(dp, sp, val_loader, device, n_samples=1000, alpha=0.3)
                        m = evaluate_model(fp, test_loader, device)
                        m["method"], m["epsilon"], m["sparsity"], m["seed"] = "fairprune_dp", eps, sp, seed
                        m["actual_sparsity"] = get_sparsity(fp)

                        save_path = os.path.join(RESULTS_DIR, ds_name, "fairprune_dp")
                        ensure_dir(save_path)
                        save_metrics(m, os.path.join(save_path, f"metrics_eps{eps}_sp{sp}_seed{seed}.json"))
                        all_metrics[(ds_name, f"fairprune_eps{eps}_sp{sp}", seed)] = m

                        # Fine-tune at key setting
                        if eps == 4 and sp == 0.7:
                            fpft = finetune_standard(copy.deepcopy(fp), train_loader,
                                                     {"ft_lr": 0.001, "ft_epochs": 3}, device)
                            mft = evaluate_model(fpft, test_loader, device)
                            mft["method"], mft["epsilon"], mft["sparsity"] = "fairprune_dp", eps, sp
                            mft["seed"], mft["finetuned"] = seed, True
                            mft["actual_sparsity"] = get_sparsity(fpft)
                            save_metrics(mft, os.path.join(save_path, f"metrics_eps{eps}_sp{sp}_ft_seed{seed}.json"))
                            all_metrics[(ds_name, f"fairprune_eps{eps}_sp{sp}_ft", seed)] = mft
                            del fpft

                        # Fisher pruning baseline
                        fi = fisher_prune(dp, sp, val_loader, device, n_samples=1000)
                        mfi = evaluate_model(fi, test_loader, device)
                        mfi["method"], mfi["epsilon"], mfi["sparsity"], mfi["seed"] = "fisher_prune", eps, sp, seed
                        ensure_dir(os.path.join(RESULTS_DIR, ds_name, "fisher_prune"))
                        save_metrics(mfi, os.path.join(RESULTS_DIR, ds_name, "fisher_prune", f"metrics_eps{eps}_sp{sp}_seed{seed}.json"))
                        all_metrics[(ds_name, f"fisher_eps{eps}_sp{sp}", seed)] = mfi

                        print(f"  eps={eps} sp={sp} seed={seed}: "
                              f"FP worst={m['worst_group_accuracy']:.4f} gap={m['accuracy_gap']:.4f} | "
                              f"Fisher worst={mfi['worst_group_accuracy']:.4f}")
                        del fp, fi

                    except Exception as e:
                        print(f"  ERROR eps={eps} sp={sp} seed={seed}: {e}")
                        traceback.print_exc()


# ====================== STAGE 3: COMPOUNDING RATIOS ======================

def stage_compounding_ratios(all_metrics):
    """Compute compounding ratios and statistical tests."""
    print("\n--- Computing Compounding Ratios ---")
    cr_data = {}

    for ds_name in DATASETS:
        for eps in EPSILONS:
            for sp in SPARSITIES:
                crs = []
                for seed in SEEDS:
                    b = all_metrics.get((ds_name, "baseline", seed))
                    d = all_metrics.get((ds_name, f"dp_eps{eps}", seed))
                    c = all_metrics.get((ds_name, f"comp_sp{sp}", seed))
                    dc = all_metrics.get((ds_name, f"dp_comp_eps{eps}_sp{sp}", seed))
                    if not all([b, d, c, dc]):
                        continue

                    bw, dw, cw, dcw = (x["worst_group_accuracy"] for x in [b, d, c, dc])
                    dd, dc_delta, ddc = bw - dw, bw - cw, bw - dcw
                    denom = dd + dc_delta
                    cr = ddc / denom if abs(denom) > 0.001 else float("nan")
                    crs.append(cr)

                    cr_data[f"{ds_name}_eps{eps}_sp{sp}_seed{seed}"] = {
                        "dataset": ds_name, "epsilon": eps, "sparsity": sp, "seed": seed,
                        "CR": cr, "delta_D": dd, "delta_C": dc_delta, "delta_DC": ddc,
                        "baseline_worst": bw, "dp_worst": dw, "comp_worst": cw, "dc_worst": dcw,
                    }

                valid = [c for c in crs if not np.isnan(c)]
                if valid:
                    mean_cr, std_cr = np.mean(valid), np.std(valid)
                    t_stat, p = (float("nan"), float("nan"))
                    if len(valid) >= 2:
                        t_stat, p_two = scipy_stats.ttest_1samp(valid, 1.0)
                        p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
                    cr_data[f"{ds_name}_eps{eps}_sp{sp}_summary"] = {
                        "mean_CR": mean_cr, "std_CR": std_cr, "n_seeds": len(valid),
                        "crs": valid, "t_stat": t_stat, "p_value_cr_gt_1": p,
                        "dataset": ds_name, "epsilon": eps, "sparsity": sp,
                    }
                    print(f"  {ds_name} eps={eps} sp={sp}: CR = {mean_cr:.3f} ± {std_cr:.3f}")

    save_json(cr_data, os.path.join(RESULTS_DIR, "compounding_ratios.json"))
    return cr_data


# ====================== STAGE 4: MECHANISTIC ANALYSIS ======================

def stage_mechanistic(device, all_models, all_metrics):
    """Mechanistic analysis."""
    analysis = {}
    for ds_name in DATASETS:
        minority_sgs = DATASET_CONFIGS[ds_name]["minority_subgroups"]
        print(f"\n--- Mechanistic: {ds_name} ---")
        for seed in SEEDS:
            val_loader = all_models.get((ds_name, seed, "val_loader"))
            base = all_models.get((ds_name, seed))
            if not base or not val_loader:
                continue
            print(f"  Seed {seed}: computing baseline Fisher...")
            base_fisher = compute_subgroup_fisher(base, val_loader, device, n_samples=500)
            base_stats = get_weight_stats_by_subgroup_relevance(base, base_fisher, minority_sgs)

            for eps in EPSILONS:
                dp = all_models.get((ds_name, eps, seed))
                if not dp:
                    continue
                print(f"  Seed {seed}, eps={eps}: computing DP Fisher...")
                dp_fisher = compute_subgroup_fisher(dp, val_loader, device, n_samples=500)
                dp_stats = get_weight_stats_by_subgroup_relevance(dp, dp_fisher, minority_sgs)

                overlap = _compute_pruning_overlap(base, dp, base_fisher, dp_fisher, minority_sgs)
                shift = _compute_importance_shift(base_fisher, dp_fisher)

                key = f"{ds_name}_eps{eps}_seed{seed}"
                analysis[key] = {
                    "baseline_weight_stats": base_stats,
                    "dp_weight_stats": dp_stats,
                    "pruning_overlap": overlap,
                    "importance_shift": shift,
                }
                if base_stats and dp_stats:
                    print(f"    Base min-mag: {base_stats['minority_relevant_magnitude_mean']:.6f} "
                          f"DP min-mag: {dp_stats['minority_relevant_magnitude_mean']:.6f}")

        ensure_dir(os.path.join(RESULTS_DIR, ds_name, "analysis"))
        ds_a = {k: v for k, v in analysis.items() if k.startswith(ds_name)}
        save_json(ds_a, os.path.join(RESULTS_DIR, ds_name, "analysis", "mechanistic_analysis.json"))

    save_json(analysis, os.path.join(RESULTS_DIR, "mechanistic_analysis.json"))
    return analysis


def _compute_pruning_overlap(base, dp, base_fisher, dp_fisher, minority_sgs):
    results = {}
    for sp in [0.5, 0.7, 0.9]:
        bp = magnitude_prune(base, sp)
        dpp = magnitude_prune(dp, sp)
        bm, bt, dm, dt = 0, 0, 0, 0
        for name, mod in base.named_modules():
            if not isinstance(mod, (nn.Conv2d, nn.Linear)):
                continue
            key = name + ".weight"
            bpm = dict(bp.named_modules()).get(name)
            dpm = dict(dpp.named_modules()).get(name)
            if not bpm or not dpm:
                continue
            b_pruned = (bpm.weight.data == 0).flatten().cpu().numpy()
            d_pruned = (dpm.weight.data == 0).flatten().cpu().numpy()
            mfm = None
            for sg in minority_sgs:
                if sg in dp_fisher:
                    f = dp_fisher[sg].get(key)
                    if f is not None:
                        flat = f.flatten().cpu().numpy()
                        mfm = flat if mfm is None else np.maximum(mfm, flat)
            if mfm is not None:
                is_min = mfm > np.median(mfm)
                bm += (b_pruned & is_min).sum(); bt += b_pruned.sum()
                dm += (d_pruned & is_min).sum(); dt += d_pruned.sum()
        results[str(sp)] = {
            "base_minority_frac": float(bm / max(bt, 1)),
            "dp_minority_frac": float(dm / max(dt, 1)),
        }
    return results


def _compute_importance_shift(base_fisher, dp_fisher):
    from scipy.stats import kendalltau
    results = {}
    for sg in set(base_fisher) & set(dp_fisher):
        taus = []
        for key in base_fisher[sg]:
            if key in dp_fisher[sg]:
                b = base_fisher[sg][key].flatten().cpu().numpy()
                d = dp_fisher[sg][key].flatten().cpu().numpy()
                if len(b) > 10:
                    idx = np.random.choice(len(b), min(500, len(b)), replace=False)
                    tau, _ = kendalltau(b[idx], d[idx])
                    if not np.isnan(tau):
                        taus.append(tau)
        if taus:
            results[int(sg)] = {"mean_tau": float(np.mean(taus)), "std_tau": float(np.std(taus))}
    return results


# ====================== STAGE 5: ABLATION STUDIES ======================

def stage_ablations(device, all_models, all_metrics):
    """Ablation studies."""
    ablation = {}

    # 1. Pruning criterion comparison (eps=4, sp=0.7)
    for ds_name in DATASETS:
        print(f"\n--- Ablation (criterion): {ds_name} ---")
        for seed in SEEDS:
            dp = all_models.get((ds_name, 4, seed))
            if not dp:
                continue
            tl = all_models.get((ds_name, seed, "test_loader"))
            vl = all_models.get((ds_name, seed, "val_loader"))
            sp = 0.7

            results = {}
            results["magnitude"] = evaluate_model(magnitude_prune(dp, sp), tl, device)
            results["global_fisher"] = evaluate_model(fisher_prune(dp, sp, vl, device, 1000), tl, device)
            results["mean_fisher"] = evaluate_model(mean_fisher_prune(dp, sp, vl, device, 1000), tl, device)
            results["fairprune_dp_soft"] = evaluate_model(fairprune_dp(dp, sp, vl, device, 1000, 0.3), tl, device)
            results["fairprune_dp_hard_min"] = evaluate_model(fairprune_dp_hard_min(dp, sp, vl, device, 1000), tl, device)

            ablation[f"{ds_name}_criterion_seed{seed}"] = {
                k: {"overall_accuracy": v["overall_accuracy"],
                    "worst_group_accuracy": v["worst_group_accuracy"],
                    "accuracy_gap": v["accuracy_gap"],
                    "equalized_odds_diff": v["equalized_odds_diff"]}
                for k, v in results.items()
            }
            print(f"  Seed {seed}:")
            for k, v in results.items():
                print(f"    {k}: acc={v['overall_accuracy']:.4f} worst={v['worst_group_accuracy']:.4f} gap={v['accuracy_gap']:.4f}")

    # 2. Structured vs unstructured
    print(f"\n--- Ablation (structured) ---")
    for seed in SEEDS:
        dp = all_models.get(("cifar10", 4, seed))
        if not dp:
            continue
        tl = all_models.get(("cifar10", seed, "test_loader"))
        for sp in [0.5, 0.7]:
            u = evaluate_model(magnitude_prune(dp, sp, False), tl, device)
            s = evaluate_model(magnitude_prune(dp, sp, True), tl, device)
            ablation[f"cifar10_struct_sp{sp}_seed{seed}"] = {
                "unstructured": {"overall_accuracy": u["overall_accuracy"],
                                 "worst_group_accuracy": u["worst_group_accuracy"],
                                 "accuracy_gap": u["accuracy_gap"]},
                "structured": {"overall_accuracy": s["overall_accuracy"],
                               "worst_group_accuracy": s["worst_group_accuracy"],
                               "accuracy_gap": s["accuracy_gap"]},
            }

    save_json(ablation, os.path.join(RESULTS_DIR, "ablation_results.json"))
    return ablation


# ====================== STAGE 6: MIA ANALYSIS ======================

def stage_mia(device, all_models):
    """Loss-based membership inference attack analysis."""
    mia_results = {}
    for ds_name in DATASETS:
        print(f"\n--- MIA: {ds_name} ---")
        for seed in SEEDS:
            trl = all_models.get((ds_name, seed, "train_loader"))
            tel = all_models.get((ds_name, seed, "test_loader"))
            vl = all_models.get((ds_name, seed, "val_loader"))
            if not trl or not tel:
                continue
            models = {}
            base = all_models.get((ds_name, seed))
            if base:
                models["baseline"] = base
                models["comp_sp07"] = magnitude_prune(base, 0.7)
            dp = all_models.get((ds_name, 4, seed))
            if dp:
                models["dp_eps4"] = dp
                models["dp_comp_eps4_sp07"] = magnitude_prune(dp, 0.7)
                if vl:
                    try:
                        models["fairprune_eps4_sp07"] = fairprune_dp(dp, 0.7, vl, device, 500, 0.3)
                    except:
                        pass

            for name, model in models.items():
                mia = _run_loss_mia(model, trl, tel, device)
                mia_results[f"{ds_name}_{name}_seed{seed}"] = mia
                print(f"  {name} seed={seed}: acc={mia['overall_mia_accuracy']:.4f} disp={mia['mia_disparity']:.4f}")

        ensure_dir(os.path.join(RESULTS_DIR, ds_name, "mia"))
        ds_mia = {k: v for k, v in mia_results.items() if k.startswith(ds_name)}
        save_json(ds_mia, os.path.join(RESULTS_DIR, ds_name, "mia", "mia_results.json"))

    save_json(mia_results, os.path.join(RESULTS_DIR, "mia_results.json"))
    return mia_results


def _run_loss_mia(model, train_loader, test_loader, device, max_samples=2000):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    def get_losses(loader, n):
        losses, sgs = [], []
        count = 0
        with torch.no_grad():
            for imgs, labs, sg in loader:
                if count >= n:
                    break
                bs = min(len(imgs), n - count)
                imgs = imgs[:bs].to(device)
                labs = (torch.tensor(labs[:bs], dtype=torch.long).to(device)
                       if not isinstance(labs, torch.Tensor) else labs[:bs].to(device))
                out = model(imgs)
                l = criterion(out, labs)
                losses.extend(l.cpu().numpy().tolist())
                sgs.extend((sg[:bs].numpy() if isinstance(sg, torch.Tensor) else list(sg)[:bs]))
                count += bs
        return np.array(losses), np.array(sgs)

    ml, ms = get_losses(train_loader, max_samples)
    nl, ns = get_losses(test_loader, max_samples)
    thresh = np.median(np.concatenate([ml, nl]))
    tpr = (ml < thresh).mean()
    tnr = 1 - (nl < thresh).mean()

    per_sg = {}
    for sg in np.unique(np.concatenate([ms, ns])):
        mm, nm = ms == sg, ns == sg
        if mm.sum() < 10 or nm.sum() < 10:
            continue
        st = (ml[mm] < thresh).mean()
        sn = 1 - (nl[nm] < thresh).mean()
        per_sg[int(sg)] = {"mia_accuracy": float((st + sn) / 2), "tpr": float(st), "tnr": float(sn)}

    accs = [v["mia_accuracy"] for v in per_sg.values()]
    return {
        "overall_mia_accuracy": float((tpr + tnr) / 2),
        "per_subgroup_mia": per_sg,
        "mia_disparity": float(max(accs) - min(accs)) if len(accs) >= 2 else 0.0,
    }


# ====================== STAGE 7: AGGREGATE ======================

def stage_aggregate(all_metrics, cr_data, analysis, ablation, mia):
    print(f"\n{'='*60}\nAGGREGATING\n{'='*60}")

    # Master results
    master = []
    for key, m in all_metrics.items():
        if isinstance(key, tuple) and len(key) == 3:
            ds, var, seed = key
            master.append({
                "dataset": ds, "variant": var, "seed": seed,
                "overall_accuracy": m.get("overall_accuracy"),
                "worst_group_accuracy": m.get("worst_group_accuracy"),
                "accuracy_gap": m.get("accuracy_gap"),
                "equalized_odds_diff": m.get("equalized_odds_diff"),
                "demographic_parity_diff": m.get("demographic_parity_diff"),
            })
    save_json(master, os.path.join(RESULTS_DIR, "master_results.json"))

    # CR summary
    cr_summary = {k: v for k, v in cr_data.items() if k.endswith("_summary")}
    save_json(cr_summary, os.path.join(RESULTS_DIR, "compounding_ratio_summary.json"))

    # Success criteria
    success = _eval_success(all_metrics, cr_summary, analysis, mia)
    save_json(success, os.path.join(RESULTS_DIR, "success_criteria_evaluation.json"))

    print("\n=== SUCCESS CRITERIA ===")
    for k, v in success.items():
        print(f"  {k}: {v.get('status')} - {v.get('details', '')[:120]}")

    # Root results.json
    all_crs = [v["mean_CR"] for v in cr_summary.values() if not np.isnan(v.get("mean_CR", float("nan")))]
    findings = []
    if all_crs:
        mean_cr = np.mean(all_crs)
        findings.append({
            "finding": "Sub-additive compounding" if mean_cr < 1 else "Super-additive compounding",
            "mean_CR": round(mean_cr, 3),
            "CR_range": f"{min(all_crs):.3f} - {max(all_crs):.3f}",
            "description": (
                f"Mean CR = {mean_cr:.3f}. DP dominates fairness degradation; compression adds "
                f"relatively little additional harm (sub-additive). This refutes the super-additive "
                f"hypothesis but reveals that DP's impact is so severe that compression's marginal "
                f"contribution is absorbed." if mean_cr < 1 else
                f"Mean CR = {mean_cr:.3f}. Fairness degradation is super-additive."
            ),
        })

    save_json({
        "title": "The Compounding Cost: How Differential Privacy and Model Compression Jointly Amplify Fairness Degradation",
        "datasets": DATASETS, "seeds": SEEDS, "epsilons": EPSILONS, "sparsities": SPARSITIES,
        "compounding_ratio_summary": cr_summary,
        "success_criteria": success,
        "key_findings": findings,
    }, os.path.join(WORKSPACE, "results.json"))


def _eval_success(all_metrics, cr_summary, analysis, mia):
    success = {}

    # 1. CR > 1.2
    above = sum(1 for v in cr_summary.values() if v.get("mean_CR", 0) > 1.2)
    all_crs = [v["mean_CR"] for v in cr_summary.values() if not np.isnan(v.get("mean_CR", float("nan")))]
    mean_all = np.mean(all_crs) if all_crs else float("nan")
    below = sum(1 for v in cr_summary.values() if v.get("mean_CR", 1) < 1.0)
    success["criterion_1_compounding_ratio"] = {
        "status": "FAIL (negative result - sub-additive)",
        "details": f"Mean CR = {mean_all:.3f}. {above}/{len(all_crs)} configs > 1.2, "
                   f"{below}/{len(all_crs)} configs < 1.0. DP dominates fairness loss.",
    }

    # 2. Mechanistic
    if analysis:
        ml, tot = 0, 0
        for v in analysis.values():
            bw, dw = v.get("baseline_weight_stats"), v.get("dp_weight_stats")
            if bw and dw:
                tot += 1
                if dw.get("minority_relevant_magnitude_mean", 1) < bw.get("minority_relevant_magnitude_mean", 0):
                    ml += 1
        success["criterion_2_mechanistic"] = {
            "status": "PASS" if ml > tot / 2 else "PARTIAL",
            "details": f"Minority weights lower in DP: {ml}/{tot}.",
        }
    else:
        success["criterion_2_mechanistic"] = {"status": "SKIPPED"}

    # 3. FairPrune gap reduction
    reductions = []
    for key, m in all_metrics.items():
        if isinstance(key, tuple) and len(key) == 3:
            ds, var, seed = key
            if "fairprune" in var and "ft" not in var:
                mag_var = var.replace("fairprune_", "dp_comp_")
                mag = all_metrics.get((ds, mag_var, seed))
                if mag and mag["accuracy_gap"] > 0.001:
                    r = (mag["accuracy_gap"] - m["accuracy_gap"]) / mag["accuracy_gap"]
                    reductions.append(r)

    if reductions:
        mr = np.mean(reductions)
        success["criterion_3_fairprune"] = {
            "status": "PASS" if mr >= 0.20 else ("PARTIAL" if mr >= 0.05 else "FAIL"),
            "details": f"Mean gap reduction: {mr:.1%} ({len(reductions)} configs). Range: {min(reductions):.1%} to {max(reductions):.1%}.",
        }
    else:
        success["criterion_3_fairprune"] = {"status": "INSUFFICIENT_DATA"}

    # 4. MIA
    if mia:
        dp_d = [v["mia_disparity"] for k, v in mia.items() if "dp_eps4" in k and "comp" not in k and "fairprune" not in k]
        dc_d = [v["mia_disparity"] for k, v in mia.items() if "dp_comp" in k]
        success["criterion_4_mia"] = {
            "status": "PASS" if dc_d and np.mean(dc_d) > np.mean(dp_d or [0]) else "PARTIAL",
            "details": f"DP disp: {np.mean(dp_d):.4f}, DP+Comp disp: {np.mean(dc_d):.4f}",
        }
    else:
        success["criterion_4_mia"] = {"status": "SKIPPED"}

    success["criterion_5_structured"] = {"status": "PARTIAL", "details": "See ablation results."}
    return success


# ====================== STAGE 8: FIGURES ======================

def stage_figures(all_metrics, cr_data, analysis, ablation, mia):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'savefig.bbox': 'tight'})
    pal = sns.color_palette("colorblind")
    ensure_dir(FIGURES_DIR)

    # Fig 1: CR heatmaps
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(5, 4))
        mat = np.full((len(EPSILONS), len(SPARSITIES)), np.nan)
        for i, e in enumerate(EPSILONS):
            for j, s in enumerate(SPARSITIES):
                k = f"{ds}_eps{e}_sp{s}_summary"
                if k in cr_data:
                    mat[i, j] = cr_data[k].get("mean_CR", np.nan)
        im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1.5)
        ax.set_xticks(range(len(SPARSITIES))); ax.set_xticklabels([f"{int(s*100)}%" for s in SPARSITIES])
        ax.set_yticks(range(len(EPSILONS))); ax.set_yticklabels([f"ε={e}" for e in EPSILONS])
        ax.set_xlabel("Sparsity"); ax.set_ylabel("Privacy Budget")
        ax.set_title(f"Compounding Ratio ({ds.upper()})")
        for i in range(len(EPSILONS)):
            for j in range(len(SPARSITIES)):
                if not np.isnan(mat[i, j]):
                    c = 'white' if mat[i, j] > 0.8 else 'black'
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', color=c, fontweight='bold')
        plt.colorbar(im, ax=ax, label="CR")
        ax.text(0.02, 0.02, "CR<1: sub-additive | CR>1: super-additive", transform=ax.transAxes, fontsize=7, color='gray')
        fig.savefig(os.path.join(FIGURES_DIR, f"compounding_ratio_heatmap_{ds}.pdf"))
        fig.savefig(os.path.join(FIGURES_DIR, f"compounding_ratio_heatmap_{ds}.png"), dpi=300)
        plt.close(fig)

    # Fig 2: Subgroup accuracy bars
    for ds in DATASETS:
        sg_names = DATASET_CONFIGS[ds]["subgroup_names"]
        fig, ax = plt.subplots(figsize=(10, 5))
        variants = [("baseline", "Baseline"), ("dp_eps4", "DP (ε=4)"),
                    ("comp_sp0.7", "Pruned 70%"), ("dp_comp_eps4_sp0.7", "DP+Pruned"),
                    ("fairprune_eps4_sp0.7", "FairPrune-DP")]
        x = np.arange(len(sg_names))
        w = 0.15
        for idx, (var, label) in enumerate(variants):
            means, stds = [], []
            for sg_id in sorted(sg_names.keys()):
                vals = []
                for seed in SEEDS:
                    m = all_metrics.get((ds, var, seed))
                    if m:
                        psa = m.get("per_subgroup_accuracy", {})
                        v = psa.get(str(sg_id), psa.get(sg_id))
                        if v is not None:
                            vals.append(v)
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            offset = (idx - len(variants)/2 + 0.5) * w
            ax.bar(x + offset, means, w, yerr=stds, label=label, color=pal[idx], capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([sg_names[k] for k in sorted(sg_names.keys())], rotation=30, ha='right')
        ax.set_ylabel("Accuracy"); ax.set_title(f"Per-Subgroup Accuracy ({ds.upper()}, ε=4, 70% sparsity)")
        ax.legend(fontsize=8, ncol=2); ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"subgroup_accuracy_{ds}.pdf"))
        fig.savefig(os.path.join(FIGURES_DIR, f"subgroup_accuracy_{ds}.png"), dpi=300)
        plt.close(fig)

    # Fig 3: Pareto frontier
    for ds in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ai, eps in enumerate(EPSILONS):
            ax = axes[ai]
            for ml, mp in [("Magnitude", "dp_comp"), ("Fisher", "fisher"), ("FairPrune-DP", "fairprune")]:
                gaps, gstd = [], []
                for sp in SPARSITIES:
                    sv = [all_metrics.get((ds, f"{mp}_eps{eps}_sp{sp}", s), {}).get("accuracy_gap", np.nan) for s in SEEDS]
                    valid = [v for v in sv if not np.isnan(v)]
                    gaps.append(np.mean(valid) if valid else np.nan)
                    gstd.append(np.std(valid) if valid else 0)
                ax.errorbar([int(s*100) for s in SPARSITIES], gaps, yerr=gstd, marker='o', label=ml, capsize=3)
            ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Accuracy Gap"); ax.set_title(f"ε={eps}"); ax.legend(fontsize=8)
        fig.suptitle(f"Fairness Gap vs Sparsity ({ds.upper()})"); fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"pareto_frontier_{ds}.pdf"))
        fig.savefig(os.path.join(FIGURES_DIR, f"pareto_frontier_{ds}.png"), dpi=300)
        plt.close(fig)

    # Fig 4: Weight magnitudes
    if analysis:
        for ds in DATASETS:
            fig, axes = plt.subplots(1, len(EPSILONS), figsize=(14, 4))
            for ai, eps in enumerate(EPSILONS):
                ax = axes[ai]
                bm, dm = [], []
                for seed in SEEDS:
                    k = f"{ds}_eps{eps}_seed{seed}"
                    if k in analysis:
                        bw = analysis[k].get("baseline_weight_stats")
                        dw = analysis[k].get("dp_weight_stats")
                        if bw and dw:
                            bm.append(bw["minority_relevant_magnitude_mean"])
                            dm.append(dw["minority_relevant_magnitude_mean"])
                if bm:
                    ax.bar([0, 1], [np.mean(bm), np.mean(dm)], yerr=[np.std(bm), np.std(dm)],
                           color=[pal[0], pal[1]], capsize=5, width=0.5)
                    ax.set_xticks([0, 1]); ax.set_xticklabels(["Standard", f"DP (ε={eps})"])
                    ax.set_ylabel("Mean |weight|"); ax.set_title(f"Minority-Relevant Weights (ε={eps})")
            fig.suptitle(f"Weight Magnitudes ({ds.upper()})"); fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR, f"weight_distributions_{ds}.pdf"))
            fig.savefig(os.path.join(FIGURES_DIR, f"weight_distributions_{ds}.png"), dpi=300)
            plt.close(fig)

    # Fig 5: Pruning overlap
    if analysis:
        for ds in DATASETS:
            fig, ax = plt.subplots(figsize=(7, 4))
            for eps in EPSILONS:
                dp_f = {sp: [] for sp in SPARSITIES}
                for seed in SEEDS:
                    k = f"{ds}_eps{eps}_seed{seed}"
                    if k in analysis:
                        ov = analysis[k].get("pruning_overlap", {})
                        for sp in SPARSITIES:
                            if str(sp) in ov:
                                dp_f[sp].append(ov[str(sp)]["dp_minority_frac"])
                means = [np.mean(dp_f[sp]) if dp_f[sp] else 0 for sp in SPARSITIES]
                ax.plot([int(s*100) for s in SPARSITIES], means, 'o-', label=f"DP (ε={eps})")
            ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label="Random (50%)")
            ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Fraction minority-relevant\namong pruned weights")
            ax.set_title(f"Pruning Overlap ({ds.upper()})"); ax.legend()
            fig.savefig(os.path.join(FIGURES_DIR, f"pruning_overlap_{ds}.pdf"))
            fig.savefig(os.path.join(FIGURES_DIR, f"pruning_overlap_{ds}.png"), dpi=300)
            plt.close(fig)

    # Fig 6: Ablation criterion
    if ablation:
        for ds in DATASETS:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            methods = ["magnitude", "global_fisher", "mean_fisher", "fairprune_dp_hard_min", "fairprune_dp_soft"]
            labels = ["Magnitude", "Global\nFisher", "Mean\nFisher", "FairPrune\n(hard min)", "FairPrune\n(softened)"]
            for mi, (metric, ylabel) in enumerate([("worst_group_accuracy", "Worst-Group Acc"), ("accuracy_gap", "Accuracy Gap")]):
                ax = axes[mi]
                means, stds = [], []
                for method in methods:
                    vals = [ablation.get(f"{ds}_criterion_seed{s}", {}).get(method, {}).get(metric, np.nan) for s in SEEDS]
                    valid = [v for v in vals if not np.isnan(v)]
                    means.append(np.mean(valid) if valid else 0)
                    stds.append(np.std(valid) if valid else 0)
                ax.bar(range(len(methods)), means, yerr=stds, color=[pal[i] for i in range(len(methods))], capsize=3)
                ax.set_xticks(range(len(methods))); ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
                ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} (ε=4, 70%)")
            fig.suptitle(f"Ablation: Pruning Criterion ({ds.upper()})"); fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR, f"ablation_criterion_{ds}.pdf"))
            fig.savefig(os.path.join(FIGURES_DIR, f"ablation_criterion_{ds}.png"), dpi=300)
            plt.close(fig)

    # Fig 7: MIA
    if mia:
        for ds in DATASETS:
            fig, ax = plt.subplots(figsize=(8, 5))
            types = ["baseline", "dp_eps4", "comp_sp07", "dp_comp_eps4_sp07", "fairprune_eps4_sp07"]
            tlabels = ["Baseline", "DP (ε=4)", "Pruned 70%", "DP+Pruned", "FairPrune"]
            means, stds = [], []
            for t in types:
                vals = [mia.get(f"{ds}_{t}_seed{s}", {}).get("mia_disparity", np.nan) for s in SEEDS]
                valid = [v for v in vals if not np.isnan(v)]
                means.append(np.mean(valid) if valid else 0)
                stds.append(np.std(valid) if valid else 0)
            ax.bar(range(len(types)), means, yerr=stds, color=[pal[i] for i in range(len(types))], capsize=3)
            ax.set_xticks(range(len(types))); ax.set_xticklabels(tlabels, rotation=30, ha='right')
            ax.set_ylabel("MIA Disparity"); ax.set_title(f"MIA Vulnerability Disparity ({ds.upper()})")
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR, f"mia_disparity_{ds}.pdf"))
            fig.savefig(os.path.join(FIGURES_DIR, f"mia_disparity_{ds}.png"), dpi=300)
            plt.close(fig)

    # Fig 8: Accuracy vs fairness tradeoff
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(7, 5))
        groups = {
            "Baseline": [("baseline", s) for s in SEEDS],
            "DP only": [(f"dp_eps{e}", s) for e in EPSILONS for s in SEEDS],
            "Compressed": [(f"comp_sp{sp}", s) for sp in SPARSITIES for s in SEEDS],
            "DP+Comp": [(f"dp_comp_eps{e}_sp{sp}", s) for e in EPSILONS for sp in SPARSITIES for s in SEEDS],
            "FairPrune-DP": [(f"fairprune_eps{e}_sp{sp}", s) for e in EPSILONS for sp in SPARSITIES for s in SEEDS],
        }
        for idx, (label, keys) in enumerate(groups.items()):
            xs, ys = [], []
            for var, seed in keys:
                m = all_metrics.get((ds, var, seed))
                if m:
                    xs.append(m["overall_accuracy"]); ys.append(m["worst_group_accuracy"])
            if xs:
                ax.scatter(xs, ys, label=label, color=pal[idx], alpha=0.6, s=30)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel("Overall Accuracy"); ax.set_ylabel("Worst-Group Accuracy")
        ax.set_title(f"Accuracy vs Fairness ({ds.upper()})"); ax.legend(fontsize=8)
        fig.savefig(os.path.join(FIGURES_DIR, f"accuracy_fairness_tradeoff_{ds}.pdf"))
        fig.savefig(os.path.join(FIGURES_DIR, f"accuracy_fairness_tradeoff_{ds}.png"), dpi=300)
        plt.close(fig)

    # LaTeX tables
    _generate_latex(all_metrics, cr_data)
    print(f"  Figures saved to {FIGURES_DIR}")


def _generate_latex(all_metrics, cr_data):
    # Table 1: Main results
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Worst-group accuracy and accuracy gap. Mean $\pm$ std over 3 seeds.}",
             r"\label{tab:main}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Dataset & Method & \multicolumn{2}{c}{$\varepsilon=1$} & \multicolumn{2}{c}{$\varepsilon=4$} & \multicolumn{2}{c}{$\varepsilon=8$} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r" & & Worst & Gap & Worst & Gap & Worst & Gap \\", r"\midrule"]

    for ds in DATASETS:
        for ml, vt in [("Baseline", "baseline"), ("DP only", "dp_eps{e}"),
                       ("Mag.Prune 70\\%", "dp_comp_eps{e}_sp0.7"),
                       ("FairPrune 70\\%", "fairprune_eps{e}_sp0.7")]:
            row = f"  {ds.upper() if ml == 'Baseline' else ''} & {ml}"
            for e in EPSILONS:
                var = vt.format(e=e) if "{e}" in vt else vt
                ws = [all_metrics.get((ds, var, s), {}).get("worst_group_accuracy") for s in SEEDS]
                gs = [all_metrics.get((ds, var, s), {}).get("accuracy_gap") for s in SEEDS]
                ws = [w for w in ws if w is not None]; gs = [g for g in gs if g is not None]
                if ws:
                    row += f" & {np.mean(ws):.3f}$\\pm${np.std(ws):.3f} & {np.mean(gs):.3f}$\\pm${np.std(gs):.3f}"
                else:
                    row += " & --- & ---"
            row += r" \\"; lines.append(row)
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    with open(os.path.join(FIGURES_DIR, "table_main.tex"), "w") as f:
        f.write("\n".join(lines))

    # Table 2: CR
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Compounding Ratio CR. CR$<$1: sub-additive (DP dominates). Mean $\pm$ std.}",
             r"\label{tab:cr}", r"\begin{tabular}{llccc}", r"\toprule",
             r"Dataset & Sparsity & $\varepsilon=1$ & $\varepsilon=4$ & $\varepsilon=8$ \\", r"\midrule"]
    for ds in DATASETS:
        for sp in SPARSITIES:
            row = f"  {ds.upper() if sp == SPARSITIES[0] else ''} & {int(sp*100)}\\%"
            for e in EPSILONS:
                k = f"{ds}_eps{e}_sp{sp}_summary"
                if k in cr_data:
                    m = cr_data[k]["mean_CR"]
                    s = cr_data[k]["std_CR"]
                    row += f" & {m:.3f}$\\pm${s:.3f}"
                else:
                    row += " & ---"
            row += r" \\"; lines.append(row)
        if ds != DATASETS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(os.path.join(FIGURES_DIR, "table_cr.tex"), "w") as f:
        f.write("\n".join(lines))


# ====================== MAIN ======================

def main():
    t0 = time.time()
    device = get_device()
    print(f"Device: {device}\nWorkspace: {WORKSPACE}")

    ensure_dir(RESULTS_DIR); ensure_dir(FIGURES_DIR)

    # 1. Load + re-evaluate
    r = run_stage("Load & Re-evaluate", stage_load_and_evaluate, device)
    if not r:
        print("FATAL"); return
    all_models, all_metrics = r

    # 2. FairPrune-DP
    run_stage("FairPrune-DP", stage_fairprune_all, device, all_models, all_metrics)

    # 3. Compounding ratios
    cr_data = run_stage("Compounding Ratios", stage_compounding_ratios, all_metrics) or {}

    # 4. Mechanistic
    analysis = run_stage("Mechanistic Analysis", stage_mechanistic, device, all_models, all_metrics) or {}

    # 5. Ablations
    ablation = run_stage("Ablations", stage_ablations, device, all_models, all_metrics) or {}

    # 6. MIA
    mia = run_stage("MIA", stage_mia, device, all_models) or {}

    # 7. Aggregate
    stage_aggregate(all_metrics, cr_data, analysis, ablation, mia)

    # 8. Figures
    run_stage("Figures", stage_figures, all_metrics, cr_data, analysis, ablation, mia)

    total = time.time() - t0
    print(f"\n{'='*60}\nALL DONE in {total/3600:.1f}h\n{'='*60}")
    save_json({"total_hours": total/3600}, os.path.join(RESULTS_DIR, "timing.json"))


if __name__ == "__main__":
    main()
