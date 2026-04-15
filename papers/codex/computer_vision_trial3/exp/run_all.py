from __future__ import annotations

import json
import resource
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from shared.adaptation import _run_stream, calibrate, run_experiment_grid
from shared.common import DOMAIN_ORDER_A, DOMAIN_ORDER_B, append_log, ensure_dir, mean_std_ci95, reset_path, save_json
from shared.proxy_benchmark import build_caches, create_splits


ROOT = Path(__file__).resolve().parents[1]
EXP_DIRS = [
    ROOT / "exp/00_environment",
    ROOT / "exp/01_data_prep",
    ROOT / "exp/02_calibration",
    ROOT / "exp/03_frozen",
    ROOT / "exp/04_topb_pixel",
    ROOT / "exp/05_raw_mask",
    ROOT / "exp/06_clip_verified",
    ROOT / "exp/07_ablations",
]


def prepare_dirs():
    generated_paths = [
        ROOT / "results.json",
        ROOT / "figures",
        ROOT / "exp/00_environment",
        ROOT / "exp/01_data_prep",
        ROOT / "exp/02_calibration",
        ROOT / "exp/03_frozen",
        ROOT / "exp/04_topb_pixel",
        ROOT / "exp/05_raw_mask",
        ROOT / "exp/06_clip_verified",
        ROOT / "exp/07_ablations",
    ]
    for path in generated_paths:
        reset_path(path)
    for path in EXP_DIRS:
        if path.name == "07_ablations":
            ensure_dir(path)
        else:
            ensure_dir(path / "logs")


def check_registered_assets():
    targets = {
        "cityscapes": ["Cityscapes", "cityscapes"],
        "acdc": ["ACDC", "acdc"],
        "cat_seg": ["CAT-Seg", "cat-seg", "cat_seg"],
        "mlmp": ["MLMP", "mlmp"],
    }
    search_root = Path("/home/zz865")
    findings = {}
    for key, needles in targets.items():
        matches = []
        for needle in needles:
            try:
                out = subprocess.check_output(
                    ["bash", "-lc", f"find {search_root} -maxdepth 5 -type d -iname '*{needle}*' | sed -n '1,20p'"],
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                out = ""
            if out:
                matches.extend([line for line in out.splitlines() if line])
        findings[key] = sorted(set(matches))
    report = {
        "registered_setup_available": False,
        "searched_root": str(search_root),
        "findings": findings,
        "conclusion": "CAT-Seg / Cityscapes / ACDC / MLMP assets were not available in this workspace, so Stage 2 was rerun as an explicit proxy feasibility study.",
    }
    save_json(ROOT / "exp/00_environment/compatibility_check.json", report)
    note = ROOT / "exp/00_environment/SKIPPED.md"
    note.write_text(
        "# Registered study not executable here\n\n"
        "The Stage 1 registered CAT-Seg + Cityscapes/Cityscapes-C/ACDC + MLMP setup is not present in this workspace.\n"
        "See `compatibility_check.json` for the search artifact. Stage 2 is therefore reported as a proxy feasibility study only.\n"
        "MLMP was not executed because no local MLMP code or checkpoint compatible with the registered setup was available here.\n"
    )
    return report


def freeze_env():
    lock_path = ROOT / "exp/00_environment/requirements_lock.txt"
    out = subprocess.check_output([str(ROOT / ".venv/bin/pip"), "freeze"], text=True)
    lock_path.write_text(out)
    note = {
        "python": subprocess.check_output([str(ROOT / ".venv/bin/python"), "-V"], text=True).strip(),
        "proxy_note": "Requested CAT-Seg/MLMP/Cityscapes/ACDC stack was absent locally. Executed a Pascal VOC proxy benchmark with synthetic domain shifts instead.",
        "gpu": subprocess.check_output(["bash", "-lc", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"], text=True).strip(),
        "cpu_count": int(subprocess.check_output(["bash", "-lc", "nproc"], text=True).strip()),
    }
    save_json(ROOT / "exp/00_environment/results.json", note)


def summarize_runs(runs):
    rows = []
    for run in runs:
        row = {
            "run_name": run["run_name"],
            "method": run["method"],
            "order": run["order"],
            "seed": run["seed"],
            "reduced": run["reduced"],
            "miou": run["metrics"]["miou"],
            "avg_transition_drop": run["avg_transition_drop"],
            "runtime_sec_per_image": run["runtime_sec_per_image"],
            "peak_vram_mb": run["peak_vram_mb"],
            "skip_ratio": run["skip_ratio"],
            "accepted_area_mean": run["accepted_area_mean"],
            "accepted_count_mean": run["accepted_count_mean"],
        }
        for domain, value in run["per_domain_miou"].items():
            row[f"{domain}_miou"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_main(df: pd.DataFrame):
    methods = ["frozen", "topb_pixel", "raw_mask", "clip_verified"]
    table = []
    for method in methods:
        sub = df[(df["method"] == method) & (~df["reduced"])]
        miou_by_seed = sub.groupby(["seed", "order"])["miou"].mean().groupby("seed").mean().tolist()
        row = mean_std_ci95(miou_by_seed)
        row.update(
            {
            "method": method,
            "miou_mean": row["mean"],
            "miou_std": row["std"],
            "miou_stderr": row["stderr"],
            "miou_ci95_low": row["ci95_low"],
            "miou_ci95_high": row["ci95_high"],
            "runtime_sec_per_image": float(sub["runtime_sec_per_image"].mean()),
            "peak_vram_mb": float(sub["peak_vram_mb"].mean()),
            "skip_ratio": float(sub["skip_ratio"].mean()),
            "avg_transition_drop": float(sub["avg_transition_drop"].mean()),
            }
        )
        for domain in DOMAIN_ORDER_A:
            domain_cols = [c for c in sub.columns if c == f"{domain}_miou"]
            row[f"{domain}_miou"] = float(sub[domain_cols[0]].mean()) if domain_cols else 0.0
        table.append(row)
    return pd.DataFrame(table)


def aggregate_ablations(df: pd.DataFrame):
    sub = df[df["reduced"]].copy()
    rows = []
    sub["ablation_name"] = sub.apply(_ablation_name, axis=1)
    for ablation_name, group in sub.groupby("ablation_name"):
        vals = group["miou"].tolist()
        summary = mean_std_ci95(vals)
        rows.append(
            {
                "ablation_name": ablation_name,
                "method": str(group["method"].iloc[0]),
                "miou_mean": summary["mean"],
                "miou_std": summary["std"],
                "miou_stderr": summary["stderr"],
                "miou_ci95_low": summary["ci95_low"],
                "miou_ci95_high": summary["ci95_high"],
                "runtime_sec_per_image": float(group["runtime_sec_per_image"].mean()),
                "skip_ratio": float(group["skip_ratio"].mean()),
                "avg_transition_drop": float(group["avg_transition_drop"].mean()),
                "accepted_area_mean": float(group["accepted_area_mean"].mean()),
                "accepted_count_mean": float(group["accepted_count_mean"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("ablation_name").reset_index(drop=True)


def _ablation_name(row: pd.Series) -> str:
    run_name = str(row.get("run_name", ""))
    if run_name.startswith("threshold_"):
        return run_name.rsplit("_seed", 1)[0]
    if "_budget_" in run_name:
        return run_name.rsplit("_seed", 1)[0]
    return str(row["method"])


def paired_tests(runs, method_a: str, method_b: str):
    out = {}
    for order in ["A", "B"]:
        a_runs = {
            r["seed"]: r
            for r in runs
            if r["method"] == method_a and r["order"] == order and not r["reduced"]
        }
        b_runs = {
            r["seed"]: r
            for r in runs
            if r["method"] == method_b and r["order"] == order and not r["reduced"]
        }
        common_seeds = sorted(set(a_runs).intersection(b_runs))
        if not common_seeds:
            continue
        av = []
        bv = []
        for seed in common_seeds:
            av.extend([x["miou"] for x in a_runs[seed]["per_image"]])
            bv.extend([x["miou"] for x in b_runs[seed]["per_image"]])
        stat = wilcoxon(av, bv, zero_method="pratt")
        out[order] = {
            "paired_seeds": common_seeds,
            "paired_image_count": len(av),
            "statistic": float(stat.statistic),
            "pvalue": float(stat.pvalue),
        }
    return out


def run_dry_run(root: Path, splits: dict, thresholds: dict):
    log_path = root / "exp/02_calibration/logs/dry_run.log"
    subset = dict(splits)
    ordered = []
    for domain in DOMAIN_ORDER_A:
        ordered.extend([s for s in splits["main"] if s["domain"] == domain])
    subset["main"] = ordered[:12]
    start = time.time()
    out = _run_stream(root, "dry_run_clip_verified_A_seed13", subset, thresholds, "A", 13, method="clip_verified")
    elapsed = time.time() - start
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    projection = {
        "dry_run_images": len(subset["main"]),
        "elapsed_sec": elapsed,
        "timing_breakdown_sec_per_image": out["timing_breakdown_sec_per_image"],
        "peak_vram_mb": out["peak_vram_mb"],
        "peak_cpu_ram_gb_estimate": float(peak_kb / 1024**2),
    }
    main_runs = 2 + 3 * 2 + 3 * 2 + 3 * 2
    reduced_runs = 3 + 9 + 18
    projection["projected_gpu_hours"] = float((out["runtime_sec_per_image"] * (144 * main_runs + 72 * reduced_runs)) / 3600.0)
    projection["timing_gate_passed"] = projection["projected_gpu_hours"] <= 7.0
    append_log(log_path, str(projection))
    save_json(root / "exp/02_calibration/dry_run_timing.json", projection)
    return projection


def write_stage_results(runs: list[dict]):
    mapping = {
        "03_frozen": "frozen",
        "04_topb_pixel": "topb_pixel",
        "05_raw_mask": "raw_mask",
        "06_clip_verified": "clip_verified",
        "07_ablations": None,
    }
    for exp_name, method in mapping.items():
        if method is None:
            selected = [r for r in runs if r["reduced"]]
        else:
            selected = [r for r in runs if r["method"] == method and not r["reduced"]]
        save_json(ROOT / f"exp/{exp_name}/results.json", {"runs": selected})


def make_figures(runs, df: pd.DataFrame, thresholds: dict):
    fig_root = ROOT / "figures"
    fig_root.mkdir(exist_ok=True, parents=True)

    for order in ["A", "B"]:
        plt.figure(figsize=(10, 4))
        for method in ["frozen", "topb_pixel", "raw_mask", "clip_verified"]:
            run = next(r for r in runs if r["method"] == method and r["order"] == order and r["seed"] == 13 and not r["reduced"])
            plt.plot([x["miou"] for x in run["per_image"]], label=method)
        for idx in [24, 48, 72, 96, 120]:
            plt.axvline(idx, color="gray", linestyle="--", linewidth=0.8)
        plt.xlabel("Image index")
        plt.ylabel("Per-image mIoU")
        plt.title(f"Per-image mIoU trajectory, order {order}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_root / f"trajectory_order_{order}.png", dpi=200)
        plt.savefig(fig_root / f"trajectory_order_{order}.pdf")
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["margin", "entropy", "agreement", "budget"], [thresholds["margin_threshold"], thresholds["entropy_threshold"], thresholds["agreement_threshold"], thresholds["budget_b"]])
    plt.title("Calibrated thresholds")
    plt.tight_layout()
    plt.savefig(fig_root / "calibration_thresholds.png", dpi=200)
    plt.savefig(fig_root / "calibration_thresholds.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    budget_runs = [r for r in runs if "budget_" in r["run_name"] and r["method"] == "clip_verified"]
    x = sorted({float(r["run_name"].split("_")[-2]) for r in budget_runs})
    y = []
    for val in x:
        vals = [r["metrics"]["miou"] for r in budget_runs if f"_{val:.1f}_" in r["run_name"]]
        y.append(float(np.mean(vals)))
    plt.plot(x, y, marker="o")
    plt.xlabel("Budget scale")
    plt.ylabel("Reduced-slice mIoU")
    plt.title("Budget sensitivity")
    plt.tight_layout()
    plt.savefig(fig_root / "budget_sensitivity.png", dpi=200)
    plt.savefig(fig_root / "budget_sensitivity.pdf")
    plt.close()

    plt.figure(figsize=(6, 4))
    threshold_runs = [r for r in runs if r["run_name"].startswith("threshold_")]
    threshold_x = sorted({float(r["run_name"].split("_")[1]) for r in threshold_runs})
    threshold_y = []
    for val in threshold_x:
        vals = [r["metrics"]["miou"] for r in threshold_runs if r["run_name"].startswith(f"threshold_{val:+.2f}")]
        threshold_y.append(float(np.mean(vals)))
    plt.plot(threshold_x, threshold_y, marker="o")
    plt.xlabel("Margin offset")
    plt.ylabel("Reduced-slice mIoU")
    plt.title("Threshold sensitivity")
    plt.tight_layout()
    plt.savefig(fig_root / "threshold_sensitivity.png", dpi=200)
    plt.savefig(fig_root / "threshold_sensitivity.pdf")
    plt.close()

    main = aggregate_main(df)
    plt.figure(figsize=(7, 4))
    plt.bar(
        main["method"],
        main["miou_mean"],
        yerr=[main["miou_mean"] - main["miou_ci95_low"], main["miou_ci95_high"] - main["miou_mean"]],
        capsize=4,
    )
    plt.ylabel("Mean-over-orders mIoU")
    plt.title("Main comparison with 95% CI")
    plt.tight_layout()
    plt.savefig(fig_root / "main_error_bars.png", dpi=200)
    plt.savefig(fig_root / "main_error_bars.pdf")
    plt.close()


def measured_runtime_hours(runs: list[dict], dry_run: dict) -> dict:
    main_hours = float(sum(r["runtime_sec_per_image"] * len(r["per_image"]) for r in runs if not r["reduced"]) / 3600.0)
    reduced_hours = float(sum(r["runtime_sec_per_image"] * len(r["per_image"]) for r in runs if r["reduced"]) / 3600.0)
    dry_run_hours = float(dry_run["elapsed_sec"] / 3600.0)
    return {
        "main_matrix_gpu_hours_estimate": main_hours,
        "reduced_slice_gpu_hours_estimate": reduced_hours,
        "dry_run_wall_hours": dry_run_hours,
        "total_hours_estimate": main_hours + reduced_hours + dry_run_hours,
    }


if __name__ == "__main__":
    t0 = time.time()
    prepare_dirs()
    print("[1/5] Freezing environment", flush=True)
    asset_report = check_registered_assets()
    freeze_env()
    print("[2/5] Creating splits", flush=True)
    splits = create_splits(ROOT)
    print("[3/5] Building caches", flush=True)
    build_caches(ROOT, splits)
    print("[4/5] Calibrating and running experiments", flush=True)
    thresholds = calibrate(ROOT, splits)
    dry_run = run_dry_run(ROOT, splits, thresholds)
    runs = run_experiment_grid(ROOT, splits, thresholds)
    write_stage_results(runs)
    df = summarize_runs(runs)
    main_table = aggregate_main(df)
    ablation_table = aggregate_ablations(df)
    ensure_dir(ROOT / "figures")
    main_table.to_csv(ROOT / "figures/main_table.csv", index=False)
    ablation_table.to_csv(ROOT / "figures/ablation_table.csv", index=False)
    main_table.to_markdown(ROOT / "figures/main_table.md", index=False)
    ablation_table.to_markdown(ROOT / "figures/ablation_table.md", index=False)
    make_figures(runs, df, thresholds)

    print("[5/5] Writing aggregates", flush=True)
    runtime = measured_runtime_hours(runs, dry_run)
    results = {
        "status": "completed_proxy_feasibility_study",
        "proxy_note": "Original Stage 1 plan required CAT-Seg/MLMP and Cityscapes/ACDC assets that were not present locally. This results file aggregates an explicitly reframed Pascal VOC proxy feasibility study with synthetic domain shifts.",
        "registered_study_supported": False,
        "claim_status": "negative_proxy_result",
        "metric_protocol": {
            "aggregate_miou": "dataset-level confusion-matrix mIoU over the 20 Pascal VOC foreground classes",
            "per_image_miou": "per-image mIoU averaged only over foreground classes present in that image",
            "note": "Per-image diagnostics are therefore not numerically comparable to the older zero-filled per-image logs used in the superseded artifact set.",
        },
        "executed_scope": {
            "main_comparison_methods": ["frozen", "topb_pixel", "raw_mask", "clip_verified"],
            "reduced_slice_ablations": ["slic", "no_clip", "no_consistency", "threshold_sensitivity", "budget_sensitivity"],
            "omitted_comparators": [
                {
                    "method": "MLMP",
                    "status": "not_run",
                    "reason": "Registered MLMP assets were not available in the workspace, so the proxy study reports only internally matched baselines.",
                },
                {
                    "method": "SLIC_main_table",
                    "status": "not_run_on_main_matrix",
                    "reason": "SLIC was executed only as a reduced-slice ablation in the proxy plan, not as a full 144-image main comparison.",
                },
            ],
        },
        "compatibility_check": asset_report,
        "seeds": [13, 17, 23],
        "orders": {"A": DOMAIN_ORDER_A, "B": DOMAIN_ORDER_B},
        "split_counts": splits["counts"],
        "cache_report": json.loads((ROOT / "exp/01_data_prep/cache_report.json").read_text()),
        "thresholds": thresholds,
        "dry_run": dry_run,
        "main_results": main_table.to_dict(orient="records"),
        "ablation_results": ablation_table.to_dict(orient="records"),
        "paired_tests": {
            "clip_verified_vs_topb": paired_tests(runs, "clip_verified", "topb_pixel"),
            "clip_verified_vs_raw_mask": paired_tests(runs, "clip_verified", "raw_mask"),
        },
        "negative_result_summary": "This proxy feasibility rerun does not substantiate the original registered claim about CLIP-verified object units on CAT-Seg/Cityscapes/ACDC. Any proxy ranking should be treated only as feasibility evidence under a different backbone, label space, and domain construction.",
        "runtime_hours": runtime["total_hours_estimate"],
        "runtime_breakdown_hours": runtime,
    }
    save_json(ROOT / "results.json", results)
    print("[done] Proxy experiment complete", flush=True)
