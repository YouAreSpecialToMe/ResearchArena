import json
from pathlib import Path

import pandas as pd

from exp.shared.core import RESULTS, bootstrap_ci, bootstrap_paired_prediction_delta, write_json


FINAL_ARTIFACTS = RESULTS / "final_artifacts"


def safe_load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_clean_lookup() -> dict:
    lookup = {}
    for benchmark in ["t2d_sm_wh", "wdc_products_medium"]:
        clean_dir = RESULTS / benchmark / "clean"
        if not clean_dir.exists():
            continue
        for method_dir in clean_dir.iterdir():
            for seed_dir in method_dir.iterdir():
                metrics = safe_load_json(seed_dir / "metrics.json")
                config = safe_load_json(seed_dir / "config.json") or {}
                if not metrics:
                    continue
                lookup[(benchmark, method_dir.name, seed_dir.name)] = {
                    "metrics": metrics,
                    "config": config,
                }
    return lookup


def collect_main_rows() -> pd.DataFrame:
    rows = []
    clean_lookup = load_clean_lookup()
    for benchmark in ["t2d_sm_wh", "wdc_products_medium"]:
        benchmark_dir = RESULTS / benchmark
        if not benchmark_dir.exists():
            continue
        for regime_dir in benchmark_dir.iterdir():
            if regime_dir.name in {"clean", "ablations"} or not regime_dir.is_dir():
                continue
            for method_dir in regime_dir.iterdir():
                for seed_dir in method_dir.iterdir():
                    for mode_dir in seed_dir.iterdir():
                        metrics = safe_load_json(mode_dir / "metrics.json")
                        perturb = safe_load_json(mode_dir / "perturbations.json")
                        runtime = safe_load_json(mode_dir / "runtime.json")
                        config = safe_load_json(mode_dir / "config.json")
                        clean_entry = clean_lookup.get((benchmark, method_dir.name, seed_dir.name))
                        if not (metrics and perturb and config and clean_entry):
                            continue
                        clean_metrics = clean_entry["metrics"]
                        rows.append(
                            {
                                **config,
                                "clean_precision": clean_metrics["precision"],
                                "clean_recall": clean_metrics["recall"],
                                "clean_f1": clean_metrics["f1"],
                                "worst_precision": metrics["precision"],
                                "worst_recall": metrics["recall"],
                                "worst_f1": metrics["f1"],
                                "absolute_f1_drop": clean_metrics["f1"] - metrics["f1"],
                                "relative_f1_drop": (clean_metrics["f1"] - metrics["f1"]) / max(clean_metrics["f1"], 1e-9),
                                "acceptance_rate": perturb["aux"]["acceptance_rate"],
                                "accepted_programs": perturb["aux"]["accepted_programs"],
                                "rejected_programs": perturb["aux"]["rejected_programs"],
                                "wall_clock_minutes": (runtime or {}).get("wall_clock_minutes"),
                                "peak_ram_gb": (runtime or {}).get("peak_ram_gb"),
                            }
                        )
    return pd.DataFrame(rows)


def collect_ablation_rows() -> pd.DataFrame:
    rows = []
    clean_lookup = load_clean_lookup()
    for benchmark in ["t2d_sm_wh", "wdc_products_medium"]:
        ablations_dir = RESULTS / benchmark / "ablations"
        if not ablations_dir.exists():
            continue
        for ablation_dir in ablations_dir.iterdir():
            for method_dir in ablation_dir.iterdir():
                for seed_dir in method_dir.iterdir():
                    for mode_dir in seed_dir.iterdir():
                        metrics = safe_load_json(mode_dir / "metrics.json")
                        perturb = safe_load_json(mode_dir / "perturbations.json")
                        config = safe_load_json(mode_dir / "config.json")
                        clean_entry = clean_lookup.get((benchmark, method_dir.name, seed_dir.name))
                        if not (metrics and perturb and config and clean_entry):
                            continue
                        clean_metrics = clean_entry["metrics"]
                        rows.append(
                            {
                                **config,
                                "clean_f1": clean_metrics["f1"],
                                "worst_f1": metrics["f1"],
                                "absolute_f1_drop": clean_metrics["f1"] - metrics["f1"],
                                "acceptance_rate": perturb["aux"]["acceptance_rate"],
                            }
                        )
    return pd.DataFrame(rows)


def aggregate_main_rows(main_df: pd.DataFrame) -> pd.DataFrame:
    if main_df.empty:
        return pd.DataFrame()
    rows = []
    for key, group in main_df.groupby(["benchmark", "method", "regime", "search_mode"]):
        benchmark, method, regime, search_mode = key
        rows.append(
            {
                "benchmark": benchmark,
                "method": method,
                "regime": regime,
                "search_mode": search_mode,
                "seed_count": int(len(group)),
                "clean_f1_mean": float(group["clean_f1"].mean()),
                "clean_f1_std": float(group["clean_f1"].std(ddof=0)),
                "worst_f1_mean": float(group["worst_f1"].mean()),
                "worst_f1_std": float(group["worst_f1"].std(ddof=0)),
                "absolute_f1_drop_mean": float(group["absolute_f1_drop"].mean()),
                "absolute_f1_drop_std": float(group["absolute_f1_drop"].std(ddof=0)),
                "absolute_f1_drop_ci95_low": bootstrap_ci(group["absolute_f1_drop"].tolist())["low"],
                "absolute_f1_drop_ci95_high": bootstrap_ci(group["absolute_f1_drop"].tolist())["high"],
                "acceptance_rate_mean": float(group["acceptance_rate"].mean()),
                "acceptance_rate_std": float(group["acceptance_rate"].std(ddof=0)),
                "wall_clock_minutes_mean": float(group["wall_clock_minutes"].mean()),
                "peak_ram_gb_max": float(group["peak_ram_gb"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["benchmark", "search_mode", "regime", "method"]).reset_index(drop=True)


def ranking_change_rows(main_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if main_df.empty:
        return pd.DataFrame()
    clean_rank_df = (
        main_df.groupby(["benchmark", "method"], as_index=False)["clean_f1"]
        .mean()
        .sort_values(["benchmark", "clean_f1", "method"], ascending=[True, False, True])
    )
    clean_ranking = {
        benchmark: group["method"].tolist()
        for benchmark, group in clean_rank_df.groupby("benchmark")
    }
    for (benchmark, regime, search_mode), group in main_df.groupby(["benchmark", "regime", "search_mode"]):
        ordered = (
            group.groupby("method", as_index=False)["worst_f1"]
            .mean()
            .sort_values(["worst_f1", "method"], ascending=[False, True])["method"]
            .tolist()
        )
        comparison_complete = set(ordered) == set(clean_ranking[benchmark])
        rows.append(
            {
                "benchmark": benchmark,
                "regime": regime,
                "search_mode": search_mode,
                "clean_ranking": " > ".join(clean_ranking[benchmark]),
                "worst_case_ranking": " > ".join(ordered),
                "comparison_complete": comparison_complete,
                "ranking_changed": (ordered != clean_ranking[benchmark]) if comparison_complete else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["benchmark", "search_mode", "regime"]).reset_index(drop=True)


def summarize_ablations(ablation_df: pd.DataFrame, audit_status: dict | None) -> pd.DataFrame:
    rows = []
    if not ablation_df.empty:
        for key, group in ablation_df.groupby(["benchmark", "method", "ablation", "search_mode"]):
            benchmark, method, ablation, search_mode = key
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "ablation": ablation,
                    "search_mode": search_mode,
                    "worst_f1_mean": float(group["worst_f1"].mean()),
                    "worst_f1_std": float(group["worst_f1"].std(ddof=0)),
                    "absolute_f1_drop_mean": float(group["absolute_f1_drop"].mean()),
                    "absolute_f1_drop_std": float(group["absolute_f1_drop"].std(ddof=0)),
                    "acceptance_rate_mean": float(group["acceptance_rate"].mean()),
                }
            )
    rows.append(
        {
            "benchmark": "all",
            "method": "audit",
            "ablation": "accepted_only_audit",
            "search_mode": "n/a",
            "status": (audit_status or {}).get("ablation_d_status", "blocked_without_human_audit_labels"),
        }
    )
    return pd.DataFrame(rows)


def paired_ci_rows(main_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    if main_df.empty:
        return pd.DataFrame()
    for benchmark in sorted(main_df["benchmark"].unique()):
        for method in sorted(main_df["method"].unique()):
            for seed in sorted(main_df["seed"].unique()):
                for search_mode in sorted(main_df["search_mode"].unique()):
                    base_path = RESULTS / benchmark / "ABCA" / method / f"seed_{seed}" / search_mode / "predictions.parquet"
                    for other in ["format", "naive"]:
                        other_path = RESULTS / benchmark / other / method / f"seed_{seed}" / search_mode / "predictions.parquet"
                        if base_path.exists() and other_path.exists():
                            delta = bootstrap_paired_prediction_delta(
                                pd.read_parquet(base_path),
                                pd.read_parquet(other_path),
                                seed=seed,
                            )
                            comparisons.append(
                                {
                                    "comparison": f"ABCA_vs_{other}",
                                    "benchmark": benchmark,
                                    "method": method,
                                    "seed": seed,
                                    "search_mode": search_mode,
                                    **delta,
                                }
                            )
                for regime in sorted(main_df["regime"].unique()):
                    random_path = RESULTS / benchmark / regime / method / f"seed_{seed}" / "random" / "predictions.parquet"
                    targeted_path = RESULTS / benchmark / regime / method / f"seed_{seed}" / "targeted" / "predictions.parquet"
                    if random_path.exists() and targeted_path.exists():
                        delta = bootstrap_paired_prediction_delta(
                            pd.read_parquet(targeted_path),
                            pd.read_parquet(random_path),
                            seed=seed,
                        )
                        comparisons.append(
                            {
                                "comparison": "targeted_vs_random",
                                "benchmark": benchmark,
                                "method": method,
                                "seed": seed,
                                "search_mode": regime,
                                **delta,
                            }
                        )
    return pd.DataFrame(comparisons)


def write_runtime_ledger() -> pd.DataFrame:
    rows = []
    for runtime_path in RESULTS.rglob("runtime.json"):
        runtime = safe_load_json(runtime_path)
        if not runtime:
            continue
        rel = runtime_path.relative_to(RESULTS).parts
        if rel[1] == "clean":
            benchmark, regime, method, seed_dir, _ = rel
            experiment = "baselines"
            search_mode = "clean"
        elif rel[1] == "ablations":
            benchmark, _, ablation, method, seed_dir, search_mode, _ = rel
            regime = "ABCA"
            experiment = f"ablation_{ablation}"
        else:
            benchmark, regime, method, seed_dir, search_mode, _ = rel
            experiment = "main_robustness"
        rows.append(
            {
                "experiment": experiment,
                "benchmark": benchmark,
                "method": method,
                "regime": regime,
                "search_mode": search_mode,
                "seed": int(seed_dir.split("_")[-1]),
                "worker_id": 0,
                "wall_clock_minutes": runtime.get("wall_clock_minutes"),
                "peak_ram_gb": runtime.get("peak_ram_gb"),
                "status": "completed",
            }
        )
    ledger = pd.DataFrame(rows).sort_values(["experiment", "benchmark", "method", "seed", "search_mode"])
    ledger.to_csv(RESULTS / "runtime_ledger.csv", index=False)
    return ledger


def write_tables(table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame, table4: pd.DataFrame, table5: pd.DataFrame, table6: pd.DataFrame, table7: pd.DataFrame) -> None:
    FINAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "table1_benchmark_statistics": table1,
        "table2_admissibility_rationale": table2,
        "table3_audit_status": table3,
        "table4_clean_corrupted_performance": table4,
        "table5_ablations": table5,
        "table6_paired_bootstrap_cis": table6,
        "table7_ranking_changes": table7,
    }.items():
        frame.to_csv(FINAL_ARTIFACTS / f"{name}.csv", index=False)
        frame.to_latex(FINAL_ARTIFACTS / f"{name}.tex", index=False)


def main():
    dataset_stats = safe_load_json(RESULTS / "dataset_statistics.json") or {}
    audit_status = safe_load_json(Path("exp/audit/results.json")) or {}
    main_df = collect_main_rows()
    ablation_df = collect_ablation_rows()
    table4 = aggregate_main_rows(main_df)
    ranking_df = ranking_change_rows(main_df)
    table5 = summarize_ablations(ablation_df, audit_status)
    table6 = paired_ci_rows(main_df)
    ledger = write_runtime_ledger()

    table1_rows = []
    for benchmark, split_rows in dataset_stats.items():
        for split, stats in split_rows.items():
            table1_rows.append({"benchmark": benchmark, "split": split, **stats})
    table1 = pd.DataFrame(table1_rows)
    table2 = pd.DataFrame(
        [
            {
                "benchmark": "t2d_sm_wh",
                "label_type": "column correspondence",
                "protected_evidence": "gold columns plus competitor columns",
                "competitor_rule": "header >=0.80 or value cosine >=0.90",
                "admissible_families": "header abbreviation/case, value formatting, row reorder, non-protected dropout",
                "forbidden_edits": "protected header/value changes or new competitors",
            },
            {
                "benchmark": "wdc_products_medium",
                "label_type": "binary match label",
                "protected_evidence": "identifiers, brand, model, quantities, protected title spans",
                "competitor_rule": "protected title spans immutable under ABCA",
                "admissible_families": "case/punctuation, boilerplate, non-protected dropout, numeric format, token permutation",
                "forbidden_edits": "identifier, quantity, or brand/model changes",
            },
        ]
    )
    table3 = pd.DataFrame(
        [
            {
                "status": audit_status.get("status", "missing"),
                "accepted_precision": audit_status.get("accepted_precision"),
                "rejected_violation_rate": audit_status.get("rejected_violation_rate"),
                "false_rejection_rate": audit_status.get("false_rejection_rate"),
                "cohens_kappa": audit_status.get("cohens_kappa"),
                "non_author_agreement": audit_status.get("non_author_agreement"),
                "validated_soundness_claim": audit_status.get("validated_soundness_claim"),
            }
        ]
    )
    write_tables(table1, table2, table3, table4, table5, table6, ranking_df)

    output = {
        "status": "completed",
        "claim_status": "pending_human_audit" if audit_status.get("status", "").startswith("pending") else "completed",
        "main_robustness_summary": table4.to_dict("records"),
        "ranking_changes": ranking_df.to_dict("records"),
        "ablations_summary": table5.to_dict("records"),
        "paired_bootstrap_cis": table6.to_dict("records"),
        "audit_status": audit_status,
        "runtime_ledger_rows": int(len(ledger)),
        "peak_ram_gb_max": float(ledger["peak_ram_gb"].max()) if not ledger.empty else None,
    }
    write_json(Path("results.json"), output)


if __name__ == "__main__":
    main()
