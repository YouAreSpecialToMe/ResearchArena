import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from exp.shared.core import (
    RESULTS,
    build_wdc_normalized_view,
    entity_protected_tokens,
    evaluate_entity,
    evaluate_schema,
    load_t2d_split,
    package_versions,
    perturb_entity_row,
    perturb_schema_table,
    profile_lookup,
    schema_admissibility_decision,
    schema_protected_sets,
    system_info,
    write_json,
    write_jsonl,
)


def safe_load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def enumerate_run_dirs() -> list[Path]:
    return sorted(path.parent for path in RESULTS.rglob("metrics.json"))


def backfill_run_artifacts(run_dir: Path) -> dict:
    runtime_path = run_dir / "runtime.json"
    runtime = safe_load_json(runtime_path) or {}
    if "package_versions" not in runtime or "peak_ram_gb" not in runtime:
        runtime["package_versions"] = package_versions()
        runtime["peak_ram_gb"] = runtime.get("peak_ram_gb")
        write_json(runtime_path, runtime)
    metadata_path = run_dir / "metadata.json"
    metadata = safe_load_json(metadata_path) or {}
    metadata["system_info"] = system_info()
    metadata["package_versions"] = package_versions()
    write_json(metadata_path, metadata)
    return {"run_dir": str(run_dir)}


def clean_example_pool() -> pd.DataFrame:
    rows = []
    for benchmark, method in [("t2d_sm_wh", "schema_strong"), ("wdc_products_medium", "entity_strong")]:
        for seed in [13, 29, 47]:
            preds = (
                evaluate_schema(seed, split="valid")[method]["predictions"]
                if benchmark == "t2d_sm_wh"
                else evaluate_entity(seed, split="valid")[method]["predictions"]
            ).copy()
            preds["benchmark"] = benchmark
            preds["method"] = method
            preds["seed"] = seed
            rows.append(preds)
    df = pd.concat(rows, ignore_index=True)
    df["difficulty_bucket"] = "hard"
    for (benchmark, seed), group in df.groupby(["benchmark", "seed"]):
        q1 = group["score_margin"].quantile(1 / 3)
        q2 = group["score_margin"].quantile(2 / 3)
        easy_mask = (df["benchmark"] == benchmark) & (df["seed"] == seed) & (df["score_margin"] > q2)
        medium_mask = (
            (df["benchmark"] == benchmark)
            & (df["seed"] == seed)
            & (df["score_margin"] > q1)
            & (df["score_margin"] <= q2)
        )
        df.loc[easy_mask, "difficulty_bucket"] = "easy"
        df.loc[medium_mask, "difficulty_bucket"] = "medium"
    return df


def candidate_program_rows() -> pd.DataFrame:
    rows = []
    for benchmark, method in [("t2d_sm_wh", "schema_strong"), ("wdc_products_medium", "entity_strong")]:
        for seed in [13, 29, 47]:
            for search_mode in ["random", "targeted"]:
                log_path = RESULTS / benchmark / "ABCA" / method / f"seed_{seed}" / search_mode / "admissibility_log.jsonl"
                if not log_path.exists():
                    continue
                with log_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        obj = json.loads(line)
                        program = obj.get("program") or []
                        if not program:
                            continue
                        severity_rank = {"low": 0, "medium": 1, "high": 2}
                        rows.append(
                            {
                                "benchmark": benchmark,
                                "method": method,
                                "seed": seed,
                                "search_mode": search_mode,
                                "eval_index": int(obj["eval_index"]),
                                "accepted": bool(obj["accepted"]),
                                "operator_family": "+".join(op[0] for op in program),
                                "severity": ["low", "medium", "high"][
                                    max((severity_rank.get(op[1], 0) for op in program), default=0)
                                ],
                                "program_length": len(program),
                                "program_json": json.dumps(program),
                            }
                        )
    return pd.DataFrame(rows)


def attach_examples_to_candidates(candidates: pd.DataFrame, example_pool: pd.DataFrame) -> pd.DataFrame:
    attached = []
    for row in candidates.to_dict("records"):
        pool = example_pool[
            (example_pool["benchmark"] == row["benchmark"]) & (example_pool["seed"] == row["seed"])
        ].sort_values("example_id")
        if pool.empty:
            continue
        offset = (row["eval_index"] * 997 + len(row["program_json"])) % len(pool)
        example = pool.iloc[int(offset)].to_dict()
        attached.append(
            {
                **row,
                "example_id": str(example["example_id"]),
                "difficulty_bucket": example["difficulty_bucket"],
                "label": int(example["label"]),
                "score_margin": float(example["score_margin"]),
                "pair_id": str(example["pair_id"]) if "pair_id" in example else None,
                "table_name": example.get("table_name"),
                "column_index_left": (
                    int(example["column_index_left"])
                    if "column_index_left" in example and pd.notna(example["column_index_left"])
                    else None
                ),
                "column_index_right": (
                    int(example["column_index_right"])
                    if "column_index_right" in example and pd.notna(example["column_index_right"])
                    else None
                ),
            }
        )
    return pd.DataFrame(attached)


def stratified_manifest(candidates: pd.DataFrame) -> pd.DataFrame:
    sampled = []
    for accepted_value, target in [(True, 120), (False, 120)]:
        sub = candidates[candidates["accepted"] == accepted_value].copy()
        sub["stratum"] = (
            sub["benchmark"]
            + "|"
            + sub["operator_family"]
            + "|"
            + sub["severity"]
            + "|"
            + sub["difficulty_bucket"]
        )
        chosen = []
        groups = sorted(sub.groupby("stratum"), key=lambda item: item[0])
        while len(chosen) < target and groups:
            next_groups = []
            for _, group in groups:
                group = group.sort_values(["seed", "search_mode", "eval_index", "example_id"])
                remaining = group.iloc[len([x for x in chosen if x["stratum"] == group.iloc[0]["stratum"]]) :]
                if remaining.empty:
                    continue
                chosen.append(remaining.iloc[0].to_dict())
                next_groups.append((group.iloc[0]["stratum"], group))
                if len(chosen) >= target:
                    break
            groups = next_groups
        sampled.append(pd.DataFrame(chosen[:target]))
    manifest = pd.concat(sampled, ignore_index=True)
    manifest["audit_subset_non_author"] = False
    subset_frames = []
    for _, group in manifest.groupby(["accepted", "benchmark"]):
        subset_frames.append(group.sort_values(["difficulty_bucket", "seed", "eval_index"]).head(18))
    if subset_frames:
        subset = pd.concat(subset_frames, ignore_index=True)
        subset_keys = set(zip(subset["benchmark"], subset["seed"], subset["search_mode"], subset["eval_index"], subset["example_id"]))
        manifest["audit_subset_non_author"] = [
            (b, s, m, e, x) in subset_keys
            for b, s, m, e, x in zip(
                manifest["benchmark"],
                manifest["seed"],
                manifest["search_mode"],
                manifest["eval_index"],
                manifest["example_id"],
            )
        ]
    manifest.insert(0, "manifest_id", np.arange(1, len(manifest) + 1))
    return manifest


def schema_packet(row: dict) -> dict:
    bundle = load_t2d_split("valid")
    corr = bundle["correspondences"]
    decision = corr[
        (corr["table_name"] == row["table_name"])
        & (corr["column_index_left"] == row["column_index_left"])
        & (corr["column_index_right"] == row["column_index_right"])
    ].iloc[0]
    df = bundle["tables"][("webtables", f"{row['table_name']}.csv")]
    right_df = bundle["tables"][("dbpedia_tables", f"{row['table_name']}.csv")]
    protected = schema_protected_sets("valid")[row["table_name"]]
    program = json.loads(row["program_json"])
    audit_log = []
    perturbed = perturb_schema_table(
        df,
        program,
        random.Random(row["seed"] * 10_000 + row["eval_index"]),
        "ABCA",
        protected,
        audit_log,
    )
    audit_log.extend(schema_admissibility_decision(df, perturbed, protected))
    left_idx = int(row["column_index_left"])
    right_idx = int(row["column_index_right"])
    return {
        "manifest_id": int(row["manifest_id"]),
        "benchmark": row["benchmark"],
        "example_id": row["example_id"],
        "released_label": int(decision["label"]),
        "checker_decision": "accepted" if row["accepted"] else "rejected",
        "checker_reasons": [item["reason_code"] for item in audit_log],
        "original_left_header": str(df.columns[left_idx]),
        "perturbed_left_header": str(perturbed.columns[left_idx]) if left_idx < len(perturbed.columns) else None,
        "right_header": str(right_df.columns[right_idx]),
        "original_left_samples": [str(x) for x in df.iloc[:5, left_idx].tolist()],
        "perturbed_left_samples": [str(x) for x in perturbed.iloc[:5, left_idx].tolist()] if left_idx < len(perturbed.columns) else [],
        "right_samples": [str(x) for x in right_df.iloc[:5, right_idx].tolist()],
    }


def entity_packet(row: dict) -> dict:
    wdc = build_wdc_normalized_view()
    original = wdc[(wdc["split"] == "valid") & (wdc["pair_id"].astype(str) == str(row["pair_id"]))].iloc[0].to_dict()
    program = json.loads(row["program_json"])
    perturbed, logs = perturb_entity_row(
        original,
        program,
        random.Random(row["seed"] * 10_000 + row["eval_index"]),
        "ABCA",
    )
    return {
        "manifest_id": int(row["manifest_id"]),
        "benchmark": row["benchmark"],
        "example_id": row["example_id"],
        "released_label": int(original["label"]),
        "checker_decision": "accepted" if row["accepted"] else "rejected",
        "checker_reasons": [item["reason_code"] for item in logs],
        "original_left_title": original["title_left_norm"],
        "perturbed_left_title": perturbed["title_left_norm"],
        "original_right_title": original["title_right_norm"],
        "perturbed_right_title": perturbed["title_right_norm"],
        "original_left_brand": original["brand_left_norm"],
        "perturbed_left_brand": perturbed["brand_left_norm"],
        "original_right_brand": original["brand_right_norm"],
        "perturbed_right_brand": perturbed["brand_right_norm"],
    }


def write_audit_packets(manifest: pd.DataFrame) -> Path:
    packets_dir = RESULTS / "audit_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    packets = []
    for row in manifest.to_dict("records"):
        packets.append(schema_packet(row) if row["benchmark"] == "t2d_sm_wh" else entity_packet(row))
    write_jsonl(packets_dir / "audit_packets.jsonl", packets)
    manifest.to_csv(packets_dir / "audit_manifest_with_examples.csv", index=False)
    template = manifest.copy()
    for col in ["label_still_justified", "ambiguity_materially_increased", "checker_decision_correct", "notes"]:
        template[col] = ""
    template.to_csv(packets_dir / "annotation_template_author1.csv", index=False)
    template.to_csv(packets_dir / "annotation_template_author2.csv", index=False)
    template[template["audit_subset_non_author"]].to_csv(packets_dir / "annotation_template_non_author.csv", index=False)
    rubric = "\n".join(
        [
            "# Audit Rubric",
            "",
            "For each perturbation instance, record:",
            "1. Whether the released benchmark label is still justified.",
            "2. Whether benchmark-relevant ambiguity materially increased.",
            "3. Whether the checker decision should be considered correct.",
        ]
    )
    (packets_dir / "RUBRIC.md").write_text(rubric, encoding="utf-8")
    return packets_dir


def write_pilot_report() -> dict:
    pilot_path = Path("exp/pilot/results.json")
    pilot = safe_load_json(pilot_path)
    report = {
        "experiment": "pilot_report",
        "status": "completed" if pilot else "missing",
        "artifact_path": str(pilot_path) if pilot_path.exists() else None,
        "pilot_summary": pilot,
    }
    write_json(Path(__file__).resolve().parent / "results.json", report)
    return report


def main() -> None:
    touched = [backfill_run_artifacts(run_dir) for run_dir in enumerate_run_dirs()]
    example_pool = clean_example_pool()
    candidates = attach_examples_to_candidates(candidate_program_rows(), example_pool)
    manifest = stratified_manifest(candidates)
    manifest_path = RESULTS / "audit_sample_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    packets_dir = write_audit_packets(manifest)
    write_json(RESULTS / "system_info" / "package_versions.json", package_versions())
    write_json(
        RESULTS / "protocol_deviations.json",
        {
            "python_environment": {
                "planned": "Python 3.11",
                "observed_primary_environment": system_info()["python"],
                "status": "deviation" if system_info()["python"] != "3.11" else "matched",
            },
            "audit_execution": {
                "status": "blocked_pending_human_annotations",
                "reason": "This workspace can generate frozen manifests and audit packets but cannot honestly supply two-author or non-author human labels.",
            },
            "audit_manifest": {
                "path": str(manifest_path),
                "status": "frozen",
                "difficulty_bucket_counts": manifest["difficulty_bucket"].value_counts().to_dict(),
                "non_author_subset_size": int(manifest["audit_subset_non_author"].sum()),
                "audit_packets_dir": str(packets_dir),
            },
        },
    )
    pilot_report = write_pilot_report()
    write_json(
        Path(__file__).resolve().parent / "results.json",
        {
            "experiment": "reproducibility_artifacts",
            "status": "completed",
            "run_dirs_backfilled": len(touched),
            "audit_manifest_rows": int(len(manifest)),
            "audit_packets_dir": str(packets_dir),
            "pilot_report": pilot_report,
        },
    )


if __name__ == "__main__":
    main()
