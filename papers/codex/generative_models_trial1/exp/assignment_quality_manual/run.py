from __future__ import annotations

import random

from exp.shared.utils import ensure_dir, project_path, read_jsonl, write_json, write_jsonl


TARGET_TOTAL = 100
OVERLAP_TOTAL = 25


def bucket_name(record: dict) -> str:
    category = record.get("source_category", "unknown")
    if category in {"counting", "count"}:
        return "count"
    if category in {"colors", "attribute_binding"}:
        return "attribute_or_color_binding"
    if category in {"position", "relation"}:
        return "relation"
    return "plausible_null"


def main() -> None:
    split_rows = read_jsonl(project_path("data", "splits", "test.jsonl"))
    by_prompt = {row["prompt_id"]: row for row in split_rows}
    candidate_rows = read_jsonl(project_path("results", "assign_and_verify.jsonl"))
    pool = []
    for row in candidate_rows:
        if row["dataset"] != "test":
            continue
        meta = by_prompt.get(row["prompt_id"])
        if meta is None:
            continue
        pool.append(
            {
                "dataset": row["dataset"],
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "selected_candidate_id": row["selected_candidate_id"],
                "source_dataset": meta["dataset"],
                "source_category": meta["source_category"],
                "analysis_bucket": bucket_name(meta),
                "candidate_image": str(
                    project_path(
                        "artifacts",
                        "candidate_cache",
                        row["dataset"],
                        row["prompt_id"],
                        f"seed_{row['seed']}",
                        f"cand_{row['selected_candidate_id']}.png",
                    )
                ),
                "audit_json": str(project_path("artifacts", "audits", "assign_and_verify", row["dataset"], row["prompt_id"], f"seed_{row['seed']}.json")),
            }
        )

    rng = random.Random(17)
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    for row in pool:
        key = (row["source_dataset"], row["analysis_bucket"])
        by_bucket.setdefault(key, []).append(row)
    for rows in by_bucket.values():
        rng.shuffle(rows)

    strata = sorted(by_bucket)
    selected = []
    seen = set()
    target_per_stratum = max(1, TARGET_TOTAL // max(1, len(strata)))
    for key in strata:
        for row in by_bucket[key][:target_per_stratum]:
            item_key = (row["prompt_id"], row["seed"])
            if item_key in seen:
                continue
            selected.append(row)
            seen.add(item_key)
    if len(selected) < TARGET_TOTAL:
        remainder = [row for rows in by_bucket.values() for row in rows if (row["prompt_id"], row["seed"]) not in seen]
        rng.shuffle(remainder)
        for row in remainder:
            selected.append(row)
            seen.add((row["prompt_id"], row["seed"]))
            if len(selected) >= TARGET_TOTAL:
                break
    selected = selected[:TARGET_TOTAL]

    overlap = selected[: min(OVERLAP_TOTAL, len(selected))]
    manifest_dir = ensure_dir(project_path("exp", "assignment_quality_manual", "manifests"))
    write_jsonl(manifest_dir / "primary_annotation_manifest.jsonl", selected)
    write_jsonl(manifest_dir / "adjudication_overlap_manifest.jsonl", overlap)
    write_json(
        project_path("exp", "assignment_quality_manual", "results.json"),
        {
            "status": "awaiting_human_annotations",
            "num_manifest_rows": len(selected),
            "num_overlap_rows": len(overlap),
            "source_strata": {f"{dataset}:{bucket}": sum(1 for row in selected if row["source_dataset"] == dataset and row["analysis_bucket"] == bucket) for dataset, bucket in strata},
            "annotation_files_expected": [
                str(manifest_dir / "primary_annotations.jsonl"),
                str(manifest_dir / "adjudication_annotations.jsonl"),
            ],
        },
    )


if __name__ == "__main__":
    main()
