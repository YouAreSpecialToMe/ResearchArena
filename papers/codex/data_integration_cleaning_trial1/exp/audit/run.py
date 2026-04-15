import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from exp.shared.core import write_json


PACKETS_DIR = Path("results/audit_packets")


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def normalize_label(value) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1.0
    if text in {"0", "false", "no", "n"}:
        return 0.0
    return None


def main() -> None:
    manifest_path = Path("results/audit_sample_manifest.csv")
    author1 = load_csv(PACKETS_DIR / "annotation_template_author1.csv")
    author2 = load_csv(PACKETS_DIR / "annotation_template_author2.csv")
    non_author = load_csv(PACKETS_DIR / "annotation_template_non_author.csv")
    if author1 is None or author2 is None or "label_still_justified" not in author1.columns:
        write_json(
            Path(__file__).resolve().parent / "results.json",
            {
                "experiment": "audit",
                "status": "pending_human_annotations",
                "audit_sample_manifest_path": str(manifest_path) if manifest_path.exists() else None,
                "annotation_templates": {
                    "author1": str(PACKETS_DIR / "annotation_template_author1.csv"),
                    "author2": str(PACKETS_DIR / "annotation_template_author2.csv"),
                    "non_author": str(PACKETS_DIR / "annotation_template_non_author.csv"),
                },
                "reason": "Human labels have not been entered into the generated templates.",
                "ablation_d_status": "blocked_without_human_audit_labels",
                "validated_soundness_claim": False,
            },
        )
        return

    author1_labels = author1["label_still_justified"].map(normalize_label)
    author2_labels = author2["label_still_justified"].map(normalize_label)
    valid_mask = author1_labels.notna() & author2_labels.notna()
    if valid_mask.sum() == 0:
        write_json(
            Path(__file__).resolve().parent / "results.json",
            {
                "experiment": "audit",
                "status": "pending_human_annotations",
                "audit_sample_manifest_path": str(manifest_path) if manifest_path.exists() else None,
                "reason": "Annotation templates exist but contain no completed label judgments.",
                "ablation_d_status": "blocked_without_human_audit_labels",
                "validated_soundness_claim": False,
            },
        )
        return

    adjudicated = author1.copy()
    adjudicated["author1_label"] = author1_labels
    adjudicated["author2_label"] = author2_labels
    adjudicated["adjudicated_label"] = author1_labels.where(author1_labels == author2_labels, other=pd.NA)

    accepted = adjudicated["accepted"].astype(bool)
    rejected = ~accepted
    adjudicated_valid = adjudicated["adjudicated_label"].notna()
    accepted_precision = float(adjudicated.loc[accepted & adjudicated_valid, "adjudicated_label"].mean())
    rejected_violation_rate = float(1.0 - adjudicated.loc[rejected & adjudicated_valid, "adjudicated_label"].mean())
    false_rejection_rate = float(adjudicated.loc[rejected & adjudicated_valid, "adjudicated_label"].mean())
    kappa = float(cohen_kappa_score(author1_labels[valid_mask], author2_labels[valid_mask]))

    non_author_agreement = None
    if non_author is not None and "label_still_justified" in non_author.columns:
        merged = non_author.merge(
            adjudicated[["manifest_id", "adjudicated_label"]],
            on="manifest_id",
            how="left",
        )
        non_author_labels = merged["label_still_justified"].map(normalize_label)
        subset_mask = non_author_labels.notna() & merged["adjudicated_label"].notna()
        if subset_mask.sum() > 0:
            non_author_agreement = float(
                (non_author_labels[subset_mask] == merged.loc[subset_mask, "adjudicated_label"]).mean()
            )

    by_benchmark = []
    for (benchmark, accepted_flag), group in adjudicated[adjudicated_valid].groupby(["benchmark", "accepted"]):
        by_benchmark.append(
            {
                "benchmark": benchmark,
                "accepted": bool(accepted_flag),
                "count": int(len(group)),
                "label_still_justified_rate": float(group["adjudicated_label"].mean()),
            }
        )

    write_json(
        Path(__file__).resolve().parent / "results.json",
        {
            "experiment": "audit",
            "status": "completed" if non_author_agreement is not None else "partial_without_non_author_labels",
            "audit_sample_manifest_path": str(manifest_path) if manifest_path.exists() else None,
            "accepted_precision": accepted_precision,
            "rejected_violation_rate": rejected_violation_rate,
            "false_rejection_rate": false_rejection_rate,
            "cohens_kappa": kappa,
            "non_author_agreement": non_author_agreement,
            "validated_soundness_claim": bool(
                accepted_precision >= 0.95
                and rejected_violation_rate >= 0.70
                and false_rejection_rate <= 0.20
                and kappa >= 0.70
                and (non_author_agreement or 0.0) >= 0.75
            ),
            "ablation_d_status": "completed" if non_author_agreement is not None else "partial",
            "by_benchmark": by_benchmark,
        },
    )


if __name__ == "__main__":
    main()
