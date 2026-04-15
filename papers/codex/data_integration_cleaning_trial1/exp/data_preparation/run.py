import csv
from pathlib import Path

from exp.shared.core import (
    DATA_CACHE,
    RESULTS,
    build_t2d_profiles,
    build_wdc_normalized_view,
    dataset_statistics,
    ensure_dirs,
    schema_protected_sets,
    write_json,
)


def main() -> None:
    ensure_dirs()
    profiles = build_t2d_profiles()
    wdc = build_wdc_normalized_view()
    stats = dataset_statistics()
    write_json(RESULTS / "dataset_statistics.json", stats)
    profiles.to_csv(RESULTS / "table1_t2d_profiles.csv", index=False)
    wdc.to_parquet(DATA_CACHE / "wdc_products_medium_normalized.parquet", index=False)
    configs_dir = Path("configs/admissibility")
    configs_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        configs_dir / "t2d_sm_wh.json",
        {
            "benchmark": "t2d_sm_wh",
            "rule_set": "benchmark_specific",
            "protected_sets": {
                k: {
                    "positive_left": sorted(v["positive_left"]),
                    "positive_right": sorted(v["positive_right"]),
                    "competitor_left": sorted(v["competitor_left"]),
                    "matched_pairs": sorted(list(v["matched_pairs"])),
                }
                for k, v in schema_protected_sets("test").items()
            },
        },
    )
    write_json(
        configs_dir / "wdc_products_medium.json",
        {
            "benchmark": "wdc_products_medium",
            "rule_set": "benchmark_specific",
            "protected_fields": [
                "gtin",
                "gtin13",
                "gtin14",
                "mpn",
                "sku",
                "productid",
                "brand",
                "model",
                "price",
                "title spans aligned to protected tokens",
            ],
        },
    )
    with (RESULTS / "table1_wdc_field_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "pairs",
                "positives",
                "negatives",
                "identifier_field_coverage",
                "brand_field_coverage",
                "model_token_coverage",
                "price_field_coverage",
            ],
        )
        writer.writeheader()
        for split, row in stats["wdc_products_medium"].items():
            writer.writerow({"split": split, **row})
    write_json(
        Path(__file__).resolve().parent / "results.json",
        {"experiment": "data_preparation", "status": "completed", "dataset_statistics_path": str(RESULTS / "dataset_statistics.json")},
    )


if __name__ == "__main__":
    main()
