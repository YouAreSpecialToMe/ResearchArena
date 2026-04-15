import json
from pathlib import Path

from exp.shared.pipeline import build_seed_bundle
from exp.shared.utils import RUN_VERSION, SEEDS, ensure_dir, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ensure_dir(root / "logs")
    for dataset in ["Adamson", "Norman", "Replogle"]:
        for seed in SEEDS:
            bundle = build_seed_bundle(dataset, seed, root=root)
            split_counts = {
                split: int((bundle.split_labels == split).sum())
                for split in ["train", "route_dev", "calibration", "test"]
            }
            write_json(
                root / "splits" / dataset / f"seed_{seed}.json",
                {
                    "run_version": RUN_VERSION,
                    "dataset": dataset,
                    "seed": seed,
                    "split_counts": split_counts,
                    "calibration_underpowered": split_counts["calibration"] < 20,
                    "responsive_panel_size": int(len(bundle.responsive_panel)),
                    "retained_genes": int(len(bundle.retained_genes)),
                    "split_config": bundle.split_config,
                    "split_memberships": bundle.split_memberships,
                    "preprocessing_stats_path": str(root / "features" / dataset / f"seed_{seed}" / "preprocessing_stats.json"),
                },
            )
    write_json(
        root / "config.json",
        {
            "run_version": RUN_VERSION,
            "datasets": ["Adamson", "Norman", "Replogle"],
            "seeds": SEEDS,
            "acceptance_rates": [0.2, 0.4, 0.6],
            "note": "GO and curated pathway annotations were not available locally; descriptor/pathway components use leakage-free hashed targets, co-target graph features, and training-derived gene modules instead.",
        },
    )


if __name__ == "__main__":
    main()
