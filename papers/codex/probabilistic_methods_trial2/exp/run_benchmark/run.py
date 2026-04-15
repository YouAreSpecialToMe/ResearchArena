from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.shared.config import ABLATION_DATASETS, ALPHA_MAIN, ALPHA_SECONDARY, DATASETS, RESULTS_DIR, SEEDS, THREAD_ENV
from exp.shared.data import load_dataset
from exp.shared.eval import run_method
from exp.shared.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["main", "ablation", "secondary"], required=True)
    args = parser.parse_args()
    os.environ.update(THREAD_ENV)
    pilot = read_json(RESULTS_DIR / "pilot_gate.json")
    use_datasets = ["synthetic", "anuran"] if pilot["decision"]["drop_mice"] else DATASETS
    manifest = []

    if args.mode == "main":
        methods = ["split_cp", "class_conditional_cp", "knn_rlcp", "gmm_rlcp", "batch_mcp", "chip_rlcp"]
        for dataset in use_datasets:
            for seed in SEEDS:
                bundle = load_dataset(dataset, seed)
                for method in methods:
                    manifest.append(run_method(bundle, dataset, seed, method, ALPHA_MAIN, run_label=method))
                if dataset == "synthetic":
                    manifest.append(run_method(bundle, dataset, seed, "oracle_rlcp", ALPHA_MAIN, run_label="oracle_rlcp"))
        write_json(RESULTS_DIR / "main_manifest.json", manifest)
    elif args.mode == "ablation":
        variants = [
            ("chip_rlcp_flat_only", {"method": "chip_rlcp", "chip_variant": "flat_only", "fallback_lambda": 0.1}),
            ("chip_rlcp_coarse_only", {"method": "chip_rlcp", "chip_variant": "coarse_only", "fallback_lambda": 0.1}),
            ("chip_rlcp_no_fallback", {"method": "chip_rlcp", "chip_variant": "full", "fallback_lambda": 0.0}),
            ("chip_uniform_overlap", {"method": "chip_uniform_overlap", "chip_variant": "full", "fallback_lambda": 0.1}),
        ]
        for dataset in ABLATION_DATASETS:
            for seed in SEEDS:
                bundle = load_dataset(dataset, seed)
                for label, spec in variants:
                    manifest.append(
                        run_method(
                            bundle,
                            dataset,
                            seed,
                            spec["method"],
                            ALPHA_MAIN,
                            chip_variant=spec["chip_variant"],
                            fallback_lambda=spec["fallback_lambda"],
                            run_label=label,
                        )
                    )
        write_json(RESULTS_DIR / "ablation_manifest.json", manifest)
    else:
        methods = ["split_cp", "gmm_rlcp", "batch_mcp", "chip_rlcp"]
        for dataset in ["synthetic", "anuran"]:
            bundle = load_dataset(dataset, 11)
            for method in methods:
                manifest.append(run_method(bundle, dataset, 11, method, ALPHA_SECONDARY, run_label=method))
        write_json(RESULTS_DIR / "secondary_manifest.json", manifest)


if __name__ == "__main__":
    main()
