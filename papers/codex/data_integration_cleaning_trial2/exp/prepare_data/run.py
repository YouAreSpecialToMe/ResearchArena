import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.pipeline import (
    ARTIFACTS_DIR,
    MAIN_SETTINGS,
    SEEDS,
    build_dataset_summary,
    choose_thresholds,
    prepare_one_setting,
    set_env_threads,
    write_json,
)


def main() -> None:
    set_env_threads()
    bundles = []
    for setting in MAIN_SETTINGS:
        base = setting.replace("_corrupted", "")
        corrupted = setting.endswith("_corrupted")
        for seed in SEEDS:
            bundles.append(prepare_one_setting(base, seed, corrupted))
    thresholds = choose_thresholds(bundles)
    for bundle in bundles:
        bundle.threshold = thresholds[bundle.setting]
    build_dataset_summary(bundles)
    manifest = [
        {
            "setting": b.setting,
            "seed": b.seed,
            "threshold": b.threshold,
            "validation_positives": len(b.validation_positives),
            "validation_negatives": len(b.validation_negatives),
            "final_positives": len(b.final_positives),
            "corruption_manifest_path": b.corruption_manifest_path,
        }
        for b in bundles
    ]
    write_json(ARTIFACTS_DIR / "prepared_manifest.json", manifest)


if __name__ == "__main__":
    main()
