from __future__ import annotations

from exp.shared.pipeline import (
    build_all_local_banks,
    compile_root_results,
    generate_all_datasets,
    run_ablations,
    run_baselines,
    run_environment_manifest,
    run_evaluation,
    run_subset_methods,
)


def main() -> None:
    run_environment_manifest()
    manifest = generate_all_datasets()
    subset_validity = build_all_local_banks(manifest, k=8, tag="k8")
    baselines = run_baselines(manifest)
    main_df = run_subset_methods(manifest, tag="k8")
    ablations = run_ablations(manifest)
    evaluation = run_evaluation(manifest, subset_validity, baselines, main_df, ablations)
    compile_root_results(manifest, subset_validity, baselines, main_df, ablations, evaluation)


if __name__ == "__main__":
    main()
