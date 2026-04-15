import pandas as pd

from exp.shared.pipeline import compile_root_results, run_evaluation


if __name__ == "__main__":
    manifest = pd.read_csv("exp/data/benchmark_manifest.csv")
    subset_validity = pd.read_csv("exp/evaluation/subset_validity_summary_k8.csv")
    baselines = pd.read_csv("exp/baselines/summary.csv")
    main = pd.read_csv("exp/main/summary.csv")
    ablations = pd.read_csv("exp/ablations/summary.csv")
    evaluation = run_evaluation(manifest, subset_validity, baselines, main, ablations)
    compile_root_results(manifest, subset_validity, baselines, main, ablations, evaluation)
