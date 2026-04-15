import pandas as pd

from exp.shared.pipeline import run_subset_methods


if __name__ == "__main__":
    manifest = pd.read_csv("exp/data/benchmark_manifest.csv")
    run_subset_methods(manifest, tag="k8")
