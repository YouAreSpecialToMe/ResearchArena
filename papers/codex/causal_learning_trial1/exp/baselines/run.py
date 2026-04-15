import pandas as pd

from exp.shared.pipeline import run_baselines


if __name__ == "__main__":
    manifest = pd.read_csv("exp/data/benchmark_manifest.csv")
    run_baselines(manifest)
