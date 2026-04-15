import pandas as pd

from exp.shared.pipeline import build_all_local_banks


if __name__ == "__main__":
    manifest = pd.read_csv("exp/data/benchmark_manifest.csv")
    build_all_local_banks(manifest, k=8, tag="k8")
