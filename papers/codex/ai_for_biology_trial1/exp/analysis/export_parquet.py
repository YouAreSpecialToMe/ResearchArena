from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared import config
from exp.shared.data import prepare_dataset_split


def main() -> None:
    pred_rows = []
    for stage in ["baseline_ladder", "retrieval_models"]:
        for path in sorted((ROOT / "exp" / stage / "predictions").glob("*.npz")):
            blob = np.load(path, allow_pickle=True)
            stem = path.stem
            dataset, rest = stem.split("_seed", 1)
            seed_str, model_slug = rest.split("_", 1)
            labels = blob["labels"].tolist()
            for i, pert in enumerate(labels):
                pred_rows.append(
                    {
                        "stage": stage,
                        "dataset": dataset,
                        "seed": int(seed_str),
                        "model": model_slug.replace("_", " "),
                        "perturbation": pert,
                        "prediction": json.dumps(blob["predictions"][i].tolist()),
                        "truth": json.dumps(blob["true"][i].tolist()),
                    }
                )
    pred_df = pd.DataFrame(pred_rows)
    out_dir = ROOT / "exp" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_dir / "per_perturbation_predictions.parquet", index=False)

    meta_rows = []
    for dataset in config.DATASETS:
        for seed in config.SEEDS:
            split = prepare_dataset_split(dataset, seed)
            idx = split.retrieval_cache_test["indices"]
            sims = split.retrieval_cache_test["similarities"]
            for i, pert in enumerate(split.test_perts):
                meta_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "perturbation": pert,
                        "neighbors_top20": json.dumps([split.train_perts[j] for j in idx[i].tolist()]),
                        "similarities_top20": json.dumps([float(x) for x in sims[i].tolist()]),
                    }
                )
    pd.DataFrame(meta_rows).to_parquet(out_dir / "retrieval_metadata.parquet", index=False)


if __name__ == "__main__":
    main()
