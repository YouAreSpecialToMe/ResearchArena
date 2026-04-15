from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.core import (
    SEEDS,
    append_jsonl,
    family_rank_matrix,
    init_experiment,
    load_array,
    load_json,
    log_message,
    pair_indices,
    peak_memory_mb,
    save_csv,
    save_json,
    set_thread_env,
    utc_now_iso,
)


def pair_score(
    x: np.ndarray,
    seed: int,
    pair: tuple[int, int],
    method: str,
    setting: dict,
) -> float:
    if method == "scalar":
        sub_x = x[:, :, [pair[0], pair[1]]]
        bundle = family_rank_matrix(
            x=sub_x,
            family="coord",
            method="scalar",
            seed=seed,
            transform_seeds=setting["transform_seeds"],
            score_seeds=setting["score_seeds"]["coord"],
        )
        return float(np.sum((bundle["pits"][:, 0] - 0.5) ** 2))
    scores = []
    for offset, family in enumerate(["cov", "tail"]):
        bundle = family_rank_matrix(
            x=x,
            family=family,
            method=method,
            seed=seed + offset,
            target_pair=pair,
            transform_seeds=setting["transform_seeds"],
            score_seeds=setting["score_seeds"][family],
        )
        stat = np.sum((bundle["pits"][:, 0] - 0.5) ** 2)
        scores.append(float(stat))
    return float(max(scores))


def recovery_metrics(ranking: list[tuple[tuple[int, int], float]], truth: tuple[int, int]) -> dict:
    ordered = [pair for pair, _ in sorted(ranking, key=lambda item: item[1], reverse=True)]
    rank = ordered.index(truth) + 1
    return {
        "top1": float(rank == 1),
        "top3": float(rank <= 3),
        "mean_rank": float(rank),
    }


def main() -> None:
    start = time.perf_counter()
    set_thread_env()
    dirs = init_experiment("localization")
    log_path = dirs["logs_dir"] / "run_stage2.log"
    log_path.write_text("")
    log_message(log_path, "Starting pair-localization study with paired saved specifications.")
    rows = []
    for seed in SEEDS:
        spec = load_json(f"results/setup/seed_{seed}_spec.json")
        for d in [8, 16]:
            truth = (0, 1)
            all_pairs = pair_indices(d)
            for mode in ["ZeroPair", "FlipPair"]:
                setting = spec["settings"][f"block_localization_{mode}_d{d}"]
                bundle = {"x": load_array(setting["x_path"])}
                log_message(log_path, f"Seed {seed}: localization mode {mode} at d={d}.")
                rankings = {}
                for method in ["cosbc", "enriched", "scalar"]:
                    rankings[method] = [
                        (pair, pair_score(bundle["x"], seed + idx * 10, pair, method, setting))
                        for idx, pair in enumerate(all_pairs)
                    ]
                for method, ranking in rankings.items():
                    metrics = recovery_metrics(ranking, truth)
                    rows.append(
                        {
                            "experiment": "localization",
                            "seed": seed,
                            "benchmark": "block_pair_localization",
                            "dimension": d,
                            "condition": mode,
                            "method": method,
                            "approximation": mode,
                            "R": 100,
                            "M": 24,
                            "B": 199,
                            **metrics,
                            "chance_top1": 1.0 / len(all_pairs),
                            "chance_top3": min(3, len(all_pairs)) / len(all_pairs),
                        }
                    )
    df = pd.DataFrame(rows)
    save_csv(dirs["results_dir"] / "localization_metrics.csv", df)
    append_jsonl(dirs["runtime_dir"] / "run_manifest.jsonl", rows)
    payload = {
        "experiment": "localization",
        "created_at_utc": utc_now_iso(),
        "metrics": {
            "cosbc_top1_mean": float(df.loc[df["method"] == "cosbc", "top1"].mean()),
            "cosbc_top1_std": float(df.loc[df["method"] == "cosbc", "top1"].std(ddof=0)),
            "enriched_top1_mean": float(df.loc[df["method"] == "enriched", "top1"].mean()),
        },
        "peak_memory_mb": peak_memory_mb(),
        "runtime_minutes": (time.perf_counter() - start) / 60.0,
    }
    save_json(dirs["exp_dir"] / "results.json", payload)
    save_json(dirs["results_dir"] / "results.json", payload)
    save_json(dirs["logs_dir"] / "run_metadata.json", payload)
    log_message(log_path, "Localization stage completed.")


if __name__ == "__main__":
    main()
