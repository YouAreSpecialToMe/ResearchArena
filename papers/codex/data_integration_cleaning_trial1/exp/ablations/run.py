import time
from pathlib import Path

from exp.shared.core import (
    RESULTS,
    SEEDS,
    append_jsonl,
    ensure_dirs,
    runtime_info,
    save_predictions,
    search_programs,
    write_json,
    write_jsonl,
)


ABLATIONS = [
    ("t2d_sm_wh", "schema_strong", "remove_competitors"),
    ("wdc_products_medium", "entity_strong", "weaken_em"),
    ("t2d_sm_wh", "schema_strong", "generic"),
    ("wdc_products_medium", "entity_strong", "generic"),
    ("t2d_sm_wh", "schema_strong", "search_strength"),
    ("wdc_products_medium", "entity_strong", "search_strength"),
]


def safe_load_json(path: Path):
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


def collect_completed_rows() -> list[dict]:
    rows = []
    for benchmark, method, ablation in ABLATIONS:
        for seed in SEEDS:
            for mode in ["random", "targeted"]:
                run_dir = RESULTS / benchmark / "ablations" / ablation / method / f"seed_{seed}" / mode
                metrics = safe_load_json(run_dir / "metrics.json")
                perturb = safe_load_json(run_dir / "perturbations.json")
                config = safe_load_json(run_dir / "config.json")
                if not (metrics and perturb and config):
                    continue
                rows.append(
                    {
                        **config,
                        "worst_f1": metrics["f1"],
                        "clean_f1": safe_load_json(
                            RESULTS / benchmark / "clean" / method / f"seed_{seed}" / "metrics.json"
                        )["f1"],
                        "acceptance_rate": perturb["aux"]["acceptance_rate"],
                    }
                )
    return rows


def main() -> None:
    ensure_dirs()
    exp_dir = Path(__file__).resolve().parent
    log_path = exp_dir / "logs" / "execution.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    for benchmark, method, ablation in ABLATIONS:
        for seed in SEEDS:
            for mode in ["random", "targeted"]:
                run_dir = RESULTS / benchmark / "ablations" / ablation / method / f"seed_{seed}" / mode
                run_dir.mkdir(parents=True, exist_ok=True)
                config = {
                    "benchmark": benchmark,
                    "method": method,
                    "ablation": ablation,
                    "seed": seed,
                    "search_mode": mode,
                    "regime": "ABCA",
                    "eval_budget": 40,
                }
                write_json(run_dir / "config.json", config)
                start = time.time()
                append_jsonl(log_path, {"event": "start_run", **config})
                result = search_programs(
                    benchmark,
                    method,
                    seed,
                    "ABCA",
                    mode,
                    ablation=None if ablation == "search_strength" else ablation,
                )
                save_predictions(result["predictions"], run_dir / "predictions.parquet")
                write_json(run_dir / "metrics.json", result["worst_metrics"])
                write_json(run_dir / "runtime.json", runtime_info(start))
                write_json(
                    run_dir / "perturbations.json",
                    {
                        "best_program": result["best_program"],
                        "curve": result["curve"],
                        "aux": result["aux"],
                        "eval_budget": result["eval_budget"],
                    },
                )
                write_jsonl(run_dir / "admissibility_log.jsonl", result["search_trajectory"])
                append_jsonl(
                    log_path,
                    {
                        "event": "finish_run",
                        **config,
                        "clean_f1": result["clean_metrics"]["f1"],
                        "worst_f1": result["worst_metrics"]["f1"],
                    },
                )
    write_json(exp_dir / "results.json", {"experiment": "ablations", "rows": collect_completed_rows()})


if __name__ == "__main__":
    main()
