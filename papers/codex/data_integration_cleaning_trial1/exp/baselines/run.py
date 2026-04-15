import time
from pathlib import Path

from exp.shared.core import (
    RESULTS,
    SEEDS,
    append_jsonl,
    ensure_dirs,
    evaluate_entity,
    evaluate_schema,
    package_versions,
    runtime_info,
    save_predictions,
    system_info,
    write_json,
)


def main() -> None:
    ensure_dirs()
    exp_dir = Path(__file__).resolve().parent
    log_path = exp_dir / "logs" / "execution.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    aggregate = {}
    for seed in SEEDS:
        schema_start = time.time()
        schema = evaluate_schema(seed)
        schema_runtime = {**runtime_info(schema_start), "system_info": system_info(), "package_versions": package_versions()}
        entity_start = time.time()
        entity = evaluate_entity(seed)
        entity_runtime = {**runtime_info(entity_start), "system_info": system_info(), "package_versions": package_versions()}
        for benchmark, runs in [("t2d_sm_wh", schema), ("wdc_products_medium", entity)]:
            for method, payload in runs.items():
                run_dir = RESULTS / benchmark / "clean" / method / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                save_predictions(payload["predictions"], run_dir / "predictions.parquet")
                write_json(run_dir / "metrics.json", payload["metrics"])
                write_json(run_dir / "config.json", {"benchmark": benchmark, "method": method, "seed": seed, **payload["config"]})
                write_json(run_dir / "runtime.json", schema_runtime if benchmark == "t2d_sm_wh" else entity_runtime)
                append_jsonl(log_path, {"benchmark": benchmark, "method": method, "seed": seed, "metrics": payload["metrics"]})
                aggregate.setdefault(method, []).append(payload["metrics"])
        write_json(RESULTS / "clean_seed_status.json", {"last_completed_seed": seed})
    for method, rows in aggregate.items():
        keys = sorted(rows[0].keys())
        write_json(
            exp_dir / f"{method}_results.json",
            {
                "experiment": "baselines",
                "method": method,
                "metrics": {
                    k: {
                        "mean": sum(r[k] for r in rows) / len(rows),
                        "std": float((sum((r[k] - (sum(x[k] for x in rows) / len(rows))) ** 2 for r in rows) / len(rows)) ** 0.5),
                    }
                    for k in keys
                },
            },
        )


if __name__ == "__main__":
    main()
