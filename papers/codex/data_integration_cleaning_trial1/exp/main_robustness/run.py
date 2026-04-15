import time
from pathlib import Path

from exp.shared.core import (
    RESULTS,
    SEEDS,
    append_jsonl,
    ensure_dirs,
    search_programs,
    write_run_artifacts,
    write_json,
)


BENCHMARK_METHODS = {
    "t2d_sm_wh": ["schema_simple", "schema_strong"],
    "wdc_products_medium": ["entity_simple", "entity_strong"],
}
REGIMES = ["ABCA", "format", "naive"]


def safe_load_json(path: Path):
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


def collect_completed_rows() -> list[dict]:
    rows = []
    clean_lookup = {}
    for benchmark, methods in BENCHMARK_METHODS.items():
        for method in methods:
            for seed in SEEDS:
                clean_metrics = safe_load_json(RESULTS / benchmark / "clean" / method / f"seed_{seed}" / "metrics.json")
                if clean_metrics:
                    clean_lookup[(benchmark, method, seed)] = clean_metrics.get("f1", 0.0)
    for benchmark, methods in BENCHMARK_METHODS.items():
        for method in methods:
            for seed in SEEDS:
                for regime in REGIMES:
                    modes = ["targeted"] if method.endswith("simple") else ["random", "targeted"]
                    for mode in modes:
                        run_dir = RESULTS / benchmark / regime / method / f"seed_{seed}" / mode
                        metrics = safe_load_json(run_dir / "metrics.json")
                        perturb = safe_load_json(run_dir / "perturbations.json")
                        config = safe_load_json(run_dir / "config.json")
                        if not (metrics and perturb and config):
                            continue
                        clean_f1 = clean_lookup.get((benchmark, method, seed), 0.0)
                        rows.append(
                            {
                                **config,
                                "clean_f1": clean_f1,
                                "worst_f1": metrics["f1"],
                                "absolute_f1_drop": clean_f1 - metrics["f1"],
                                "relative_f1_drop": (clean_f1 - metrics["f1"]) / max(clean_f1, 1e-9),
                                "acceptance_rate": perturb["aux"]["acceptance_rate"],
                                "accepted_programs": perturb["aux"]["accepted_programs"],
                                "rejected_programs": perturb["aux"]["rejected_programs"],
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
    for benchmark, methods in BENCHMARK_METHODS.items():
        for method in methods:
            for seed in SEEDS:
                for regime in REGIMES:
                    modes = ["targeted"] if method.endswith("simple") else ["random", "targeted"]
                    for mode in modes:
                        run_dir = RESULTS / benchmark / regime / method / f"seed_{seed}" / mode
                        run_dir.mkdir(parents=True, exist_ok=True)
                        config = {
                            "benchmark": benchmark,
                            "method": method,
                            "seed": seed,
                            "regime": regime,
                            "search_mode": mode,
                            "eval_budget": 40,
                            "program_length_max": 3,
                            "beam_width": 8 if mode == "targeted" else None,
                        }
                        start = time.time()
                        append_jsonl(log_path, {"event": "start_run", **config})
                        result = search_programs(benchmark, method, seed, regime, mode)
                        write_run_artifacts(run_dir, config, result, start)
                        append_jsonl(
                            log_path,
                            {
                                "event": "finish_run",
                                **config,
                                "clean_f1": result["clean_metrics"]["f1"],
                                "worst_f1": result["worst_metrics"]["f1"],
                                "acceptance_rate": result["aux"]["acceptance_rate"],
                            },
                        )
    summary = collect_completed_rows()
    write_json(
        exp_dir / "results.json",
        {
            "experiment": "main_robustness",
            "rows": summary,
            "pilot_simplification": "Random-search runs were dropped only for simple baselines; retained simple targeted runs now use the full 40-evaluation budget.",
        },
    )


if __name__ == "__main__":
    main()
