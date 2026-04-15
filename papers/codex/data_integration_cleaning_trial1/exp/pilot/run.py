import time
from pathlib import Path

from exp.shared.core import ensure_dirs, search_programs, write_json


PILOT_RUNS = [
    ("t2d_sm_wh", "schema_simple", 13, "ABCA", "targeted"),
    ("t2d_sm_wh", "schema_simple", 13, "naive", "targeted"),
    ("t2d_sm_wh", "schema_strong", 13, "ABCA", "targeted"),
    ("t2d_sm_wh", "schema_strong", 13, "naive", "targeted"),
    ("wdc_products_medium", "entity_simple", 13, "ABCA", "targeted"),
    ("wdc_products_medium", "entity_simple", 13, "naive", "targeted"),
    ("wdc_products_medium", "entity_strong", 13, "ABCA", "targeted"),
    ("wdc_products_medium", "entity_strong", 13, "naive", "targeted"),
]


def main() -> None:
    ensure_dirs()
    exp_dir = Path(__file__).resolve().parent
    rows = []
    start = time.time()
    for benchmark, method, seed, regime, search_mode in PILOT_RUNS:
        run_start = time.time()
        result = search_programs(benchmark, method, seed, regime, search_mode)
        rows.append(
            {
                "benchmark": benchmark,
                "method": method,
                "seed": seed,
                "regime": regime,
                "search_mode": search_mode,
                "clean_f1": result["clean_metrics"]["f1"],
                "worst_f1": result["worst_metrics"]["f1"],
                "absolute_f1_drop": result["clean_metrics"]["f1"] - result["worst_metrics"]["f1"],
                "acceptance_rate": result["aux"]["acceptance_rate"],
                "wall_clock_minutes": (time.time() - run_start) / 60.0,
            }
        )
    write_json(
        exp_dir / "results.json",
        {
            "experiment": "pilot",
            "status": "completed",
            "planned_fraction_of_main_runs": 0.10,
            "actual_run_count": len(rows),
            "rows": rows,
            "wall_clock_minutes_total": (time.time() - start) / 60.0,
            "simplification_decision": "drop random-search runs only for simple baselines",
        },
    )


if __name__ == "__main__":
    main()
