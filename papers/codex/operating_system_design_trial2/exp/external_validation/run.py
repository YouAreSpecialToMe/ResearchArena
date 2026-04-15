from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from exp.shared.common import MAX_CPU_WORKERS, RunSpec, SEEDS, WINDOW, ensure_layout, write_json
from exp.shared.replay import ReplaySimulator, load_calibration
from exp.shared.workloads import build_sqlite_external_trace


VARIANTS = ["Raw", "Bursty"]
BUDGETS = ["tight", "medium"]
METHODS = ["pCache-Account", "ShareArb"]


def _active_pages(events: list[dict]) -> int:
    counts = []
    for start in range(0, len(events), WINDOW):
        window = events[start : start + WINDOW]
        counts.append(len({(event["file_id"], event["page_id"]) for event in window}))
    return max(1, int(sum(counts) / max(1, len(counts))))


def build_specs() -> list[RunSpec]:
    calibration = load_calibration()
    sqlite_cal = calibration["manifest"]["SQLiteTraceMix-2T"]
    specs = []
    for seed in SEEDS:
        for variant in VARIANTS:
            payload = build_sqlite_external_trace(seed, variant)
            family = payload["family"]
            active_pages = _active_pages(payload["events"])
            budgets = {
                "tight": max(16, int(round(active_pages * 0.35))),
                "medium": max(16, int(round(active_pages * 0.55))),
            }
            for budget_name in BUDGETS:
                for method in METHODS:
                    specs.append(
                        RunSpec(
                            experiment="external_validation",
                            workload_family=family,
                            cache_budget=budget_name,
                            method=method,
                            seed=seed,
                            tenant_count=2,
                            trace_path=str(Path("traces") / f"{family}__seed{seed}.json"),
                            budget_pages=budgets[budget_name],
                            input_artifacts=["calibration/manifest.json"],
                        )
                    )
    return specs


def run_one(spec: RunSpec) -> dict:
    calibration = load_calibration()
    calibration["latency_by_family"][spec.workload_family] = calibration["latency_by_family"]["SQLiteTraceMix-2T"]
    calibration["isolated_avg_latency"][spec.workload_family] = calibration["isolated_avg_latency"]["SQLiteTraceMix-2T"]
    calibration["manifest"][spec.workload_family] = {
        **calibration["manifest"]["SQLiteTraceMix-2T"],
        "workload_family": spec.workload_family,
        "budgets": {spec.cache_budget: spec.budget_pages},
    }
    return ReplaySimulator(spec, calibration).run()


def main() -> None:
    ensure_layout()
    specs = build_specs()
    with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
        rows = list(pool.map(run_one, specs))
    write_json(Path("exp/external_validation/results.json"), {"runs": rows})


if __name__ == "__main__":
    main()
