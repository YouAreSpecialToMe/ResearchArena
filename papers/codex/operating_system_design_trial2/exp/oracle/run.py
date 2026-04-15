from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from exp.shared.common import MAX_CPU_WORKERS, RunSpec, SEEDS, ensure_layout, write_json
from exp.shared.replay import ReplaySimulator, load_calibration


def run_one(spec: RunSpec) -> dict:
    return ReplaySimulator(spec, load_calibration()).run()


def main() -> None:
    ensure_layout()
    calibration = load_calibration()
    manifest = calibration["manifest"]
    specs = []
    for family in ["OverlapShift-2T", "SQLiteTraceMix-2T"]:
        cfg = manifest[family]
        for seed in SEEDS:
            specs.append(
                RunSpec(
                    experiment="oracle",
                    workload_family=family,
                    cache_budget="medium",
                    method="OracleOverlap",
                    seed=seed,
                    tenant_count=cfg["tenant_count"],
                    trace_path=str(Path("traces") / f"{family}__seed{seed}.json"),
                    budget_pages=cfg["budgets"]["medium"],
                    oracle=True,
                )
            )
    with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
        rows = list(pool.map(run_one, specs))
    write_json(Path("exp/oracle/results.json"), {"runs": rows})


if __name__ == "__main__":
    main()
