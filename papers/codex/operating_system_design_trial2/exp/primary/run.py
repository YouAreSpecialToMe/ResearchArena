from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from exp.shared.common import MAX_CPU_WORKERS, RunSpec, SEEDS, ensure_layout, write_json
from exp.shared.replay import ReplaySimulator, load_calibration


PRIMARY_METHODS = ["PrivateOnly-Utility", "pCache-Account", "pCache-Account+Policy", "ShareArb"]


def run_one(spec: RunSpec) -> dict:
    return ReplaySimulator(spec, load_calibration()).run()


def main() -> None:
    ensure_layout()
    calibration = load_calibration()
    manifest = calibration["manifest"]
    specs = []
    for family, family_cfg in manifest.items():
        if "LowOverlap" in family:
            continue
        for budget_name, budget_pages in family_cfg["budgets"].items():
            for seed in SEEDS:
                for method in PRIMARY_METHODS:
                    specs.append(
                        RunSpec(
                            experiment="primary",
                            workload_family=family,
                            cache_budget=budget_name,
                            method=method,
                            seed=seed,
                            tenant_count=family_cfg["tenant_count"],
                            trace_path=str(Path("traces") / f"{family}__seed{seed}.json"),
                            budget_pages=budget_pages,
                        )
                    )
    with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
        rows = list(pool.map(run_one, specs))
    write_json(Path("exp/primary/results.json"), {"runs": rows})


if __name__ == "__main__":
    main()
