from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from exp.shared.common import MAX_CPU_WORKERS, RunSpec, SEEDS, ensure_layout, write_json
from exp.shared.replay import ReplaySimulator, load_calibration


OVERLAP_FAMILIES = ["OverlapShift-2T", "ScanVsLoop-2T", "SQLiteTraceMix-2T", "SQLiteTraceMix-3T"]
TWO_TENANT_OVERLAP = ["OverlapShift-2T", "ScanVsLoop-2T", "SQLiteTraceMix-2T"]


def run_one(spec: RunSpec) -> dict:
    return ReplaySimulator(spec, load_calibration()).run()


def main() -> None:
    ensure_layout()
    calibration = load_calibration()
    manifest = calibration["manifest"]
    specs = []
    for family in OVERLAP_FAMILIES:
        cfg = manifest[family]
        for budget_name, budget_pages in cfg["budgets"].items():
            for seed in SEEDS:
                specs.append(RunSpec("ablations", family, budget_name, "ShareArb-NoDebt", seed, cfg["tenant_count"], str(Path("traces") / f"{family}__seed{seed}.json"), budget_pages))
                specs.append(RunSpec("ablations", family, budget_name, "ShareArb-UnitCost", seed, cfg["tenant_count"], str(Path("traces") / f"{family}__seed{seed}.json"), budget_pages, miss_cost_mode="UnitCost"))
                if budget_name == "medium":
                    specs.append(RunSpec("ablations", family, budget_name, "UniformSRV", seed, cfg["tenant_count"], str(Path("traces") / f"{family}__seed{seed}.json"), budget_pages, srv_mode="uniform"))
        for seed in SEEDS:
            for half_life, label in [(0.5, "ShareArb-HalfLife0.5"), (2.0, "ShareArb-HalfLife2.0")]:
                if family in TWO_TENANT_OVERLAP:
                    specs.append(RunSpec("ablations", family, "medium", label, seed, cfg["tenant_count"], str(Path("traces") / f"{family}__seed{seed}.json"), cfg["budgets"]["medium"], debt_half_life_turnovers=half_life))
    for family in ["DisjointPhase-2T", "OverlapShiftLowOverlap-2T"]:
        cfg = manifest[family]
        for budget_name, budget_pages in cfg["budgets"].items():
            for seed in SEEDS:
                specs.append(RunSpec("ablations", family, budget_name, "NoReduction", seed, cfg["tenant_count"], str(Path("traces") / f"{family}__seed{seed}.json"), budget_pages, reduction_enabled=False))
    with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
        rows = list(pool.map(run_one, specs))
    write_json(Path("exp/ablations/results.json"), {"runs": rows})


if __name__ == "__main__":
    main()
