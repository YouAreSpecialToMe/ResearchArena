from __future__ import annotations

import json
from pathlib import Path

from exp.shared.common import RunSpec, ensure_layout, write_json
from exp.shared.calibration import prepare_and_calibrate
from exp.shared.replay import ReplaySimulator, load_calibration


def main() -> None:
    ensure_layout()
    if not Path("calibration/manifest.json").exists():
        prepare_and_calibrate()
    calibration = load_calibration()
    manifest = calibration["manifest"]
    families = ["OverlapShift-2T", "ScanVsLoop-2T", "SQLiteTraceMix-2T", "SQLiteTraceMix-3T", "DisjointPhase-2T"]
    rows = []
    for family in families:
        trace_path = Path("traces") / f"{family}__seed11.json"
        spec = RunSpec(
            experiment="smoke",
            workload_family=family,
            cache_budget="medium",
            method="ShareArb",
            seed=11,
            tenant_count=manifest[family]["tenant_count"],
            trace_path=str(trace_path),
            budget_pages=manifest[family]["budgets"]["medium"],
        )
        metrics = ReplaySimulator(spec, calibration).run()
        rows.append(metrics)
    summary = {"runs": rows}
    write_json(Path("exp/smoke/results.json"), summary)


if __name__ == "__main__":
    main()
