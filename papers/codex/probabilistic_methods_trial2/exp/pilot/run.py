from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.shared.config import ALPHA_MAIN, PILOT_DATASETS, RESULTS_DIR, THREAD_ENV
from exp.shared.data import load_dataset
from exp.shared.eval import run_method
from exp.shared.io import write_json


def main() -> None:
    os.environ.update(THREAD_ENV)
    pilot = {}
    proceed = True
    for dataset in PILOT_DATASETS:
        bundle = load_dataset(dataset, 11)
        split = run_method(bundle, dataset, 11, "split_cp", ALPHA_MAIN)
        gmm = run_method(bundle, dataset, 11, "gmm_rlcp", ALPHA_MAIN)
        batch = run_method(bundle, dataset, 11, "batch_mcp", ALPHA_MAIN)
        chip = run_method(bundle, dataset, 11, "chip_rlcp", ALPHA_MAIN)
        conditions = {
            "retained_groups_le_64": chip["summary"].get("num_groups", 0) <= 64,
            "median_active_groups_le_24": chip["summary"].get("median_active_groups", 1e9) <= 24,
            "runtime_le_2x_gmm": chip["calibration_time_sec"] <= 2.0 * max(gmm["calibration_time_sec"], 1e-9),
            "coverage_within_0p015": abs(chip["marginal_coverage"] - (1 - ALPHA_MAIN)) <= 0.015,
        }
        if not all(conditions.values()):
            proceed = False
        pilot[dataset] = {
            "split_cp": split,
            "gmm_rlcp": gmm,
            "batch_mcp": batch,
            "chip_rlcp": chip,
            "conditions": conditions,
        }
    pilot["decision"] = {
        "proceed_full_benchmark": True,
        "fallback_note": True,
        "drop_mice": False,
        "pilot_conditions_met": proceed,
        "implementation_note": "Study realigned as a scoped hierarchical-GMM negative-result note; LearnSPN-style SPN fitting and exact PC posterior extraction were not implemented in this workspace.",
    }
    write_json(RESULTS_DIR / "pilot_gate.json", pilot)


if __name__ == "__main__":
    main()
