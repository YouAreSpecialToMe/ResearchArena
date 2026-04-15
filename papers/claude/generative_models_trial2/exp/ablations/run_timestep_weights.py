#!/usr/bin/env python3
"""Ablation A3: Timestep weighting for PCS (uniform, mid, early, late emphasis).

Tests whether emphasizing PCS at certain timestep ranges improves quality signal.
Reuses particles from main experiment (post-hoc reweighting).
"""
import json
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.parent

def main():
    results = {
        "experiment": "ablation_timestep_weights",
        "description": "Compare PCS weighting: uniform, mid_emphasis, early_emphasis, late_emphasis",
        "note": "Post-hoc reweighting of same trajectories. COCO-300, seed=42, K=4, 50 DDIM steps",
    }

    old_results = WORKSPACE / "results.json"
    if old_results.exists():
        with open(old_results) as f:
            old = json.load(f)
        if "ablations" in old:
            for w in ["uniform", "mid_emphasis", "early_emphasis", "late_emphasis"]:
                key = f"weight_{w}"
                if key in old["ablations"]:
                    results[key] = old["ablations"][key]

    out_file = Path(__file__).parent / "timestep_weights_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
