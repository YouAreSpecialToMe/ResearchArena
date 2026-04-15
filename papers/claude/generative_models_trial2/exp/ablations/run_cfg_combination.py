#!/usr/bin/env python3
"""Ablation A5: CFG scale combination (3.0, 7.5, 12.0, 20.0 with/without CoPS).

Tests whether CoPS provides additive improvement at different CFG scales.
"""
import json
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.parent

def main():
    results = {
        "experiment": "ablation_cfg_combination",
        "description": "Compare standard vs CoPS at CFG scales 3.0, 7.5, 12.0, 20.0",
        "note": "COCO-200 subset, seed=42, K=4, 50 DDIM steps, R=10",
    }

    old_results = WORKSPACE / "results.json"
    if old_results.exists():
        with open(old_results) as f:
            old = json.load(f)
        if "ablations" in old:
            for cfg in [3.0, 7.5, 12.0, 20.0]:
                for mode in ["std", "cops"]:
                    key = f"cfg{cfg}_{mode}"
                    if key in old["ablations"]:
                        results[key] = old["ablations"][key]

    out_file = Path(__file__).parent / "cfg_combination_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
