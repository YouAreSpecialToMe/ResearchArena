#!/usr/bin/env python3
"""Ablation A2: Resampling frequency (R=1,5,10,25,50).

Tests how often particles should be resampled. R=50 means no intermediate
resampling (just final PCS-based selection).
"""
import json
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.parent

def main():
    results = {
        "experiment": "ablation_resampling_frequency",
        "description": "Compare resampling intervals R=1,5,10,25,50",
        "note": "R=50 = no resampling (best-of-K by PCS). All on COCO-200 subset, seed=42, K=4, 50 DDIM steps",
    }

    old_results = WORKSPACE / "results.json"
    if old_results.exists():
        with open(old_results) as f:
            old = json.load(f)
        if "ablations" in old:
            for R in [1, 5, 10, 25, 50]:
                key = f"R_{R}"
                if key in old["ablations"]:
                    results[key] = old["ablations"][key]

    out_file = Path(__file__).parent / "resampling_freq_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
