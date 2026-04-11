#!/usr/bin/env python3
"""Submit ICLR 2025 baseline papers to Stanford AI Reviewer via SLURM.

Submits 3 groups: accepted (100), rejected (100), random (100).
Distributes across SLURM nodes for IP diversity.

Usage:
    python scripts/submit_iclr_baseline.py [--dry-run] [--batch-size 5] [--group accepted|rejected|random|all]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "analysis" / "iclr2025_baseline"
LOG_DIR = OUTDIR / "stanford_reviews"

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=iclr_{group}_{idx:03d}
#SBATCH --partition=default_partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --time=01:00:00
#SBATCH --output={log_dir}/{group}/submit_{idx:03d}_%j.out
#SBATCH --error={log_dir}/{group}/submit_{idx:03d}_%j.err

cd {base}
pip install playwright -q 2>/dev/null
playwright install chromium 2>/dev/null

python -u scripts/stanford_submit_one.py "{pdf_path}" {global_idx} --venue iclr --agent iclr2025_{group} --seed "{area}" --trial "{decision}"
"""


def submit_group(group, batch_size, start_from, dry_run):
    manifest_path = OUTDIR / f"manifest_{group}.json"
    if not manifest_path.exists():
        print(f"No manifest for {group}: {manifest_path}")
        return

    papers = json.loads(manifest_path.read_text())
    print(f"\n=== {group.upper()}: {len(papers)} papers ===")

    group_log = LOG_DIR / group
    group_log.mkdir(parents=True, exist_ok=True)
    script_dir = group_log / "slurm_scripts"
    script_dir.mkdir(exist_ok=True)

    if dry_run:
        for p in papers[:5]:
            print(f"  [{p['index']:3d}] score={p['avg_score']:.1f} | {p['title'][:55]}")
        print(f"  ... ({len(papers)} total)")
        return

    submitted = 0
    for i, p in enumerate(papers):
        if i < start_from:
            continue

        idx = p["index"]
        # Global index for unique tracking across groups
        global_idx = {"accepted": 0, "rejected": 100, "random": 200}[group] + idx

        script_content = SLURM_TEMPLATE.format(
            group=group,
            idx=idx,
            global_idx=global_idx,
            log_dir=str(LOG_DIR),
            base=str(BASE),
            pdf_path=p["pdf_path"],
            area=p.get("area", "unknown"),
            decision=p["decision"],
        )

        script_path = script_dir / f"submit_{idx:03d}.sh"
        script_path.write_text(script_content)

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True,
        )
        job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else "FAIL"
        print(f"  [{group}_{idx:03d}] {job_id} | {p['title'][:45]}")
        submitted += 1

        if submitted % batch_size == 0 and i < len(papers) - 1:
            print(f"  --- Batch done, waiting 5min ---")
            time.sleep(300)

    print(f"  Submitted {submitted} jobs for {group}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--group", default="all", choices=["accepted", "rejected", "random", "all"])
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    groups = ["accepted", "rejected", "random"] if args.group == "all" else [args.group]
    for group in groups:
        submit_group(group, args.batch_size, args.start_from, args.dry_run)


if __name__ == "__main__":
    main()
