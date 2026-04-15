from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.benchmark_spec import CURRENT_ENV, ITEMS, OLD_ENV
from exp.shared.build_benchmark import main as build_benchmark
from exp.shared.utils import ROOT, environment_metadata, run, write_json


def ensure_old_env() -> None:
    if not Path(OLD_ENV).exists():
        run(["uv", "venv", ".venv_old", "--python", "3.10"], cwd=ROOT, timeout=600)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            ".venv_old/bin/python",
            "pandas==1.5.3",
            "scikit-learn==1.1.3",
            "numpy==1.26.4",
            "scipy==1.11.4",
            "requests==2.32.5",
            "rank-bm25==0.2.2",
            "matplotlib==3.10.7",
            "seaborn==0.13.2",
        ],
        cwd=ROOT,
        timeout=1800,
    )


def execute_versions() -> None:
    for item in ITEMS:
        item_dir = ROOT / "benchmark" / "items" / item["item_id"]
        print(f"[benchmark_build] old_run item={item['item_id']}", flush=True)
        run([OLD_ENV, str(item_dir / "harness.py"), "--output", str(item_dir / "results_old.json")], cwd=ROOT)
        print(f"[benchmark_build] current_run item={item['item_id']}", flush=True)
        run([CURRENT_ENV, str(item_dir / "harness.py"), "--output", str(item_dir / "results_current.json")], cwd=ROOT)
        if item["status"] == "needs_update":
            print(f"[benchmark_build] repair_run item={item['item_id']}", flush=True)
            run(
                [
                    CURRENT_ENV,
                    str(item_dir / "harness.py"),
                    "--code-file",
                    str(item_dir / "reference_repair.py"),
                    "--output",
                    str(item_dir / "results_repair.json"),
                ],
                cwd=ROOT,
            )


def write_env_config() -> None:
    meta = environment_metadata()
    meta["package_versions"] = {}
    for pkg in [
        "torch",
        "vllm",
        "transformers",
        "accelerate",
        "pandas",
        "scikit-learn",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "rank-bm25",
        "jsonschema",
        "pytest",
        "requests",
    ]:
        try:
            meta["package_versions"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            meta["package_versions"][pkg] = None
    write_json(ROOT / "results" / "env_config.json", meta)


def main() -> None:
    print("[benchmark_build] rebuilding benchmark artifact", flush=True)
    build_benchmark()
    ensure_old_env()
    execute_versions()
    write_env_config()
    build_result = {
        "experiment": "benchmark_build",
        "config": {
            "current_env": CURRENT_ENV,
            "old_env": OLD_ENV,
            "item_count": len(ITEMS),
        },
        "metrics": {"item_count": len(ITEMS)},
    }
    write_json(ROOT / "exp" / "benchmark_build" / "results.json", build_result)
    log_dir = ROOT / "exp" / "benchmark_build" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "latest_run_summary.log").write_text(
        "\n".join(
            [
                "experiment=benchmark_build",
                "action=rebuild_benchmark_and_execute",
                f"current_env={CURRENT_ENV}",
                f"old_env={OLD_ENV}",
                f"item_count={len(ITEMS)}",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
