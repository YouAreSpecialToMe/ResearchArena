from __future__ import annotations

import json
import os
import random
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any
from math import sqrt


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmark"
ITEMS_DIR = BENCHMARK_DIR / "items"
RESULTS_DIR = ROOT / "results"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True, timeout=timeout)


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


def bootstrap_diff(values_a: list[int], values_b: list[int], n_boot: int = 5000, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idxs = [rng.randrange(len(values_a)) for _ in range(len(values_a))]
        a = statistics.mean(values_a[i] for i in idxs)
        b = statistics.mean(values_b[i] for i in idxs)
        diffs.append(a - b)
    diffs.sort()
    return {
        "mean_diff": statistics.mean(diffs),
        "ci_low": diffs[int(0.025 * len(diffs))],
        "ci_high": diffs[int(0.975 * len(diffs))],
    }


def summarize_seed_metric(per_seed: dict[int, float]) -> dict[str, Any]:
    values = list(per_seed.values())
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"per_seed": per_seed, "mean": statistics.mean(values), "std": std}


def wilson_interval(successes: int, total: int, confidence_z: float = 1.96) -> dict[str, float]:
    if total == 0:
        return {"successes": 0, "total": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    phat = successes / total
    denom = 1 + (confidence_z**2 / total)
    center = (phat + confidence_z**2 / (2 * total)) / denom
    margin = confidence_z * sqrt((phat * (1 - phat) + confidence_z**2 / (4 * total)) / total) / denom
    return {
        "successes": successes,
        "total": total,
        "rate": phat,
        "ci_low": max(0.0, center - margin),
        "ci_high": min(1.0, center + margin),
    }


def package_versions() -> dict[str, str]:
    packages = [
        "accelerate",
        "jsonschema",
        "matplotlib",
        "numpy",
        "pandas",
        "pytest",
        "rank-bm25",
        "requests",
        "scikit-learn",
        "scipy",
        "seaborn",
        "torch",
        "transformers",
        "vllm",
    ]
    versions = {}
    for package in packages:
        module_name = {"scikit-learn": "sklearn", "rank-bm25": "rank_bm25"}.get(package, package)
        try:
            module = __import__(module_name)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[package] = "unavailable"
    return versions


def environment_metadata() -> dict[str, Any]:
    import platform
    import sys

    cpu_count = os.cpu_count()
    py = run(["bash", "-lc", f"{sys.executable} --version"])
    free_h = run(["bash", "-lc", "free -h"])
    nvidia = run(["bash", "-lc", "nvidia-smi"])
    git = run(["bash", "-lc", "git rev-parse HEAD || true"])
    gpu_report = nvidia.stdout.strip()
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", gpu_report)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", gpu_report)
    gpu_name_match = re.search(r"\|\s*\d+\s+([A-Za-z0-9 ].*?)\s{2,}\|", gpu_report)
    vram_match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", gpu_report)
    return {
        "current_date": "2026-03-21",
        "os": platform.platform(),
        "python": py.stdout.strip(),
        "python_executable": sys.executable,
        "cpu_count": cpu_count,
        "effective_cpu_budget": 4,
        "gpu_model": gpu_name_match.group(1).strip() if gpu_name_match else None,
        "gpu_vram_total_mib": int(vram_match.group(2)) if vram_match else None,
        "cuda_version": cuda_match.group(1) if cuda_match else None,
        "driver_version": driver_match.group(1) if driver_match else None,
        "ram_report": free_h.stdout.strip(),
        "gpu_report": gpu_report,
        "git_commit": git.stdout.strip(),
        "package_versions": package_versions(),
    }


def mean_latency(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(r.get("latency_sec", 0.0) for r in records) / len(records)


def extract_code_blocks(html: str) -> list[str]:
    return re.findall(r"<pre><code>(.*?)</code></pre>", html, flags=re.S)


def now() -> float:
    return time.time()


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())
