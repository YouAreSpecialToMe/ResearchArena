from __future__ import annotations

import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PAGE_SIZE = 4096
SEEDS = [11, 17, 23]
TRACE_MODES = ["ExtendedHinted", "CompactState", "NoDirty", "AccessOnly"]
ABLATION_TRACE_MODES = ["NoReclaim"]
POLICIES = ["LinuxDefault", "FIFO", "CLOCK", "LFU", "S3FIFO", "Hyperbolic"]
WORKLOADS = ["stream_scan", "sqlite_zipf", "filebench_fileserver"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_layout() -> dict[str, Path]:
    root = repo_root()
    paths = {
        "artifacts": ensure_dir(root / "artifacts"),
        "datasets": ensure_dir(root / "artifacts" / "datasets"),
        "env": ensure_dir(root / "artifacts" / "env"),
        "logs": ensure_dir(root / "artifacts" / "logs"),
        "traces_extended": ensure_dir(root / "artifacts" / "traces" / "extended"),
        "traces_compact": ensure_dir(root / "artifacts" / "traces" / "compact"),
        "replay": ensure_dir(root / "artifacts" / "replay"),
        "live": ensure_dir(root / "artifacts" / "live"),
        "tables": ensure_dir(root / "artifacts" / "tables"),
        "plots": ensure_dir(root / "artifacts" / "plots"),
        "figures": ensure_dir(root / "figures"),
        "exp": ensure_dir(root / "exp"),
    }
    for name in [
        "environment_config",
        "data_preparation",
        "main_ranking_study",
        "reference_stability",
        "live_anchors",
        "audits",
        "visualization",
    ]:
        ensure_dir(paths["exp"] / name)
        ensure_dir(paths["exp"] / name / "logs")
    return paths


def write_json(path: Path, obj: Any) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / float(1024**3)


def pages_to_gb(num_pages: float) -> float:
    return num_pages * PAGE_SIZE / float(1024**3)


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return {"mean": float(mean), "std": float(math.sqrt(var))}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def bootstrap_ci(values: list[float], seed: int = 0, rounds: int = 1000) -> dict[str, float]:
    import random

    if not values:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    return {
        "mean": float(sum(values) / len(values)),
        "ci_low": percentile(means, 0.025),
        "ci_high": percentile(means, 0.975),
    }


def paired_bootstrap_ci(
    left: list[float],
    right: list[float],
    seed: int = 0,
    rounds: int = 1000,
) -> dict[str, float]:
    import random

    if not left or not right or len(left) != len(right):
        return {"mean_delta": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = random.Random(seed)
    deltas = [float(a - b) for a, b in zip(left, right)]
    samples = []
    for _ in range(rounds):
        indices = [rng.randrange(len(deltas)) for _ in range(len(deltas))]
        sample = [deltas[idx] for idx in indices]
        samples.append(sum(sample) / len(sample))
    return {
        "mean_delta": float(sum(deltas) / len(deltas)),
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
    }


def detect_peak_rss_mb() -> float:
    scale = 1024.0
    if sys.platform == "darwin":
        scale = 1024.0 * 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def capture_package_versions() -> dict[str, Any]:
    packages = {}
    for name in ["numpy", "pandas", "scipy", "pyarrow", "matplotlib", "seaborn", "psutil"]:
        try:
            mod = __import__(name)
            packages[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            packages[name] = f"MISSING:{type(exc).__name__}"
    tools = {}
    for name in ["fio", "filebench", "hyperfine"]:
        path = shutil.which(name)
        tools[name] = path if path else "MISSING"
    return {
        "python": platform.python_version(),
        "sqlite3": command_output(
            [
                sys.executable,
                "-c",
                "import sqlite3; print(sqlite3.sqlite_version)",
            ]
        ),
        "packages": packages,
        "tools": tools,
    }


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def run_logged(path: Path, argv: list[str]) -> int:
    with path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False)
    return proc.returncode


def stage_log_path(stage_name: str) -> Path:
    paths = ensure_layout()
    return paths["exp"] / stage_name / "logs" / "run.log"


def reset_stage_log(stage_name: str, header: str) -> Path:
    path = stage_log_path(stage_name)
    save_text(path, f"{header}\n")
    return path


def append_stage_log(stage_name: str, message: str) -> None:
    path = stage_log_path(stage_name)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{now_ts()} {message}\n")


def system_report() -> str:
    uname = command_output(["uname", "-a"])
    mount = command_output(["mount"])
    cpu = ""
    mem = ""
    try:
        cpu = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except Exception:
        pass
    try:
        mem = Path("/proc/meminfo").read_text(encoding="utf-8")
    except Exception:
        pass
    return "\n".join(
        [
            f"timestamp: {now_ts()}",
            f"uname: {uname}",
            f"python: {platform.python_version()}",
            f"filesystem_mounts:\n{mount}",
            f"cpuinfo:\n{cpu}",
            f"meminfo:\n{mem}",
        ]
    )
