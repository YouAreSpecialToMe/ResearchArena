from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.common import ensure_dir, save_json, set_thread_env


def main() -> None:
    set_thread_env()
    out_dir = ensure_dir(__import__("pathlib").Path(__file__).resolve().parent)
    logs_dir = ensure_dir(out_dir / "logs")
    payload = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "thread_env": {k: os.environ[k] for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "nproc": str(len(os.sched_getaffinity(0))),
        "nproc_shell": subprocess.check_output(["bash", "-lc", "nproc"], text=True).strip(),
        "memory": subprocess.check_output(["bash", "-lc", "free -h | sed -n '2p'"], text=True).strip(),
        "planned_cpu_workers": 2,
        "logs_dir": str(logs_dir),
    }
    save_json(out_dir / "results.json", payload)


if __name__ == "__main__":
    main()
