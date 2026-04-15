from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.shared.common import ensure_dir, save_json, set_thread_env
from exp.shared.sem import build_benchmark_sets


def main() -> None:
    set_thread_env()
    ensure_dir(Path(__file__).resolve().parent / "logs")
    payload = build_benchmark_sets()
    save_json(__import__("pathlib").Path(__file__).resolve().parent / "results.json", payload)


if __name__ == "__main__":
    main()
