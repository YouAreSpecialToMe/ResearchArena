#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp" / "shared"))

from llvm_cleanup_harness import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "run", "--config", "throttle", "--mode", "benchmarks", "--repeats", "3"]
    main()
