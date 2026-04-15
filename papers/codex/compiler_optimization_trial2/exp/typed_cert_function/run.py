#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp" / "shared"))

from llvm_cleanup_harness import write_skipped_docs


if __name__ == "__main__":
    write_skipped_docs()
